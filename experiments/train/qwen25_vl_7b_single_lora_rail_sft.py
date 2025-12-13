import os
import math
import random
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torch.nn.functional as F_nn

from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    Qwen2_5_VLForConditionalGeneration,
    EarlyStoppingCallback,
    TrainerCallback,
)
import wandb
import logging
import sys
import gc
import json
import csv
from sklearn.metrics import mean_squared_error

# utils (assumed available in your project)
from utils.data_process import DIMENSION_LIST
from utils.prompt_lib import qwen_prompt, qwen_prompt_with_comment
from qwen_vl_utils import process_vision_info


# --------------------- Config ---------------------
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH = ""
IMAGE_DIR = ""
IMAGE_SIZE = 224
EPOCH = 100
AUG = False
AUG_MODE = "colorjitter"
COMMENT = False

COMMENT_JSON_PATH = ""

PROJECT_NAME = ""

if AUG:
    PROJECT_NAME += f"_{AUG_MODE}"

if COMMENT:
    PROJECT_NAME += f"_comment"

OUTPUT_DIR = ""
METRIC_NAME = "pearson"  # choices=["mse", "pearson"]
GREATER_IS_BETTER = True if METRIC_NAME == "pearson" else False
os.makedirs(OUTPUT_DIR, exist_ok=True)
LEARNING_RATE = 1e-5


# --------------------- Helpers ---------------------
def resize_image(image: Image.Image, target_size=224):
    """保持宽高比缩放+灰色填充正方形，接受 PIL.Image"""
    if not isinstance(image, Image.Image):
        raise TypeError("resize_image expects a PIL.Image.Image")
    w, h = image.size
    ratio = min(target_size / w, target_size / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized = image.resize((new_w, new_h), Image.BICUBIC)
    padded = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    padded.paste(resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return padded


def apply_augmentation(image: Image.Image, mode: str = 'rotation_blur'):
    if mode == 'rotation_blur':
        angle = random.uniform(-15, 15)
        image = F.rotate(image, angle)
        if random.random() < 0.2:
            image = F.gaussian_blur(image, kernel_size=(5, 5))
    elif mode == 'rotation':
        angle = random.uniform(-15, 15)
        image = F.rotate(image, angle)
    elif mode == 'blur':
        if random.random() < 0.2:
            image = F.gaussian_blur(image, kernel_size=(5, 5))
    elif mode == 'hflip':
        if random.random() < 0.5:
            image = F.hflip(image)
    elif mode == 'colorjitter':
        jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        image = jitter(image)

    return image


def compute_score_loss(outputs, labels, tokenizer, num_items_in_batch=None, score_range=(1, 5)):
    logits = outputs.logits
    device = labels.device
    num_seq = labels.size(0)

    # 1) 收集分数 token
    score_tokens, score_to_indices, valid_score_tokens = [], [], []
    for s in range(score_range[0], score_range[1] + 1):
        token = str(s)
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            score_to_indices.append(ids[0])
            valid_score_tokens.append(token)

    if not score_to_indices:
        return torch.tensor(0.0, device=device, requires_grad=True)

    score_grids = torch.tensor([float(s) for s in valid_score_tokens], device=device, dtype=logits.dtype)
    id2score = {tid: float(s) for tid, s in zip(score_to_indices, valid_score_tokens)}

    # 2) 找 labels 里的分数 token
    score_label_token_ids, score_pos = extract_label_from_sequence(labels, tokenizer)
    if score_label_token_ids.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    valid_indices, score_labels_list = [], []
    for i, tok in enumerate(score_label_token_ids):
        t = tok.item()
        if t in id2score:
            valid_indices.append(i)
            score_labels_list.append(id2score[t])

    if not valid_indices:
        return torch.tensor(0.0, device=device, requires_grad=True)

    valid_indices = torch.tensor(valid_indices, device=device, dtype=torch.long)
    score_labels = torch.tensor(score_labels_list, device=device, dtype=logits.dtype)
    valid_score_pos = score_pos[valid_indices]

    # 3) 取 logits
    valid_score_pos_shifted = torch.clamp(valid_score_pos - 1, min=0)
    batch_indices = valid_indices
    score_logits = logits[batch_indices, valid_score_pos_shifted, :]  # (k, vocab)

    probs = torch.softmax(score_logits, dim=-1)
    score_grid_probs = probs.index_select(dim=-1, index=torch.tensor(score_to_indices, device=device))

    weighted_scores = (score_grid_probs * score_grids).sum(dim=-1)

    score_loss = torch.nn.functional.mse_loss(
        input=weighted_scores,
        target=score_labels,
        reduction="mean"
    )

    # 如果你要按 batch size normalize，打开下面两行
    # if num_items_in_batch is not None:
    #     score_loss = score_loss * (len(valid_indices) / num_seq)

    return score_loss


def extract_label_from_sequence(labels, tokenizer):
    batch_size = labels.size(0)
    device = labels.device

    # 允许的分数 token id
    score_token_ids = set()
    for s in map(str, range(1, 6)):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            score_token_ids.add(ids[0])

    score_positions, score_tokens = [], []
    for i in range(batch_size):
        valid_pos = (labels[i] != -100).nonzero(as_tuple=True)[0]
        for pos in reversed(valid_pos):  # 从后往前找
            tid = labels[i, pos].item()
            if tid in score_token_ids:
                score_positions.append(pos)
                score_tokens.append(labels[i, pos])
                break  # 找到就停
        # 没找到就跳过

    if score_tokens:
        return (
            torch.stack(score_tokens).to(device),
            torch.tensor(score_positions, device=device, dtype=torch.long),
        )
    return torch.tensor([], device=device, dtype=torch.long), torch.tensor([], device=device, dtype=torch.long)


# --------------------- Dataset ---------------------
class ScoreDataset(Dataset):
    def __init__(self, csv_path, images_path, processor, size=224, row_indices=None, target_dim=None, augment=False, comment_json=None):
        """
        row_indices: list of dataframe row indices to keep (these are row indices, not expanded per-dimension indices)
        target_dim: if provided, only include that dimension
        """
        self.processor = processor
        self.size = size
        self.augment = augment
        df = pd.read_csv(csv_path)
        self.data = []

        comment_data = None
        if comment_json is not None:
            with open(comment_json, 'r', encoding='utf-8') as f:
                comment_data = json.load(f)        

        for ridx, row in df.iterrows():
            if row_indices is not None and ridx not in row_indices:
                continue
            
            image_name = row["name"] if "name" in row else row[0]
            
            img_path = os.path.join(images_path, row["name"]) if "name" in row else os.path.join(images_path, row[0])
            
            comment = None

            if comment_data is not None and image_name in comment_data['ArtEdu']['Qwen/Qwen2.5-VL-7B-Instruct']:
                comments = comment_data['ArtEdu']['Qwen/Qwen2.5-VL-7B-Instruct'][image_name]

                all_comment_keys = [key for key in comments.keys() 
                                if key.startswith('generated_comment_') or key == 'gold_label']
                        
                if all_comment_keys:
                    selected_key = random.choice(all_comment_keys)
                    comment = comments[selected_key]

            for dim in DIMENSION_LIST:
                if target_dim is not None and dim != target_dim:
                    continue
                raw_score = row.get(dim, None)
                score = int(raw_score) if (raw_score is not None and not math.isnan(raw_score)) else 3
                # clamp
                score = max(1, min(5, score))
                # 将分数转换为字符串形式，以便模型学习生成对应的token
                self.data.append({
                    "image_path": img_path,
                    "dim": dim,
                    "comment": comment,
                    "score": str(score),  # 改为字符串形式
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if not COMMENT:
            messages = qwen_prompt(item["dim"], item["image_path"])  # returns chat messages expected by processor
        else:
            messages = qwen_prompt_with_comment(item["dim"], item["image_path"], item["comment"])

        # process vision info (this helper should return (list_of_images_or_tensors, image_grid_thw))
        image_inputs, image_grid_thw = process_vision_info(messages)

        # ensure we have a PIL image to resize
        img = image_inputs[0]
        if isinstance(img, Image.Image):
            resized = resize_image(img, target_size=self.size)
        elif isinstance(img, torch.Tensor):
            # assume CHW or HWC tensor
            try:
                pil = T.ToPILImage()(img.cpu())
                resized = resize_image(pil, target_size=self.size)
            except Exception:
                # fallback: convert via numpy
                arr = img.cpu().numpy()
                if arr.ndim == 3 and arr.shape[0] <= 4:  # CHW
                    arr = np.transpose(arr, (1, 2, 0))
                pil = Image.fromarray((arr * 255).astype('uint8')) if arr.max() <= 1.0 else Image.fromarray(arr.astype('uint8'))
                resized = resize_image(pil, target_size=self.size)
        else:
            # try to coerce
            try:
                pil = Image.open(item["image_path"]).convert('RGB')
                resized = resize_image(pil, target_size=self.size)
            except Exception as e:
                raise RuntimeError(f"Cannot convert image input for idx {idx}: {e}")

        # optional augmentation
        if self.augment:
            resized = apply_augmentation(resized, mode=AUG_MODE)

        text_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 修改这里：使用文本形式的标签进行tokenization
        inputs = self.processor(
            images=resized,
            text=text_inputs,
            return_tensors="pt",
        )
        
        # 创建labels，将分数作为文本添加到输入中
        score_text = item["score"]
        score_inputs = self.processor(
            text=score_text,
            return_tensors="pt",
            add_special_tokens=False
        )
        
        # 合并输入和分数标签
        input_ids = inputs["input_ids"].squeeze(0)
        score_ids = score_inputs["input_ids"].squeeze(0)
        combined_input_ids = torch.cat([input_ids, score_ids], dim=0)
        
        # 创建对应的attention_mask
        attention_mask = torch.ones_like(combined_input_ids)
        
        # 创建labels，只保留分数token位置的标签，其他位置为-100
        labels = torch.full_like(combined_input_ids, -100)
        labels[-len(score_ids):] = score_ids

        return {
            "pixel_values": inputs['pixel_values'].squeeze(0),
            "input_ids": combined_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "image_grid_thw": inputs.image_grid_thw[0],
            "dim": item["dim"],
            "index": idx,
        }


@dataclass
class DataCollator:
    def __call__(self, batch: List[Dict]) -> Dict:
        pad_id = processor.tokenizer.pad_token_id or 0
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [x["input_ids"] for x in batch], batch_first=True, padding_value=pad_id
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [x["attention_mask"] for x in batch], batch_first=True, padding_value=0
            ),
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.nn.utils.rnn.pad_sequence(
                [x["labels"] for x in batch], batch_first=True, padding_value=-100
            ),
            # 将 numpy arrays 转换为 tensors
            "image_grid_thw": torch.stack([torch.tensor(x["image_grid_thw"]) for x in batch]),
        }


# --------------------- Metrics & Callbacks ---------------------
def compute_metrics(eval_pred):
    """
    Fixed compute_metrics function to properly handle predictions and labels
    """
    preds, labels = eval_pred
    
    # preds should be a tuple where the first element is loss and second is predictions
    # Or directly the predictions if return_outputs=False in compute_loss
    if isinstance(preds, tuple):
        preds = preds[1]  # Get predictions from (loss, predictions) tuple
    
    # Convert to numpy if they're tensors
    if hasattr(preds, 'detach'):
        preds = preds.detach().cpu().numpy()
    if hasattr(labels, 'detach'):
        labels = labels.detach().cpu().numpy()
    
    # Flatten predictions
    preds = np.array(preds).flatten()
    
    # labels are token ids, need to convert to real scores
    labels = np.array(labels)
    
    # Filter out -100 (ignored) labels
    mask = labels != -100
    valid_labels = labels[mask]
    
    # Map token ids to scores (1-5)
    id2score = {}
    for s in range(1, 6):
        token_ids = processor.tokenizer.encode(str(s), add_special_tokens=False)
        if len(token_ids) == 1:
            id2score[token_ids[0]] = float(s)
    
    # Convert label token ids to scores
    valid_labels_scores = []
    for tid in valid_labels:
        if tid in id2score:
            valid_labels_scores.append(id2score[tid])
    
    valid_labels = np.array(valid_labels_scores)
    
    # Align lengths
    min_len = min(len(preds), len(valid_labels))
    preds = preds[:min_len]
    valid_labels = valid_labels[:min_len]
    
    print(f"Validation - Predicted scores: {preds[:10]}...")  # Show first 10 predictions
    print(f"Validation - True scores: {valid_labels[:10]}...")  # Show first 10 labels
    
    if len(preds) == 0 or len(valid_labels) == 0:
        return {METRIC_NAME: 0.0}
    
    if METRIC_NAME == "mse":
        mse = float(((preds - valid_labels) ** 2).mean())
        print(f"Validation MSE: {mse}")
        return {"mse": mse}
    elif METRIC_NAME == "pearson":
        if len(preds) < 2:
            return {"pearson": 0.0}
        vx = preds - preds.mean()
        vy = valid_labels - valid_labels.mean()
        corr = (vx * vy).sum() / (np.sqrt((vx**2).sum()) * np.sqrt((vy**2).sum()) + 1e-8)
        print(f"Validation Pearson correlation: {corr}")
        return {"pearson": float(corr)}
    
    return {METRIC_NAME: 0.0}


# ------------------ Save Model ------------------
class SaveBestCallback(TrainerCallback):
    def __init__(self, model, metric_name, output_dir, greater_is_better=True):
        self.best_metric = None
        self.model = model
        self.metric_name = metric_name
        self.output_dir = output_dir
        self.greater_is_better = greater_is_better
        self.best_model_state = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metric = metrics.get(self.metric_name)
        if metric is None:
            # Try with eval_ prefix
            metric = metrics.get(f"eval_{self.metric_name}")
            
        if metric is None:
            print(f"Metric {self.metric_name} not found in metrics: {list(metrics.keys())}")
            return

        update = False
        if self.best_metric is None:
            update = True
        else:
            if self.greater_is_better and metric > self.best_metric:
                update = True
            elif not self.greater_is_better and metric < self.best_metric:
                update = True

        if update:
            self.best_metric = metric
            print(f"New best {self.metric_name}: {metric:.4f}, saving model...")
            
            # Save the model state dict instead of relying on save_pretrained during training
            try:
                # Save model weights
                self.model.save_pretrained(f"{self.output_dir}/best_model")
                print(f"Model saved to {self.output_dir}/best_model")
            except Exception as e:
                print(f"Error saving model: {e}")
                # Fallback: save just the state dict
                try:
                    model_path = f"{self.output_dir}/best_model"
                    os.makedirs(model_path, exist_ok=True)
                    torch.save(self.model.state_dict(), f"{model_path}/pytorch_model.bin")
                    self.model.config.save_pretrained(model_path)
                    print(f"Model state dict saved to {model_path}")
                except Exception as e2:
                    print(f"Error saving model state dict: {e2}")


# --------------------- Training routine (single model for all dimensions) ---------------------
def train_single_model_all_dimensions(train_dataset, val_dataset, base_output_dir, model_name, processor, device_map='auto'):
    """
    Train a single model for all dimensions instead of separate models per dimension
    """
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\n=== Training single model for all dimensions ===")
    dim_output_dir = os.path.join(base_output_dir, f"all_dimensions")
    os.makedirs(dim_output_dir, exist_ok=True)

    # 1. load base model
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        use_cache=False,
    )

    # 2. configure LoRA - single configuration for all dimensions
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    logging.info(f"{'='*15} LoRA Configuration for all dimensions {'='*15}")
    logging.info(f"  r: {lora_config.r}")
    logging.info(f"  lora_alpha: {lora_config.lora_alpha}")
    logging.info(f"  target_modules: {lora_config.target_modules}")
    logging.info(f"  lora_dropout: {lora_config.lora_dropout}")
    logging.info(f"  bias: {lora_config.bias}")
    logging.info(f"  task_type: {lora_config.task_type}")
    logging.info('='*40)

    model = get_peft_model(base_model, lora_config)
    
    # Attach tokenizer to model for loss computation
    model.tokenizer = processor.tokenizer

    # 3. freeze all except LoRA parameters
    for n, p in model.named_parameters():
        if "lora" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # 4. compute_loss for Trainer
    def compute_loss(model_, inputs, return_outputs=False, **kwargs):
        device = next(model_.parameters()).device
        for k in ["input_ids", "attention_mask", "pixel_values", "labels", "image_grid_thw"]:
            if k in inputs and isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to(device)

        outputs = model_(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            return_dict=True,
            image_grid_thw=inputs["image_grid_thw"],
            output_hidden_states=True,
        )

        # Compute score loss
        score_loss = compute_score_loss(
            outputs=outputs,
            labels=inputs["labels"].to(device),
            tokenizer=model_.tokenizer,
            score_range=(1, 5)
        )

        # Compute predicted scores for metrics
        logits = outputs.logits
        score_label_token_ids, score_pos = extract_label_from_sequence(inputs["labels"], model_.tokenizer)

        if score_label_token_ids.numel() > 0:
            valid_score_pos_shifted = torch.clamp(score_pos - 1, min=0)
            batch_indices = torch.arange(len(valid_score_pos_shifted), device=device)
            score_logits = logits[batch_indices, valid_score_pos_shifted, :]

            # Get token ids for scores 1-5
            score_token_ids = []
            for s in range(1, 6):
                ids = model_.tokenizer.encode(str(s), add_special_tokens=False)
                if len(ids) == 1:
                    score_token_ids.append(ids[0])
            
            if len(score_token_ids) > 0:
                score_grids = torch.tensor([float(s) for s in range(1, 6) if len(model_.tokenizer.encode(str(s), add_special_tokens=False)) == 1], 
                                         device=device, dtype=logits.dtype)
                
                probs = torch.softmax(score_logits, dim=-1)
                score_grid_probs = probs.index_select(dim=-1, index=torch.tensor(score_token_ids, device=device))
                pred_scores = (score_grid_probs * score_grids[:score_grid_probs.shape[1]]).sum(dim=-1)
            else:
                pred_scores = torch.zeros(inputs["labels"].size(0), device=device)
        else:
            pred_scores = torch.zeros(inputs["labels"].size(0), device=device)

        if return_outputs:
            return score_loss, pred_scores
        return score_loss

    # 5. training args
    training_args = TrainingArguments(
        output_dir=dim_output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=EPOCH,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="no",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_total_limit=1,
        load_best_model_at_end=False,
        metric_for_best_model=f"eval_{METRIC_NAME}",
        greater_is_better=GREATER_IS_BETTER,
        bf16=True,
        report_to="wandb",
        remove_unused_columns=False,
        max_grad_norm=1.0,
        label_names=["labels"],
        run_name=f"all_dimensions", 
    )

    class CustomTrainer(Trainer):
        def compute_loss(self, model__, inputs, return_outputs=False, **kwargs):
            return compute_loss(model__, inputs, return_outputs)

        def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
            """
            Override prediction_step to properly handle our custom loss and predictions
            """
            with torch.no_grad():
                loss, pred_scores = compute_loss(model, inputs, return_outputs=True)
                
                # Return format expected by Trainer: (loss, logits, labels)
                return (loss, pred_scores, inputs["labels"])

    save_best_callback = SaveBestCallback(
        model=model,
        metric_name=METRIC_NAME,
        output_dir=dim_output_dir,
        greater_is_better=GREATER_IS_BETTER
    )

    early_stop_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.0
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[early_stop_callback, save_best_callback],
    )

    trainer.train()

    # save peft+model
    model.save_pretrained(dim_output_dir)

    # free memory
    try:
        del trainer, model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    except Exception as e:
        print(f"[All dimensions] Error during cleanup: {e}")
        pass

    return dim_output_dir


# --------------------- Main ---------------------
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "training.log")

    class InfoOnlyFilter(logging.Filter):
        def filter(self, record):
            return record.levelno == logging.INFO

    # 创建 handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(InfoOnlyFilter())

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.addFilter(InfoOnlyFilter())

    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,  # 默认最低级别
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            file_handler,
            stream_handler
        ]
    )

    logger = logging.getLogger(__name__)

    class LoggerWriter:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level

        def write(self, message):
            try:
                if message.strip():
                    self.logger.log(self.level, message.strip())
            except Exception:
                pass  # 忽略日志关闭时的写入错误

        def flush(self):
            pass

    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)

    logging.info(f"{'='*15} Training Configuration {'='*15}")
    logging.info(f"MODEL_NAME: {MODEL_NAME}")
    logging.info(f"DATA_PATH: {DATA_PATH}")
    logging.info(f"IMAGE_DIR: {IMAGE_DIR}")
    logging.info(f"IMAGE_SIZE: {IMAGE_SIZE}")
    logging.info(f"EPOCH: {EPOCH}")
    logging.info(f"AUG: {AUG}")
    if AUG:
        logging.info(f"AUG_MODE: {AUG_MODE}")
    logging.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
    logging.info('='*40)

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    df = pd.read_csv(DATA_PATH)
    original_row_count = len(df)
    original_indices = list(range(original_row_count))
    random.shuffle(original_indices)
    train_size = int(0.9 * original_row_count)
    train_row_indices = original_indices[:train_size]
    val_row_indices = original_indices[train_size:]

    # For single model training, we don't loop through dimensions
    print(f"\n{'='*20} Training single model for all dimensions {'='*20}")
    
    wandb.init(
        project=f"{PROJECT_NAME}",
        name=f"all_dimensions",
        config={
            "model": MODEL_NAME,
            "dataset": DATA_PATH,
            "image_size": IMAGE_SIZE,
            "epochs": EPOCH,
        },
        reinit=True,
    )        
    
    try:
        # Create datasets without target_dim restriction (include all dimensions)
        train_dataset = ScoreDataset(
            DATA_PATH, IMAGE_DIR, processor,
            size=IMAGE_SIZE,
            row_indices=train_row_indices,
            augment=AUG,
            comment_json=COMMENT_JSON_PATH,
        )
        val_dataset = ScoreDataset(
            DATA_PATH, IMAGE_DIR, processor,
            size=IMAGE_SIZE,
            row_indices=val_row_indices,
            augment=False,
            comment_json=COMMENT_JSON_PATH,
        )

        output_dir = train_single_model_all_dimensions(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            base_output_dir=OUTPUT_DIR,
            model_name=MODEL_NAME,
            processor=processor,
            device_map='auto',
        )

        print(f"Successfully trained single model for all dimensions")

    except Exception as e:
        print(f"Error training single model: {str(e)}")

    try:
        wandb.finish()
    except:
        pass
