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

from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    Qwen2_5_VLForConditionalGeneration,
    TrainerCallback,
)
import wandb
import logging
import sys
import gc
from sklearn.metrics import mean_squared_error

# utils (assumed available in your project)
from utils.data_process import DIMENSION_LIST
from utils.prompt_lib import qwen_prompt
from qwen_vl_utils import process_vision_info


# --------------------- Config ---------------------
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH = ""
IMAGE_DIR = ""
IMAGE_SIZE = 224
EPOCH = 100
PROJECT_NAME = ""
OUTPUT_DIR = ""
os.makedirs(OUTPUT_DIR, exist_ok=True)
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LORA_R = 8
LORA_ALPHA = 16
NUM_CLASSES = 5
NUM_THRESHOLDS = NUM_CLASSES - 1


# --------------------- Helpers ---------------------
class CoralHead(nn.Module):
    """
    CORAL head: outputs K-1 logits for thresholds y > 1, y > 2, ..., y > K-1
    """
    def __init__(self, in_dim=3584, num_thresholds=4):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_thresholds),
        )
        self._init_weights()

    def _init_weights(self):
        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.seq.apply(init)

    def forward(self, x):
        # x: [B, hidden]
        return self.seq(x)   # [B, K-1]


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
    """对 PIL.Image 做简单增强，并返回 PIL.Image"""
    if not isinstance(image, Image.Image):
        image = T.ToPILImage()(image)

    if mode == 'rotation_blur':
        angle = random.uniform(-15, 15)
        image = F.rotate(image, angle)
        if random.random() < 0.2:
            image = image.filter(Image.Filter.GaussianBlur(radius=1.5))
    elif mode == 'rotation':
        angle = random.uniform(-15, 15)
        image = F.rotate(image, angle)
    elif mode == 'blur':
        if random.random() < 0.2:
            image = image.filter(Image.Filter.GaussianBlur(radius=1.5))
    elif mode == 'hflip':
        if random.random() < 0.5:
            image = F.hflip(image)
    elif mode == 'colorjitter':
        jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        image = jitter(image)
    return image


# --------------------- Dataset ---------------------
class ScoreDataset(Dataset):
    def __init__(self, csv_path, images_path, processor, size=224, row_indices=None, target_dim=None, augment=False):
        """
        row_indices: list of dataframe row indices to keep (these are row indices, not expanded per-dimension indices)
        target_dim: if provided, only include that dimension
        """
        self.processor = processor
        self.size = size
        self.augment = augment
        df = pd.read_csv(csv_path)
        self.data = []

        for ridx, row in df.iterrows():
            if row_indices is not None and ridx not in row_indices:
                continue
            img_path = os.path.join(images_path, row["name"]) if "name" in row else os.path.join(images_path, row[0])
            for dim in DIMENSION_LIST:
                if target_dim is not None and dim != target_dim:
                    continue
                raw_score = row.get(dim, None)
                score = int(raw_score) if (raw_score is not None and not math.isnan(raw_score)) else 3
                # clamp to [1,5]
                score = max(1, min(NUM_CLASSES, score))
                self.data.append({
                    "image_path": img_path,
                    "dim": dim,
                    "score": score,
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = qwen_prompt(item["dim"], item["image_path"])  # returns chat messages expected by processor

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
            resized = apply_augmentation(resized, mode='rotation_blur')

        text_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            images=resized,
            text=text_inputs,
            return_tensors="pt",
        )

        label = float(item["score"])  # still存 1-5 分数，在 CORAL 里转成多 binary target

        return {
            "pixel_values": inputs['pixel_values'].squeeze(0),
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
            "image_grid_thw": inputs.image_grid_thw[0],
            "dim": item["dim"],
            "index": idx,
        }


@dataclass
class DataCollator:
    def __call__(self, batch: List[Dict]) -> Dict:
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence([x["input_ids"] for x in batch], batch_first=True, padding_value=0),
            "attention_mask": torch.nn.utils.rnn.pad_sequence([x["attention_mask"] for x in batch], batch_first=True, padding_value=0),
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
            "image_grid_thw": torch.stack([x["image_grid_thw"] for x in batch]),
        }


class DimensionBatchSampler(torch.utils.data.Sampler):
    """按照维度分批次，保持同一batch内为同一维度（用于分析/调试）"""
    def __init__(self, data_source, batch_size=4, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

        # resolve to base dataset
        self._dataset = data_source
        if isinstance(data_source, torch.utils.data.Subset):
            self._dataset = data_source.dataset

        # build dim -> indices
        self.dim_to_indices = defaultdict(list)
        for idx in range(len(data_source)):
            try:
                item = data_source[idx]
                self.dim_to_indices[item['dim']].append(idx)
            except Exception as e:
                print(f"[Sampler Error] idx {idx}: {e}")

        self.batches = []
        for dim, indices in self.dim_to_indices.items():
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                self.batches.append(indices[i:i + self.batch_size])

        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


def get_dataloader(dataset, batch_size=4, shuffle=False):
    sampler = DimensionBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=DataCollator(),
        num_workers=0,
    )


# --------------------- Metrics & Callbacks ---------------------
def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # preds 可能是 dict({"logits": tensor})，也可能直接是 tensor
    if isinstance(preds, dict) and "logits" in preds:
        preds = preds["logits"]

    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    else:
        preds = np.array(preds)

    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    else:
        labels = np.array(labels)

    preds = preds.squeeze()   # predicted scores 1-5
    labels = labels.squeeze() # true scores 1-5

    mse = mean_squared_error(labels, preds)
    return {"eval_mse": float(mse)}


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, min_delta=0.001, dimension="unknown"):
        """
        Args:
            patience (int): 连续多少次 eval 没有提升后停止
            min_delta (float): 最小改进幅度
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_mse = float('inf')
        self.dimension = dimension
        self.counter = 0

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get('metrics', {})
        if not metrics or 'eval_mse' not in metrics:
            return

        current_mse = metrics["eval_mse"]
        print(f"[EarlyStopping-{self.dimension}] Step: {state.global_step}, Current MSE: {current_mse:.6f}, Best MSE: {self.best_mse:.6f}, Counter: {self.counter}")

        if current_mse < self.best_mse - self.min_delta:
            # 有显著提升，重置 patience
            self.best_mse = current_mse
            self.counter = 0
            print(f"[EarlyStopping-{self.dimension}] Improvement found. Reset counter.")
        else:
            # 没有提升，增加计数
            self.counter += 1
            print(f"[EarlyStopping-{self.dimension}] No improvement. Counter increased to {self.counter}.")
            if self.counter >= self.patience:
                print(
                    f"[EarlyStopping-{self.dimension}] Patience reached ({self.patience}), "
                    f"stopping training at step {state.global_step}."
                )
                control.should_training_stop = True


# --------------------- Training routine (single-dimension per model) ---------------------
def train_single_dimension(dimension, train_dataset, val_dataset, base_output_dir, model_name, processor, device_map='auto'):
    seed = hash(dimension) % (2**32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)    
    
    print(f"\n=== Training dimension: {dimension} (CORAL) ===")
    dim_output_dir = os.path.join(base_output_dir, f"dimension_{dimension}")
    os.makedirs(dim_output_dir, exist_ok=True)

    # 1. load base model
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        use_cache=False,
    )

    # 2. configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)

    logging.info(f"{'='*15} LoRA Configuration for dimension {dimension} {'='*15}")
    logging.info(f"  r: {lora_config.r}")
    logging.info(f"  lora_alpha: {lora_config.lora_alpha}")
    logging.info(f"  target_modules: {lora_config.target_modules}")
    logging.info(f"  lora_dropout: {lora_config.lora_dropout}")
    logging.info(f"  bias: {lora_config.bias}")
    logging.info(f"  task_type: {lora_config.task_type}")
    logging.info('='*40)  

    # 3. add CORAL head with correct input dim
    hidden_size = getattr(model.config, 'hidden_size', None) or getattr(getattr(model, 'config', {}), 'hidden_size', 3584)
    model.coral_head = CoralHead(in_dim=hidden_size, num_thresholds=NUM_THRESHOLDS)
    model.coral_head.to(next(model.parameters()).device)

    # 4. freeze all except LoRA and coral head
    for n, p in model.named_parameters():
        if ("lora" in n) or ("coral_head" in n):
            p.requires_grad = True
        else:
            p.requires_grad = False

    bce_loss = nn.BCEWithLogitsLoss()

    # 5. compute_loss for Trainer (CORAL: cumulative BCE)
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

        hidden_states = outputs.hidden_states
        cls_token = hidden_states[-1][:, -1, :].to(device)   # [B, hidden]

        logits = model_.coral_head(cls_token)               # [B, K-1]

        # labels: original scores in {1,...,K}
        labels = inputs["labels"].to(device)                # [B]
        # create cumulative targets: t_k = 1 if y > k else 0, for k=1..K-1
        # thresholds: tensor([1,2,3,4])
        thresholds = torch.arange(1, NUM_CLASSES, device=device).unsqueeze(0)  # [1, K-1]
        label_expanded = labels.unsqueeze(-1)                                   # [B,1]
        targets = (label_expanded > thresholds).float()                         # [B, K-1]

        loss = bce_loss(logits, targets)

        # prediction: number of thresholds passed (sigmoid(logit_k) > 0.5) + 1
        probs = torch.sigmoid(logits)                          # [B, K-1]
        passes = (probs > 0.5).float().sum(dim=-1)             # [B]
        pred_scores = passes + 1.0                             # [B] in {1,...,K}

        if return_outputs:
            # 返回张量以便 compute_metrics 使用
            return loss, {"logits": pred_scores}

        return loss


    # 6. training args
    dim_training_args = TrainingArguments(
        output_dir=dim_output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
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
        metric_for_best_model="eval_mse",
        greater_is_better=False,
        bf16=True,
        report_to="wandb",
        remove_unused_columns=False,
        max_grad_norm=1.0,
        label_names=["labels"],
        run_name=f"dimension_{dimension}", 
    )

    logging.info(f"{'='*15} Training Arguments for dimension {dimension} {'='*15}")
    logging.info(f"Training Arguments for dimension {dimension}:")
    logging.info(f"  output_dir: {dim_training_args.output_dir}")
    logging.info(f"  per_device_train_batch_size: {dim_training_args.per_device_train_batch_size}")
    logging.info(f"  gradient_accumulation_steps: {dim_training_args.gradient_accumulation_steps}")
    logging.info(f"  num_train_epochs: {dim_training_args.num_train_epochs}")
    logging.info(f"  learning_rate: {dim_training_args.learning_rate}")
    logging.info(f"  warmup_steps: {dim_training_args.warmup_steps}")
    logging.info('='*40)

    class CustomTrainer(Trainer):
        def compute_loss(self, model__, inputs, return_outputs=False, **kwargs):
            return compute_loss(model__, inputs, return_outputs)

    # callbacks
    early_stopping = EarlyStoppingCallback(patience=3, min_delta=0.001, dimension=dimension)
    save_best_callback = SaveBestModelCallback(model=model, output_dir=dim_output_dir, dimension=dimension)

    trainer = CustomTrainer(
        model=model,
        args=dim_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[early_stopping, save_best_callback],
    )

    trainer.train()

    # save peft+model and coral head
    model.save_pretrained(dim_output_dir)
    torch.save(model.coral_head.state_dict(), os.path.join(dim_output_dir, "coral_head.pth"))

    # free memory
    try:
        torch.cuda.empty_cache()
        del trainer, model, base_model
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
    except Exception:
        print(f"[{dimension}] Error during cleanup: {e}")
        pass

    return dim_output_dir


# ------------------ Save Model ------------------
class SaveBestModelCallback(TrainerCallback):
    def __init__(self, model, output_dir, dimension="unknown"):
        self.model = model
        self.output_dir = output_dir
        self.best_mse = float('inf')
        self.dimension = dimension

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics", {})
        if not metrics or "eval_mse" not in metrics:
            return

        current_mse = metrics["eval_mse"]
        print(f"[SaveBestModelCallback] Current MSE: {current_mse:.4f}, Best MSE: {self.best_mse:.4f}")

        if current_mse < self.best_mse:
            print("New best model found! Saving model and coral head...")
            self.best_mse = current_mse

            # 保存 PEFT 模型（包括 LoRA 权重）
            self.model.save_pretrained(self.output_dir)
            
            # 保存 CORAL 头
            coral_head_path = os.path.join(self.output_dir, "coral_head.pth")
            torch.save(self.model.coral_head.state_dict(), coral_head_path)
            
            # 保存最佳 MSE 值
            mse_file = os.path.join(self.output_dir, "best_mse.txt")
            with open(mse_file, "w") as f:
                f.write(f"{self.best_mse:.6f}\n")
            
            print(f"[{self.dimension}] Saved best model to: {self.output_dir}")


# --------------------- Main ---------------------
if __name__ == '__main__':
    # log
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

    DIMENSION_LIST = [
        # "realism",
        "deformation",
        "imagination",
        "color_richness",
        "color_contrast",
        "line_combination",
        "line_texture",
        "picture_organization",
        "transformation"
    ]

    for dimension in DIMENSION_LIST:
        print(f"\n{'='*20} Training {dimension} (CORAL) {'='*20}")
        wandb.init(
            project=f"{PROJECT_NAME}",
            name=f"{dimension}",
            config={
                "model": MODEL_NAME,
                "dataset": DATA_PATH,
                "image_size": IMAGE_SIZE,
                "epochs": EPOCH,
                "dimension": dimension,
            },
            reinit=True,  # 重新初始化
        )        
        
        try:
            train_dataset = ScoreDataset(
                DATA_PATH, IMAGE_DIR, processor,
                size=IMAGE_SIZE,
                row_indices=train_row_indices,
                target_dim=dimension,
                augment=False,
            )
            val_dataset = ScoreDataset(
                DATA_PATH, IMAGE_DIR, processor,
                size=IMAGE_SIZE,
                row_indices=val_row_indices,
                target_dim=dimension,
                augment=False,
            )

            dim_output_dir = train_single_dimension(
                dimension=dimension,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                base_output_dir=OUTPUT_DIR,
                model_name=MODEL_NAME,
                processor=processor,
                device_map='auto',
            )

            print(f"Successfully trained dimension (CORAL): {dimension}")

        except Exception as e:
            print(f"Error training dimension {dimension}: {str(e)}")
            continue

        try:
            wandb.finish()  # 显式结束当前运行
        except:
            pass
