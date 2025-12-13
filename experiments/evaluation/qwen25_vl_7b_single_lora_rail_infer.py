# qwen25_vl_7b_single_lora_infer.py

import torch
import pandas as pd
import os
import numpy as np
import json
from PIL import Image
from tqdm.auto import tqdm
from peft import PeftModel
from utils.data_process import DIMENSION_LIST
from utils.prompt_lib import qwen_prompt, qwen_prompt_with_comment
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# --------------------- Config ---------------------
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
PROJECT_NAME = ""
OUTPUT_DIR = f"../{PROJECT_NAME}"
BEST_MODEL_DIR = f"{OUTPUT_DIR}/all_dimensions/best_model"
DATA_PATH = ""
IMAGE_DIR = ""
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_JSON_PATH = f"../{PROJECT_NAME}.json"
FLOAT = False

# Comment-related configuration
COMMENT = False
COMMENT_JSON_PATH = ""

# --------------------- Helpers ---------------------
def resize_image(image, target_size=224):
    """PIL.Image"""
    w, h = image.size
    ratio = min(target_size / w, target_size / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized = image.resize((new_w, new_h), Image.BICUBIC)
    padded = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    padded.paste(resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return padded

def preprocess(image_path, dimension, processor, comment=None):
    """Preprocess image and prompt for model input"""
    if not COMMENT or comment is None:
        prompt = qwen_prompt(dimension, image_path)
    else:
        prompt = qwen_prompt_with_comment(dimension, image_path, comment)
        
    image_inputs, _ = process_vision_info(prompt)
    resized_image = resize_image(image_inputs[0], target_size=IMAGE_SIZE)
    text_inputs = processor.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(
        images=resized_image,
        text=text_inputs,
        return_tensors="pt",
    )
    return {
        "pixel_values": inputs['pixel_values'].squeeze(0).to(DEVICE),
        "input_ids": inputs['input_ids'].squeeze().to(DEVICE),
        "attention_mask": inputs['attention_mask'].squeeze().to(DEVICE),
        "image_grid_thw": inputs['image_grid_thw'][0].to(DEVICE) if 'image_grid_thw' in inputs else None,
    }

def load_model():
    """Load the base model with single LoRA adapter"""
    print("Loading base model...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print(f"Loading LoRA adapter from {BEST_MODEL_DIR}...")
    model = PeftModel.from_pretrained(
        base_model,
        BEST_MODEL_DIR,
        torch_dtype=torch.bfloat16,
    )
    
    # Attach tokenizer to model for prediction
    model.tokenizer = processor.tokenizer
    return model, processor

def extract_label_from_sequence(labels, tokenizer):
    """Extract score tokens from label sequence"""
    batch_size = labels.size(0)
    device = labels.device

    # token id
    score_token_ids = set()
    for s in map(str, range(1, 6)):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            score_token_ids.add(ids[0])

    score_positions, score_tokens = [], []
    for i in range(batch_size):
        valid_pos = (labels[i] != -100).nonzero(as_tuple=True)[0]
        for pos in reversed(valid_pos):
            tid = labels[i, pos].item()
            if tid in score_token_ids:
                score_positions.append(pos)
                score_tokens.append(labels[i, pos])
                break

    if score_tokens:
        return (
            torch.stack(score_tokens).to(device),
            torch.tensor(score_positions, device=device, dtype=torch.long),
        )
    return torch.tensor([], device=device, dtype=torch.long), torch.tensor([], device=device, dtype=torch.long)

def predict_score_with_model(image_path, dimension, processor, model, comment=None):
    """Predict score for a single image and dimension"""
    # Preprocess inputs
    inputs = preprocess(image_path, dimension, processor, comment)
    
    # Ensure all inputs are on the correct device
    input_ids = inputs["input_ids"].unsqueeze(0).to(DEVICE)
    attention_mask = inputs["attention_mask"].unsqueeze(0).to(DEVICE)
    pixel_values = inputs["pixel_values"].unsqueeze(0).to(DEVICE)
    image_grid_thw = inputs["image_grid_thw"].unsqueeze(0).to(DEVICE) if inputs["image_grid_thw"] is not None else None
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        
        logits = outputs.logits  # (batch, seq_len, vocab)
        
        # Get score token IDs
        score_token_ids = []
        id2score = {}
        for s in range(1, 6):
            token = str(s)
            ids = model.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                score_token_ids.append(ids[0])
                id2score[ids[0]] = float(s)

        if not score_token_ids:
            print("No valid score tokens found")
            return 3.0  # default score if no valid tokens found

        # Get logits at the last position (where score should be)
        score_logits = logits[0, -1, :]  # (vocab)
        
        # Extract probabilities for score tokens
        score_probs = torch.softmax(score_logits[score_token_ids], dim=-1)
        score_values = torch.tensor([id2score[tid] for tid in score_token_ids], device=score_probs.device)
        
        # Calculate weighted score
        predicted_score = (score_probs * score_values).sum().item()
        
    # Clean up intermediate variables
    del outputs, logits, score_logits, score_probs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if FLOAT:
        return round(predicted_score, 2) 
    else:
        rounded_score = round(predicted_score)
        return max(1, min(5, rounded_score))  # Clamp to 1-5 range

def batch_predict_scores(data_path, images_path, output_json_path):
    """Perform batch inference on test data"""
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    df = pd.read_csv(data_path)
    first_col_name = df.columns[0]
    
    # Load comments if available
    comment_data = None
    if COMMENT and COMMENT_JSON_PATH and os.path.exists(COMMENT_JSON_PATH):
        with open(COMMENT_JSON_PATH, 'r', encoding='utf-8') as f:
            comment_data = json.load(f)
    
    # Load model once for all predictions
    print("Loading model...")
    model, processor = load_model()
    
    # Initialize results dictionary
    output_json = {
        "ArtEdu": {
            MODEL_NAME: {}
        }
    }
    
    total_images = len(df)
    print(f"Processing {total_images} images across {len(DIMENSION_LIST)} dimensions...")
    
    with tqdm(total=total_images * len(DIMENSION_LIST), desc="Processing") as pbar:
        for row in df.itertuples(index=False):
            image_name = getattr(row, first_col_name)
            image_path = os.path.join(images_path, image_name)
            
            # Get comment if available
            comment = None
            if comment_data and image_name in comment_data['ArtEdu'][MODEL_NAME]:
                comments = comment_data['ArtEdu'][MODEL_NAME][image_name]
                # Use gold label if available, otherwise use a generated comment
                if 'gold_label' in comments:
                    comment = comments['gold_label']
                else:
                    # Select any available generated comment
                    comment_keys = [k for k in comments.keys() if k.startswith('generated_comment_')]
                    if comment_keys:
                        comment = comments[comment_keys[0]]
            
            # Initialize entry for this image if not exists
            if image_name not in output_json["ArtEdu"][MODEL_NAME]:
                output_json["ArtEdu"][MODEL_NAME][image_name] = {}
            
            # Process each dimension
            for dim in DIMENSION_LIST:
                try:
                    predicted = predict_score_with_model(image_path, dim, processor, model, comment)
                    origin_score = int(getattr(row, dim)) if not pd.isna(getattr(row, dim)) else 3
                    output_json["ArtEdu"][MODEL_NAME][image_name][dim] = {
                        "original": origin_score,
                        "predicted": predicted
                    }
                except Exception as e:
                    print(f"Error processing: {image_name} - {dim} | Error: {str(e)}")
                    origin_score = int(getattr(row, dim)) if not pd.isna(getattr(row, dim)) else 3
                    output_json["ArtEdu"][MODEL_NAME][image_name][dim] = {
                        "original": origin_score,
                        "predicted": None
                    }
                
                pbar.update(1)
    
    # Save results to specified path
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)
    print(f"Predictions completed, results saved to: {output_json_path}")
    
    # Clean up model memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    print(f"Starting inference with single LoRA model")
    print(f"Model: {MODEL_NAME}")
    print(f"Project: {PROJECT_NAME}")
    print(f"Data path: {DATA_PATH}")
    print(f"Output path: {OUTPUT_JSON_PATH}")
    
    try:
        batch_predict_scores(DATA_PATH, IMAGE_DIR, OUTPUT_JSON_PATH)
        print("Inference completed successfully!")
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise
