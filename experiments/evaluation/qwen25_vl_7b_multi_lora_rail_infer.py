# qwen25_vl_7b_multi_lora_rail_infer.py

import torch
import pandas as pd
import os
import numpy as np
import json
import math
import cv2
from PIL import Image
from tqdm.auto import tqdm
from peft import PeftModel
from utils.data_process import DIMENSION_LIST
from utils.prompt_lib import qwen_prompt
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
PROJECT_NAME = ""
OUTPUT_DIR = f"../{PROJECT_NAME}"
DATA_PATH = ""
IMAGE_DIR = ""
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_JSON_PATH = f"../{PROJECT_NAME}.json"
FLOAT = True
if FLOAT:
    OUTPUT_JSON_PATH = f"/scratch/users/k24015729/creativity_assessment/mllms/output/{PROJECT_NAME}_float.json"
PLOT_SAVE_PATH = f"/scratch/users/k24015729/creativity_assessment/mllms/plot/qwen25_vl_7b_multi_lora_rail_224_4_subspace_overlap_analysis.png"


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


def preprocess(image_path, dimension, processor):
    prompt = qwen_prompt(dimension, image_path)
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


def load_single_model(dimension):
    # Load base model first
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load processor to get tokenizer
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Load LoRA adapters using PEFT
    dim_dir = os.path.join(OUTPUT_DIR, f"dimension_{dimension}", "best_model")
    model = PeftModel.from_pretrained(
        base_model,
        dim_dir,
        torch_dtype=torch.bfloat16,
    )
    
    # Attach tokenizer to model for prediction
    model.tokenizer = processor.tokenizer
    return model


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


def predict_score_with_model(image_path, dimension, processor, model):
    inputs = preprocess(image_path, dimension, processor)
    
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
        
    del outputs, logits, score_logits, score_probs
    torch.cuda.empty_cache()

    if FLOAT:
        return round(predicted_score, 2) 
    else:
        rounded_score = round(predicted_score)
        return max(1, min(5, rounded_score))


def batch_predict_scores(data_path, images_path, output_json_path):
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    df = pd.read_csv(data_path)
    first_col_name = df.columns[0]
    
    output_json = {
        "ArtEdu": {
            MODEL_NAME: {}
        }
    }
    
    for dim in DIMENSION_LIST:
        print(f"Processing dimension: {dim}")
        model = load_single_model(dim)
        
        total_images = len(df)
        with tqdm(total=total_images, desc=f"Processing {dim}") as pbar:
            for row in df.itertuples(index=False):
                image_name = getattr(row, first_col_name)
                image_path = os.path.join(images_path, image_name)
                
                if image_name not in output_json["ArtEdu"][MODEL_NAME]:
                    output_json["ArtEdu"][MODEL_NAME][image_name] = {}
                
                try:
                    predicted = predict_score_with_model(image_path, dim, processor, model)
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
        
        del model
        torch.cuda.empty_cache()
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)
    print(f"Predictions completed, results saved to: {output_json_path}")


def predict_scores_from_folder(images_folder, output_json_path, output_csv_path=None):
    """
    Predict scores for all images in a folder and save results to both JSON and CSV formats.
    CSV format has images as rows and dimensions as columns for easier comparison.
    
    Args:
        images_folder (str): Path to the folder containing images
        output_json_path (str): Path to save the prediction results as JSON
        output_csv_path (str, optional): Path to save the prediction results as CSV. 
                                        If None, derives from JSON path.
    
    Returns:
        None: Results are saved to both JSON and CSV files
    """
    import os
    import json
    import pandas as pd
    import torch
    from tqdm import tqdm
    
    # If CSV path not provided, derive it from JSON path
    if output_csv_path is None:
        base_name = os.path.splitext(output_json_path)[0]
        output_csv_path = f"{base_name}.csv"
    
    # Initialize processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    # Get all image files
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    image_paths = [os.path.join(images_folder, f) for f in os.listdir(images_folder) 
                   if os.path.splitext(f)[-1].lower() in exts]
    image_paths.sort()
    
    if not image_paths:
        raise ValueError(f"No images found in folder: {images_folder}")
    
    print(f"Found {len(image_paths)} images in {images_folder}")
    
    # Initialize results dictionary (for JSON)
    output_json = {
        "ArtEdu": {
            MODEL_NAME: {}
        }
    }
    
    # Initialize dictionary for CSV (image_name -> {dimension: predicted_score})
    csv_results = {}
    
    # Process each dimension
    for dim in DIMENSION_LIST:
        print(f"Processing dimension: {dim}")
        # Load model for current dimension
        model = load_single_model(dim)
        
        # Process all images for this dimension
        with tqdm(total=len(image_paths), desc=f"Processing {dim}") as pbar:
            for image_path in image_paths:
                image_name = os.path.basename(image_path)
                
                # Initialize entry for this image if needed (for both JSON and CSV)
                if image_name not in output_json["ArtEdu"][MODEL_NAME]:
                    output_json["ArtEdu"][MODEL_NAME][image_name] = {}
                    
                if image_name not in csv_results:
                    csv_results[image_name] = {"image": image_name}
                    # Initialize all dimensions with None
                    for d in DIMENSION_LIST:
                        csv_results[image_name][d] = None
                
                try:
                    # Predict score
                    predicted = predict_score_with_model(image_path, dim, processor, model)
                    
                    # Save to JSON structure
                    output_json["ArtEdu"][MODEL_NAME][image_name][dim] = {
                        "predicted": predicted
                    }
                    
                    # Save to CSV structure
                    csv_results[image_name][dim] = predicted
                    
                except Exception as e:
                    print(f"Error processing: {image_name} - {dim} | Error: {str(e)}")
                    
                    # Save error result to JSON structure
                    output_json["ArtEdu"][MODEL_NAME][image_name][dim] = {
                        "predicted": None
                    }
                    
                    # CSV already has None as default
                
                pbar.update(1)
        
        # Clean up model memory
        del model
        torch.cuda.empty_cache()
    
    # Save results to JSON file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)
    print(f"JSON results saved to: {output_json_path}")
    
    # Save results to CSV file (images as rows, dimensions as columns)
    csv_data = list(csv_results.values())
    df = pd.DataFrame(csv_data)
    # Reorder columns to have 'image' first, then dimensions in order
    columns_order = ["image"] + DIMENSION_LIST
    df = df[columns_order]
    df.to_csv(output_csv_path, index=False)
    print(f"CSV results saved to: {output_csv_path}")
    
    print(f"Total predictions: {len(csv_data)}")
    print(f"Dimensions processed: {len(DIMENSION_LIST)}")
    print(f"Images processed: {len(image_paths)}")


if __name__ == '__main__':
    ### score
    batch_predict_scores(DATA_PATH, IMAGE_DIR, OUTPUT_JSON_PATH)
