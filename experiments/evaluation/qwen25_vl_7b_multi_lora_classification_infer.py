import torch
import pandas as pd
import os
import json
from tqdm.auto import tqdm
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from utils.data_process import DIMENSION_LIST
from utils.prompt_lib import qwen_prompt
from PIL import Image

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
PROJECT_NAME = ""
OUTPUT_DIR = f"../{PROJECT_NAME}"
DATA_PATH = ""
IMAGE_DIR = ""
OUTPUT_JSON_PATH = f"../{PROJECT_NAME}.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

FLOAT = False
if FLOAT:
    OUTPUT_JSON_PATH = f"/scratch/users/k24015729/creativity_assessment/mllms/output/{PROJECT_NAME}_float.json"


# ----------------------------
# Image preprocess
# ----------------------------

def resize_image(image, target_size=224):
    w, h = image.size
    ratio = min(target_size / w, target_size / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized = image.resize((new_w, new_h), Image.BICUBIC)
    padded = Image.new('RGB', (target_size, target_size), (128,128,128))
    padded.paste(resized, ((target_size-new_w)//2, (target_size-new_h)//2))
    return padded


def preprocess(image_path, dimension, processor):
    prompt = qwen_prompt(dimension, image_path)
    image_inputs, _ = process_vision_info(prompt)
    resized = resize_image(image_inputs[0], target_size=IMAGE_SIZE)

    text_inputs = processor.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        images=resized,
        text=text_inputs,
        return_tensors="pt"
    )

    return {
        "pixel_values": inputs["pixel_values"].squeeze(0).to(DEVICE),
        "input_ids": inputs["input_ids"].squeeze().to(DEVICE),
        "attention_mask": inputs["attention_mask"].squeeze().to(DEVICE),
        "image_grid_thw": inputs["image_grid_thw"][0].to(DEVICE)
            if "image_grid_thw" in inputs else None,
    }


# ----------------------------
# Load Model
# ----------------------------

def load_single_model(dimension):
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    lora_dir = os.path.join(OUTPUT_DIR, f"dimension_{dimension}")
    model = PeftModel.from_pretrained(base, lora_dir, torch_dtype=torch.bfloat16)

    model.tokenizer = processor.tokenizer
    return model, processor


# ----------------------------
# Classification prediction
# ----------------------------

def predict_classification(image_path, dimension, processor, model):
    """Return score 1~5 via 5-way classification"""
    inputs = preprocess(image_path, dimension, processor)

    input_ids = inputs["input_ids"].unsqueeze(0)
    attention_mask = inputs["attention_mask"].unsqueeze(0)
    pixel_values = inputs["pixel_values"].unsqueeze(0)
    image_grid_thw = (
        inputs["image_grid_thw"].unsqueeze(0)
        if inputs["image_grid_thw"] is not None else None
    )

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        logits = out.logits[0]     # (seq_len, vocab)

    # ------------------------------
    # 1~5 token logits
    # ------------------------------
    score_token_ids = []
    id2score = {}
    for s in range(1, 6):
        tok = str(s)
        ids = model.tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) == 1:
            score_token_ids.append(ids[0])
            id2score[ids[0]] = float(s)

    last_logits = logits[-1]
    score_logits = last_logits[score_token_ids]        # (5,)
    probs = torch.softmax(score_logits, dim=-1)

    values = torch.tensor([id2score[t] for t in score_token_ids], device=probs.device)
    pred = (probs * values).sum().item()

    if FLOAT:
        return round(pred, 2)
    else:
        return int(max(1, min(5, round(pred))))


# ----------------------------
# Batch predict
# ----------------------------

def batch_predict(data_path, image_dir, output_json_path):
    df = pd.read_csv(data_path)
    first_col = df.columns[0]

    results = { "ArtEdu": { MODEL_NAME: {} } }

    for dim in DIMENSION_LIST:
        print(f"\n========== Predicting dimension: {dim} ==========")

        model, processor = load_single_model(dim)

        for row in tqdm(df.itertuples(index=False)):
            name = getattr(row, first_col)
            img_path = os.path.join(image_dir, name)

            if name not in results["ArtEdu"][MODEL_NAME]:
                results["ArtEdu"][MODEL_NAME][name] = {}

            try:
                pred = predict_classification(img_path, dim, processor, model)
                orig = int(getattr(row, dim)) if not pd.isna(getattr(row, dim)) else None
            except:
                pred = None
                orig = None

            results["ArtEdu"][MODEL_NAME][name][dim] = {
                "original": orig,
                "predicted": pred,
            }

        del model
        torch.cuda.empty_cache()

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("Done! Saved to:", output_json_path)


if __name__ == '__main__':
    print(f"Starting inference with multi LoRA classification models")
    print(f"Project: {PROJECT_NAME}")
    print(f"Model: {MODEL_NAME}")
    print(f"Data path: {DATA_PATH}")
    print(f"Output path: {OUTPUT_JSON_PATH}")
    
    try:
        batch_predict(DATA_PATH, IMAGE_DIR, OUTPUT_JSON_PATH)
        print("Inference completed successfully!")
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise
