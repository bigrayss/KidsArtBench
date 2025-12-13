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
import torch.nn as nn


# ----------------------
# Config
# ----------------------

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
PROJECT_NAME = ""
OUTPUT_DIR = f"../{PROJECT_NAME}"

DATA_PATH = ""
IMAGE_DIR = ""

OUTPUT_JSON_PATH = f""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

FLOAT = False
if FLOAT:
    OUTPUT_JSON_PATH = OUTPUT_JSON_PATH.replace(".json", "_float.json")


# ----------------------
# Preprocess
# ----------------------

def resize_image(image, target_size=224):
    w, h = image.size
    ratio = min(target_size/w, target_size/h)
    new_w = int(w*ratio)
    new_h = int(h*ratio)
    resized = image.resize((new_w,new_h), Image.BICUBIC)
    pad = Image.new('RGB', (target_size,target_size), (128,128,128))
    pad.paste(resized, ((target_size-new_w)//2, (target_size-new_h)//2))
    return pad


def preprocess(image_path, dimension, processor):
    prompt = qwen_prompt(dimension, image_path)
    img_inputs, _ = process_vision_info(prompt)
    resized = resize_image(img_inputs[0], IMAGE_SIZE)

    txt = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    inp = processor(
        images=resized,
        text=txt,
        return_tensors="pt"
    )

    return {
        "pixel_values": inp["pixel_values"].squeeze(0).to(DEVICE),
        "input_ids": inp["input_ids"].squeeze().to(DEVICE),
        "attention_mask": inp["attention_mask"].squeeze().to(DEVICE),
        "image_grid_thw": inp["image_grid_thw"][0].to(DEVICE)
            if "image_grid_thw" in inp else None,
    }


# ----------------------
# CLM head
# ----------------------

class CLMHead(nn.Module):
    def __init__(self, in_dim=3584, num_thresholds=4):
        super().__init__()
        self.score = nn.Linear(in_dim, 1)
        self.thresholds = nn.Parameter(torch.zeros(num_thresholds))

    def forward(self, x):
        f = self.score(x)                         # [B,1]
        return f - self.thresholds.view(1,-1)     # [B,K-1]


# ----------------------
# Load model + LoRA + head
# ----------------------

def load_single_model(dimension):
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        return_dict=True
    )

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    dim_dir = os.path.join(OUTPUT_DIR, f"dimension_{dimension}")

    # load LoRA
    model = PeftModel.from_pretrained(
        base, dim_dir, torch_dtype=torch.bfloat16
    )
    model.eval()
    model.tokenizer = processor.tokenizer

    # 🔥 cast LoRA to bf16
    for n, p in model.named_parameters():
        if "lora" in n:
            p.data = p.data.to(torch.bfloat16)

    hidden = getattr(model.config, "hidden_size", 3584)

    clm_head = CLMHead(in_dim=hidden, num_thresholds=4)
    clm_path = os.path.join(dim_dir, "clm_head.pth")
    clm_head.load_state_dict(torch.load(clm_path, map_location="cpu"))

    # cast to bf16
    clm_head = clm_head.to(dtype=torch.bfloat16, device=DEVICE)

    clm_head.eval()
    return model, clm_head, processor


# ----------------------
# Predict (CLM)
# ----------------------

def predict_clm(image_path, dimension, processor, model, clm_head):
    inp = preprocess(image_path, dimension, processor)

    with torch.no_grad():
        out = model(
            input_ids=inp["input_ids"].unsqueeze(0),
            attention_mask=inp["attention_mask"].unsqueeze(0),
            pixel_values=inp["pixel_values"].unsqueeze(0),
            image_grid_thw=(
                inp["image_grid_thw"].unsqueeze(0)
                if inp["image_grid_thw"] is not None else None
            ),
            return_dict=True,
            output_hidden_states=True,
        )

    hidden = out.hidden_states[-1][:, -1, :].to(torch.bfloat16)

    logits = clm_head(hidden)   # [1,4]
    probs = torch.sigmoid(logits)[0]

    passes = (probs > 0.5).sum().item()
    score = passes + 1

    return float(score) if FLOAT else int(score)


# ----------------------
# Batch Predict
# ----------------------

def batch_predict(data_path, img_dir, output_json):
    df = pd.read_csv(data_path)
    name_col = df.columns[0]

    results = {"ArtEdu": {MODEL_NAME: {}}}

    for dim in DIMENSION_LIST:
        print(f"\n---- CLM inference for: {dim}")

        model, clm_head, processor = load_single_model(dim)

        for row in tqdm(df.itertuples(index=False)):
            name = getattr(row, name_col)

            img_path = os.path.join(img_dir, name)
            if name not in results["ArtEdu"][MODEL_NAME]:
                results["ArtEdu"][MODEL_NAME][name] = {}

            pred = predict_clm(img_path, dim, processor, model, clm_head)
            orig_raw = getattr(row, dim)
            orig = int(orig_raw) if not pd.isna(orig_raw) else None

            results["ArtEdu"][MODEL_NAME][name][dim] = {
                "original": orig,
                "predicted": pred,
            }

        del model, clm_head
        torch.cuda.empty_cache()

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("\nSaved:", output_json)


# ----------------------
# Main
# ----------------------

if __name__ == "__main__":
    print("=== CLM INFERENCE START ===")
    print("Model:", MODEL_NAME)
    print("Data:", DATA_PATH)

    batch_predict(DATA_PATH, IMAGE_DIR, OUTPUT_JSON_PATH)

    print("=== DONE ===")
