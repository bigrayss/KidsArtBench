from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

import pandas as pd
import os
import re
import json
import torch

from utils.prompt_lib import rubric_prompt
from utils.data_process import DIMENSION_LIST, resize_image

MODEL_PATH = "moonshotai/Kimi-VL-A3B-Instruct"
DATA_PATH = ""
IMAGES_PATH = ""
OUTPUT_JSON_PATH = ""

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
).eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

def kimi_get_score(dimension, image_path):
    messages = rubric_prompt(dimension, image_path)
    
    try:
        image = Image.open(image_path).convert("RGB")
        
        image = resize_image(image, target_size=448)

        text_input = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            return_dict=False
        )

        inputs = processor(
            images=image,
            text=text_input,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=10)
            decoded = processor.batch_decode(
                generation[:, input_len:], 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
        
        print(f"[{dimension}] {decoded}")

        match = re.search(r'\d+', decoded.strip())
        score = int(match.group()) if match else None
        return score
    
    except Exception as e:
        print(f"Error processing {image_path} for {dimension}: {e}")
        return None


def request_artedu_kimi(data_path, images_path, output_json_path):
    df = pd.read_csv(data_path)
    first_col_name = df.columns[0]
    output_json = {"ArtEdu": {"kimi_vl_a3b": {}}}

    for row in df.itertuples(index=False):
        image_name = getattr(row, first_col_name)
        image_path = os.path.join(images_path, image_name)
        result_dict = {}

        for dimension in DIMENSION_LIST:
            score = kimi_get_score(dimension.replace("_", " ").title(), image_path)
            origin_score = getattr(row, dimension)
            score_dict = {"original": origin_score, "current": score}
            result_dict[dimension] = score_dict

        output_json["ArtEdu"]["kimi_vl_a3b"][image_name] = result_dict

    with open(output_json_path, "w") as file:
        json.dump(output_json, file, indent=4)

    print("Processing complete.")


if __name__ == "__main__":
    request_artedu_kimi(DATA_PATH, IMAGES_PATH, OUTPUT_JSON_PATH)
