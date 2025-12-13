import requests
import torch
import pandas as pd
import os
import re
import json
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from utils.data_process import DIMENSION_LIST, resize_image
from utils.prompt_lib import rubric_prompt


MODEL_NAME = "XiaomiMiMo/MiMo-VL-7B-RL"
DATA_PATH = ""
IMAGES_PATH = ""
OUTPUT_JSON_PATH = ""

def mimo_vl_get_score(model, processor, dimension, image_path):
    """
    Generate score for a given dimension using Xiaomi MiMo-VL-7B-RL.
    
    Args:
        model: Loaded MiMo model.
        processor: Loaded MiMo processor.
        dimension: Dimension to evaluate (e.g., "originality").
        image_path: Path to the image file.
    
    Returns:
        clean_answer: Extracted integer score from model output.
    """
    # Build prompt dynamically based on dimension
    messages = rubric_prompt(dimension=dimension, image_path=image_path)

    image = Image.open(image_path).convert("RGB")

    image = resize_image(image, target_size=448)

    # Tokenize and prepare inputs
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(output_text)
    
    # Extract numerical answer
    match = re.findall(r'\d+', output_text)
    score = int(match[-1]) if match else None

    return score


def request_artedu_mimo(model_name, data_path, images_path, output_json_path):
    model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto")
    print(f"Model loaded on device: {next(model.parameters()).device}")  # DEBUG
    processor = AutoProcessor.from_pretrained(model_name)

    df = pd.read_csv(data_path)
    first_col_name = df.columns[0]
    output_json = {"ArtEdu": {model_name: {}}}

    for row in df.itertuples(index=False):
        image_name = getattr(row, first_col_name)
        image_path = os.path.join(images_path, image_name)
        result_dict = {}

        for dimension in DIMENSION_LIST:
            score = mimo_vl_get_score(model=model, processor=processor, dimension=dimension.replace("_", " ").title(), image_path=image_path)
            origin_score = getattr(row, dimension)
            result_dict[dimension] = {
                "original": origin_score,
                "current": score
            }

        output_json["ArtEdu"][model_name][image_name] = result_dict

    # Save results
    with open(output_json_path, "w") as f:
        json.dump(output_json, f, indent=4)

    print("Finished.")


if __name__ == "__main__":
    request_artedu_mimo(model_name=MODEL_NAME, data_path=DATA_PATH, images_path=IMAGES_PATH, output_json_path=OUTPUT_JSON_PATH)
