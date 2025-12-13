from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import torch
import pandas as pd
import os
import re
import json

from utils.prompt_lib import gemma_prompt
from utils.data_process import DIMENSION_LIST, resize_image

# Configuration
MODEL_ID = "google/gemma-3-12b-it"
DATA_PATH = ""
IMAGES_PATH = ""
OUTPUT_JSON_PATH = ""

# Load Model and Processor
model = Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Chat Template:", processor.chat_template)

def gemma3_get_score(dimension, image_path):
    # Open image
    messages = gemma_prompt(dimension, image_path)
    
    image = Image.open(image_path).convert("RGB")

    image = resize_image(image, target_size=448)

    # Prepare inputs
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    # Generate response
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=10)
        decoded = processor.decode(generation[0][input_len:], skip_special_tokens=True)

    print(decoded)

    # Extract numerical score
    match = re.search(r'\d+', decoded.strip())
    clean_answer = int(match.group()) if match else None

    return clean_answer

def request_artedu_gemma3(data_path, images_path, output_json_path):
    df = pd.read_csv(data_path)
    first_col_name = df.columns[0]
    output_json = {"ArtEdu": {"gemma3_4b": {}}}

    for row in df.itertuples(index=False):
        image_name = getattr(row, first_col_name)
        image_path = os.path.join(images_path, image_name)
        result_dict = {}

        for dimension in DIMENSION_LIST:
            score = gemma3_get_score(dimension.replace("_", " ").title(), image_path)
            origin_score = getattr(row, dimension)
            score_dict = {"original": origin_score, "current": score}
            result_dict[dimension] = score_dict

        output_json["ArtEdu"]["gemma3_4b"][image_name] = result_dict

    # Save results
    with open(output_json_path, "w") as file:
        json.dump(output_json, file, indent=4)

    print("Processing complete.")

if __name__ == "__main__":
    request_artedu_gemma3(DATA_PATH, IMAGES_PATH, OUTPUT_JSON_PATH)
