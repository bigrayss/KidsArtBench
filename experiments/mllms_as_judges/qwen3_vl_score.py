import json
import pandas as pd
import os
import re
import torch
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from utils.data_process import DIMENSION_LIST
from utils.prompt_lib import qwen_prompt, rubric_prompt

MODEL_PATH = {
    "Qwen3-VL-30B-A3B-Instruct": "Qwen/Qwen3-VL-30B-A3B-Instruct",
}
MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DATA_PATH = ""
IMAGES_PATH = ""
OUTPUT_JSON_PATH = ""


def qwen3_vl_get_score(model, processor, dimension, image_path):
    # Create prompt using rubric_prompt
    messages = rubric_prompt(dimension=dimension, image_path=image_path)

    print("-" * 10)
    print(messages)

    # Prepare inputs for the model
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        answer = re.search(r"\d+", output_text[0])
        clean_answer = int(answer.group()) if answer else None

    return clean_answer


def request_ArtEdu(model_name, data_path, images_path, output_json_path):
    # Load model with recommended settings
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto"
    )

    min_pixels = 448 * 448  # 200704
    max_pixels = 448 * 448

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels
        )

    # Prepare data
    df = pd.read_csv(data_path)
    first_col_name = df.columns[0]
    output_json = {
        "ArtEdu": {
            model_name: {}
        }
    }

    # Process each image and get scores
    for row in df.itertuples(index=False):
        image_name = getattr(row, first_col_name)
        image_path = os.path.join(images_path, image_name)
        result_dict = {}
        
        for dimension in DIMENSION_LIST:
            score = qwen3_vl_get_score(
                model=model, 
                processor=processor, 
                dimension=dimension.replace("_", " ").title(), 
                image_path=image_path
            )
            origin_score = getattr(row, dimension)
            score_dict = {
                "original": origin_score,
                "current": score,
            }
            result_dict[dimension] = score_dict
            
        output_json["ArtEdu"][model_name][image_name] = result_dict

    # Save results
    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(output_json, file, ensure_ascii=False, indent=4)

    print("finish!")


def test_single_image_qwen3(model_name, image_path, dimension):
    """
    Test scoring a single image for a specific dimension using Qwen3-VL model
    
    Args:
        model_name (str): Name/path of the Qwen3-VL model to use
        image_path (str): Path to the image file
        dimension (str): The dimension to evaluate (e.g., "Creativity", "Technique")
    
    Returns:
        int: The score extracted from the model's response
    """
    
    # Load model with recommended settings
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto"
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)

    # Get score using existing function
    score = qwen3_vl_get_score(
        model=model, 
        processor=processor, 
        dimension=dimension.replace("_", " ").title(), 
        image_path=image_path
    )
    
    print(f"Image: {image_path}")
    print(f"Dimension: {dimension}")
    print(f"Score: {score}")
    
    return score



if __name__ == '__main__':
    request_ArtEdu(
        model_name=MODEL_NAME, 
        data_path=DATA_PATH,
        images_path=IMAGES_PATH, 
        output_json_path=OUTPUT_JSON_PATH
    )

