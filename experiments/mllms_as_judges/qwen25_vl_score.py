import json
import pandas as pd
import os
import re
import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from utils.data_process import DIMENSION_LIST
from utils.prompt_lib import qwen_prompt, rubric_prompt


MODEL_PATH = {
    "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-72B-Instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
    "QVQ-72B-Preview": "Qwen/QVQ-72B-Preview"
}
MODEL_NAME = "Qwen/Qwen2.5-VL-32B-Instruct"
DATA_PATH = ""
IMAGES_PATH = ""
OUTPUT_JSON_PATH = ""
REFERENCE_IMAGES_DIR = ""


def get_few_shot_examples(dimension, reference_images_dir=REFERENCE_IMAGES_DIR):
    """
    Get few-shot examples for a given dimension from reference images directory.
    
    Args:
        dimension (str): The evaluation dimension (e.g., "Realism", "Deformation")
        reference_images_dir (str): Path to the reference images directory
    
    Returns:
        list: A list of few-shot examples with low and high score images
    """
    dimension_formatted = dimension.lower().replace(" ", "_")
    
    # Define the example images (score 1 and score 5)
    low_score_image = os.path.join(reference_images_dir, f"{dimension_formatted}_1.jpg")
    high_score_image = os.path.join(reference_images_dir, f"{dimension_formatted}_5.jpg")
    
    # Check if files exist with .jpg extension
    if not os.path.exists(low_score_image):
        # Try with .png extension
        low_score_image = os.path.join(reference_images_dir, f"{dimension_formatted}_1.png")
    
    if not os.path.exists(high_score_image):
        # Try with .png extension
        high_score_image = os.path.join(reference_images_dir, f"{dimension_formatted}_5.png")
    
    examples = []
    
    # Add low score example (score 1)
    if os.path.exists(low_score_image):
        examples.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": low_score_image,
                },
                {
                    "type": "text",
                    "text": f"This is an example of {dimension} with a score of 1 (poor quality)."
                }
            ]
        })
        examples.append({
            "role": "assistant",
            "content": "1"
        })
    
    # Add high score example (score 5)
    if os.path.exists(high_score_image):
        examples.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": high_score_image,
                },
                {
                    "type": "text",
                    "text": f"This is an example of {dimension} with a score of 5 (excellent quality)."
                }
            ]
        })
        examples.append({
            "role": "assistant",
            "content": "5"
        })
    
    return examples


def qwen_prompt_few_shot(dimension, image_path):
    """
    Create a prompt with few-shot examples for Qwen model.
    
    Args:
        dimension (str): The evaluation dimension
        image_path (str): Path to the student's artwork image
    
    Returns:
        list: Conversation messages with few-shot examples
    """
    # Get few-shot examples
    few_shot_examples = get_few_shot_examples(dimension)
    
    # Create main prompt
    dimension_prompt = f"<image>Assess the student's artwork based on the '{dimension}' criterion and provide a score (1-5)."
    
    # Add rubric based on dimension
    if dimension == "Realism":
        dimension_prompt += (
            " This criterion assesses the accuracy of proportions, textures, lighting, and perspective to create a lifelike depiction."
            "\n5: The artwork exhibits exceptional detail and precision in depicting Realism features. Textures and lighting are used masterfully to mimic real-life appearances with accurate proportions and perspective. The representation is strikingly lifelike, demonstrating advanced skills in realism."
            "\n4: The artwork presents a high level of detail and accuracy in the portrayal of subjects. Proportions and textures are very well executed, and the lighting enhances the realism. Although highly Realism, minor discrepancies in perspective or detail might be noticeable."
            "\n3: The artwork represents subjects with a moderate level of realism. Basic proportions are correct, and some textures and lighting effects are used to enhance realism. However, the depiction may lack depth or detail in certain areas."
            "\n2: The artwork attempts realism but struggles with accurate proportions and detailed textures. Lighting and perspective may be inconsistently applied, resulting in a less convincing depiction."
            "\n1: The artwork shows minimal attention to Realism details. Proportions, textures, and lighting are poorly executed, making the depiction far from lifelike."
        )
    elif dimension == "Deformation":
        dimension_prompt += (
            " This criterion evaluates the artist's ability to creatively and intentionally deform reality to convey a message, emotion, or concept."
            "\n5: The artwork demonstrates masterful use of deformation to enhance the emotional or conceptual impact of the piece. The transformations are thoughtful and integral to the artwork's message, seamlessly blending with the composition to engage viewers profoundly."
            "\n4: The artwork effectively uses deformation to express artistic intentions. The modifications are well-integrated and contribute significantly to the viewer's understanding or emotional response. Minor elements of the deformation might detract from its overall effectiveness."
            "\n3: The artwork includes noticeable deformations that add to its artistic expression. While these elements generally support the artwork's theme, they may be somewhat disjointed from the composition, offering mixed impact on the viewer."
            "\n2: The artwork attempts to use deformation but does so with limited success. The deformations are present but feel forced or superficial, only marginally contributing to the artwork's expressive goals."
            "\n1: The artwork features minimal or ineffective deformation, with little to no enhancement of the artwork's message or emotional impact. The attempts at deformation seem disconnected from the artwork's overall intent."
        )
    elif dimension == "Imagination":
        dimension_prompt += (
            " This criterion evaluates the artist's ability to use their creativity to form unique and original ideas within their artwork."
            "\n5: The artwork displays a profound level of originality and creativity, introducing unique concepts or interpretations that are both surprising and thought-provoking."
            "\n4: The artwork presents creative ideas that are both original and nicely executed, though they may be similar to conventional themes."
            "\n3: The artwork shows some creative ideas, but they are somewhat predictable and do not stray far from traditional approaches."
            "\n2: The artwork has minimal creative elements, with ideas that are largely derivative and lack originality."
            "\n1: The artwork lacks imagination, with no discernible original ideas or creative concepts."
        )
    elif dimension == "Color Richness":
        dimension_prompt += (
            " This criterion assesses the use and range of colors to create a visually engaging experience."
            "\n5: The artwork uses a wide and harmonious range of colors, each contributing to a vivid and dynamic composition."
            "\n4: The artwork features a good variety of colors that are well-balanced, enhancing the visual appeal of the piece."
            "\n3: The artwork includes a moderate range of colors, but the palette may not fully enhance the subject matter."
            "\n2: The artwork has limited color variety, with a palette that does not significantly contribute to the piece's impact."
            "\n1: The artwork shows poor use of colors, with a very restricted range that detracts from the visual experience."
        )
    elif dimension == "Color Contrast":
        dimension_prompt += (
            " This criterion evaluates the effective use of contrasting colors to enhance artistic expression."
            "\n5: The artwork masterfully employs contrasting colors to create a striking and effective visual impact."
            "\n4: The artwork effectively uses contrasting colors to enhance visual interest, though the contrast may be less pronounced."
            "\n3: The artwork has some contrast in colors, but it is not used effectively to enhance the artwork's overall appeal."
            "\n2: The artwork makes minimal use of color contrast, resulting in a lackluster visual impact."
            "\n1: The artwork lacks effective color contrast, making the piece visually unengaging."
        )
    elif dimension == "Line Combination":
        dimension_prompt += (
            " This criterion assesses the integration and interaction of lines within the artwork."
            "\n5: The artwork exhibits exceptional integration of line combinations, creating a harmonious and engaging visual flow."
            "\n4: The artwork displays good use of line combinations that contribute to the overall composition, though some areas may lack cohesion."
            "\n3: The artwork shows average use of line combinations, with some effective sections but overall lacking in cohesiveness."
            "\n2: The artwork has minimal effective use of line combinations, with lines that often clash or do not contribute to a unified composition."
            "\n1: The artwork shows poor integration of lines, with combinations that disrupt the visual harmony of the piece."
        )
    elif dimension == "Line Texture":
        dimension_prompt += (
            " This criterion evaluates the variety and execution of line textures within the artwork."
            "\n5: The artwork demonstrates a wide variety of line textures, each skillfully executed to enhance the piece's aesthetic and thematic elements."
            "\n4: The artwork includes a good range of line textures, well executed but with some areas that may lack definition."
            "\n3: The artwork features moderate variety in line textures, with generally adequate execution but lacking in detail."
            "\n2: The artwork has limited line textures, with execution that does not significantly contribute to the artwork's quality."
            "\n1: The artwork lacks variety and sophistication in line textures, resulting in a visually dull piece."
        )
    elif dimension == "Picture Organization":
        dimension_prompt += (
            " This criterion evaluates the overall composition and spatial arrangement within the artwork."
            "\n5: The artwork is impeccably organized, with each element thoughtfully placed to create a balanced and compelling composition."
            "\n4: The artwork has a good organization, with a well-arranged composition that effectively guides the viewer's eye, though minor elements may disrupt the flow."
            "\n3: The artwork has an adequate organization, but the composition may feel somewhat unbalanced or disjointed."
            "\n2: The artwork shows poor organization, with a composition that lacks coherence and does not effectively engage the viewer."
            "\n1: The artwork is poorly organized, with a chaotic composition that detracts from the piece's overall impact."
        )
    elif dimension == "Transformation":
        dimension_prompt += (
            " This criterion assesses the artist's ability to transform traditional or familiar elements into something new and unexpected."
            "\n5: The artwork is transformative, offering a fresh and innovative take on traditional elements, significantly enhancing the viewer's experience."
            "\n4: The artwork successfully transforms familiar elements, providing a new perspective, though the innovation may not be striking."
            "\n3: The artwork shows some transformation of familiar elements, but the changes are somewhat predictable and not highly innovative."
            "\n2: The artwork attempts transformation but achieves only minimal success, with changes that are either too subtle or not effectively executed."
            "\n1: The artwork lacks transformation, with traditional elements that are replicated without any significant innovation or creative reinterpretation."
        )
    
    dimension_prompt += "\nOnly output a score(1-5)."
    
    # Build conversation with few-shot examples
    conversation = []
    
    # Add few-shot examples
    conversation.extend(few_shot_examples)
    
    # Add main task
    conversation.append({
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {
                "type": "text",
                "text": dimension_prompt,
            },
        ]
    })
    
    return conversation


def qwen25_vl_get_score(model, processor, dimension, image_path):
    # prompt
    messages = rubric_prompt(dimension=dimension, image_path=image_path)
    # messages = qwen_prompt_few_shot(dimension=dimension, image_path=image_path)

    print("-"*10)
    print(messages)

    # load images and prepare for inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128, use_cache=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        answer = re.search(r"\d+", output_text[0])
        clean_answer = int(answer.group()) if answer else None

    return clean_answer


def request_ArtEdu(model_name, data_path, images_path, output_json_path):

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # quantization_config=quantization_config,
        device_map="auto",
    )

    # if torch.cuda.device_count() > 1:
    #     model = DataParallel(model)

    min_pixels = 200704  # 50176  # 256*28*28
    max_pixels = 200704  # 50176  # 1280*28*28
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True)
    # processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    # prepare for the data
    df = pd.read_csv(data_path)
    first_col_name = df.columns[0]
    output_json = {
        "ArtEdu":
            {
                model_name: {}
            }
    }

    # get the score
    for row in df.itertuples(index=False):
        image_name = getattr(row, first_col_name)
        image_path = os.path.join(images_path, image_name)
        result_dict = {}
        for dimension in DIMENSION_LIST:
            score = qwen25_vl_get_score(model=model, processor=processor, dimension=dimension.replace("_", " ").title(), image_path=image_path)
            origin_score = getattr(row, dimension)
            score_dict = {
                "original": origin_score,
                "current": score,
            }
            result_dict[dimension] = score_dict
        output_json["ArtEdu"][model_name][image_name] = result_dict

    # save the output
    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(output_json, file, ensure_ascii=False, indent=4)

    print("finish!")
    return


if __name__ == '__main__':
    request_ArtEdu(model_name=MODEL_NAME, data_path=DATA_PATH,
                   images_path=IMAGES_PATH, output_json_path=OUTPUT_JSON_PATH)
    
