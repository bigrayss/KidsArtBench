import torch
import pandas as pd
import os
import json
from PIL import Image
from tqdm.auto import tqdm
from peft import PeftModel
from utils.data_process import DIMENSION_LIST
from utils.prompt_lib import qwen_prompt
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
PROJECT_NAME = ""
OUTPUT_DIR = f"../{PROJECT_NAME}"
DATA_PATH = ""
IMAGE_DIR = ""
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_JSON_PATH = f"../{PROJECT_NAME}.json"
PLOT_SAVE_PATH = f""


class RegressionHead(torch.nn.Module):
    def __init__(self, in_dim=3584):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 512),
            torch.nn.LayerNorm(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.seq(x)


def load_model_and_heads():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    models = {}
    regression_heads = {}
    for dim in DIMENSION_LIST:
        dim_dir = os.path.join(OUTPUT_DIR, f"dimension_{dim}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            dim_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        regression_head = RegressionHead(in_dim=getattr(model.config, 'hidden_size', 3584))
        regression_head_path = os.path.join(dim_dir, "regression_head.pth")
        regression_head.load_state_dict(torch.load(regression_head_path, map_location=DEVICE))
        regression_head = regression_head.to(DEVICE).to(torch.bfloat16)
        model.regression_head = regression_head
        models[dim] = model
        regression_heads[dim] = regression_head
    return models, processor


def resize_image(image, target_size=224):
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
    dim_dir = os.path.join(OUTPUT_DIR, f"dimension_{dimension}")

    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     dim_dir,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )

    # Load base model first
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Load LoRA adapters using PEFT
    model = PeftModel.from_pretrained(
        base_model,
        dim_dir,
        torch_dtype=torch.bfloat16,
    )

    regression_head = RegressionHead(in_dim=getattr(model.config, 'hidden_size', 3584))
    regression_head_path = os.path.join(dim_dir, "regression_head.pth")
    regression_head.load_state_dict(torch.load(regression_head_path, map_location=DEVICE))
    regression_head = regression_head.to(DEVICE).to(torch.bfloat16)
    model.regression_head = regression_head
    return model


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
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        cls_token = hidden_states[-1][:, -1, :].to(DEVICE)
        logits = model.regression_head(cls_token).squeeze(-1)
        score = torch.sigmoid(logits) * 4.0 + 1.0
    
    del outputs, hidden_states, cls_token, logits
    torch.cuda.empty_cache()
    
    return round(score.item())


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


# --------------------- PLOT ---------------------


def compute_subspace_overlap_score():
    import numpy as np
    from scipy.linalg import subspace_angles

    TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

    # {dim_name: {layer_name: merged_lora_matrix}}
    dim_lora = {}

    for dim in DIMENSION_LIST:
        print(f"Loading LoRA for: {dim}")
        dim_dir = os.path.join(OUTPUT_DIR, f"dimension_{dim}")

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, dim_dir, torch_dtype=torch.bfloat16)

        state_dict = model.state_dict()
        layer_dict = {}

        for name in state_dict:
            for tgt in TARGET_MODULES:
                if tgt in name and "lora_A" in name:
                    # lora_B
                    b_name = name.replace("lora_A", "lora_B")
                    if b_name in state_dict:
                        A = state_dict[name].detach().cpu().float().numpy()
                        B = state_dict[b_name].detach().cpu().float().numpy()
                        merged = B @ A
                        layer_dict[tgt] = merged.reshape(-1, 1) if merged.ndim == 1 else merged

        dim_lora[dim] = layer_dict
        del model, base_model
        torch.cuda.empty_cache()

    def subspace_overlap(A, B, k=5):
        Ua, _, _ = np.linalg.svd(A, full_matrices=False)
        Ub, _, _ = np.linalg.svd(B, full_matrices=False)
        k = min(k, Ua.shape[1], Ub.shape[1])
        angles = subspace_angles(Ua[:, :k], Ub[:, :k])
        return float(np.mean(np.cos(angles) ** 2))

    dims = list(dim_lora.keys())
    n = len(dims)
    overall_scores = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                overall_scores[i, j] = 1.0
            else:
                overlaps = []
                for tgt in TARGET_MODULES:
                    if tgt in dim_lora[dims[i]] and tgt in dim_lora[dims[j]]:
                        overlaps.append(subspace_overlap(dim_lora[dims[i]][tgt], dim_lora[dims[j]][tgt]))
                overall_scores[i, j] = np.mean(overlaps) if overlaps else 0.0

    print("=== Subspace Overlap Matrix ===")
    print(overall_scores)
    return {"dimensions": dims, "overlap_scores": overall_scores}


def visualize_subspace_overlap(overlap_results):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    import matplotlib
    matplotlib.use('Agg')
    
    overlap_scores = overlap_results.get("overlap_scores", None)
    overlap_scores_pca = overlap_results.get("overlap_scores_pca", None)
    dimensions = overlap_results["dimensions"]
    
    print(f"Visualizing results with dimensions: {dimensions}")
    print(f"Overlap scores shape: {overlap_scores.shape if overlap_scores is not None else 'None'}")
    print(f"PCA overlap scores shape: {overlap_scores_pca.shape if overlap_scores_pca is not None else 'None'}")
    
    if overlap_scores is not None or overlap_scores_pca is not None:
        n_plots = sum([1 for x in [overlap_scores, overlap_scores_pca] if x is not None])
        fig, axes = plt.subplots(1, n_plots, figsize=(8*n_plots, 6))
        if n_plots == 1:
            axes = [axes]
        elif n_plots == 0:
            print("No data to visualize")
            return
        
        idx = 0
        if overlap_scores is not None:
            print(f"Plotting basic overlap scores with shape {overlap_scores.shape}")
            sns.heatmap(overlap_scores,
                        xticklabels=dimensions,
                        yticklabels=dimensions,
                        annot=True,
                        fmt=".3f",
                        cmap="viridis",
                        vmin=0,
                        vmax=1,
                        ax=axes[idx])
            axes[idx].set_title("subspace overlap scores")
            axes[idx].tick_params(axis='x', rotation=45)    
            axes[idx].tick_params(axis='y', rotation=0)
            idx += 1
            
        if overlap_scores_pca is not None:
            print(f"Plotting PCA overlap scores with shape {overlap_scores_pca.shape}")
            sns.heatmap(overlap_scores_pca,
                        xticklabels=dimensions,
                        yticklabels=dimensions,
                        annot=True,
                        fmt=".3f",
                        cmap="viridis",
                        vmin=0,
                        vmax=1,
                        ax=axes[idx if n_plots > 1 else 0])
            axes[idx if n_plots > 1 else 0].set_title("子空间重叠分数 (PCA方法)")
            axes[idx if n_plots > 1 else 0].tick_params(axis='x', rotation=45)
            axes[idx if n_plots > 1 else 0].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
  
        output_file = PLOT_SAVE_PATH
        try:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"子空间重叠分析图已保存为 {output_file}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        print("No overlap scores to visualize")


if __name__ == '__main__':
    batch_predict_scores(DATA_PATH, IMAGE_DIR, OUTPUT_JSON_PATH)
  
    # overlap score
    # overlap_results = compute_subspace_overlap_score()
    
    # visualize_subspace_overlap(overlap_results)

    print("Finished!")
