import os
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from transformers import TrainerCallback
from utils.data_process import DIMENSION_LIST


def compute_metrics(pred):
    logits, labels = pred
    print("Logits:", logits)
    print("Labels:", labels)

    logits = np.array(logits).squeeze()
    preds = logits
    mse = mean_squared_error(labels, preds)   # mse
    return {"eval_mse": mse}


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_mse = float('inf')
        self.counter = 0
        self.early_stop = False

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics", {})
        if not metrics or "eval_mse" not in metrics:
            return

        current_mse = metrics["eval_mse"]
        print(f"Current MSE: {current_mse:.4f}, Best MSE: {self.best_mse:.4f}")

        if current_mse < self.best_mse - self.min_delta:
            self.best_mse = current_mse
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                control.should_training_stop = True
                self.early_stop = True


class SaveBestModelCallback(TrainerCallback):
    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir
        self.best_mse = float('inf')

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics", {})
        if not metrics or "eval_mse" not in metrics:
            return

        current_mse = metrics["eval_mse"]
        print(f"Current MSE: {current_mse:.4f}, Best MSE: {self.best_mse:.4f}")

        if current_mse < self.best_mse:
            print("New best model found. Saving...")
            self.best_mse = current_mse

            for dim in DIMENSION_LIST:
                # 保存该维度的 LoRA 模块
                lora_output_path = os.path.join(self.output_dir, f"lora_{dim}_best")
                self.model.lora_modules[dim].save_pretrained(lora_output_path)

                # 保存该维度的 regression head
                regression_head_output_path = os.path.join(self.output_dir, f"regression_head_{dim}_best.pth")
                torch.save(
                    self.model.regression_heads[dim].state_dict(),
                    regression_head_output_path
                )
                

class SaveBestModelCallback_Multilora(TrainerCallback):
    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir
        self.best_mse = float('inf')

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics", {})
        if not metrics or "eval_mse" not in metrics:
            return

        current_mse = metrics["eval_mse"]
        print(f"[Eval] Current MSE: {current_mse:.4f}, Best MSE: {self.best_mse:.4f}")

        if current_mse < self.best_mse:
            print("🎯 New best model found! Saving all components...")
            self.best_mse = current_mse

            # 保存主干模型（不包括 LoRA）
            base_path = os.path.join(self.output_dir, "base_model_best")
            self.model.base.save_pretrained(base_path)

            # 保存每个维度的 LoRA adapter 和 regression head
            for dim in DIMENSION_LIST:
                lora_output_path = os.path.join(self.output_dir, f"lora_{dim}_best")
                self.model.lora_modules[dim].save_pretrained(lora_output_path)

                regression_head_output_path = os.path.join(self.output_dir, f"regression_head_{dim}_best.pth")
                torch.save(
                    self.model.regression_heads[dim].state_dict(),
                    regression_head_output_path
                )

            # 保存当前 best MSE
            mse_file = os.path.join(self.output_dir, "best_mse.txt")
            with open(mse_file, "w") as f:
                f.write(f"{self.best_mse:.6f}\n")
