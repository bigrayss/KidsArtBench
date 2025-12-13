import os
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.transforms import functional as F
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.prompt_lib import qwen_prompt


DIMENSION_LIST = [
    "realism",
    "deformation",
    "imagination",
    "color_richness",
    "color_contrast",
    "line_combination",
    "line_texture",
    "picture_organization",
    "transformation"
]


def resize_image(image: Image.Image, target_size=448):
    """保持宽高比缩放+灰色填充正方形"""
    # 计算缩放比例
    w, h = image.size
    ratio = min(target_size/w, target_size/h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    
    # 先缩放
    resized = image.resize((new_w, new_h), Image.BICUBIC)
    
    # 再填充为正方形（灰色背景）
    padded = Image.new('RGB', (target_size, target_size), (128, 128, 128))
    padded.paste(resized, ((target_size-new_w)//2, (target_size-new_h)//2))
    return padded

def apply_augmentation(image: Image.Image, mode: str = 'rotation_blur'):
    if mode == 'rotation_blur':
        angle = random.uniform(-15, 15)
        image = F.rotate(image, angle)
        if random.random() < 0.2:
            image = F.gaussian_blur(image, kernel_size=(5, 5))
    elif mode == 'rotation':
        angle = random.uniform(-15, 15)
        image = F.rotate(image, angle)
    elif mode == 'blur':
        if random.random() < 0.2:
            image = F.gaussian_blur(image, kernel_size=(5, 5))
    elif AUGMENTATION_MODE == 'hflip':
        if random.random() < 0.5:
            resized_images = F.hflip(resized_images)
    elif AUGMENTATION_MODE == 'colorjitter':
        jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        resized_images = jitter(resized_images)

    return image


class ScoreDataset(Dataset):
    def __init__(self, csv_path, images_path, processor, size=448, train=True, indices=None, double_data=False):
        self.processor = processor
        self.size = size
        self.data = []
        self.train = train

        df = pd.read_csv(csv_path)

        all_data = []
        for idx, row in df.iterrows():
            for dim in DIMENSION_LIST:
                score = int(row[dim]) if not math.isnan(row[dim]) else 3
                assert 1 <= score <= 5, f"Label out of range: {score}"
                all_data.append({
                    "image_path": os.path.join(images_path, row["name"]),
                    "dim": dim,
                    "score": score,
                })

        if double_data:
            all_data *= 2

        if indices is None:
            self.data = all_data
        else:
            self.data = []
            for row_idx in indices:
                start_idx = row_idx * len(DIMENSION_LIST)
                end_idx = (row_idx + 1) * len(DIMENSION_LIST)
                self.data.extend(all_data[start_idx:end_idx])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        # print(f"[Dataset] Loading index {idx}")
        item = self.data[idx]
        
        messages = qwen_prompt(item["dim"], item["image_path"])

        text_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, _ = process_vision_info(messages)
        resized_images = resize_image(image_inputs[0], target_size=self.size)

        # if self.train and random.random() < 0.5:
        #     resized_images = transforms.functional.hflip(resized_images)

        # if self.train:
        #     if AUGMENTATION_MODE == 'hflip':
        #         if random.random() < 0.5:
        #             resized_images = F.hflip(resized_images)

        #     elif AUGMENTATION_MODE == 'rotation':
        #         angle = random.uniform(-15, 15)
        #         resized_images = F.rotate(resized_images, angle)

        #     elif AUGMENTATION_MODE == 'colorjitter':
        #         jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        #         resized_images = jitter(resized_images)

        #     elif AUGMENTATION_MODE == 'blur':
        #         if random.random() < 0.2:
        #             resized_images = F.gaussian_blur(resized_images, kernel_size=(5, 5))

        inputs = self.processor(
            images=resized_images,
            text=text_inputs,
            return_tensors="pt",
        )

        label = int(item["score"])

        return {
            "pixel_values": inputs['pixel_values'].squeeze(0),
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": torch.tensor(label, dtype=torch.float),
            "image_grid_thw": inputs.image_grid_thw[0],
            "dim": item["dim"],
            "index": idx,
        }


class ScoreDataset_Aug(Dataset):
    def __init__(self, csv_path, processor, augmentation='rotation_blur', size=448, train=True, indices=None):
        self.processor = processor
        self.size = size
        self.data = []
        self.train = train
        self.aug = augmentation

        df = pd.read_csv(csv_path)
        all_data = []
        for idx, row in df.iterrows():
            for dim in DIMENSION_LIST:
                score = int(row[dim]) if not math.isnan(row[dim]) else 3
                assert 1 <= score <= 5, f"Label out of range: {score}"
                all_data.append({
                    "image_path": os.path.join(IMAGE_DIR, row["name"]),
                    "dim": dim,
                    "score": score,
                })

        if indices is None:
            self.data = all_data
        else:
            self.data = []
            for row_idx in indices:
                start_idx = row_idx * len(DIMENSION_LIST)
                end_idx = (row_idx + 1) * len(DIMENSION_LIST)
                self.data.extend(all_data[start_idx:end_idx])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        # print(f"[Dataset] Loading index {idx}")
        item = self.data[idx]
        
        messages = qwen_prompt(item["dim"], item["image_path"])

        text_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, _ = process_vision_info(messages)
        resized_images = resize_image(image_inputs[0], target_size=self.size)

        if self.train:
            resized_images = apply_augmentation(resized_images, mode=self.aug)

        inputs = self.processor(
            images=resized_images,
            text=text_inputs,
            return_tensors="pt",
        )

        label = int(item["score"])

        return {
            "pixel_values": inputs['pixel_values'].squeeze(0),
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": torch.tensor(label, dtype=torch.float),
            "image_grid_thw": inputs.image_grid_thw[0],
            "dim": item["dim"],
            "index": idx,
        }


@dataclass
class DataCollator:
    def __call__(self, batch: List[Dict]) -> Dict:
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence([x["input_ids"] for x in batch], batch_first=True),  # padding_value=processor.tokenizer.pad_token_id),
            "attention_mask": torch.nn.utils.rnn.pad_sequence([x["attention_mask"] for x in batch], batch_first=True, padding_value=0),
            "pixel_values": torch.stack([x["pixel_values"].contiguous() for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
            "image_grid_thw": torch.stack([x["image_grid_thw"] for x in batch]),
            "dim": [x["dim"] for x in batch],
            "index": torch.tensor([x["index"] for x in batch]),
        }


class DimensionBatchSampler(Sampler):
    def __init__(self, data_source, batch_size=4, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 解析出最底层的 ScoreDataset 和 对应的真实索引
        self._dataset, self._indices = self.resolve_dataset_and_indices(data_source)

        # 构建维度到索引的映射
        self.dim_to_indices = defaultdict(list)
        for idx in self._indices:
            try:
                item = self._dataset[idx]
                self.dim_to_indices[item["dim"]].append(idx)
            except Exception as e:
                print(f"[ERROR] Invalid index {idx}: {e}")

        # 构建 batches
        self.all_dims = list(self.dim_to_indices.keys())
        self.batches = []

        for dim in self.all_dims:
            indices = self.dim_to_indices[dim]
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                self.batches.append((dim, indices[i:i + self.batch_size]))

        if self.shuffle:
            random.shuffle(self.batches)

    @staticmethod
    def resolve_dataset_and_indices(ds):
        indices = []
        subset_stack = []

        while isinstance(ds, torch.utils.data.Subset):
            subset_stack.append(ds)
            indices.append(ds.indices)
            ds = ds.dataset  # 进入下一层

        base_dataset = ds  # 最底层的真实 Dataset（ScoreDataset）

        final_indices = list(range(len(base_dataset)))

        for idx, subset in enumerate(reversed(subset_stack)):
            idx_list = indices[-(idx + 1)]
            max_accessed = max(final_indices) if final_indices else -1

            if max_accessed >= len(idx_list):
                raise IndexError(f"""
                    Index out of range at level {idx}
                    Indices length: {len(idx_list)}
                    Trying to access index: {max_accessed}
                    """
                    )

            final_indices = [idx_list[i] for i in final_indices]

        return base_dataset, final_indices

    def __iter__(self):
        for _, batch_indices in self.batches:
            yield batch_indices

    def __len__(self):
        return len(self.batches)


def get_dataloader(dataset, batch_size=4, shuffle=False):
    sampler = DimensionBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=DataCollator(),
        num_workers=0,
    )


class ScoreDataset_Batch(Dataset):
    def __init__(self, csv_path, processor, size=448):
        self.processor = processor
        self.size = size
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(IMAGE_DIR, row["name"])
        image = Image.open(image_path).convert("RGB")
        image = resize_image(image, self.size)

        # Generate prompts for each dimension
        messages_list = [[{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": qwen_prompt(dim)}]
        }] for dim in DIMENSION_LIST]

        # Process inputs
        images, _ = process_vision_info(messages_list[0])
        inputs = self.processor(
            images=image,
            text=[m[0]["content"][1]["text"] for m in messages_list],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

        # Collect labels
        labels = torch.tensor([row[dim] for dim in DIMENSION_LIST], dtype=torch.float)

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pixel_values": inputs.pixel_values,
            "labels": labels
        }