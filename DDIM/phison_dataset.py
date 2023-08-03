import os
from typing import Dict
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision.transforms import Compose, ColorJitter, ToTensor
import torch
import torch.nn as nn
from torchvision import models, transforms
from types import SimpleNamespace
from datasets import load_dataset, Image
import random

cfg = SimpleNamespace(    
#     df_dataset_name = "barry556652/try_DF1",
#     g_dataset_name = "barry556652/try_good1",
    dataset_name = "barry556652/try",
    image_column = "image",
    caption_column = "text",
    max_train_samples = 1,
    seed = 42,
    random_flip = "store_true",
    train_batch_size = 64,
    dataloader_num_workers = 10,
)

dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

def get_phison(image_size):
    
    if cfg.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            cfg.dataset_name,
        )
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(cfg.dataset_name, None)

    image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]

    caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    
    train_transforms = transforms.Compose(
        [
            transforms.Resize([image_size,image_size], interpolation=transforms.InterpolationMode.BILINEAR),
#             #centercrop越小速度越快
#             transforms.CenterCrop(image_size),
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    
    def class_label_tensor(examples, is_train=True):
        
        def class_tokenizer(text):
            #special 200
            #class_names = ['C0201', 'SOT23', 'BGA4P', 'TVS523', 'CRYSTAL', 'BGA4P', 'R0805', 'LED0603']
            
            #without SOT
            class_names = ['C0201', 'SOT23', 'L2016', 'C0604', 'R0402', 'C0402']
            class_label = text.split(' ')
            num_classes = len(class_names)
            class_vector = torch.zeros(num_classes, dtype=torch.float)
            class_index = class_names.index(class_label[1])
            class_vector[class_index] = 1
            class_tensor = class_vector.view(1, num_classes)
            return torch.unsqueeze(torch.tensor(class_index), 0).to(torch.float)
        
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        label_tensor = class_tokenizer(captions[0])
        return label_tensor
    
    def condition_label_tensor(examples, is_train=True):
        
        def condition_tokenizer(text):
            condition_names = ['good', 'broke', 'shift', "short"]
            class_label = text.split(' ')
            num_condition = len(condition_names)
            condition_vector = torch.zeros(num_condition, dtype=torch.float)
            condition_index = condition_names.index(class_label[0])
            condition_vector[condition_index] = 1
            condition_tensor = condition_vector.view(1, num_condition)
            return torch.unsqueeze(torch.tensor(condition_index), 0).to(torch.float)
        
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        condition_tensor = condition_tokenizer(captions[0])
        return condition_tensor
    
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["class_label"] = class_label_tensor(examples)
        examples["condition_label"] = condition_label_tensor(examples)
        return examples
    
    train_dataset = dataset["train"].with_transform(preprocess_train)
    
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        class_label = torch.stack([example["class_label"] for example in examples])
        condition_label = torch.stack([example["condition_label"] for example in examples])
        return {"pixel_values": pixel_values, "condition_label": condition_label, "class_label": class_label}
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
    )
    
    
    return train_dataloader

def get_phison_good(image_size):
    
    if cfg.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            cfg.dataset_name,
        )
        dataset = dataset.filter(lambda example: example["text"].startswith("good"))
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(cfg.dataset_name, None)

    image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]

    caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    
    train_transforms = transforms.Compose(
        [
            transforms.Resize([image_size,image_size], interpolation=transforms.InterpolationMode.BILINEAR),
#             #centercrop越小速度越快
#             transforms.CenterCrop(image_size),
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    
    def class_label_tensor(examples, is_train=True):
        
        def class_tokenizer(text):
            #special 200
            #class_names = ['C0201', 'SOT23', 'BGA4P', 'TVS523', 'CRYSTAL', 'BGA4P', 'R0805', 'LED0603']
            
            #without SOT
            class_names = ['C0201', 'SOT23', 'L2016', 'C0604', 'R0402', 'C0402']
            class_label = text.split(' ')
            num_classes = len(class_names)
            class_vector = torch.zeros(num_classes, dtype=torch.float)
            class_index = class_names.index(class_label[1])
            class_vector[class_index] = 1
            class_tensor = class_vector.view(1, num_classes)
            return torch.unsqueeze(torch.tensor(class_index), 0).to(torch.float)
        
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        label_tensor = class_tokenizer(captions[0])
        return label_tensor
    
    def condition_label_tensor(examples, is_train=True):
        
        def condition_tokenizer(text):
            condition_names = ['good', 'broke', 'shift', "short"]
            class_label = text.split(' ')
            num_condition = len(condition_names)
            condition_vector = torch.zeros(num_condition, dtype=torch.float)
            condition_index = condition_names.index(class_label[0])
            condition_vector[condition_index] = 1
            condition_tensor = condition_vector.view(1, num_condition)
            return torch.unsqueeze(torch.tensor(condition_index), 0).to(torch.float)
        
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        condition_tensor = condition_tokenizer(captions[0])
        return condition_tensor
    
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["class_label"] = class_label_tensor(examples)
        examples["condition_label"] = condition_label_tensor(examples)
        return examples
    
    train_dataset = dataset["train"].with_transform(preprocess_train)
    
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        class_label = torch.stack([example["class_label"] for example in examples])
        condition_label = torch.stack([example["condition_label"] for example in examples])
        return {"pixel_values": pixel_values, "condition_label": condition_label, "class_label": class_label}
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
    )
    
    
    return train_dataloader

# images = batch['pixel_values']
# labels = batch['condition_label']
# labels_1 = batch['class_label']
# good_image = []
# for i in range(64):
#     j = 0
#     for image, data in zip(batch2['pixel_values'], batch2['class_label']):
#         if data.item() == labels_1[i].item():
#             good_image.append(image)
#             break
#         else:
#             j+=1
# good_image = torch.stack(good_image)
# good_image.size()