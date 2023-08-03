# -*- coding: utf-8 -*-


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

from DDIM.DiffusionCondition import GaussianDiffusionSampler, GaussianDiffusionTrainer, DDIMSampler, extract
from DDIM.ModelCondition import UNet
from DDIM.LabML_Unet import UNetModel
from Scheduler import GradualWarmupScheduler

from torchvision.transforms import Compose, ColorJitter, ToTensor
import torch
import torch.nn as nn
from torchvision import models, transforms
from types import SimpleNamespace
from datasets import load_dataset, Image

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

import wandb

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

cfg = SimpleNamespace(    
    dataset_name = "barry556652/special_img_200",
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
        dataset = dataset.filter(lambda example: not example["text"].endswith("CRYSTAL"))
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
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    def class_label_tensor(examples, is_train=True):
        
        def class_tokenizer(text):
            #special 200
            class_names = ['C0201', 'SOT23', 'BGA4P', 'R0805', 'TVS523', 'LED0603']
            
            #without SOT
            #class_names = ['C0201', 'SOT23', 'L2016', 'C0604', 'R0402', 'C0402']
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


def train(modelConfig: Dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset
#     dataset = CIFAR10(
#         root='./CIFAR10', train=True, download=True,
#         transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]))
#     dataloader = DataLoader(
#         dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    wandb.init(
            project="CDDIM",
            config={
                "learning_rate": modelConfig["lr"],
                "epochs": modelConfig["epoch"],
                "dataset": "phison_image_without_SOT",
                "architecture": "classifier-free conditional DDPM",
                "num_res_blocks": modelConfig["num_res_blocks"],
                "img_size": modelConfig["img_size"],
                "batch_size": modelConfig["batch_size"],
                "T": modelConfig["T"],
                "ch_mult": [1, 2, 2, 2]
            },
            job_type="training"
        )
    dataloader = get_phison(modelConfig["img_size"])
    
    # model setup
    net_model = UNetModel(
        T=modelConfig["T"], 
        in_channels=3, 
        out_channels=3, 
        channels=modelConfig["channel"], 
        channel_multipliers=modelConfig["channel_mult"], 
        attention_levels=modelConfig["attention_levels"], 
        n_heads=modelConfig["n_heads"],
        n_res_blocks=modelConfig["num_res_blocks"],
        num_labels=modelConfig["num_labels"],
    )
    
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"])
    
    dataloader, net_model, optimizer, trainer= accelerator.prepare(dataloader, net_model, optimizer, trainer)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for batch in tqdmDataLoader:
                # train
                images = batch['pixel_values']
                labels = batch['condition_label']
                labels_1 = batch['class_label']
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device) + 1
                labels_1 = labels_1.to(device) + 1
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                    labels_1 = torch.zeros_like(labels_1).to(device)
                loss = trainer(x_0, labels, labels_1).sum() / b ** 2.
                accelerator.backward(loss)
                wandb.log({"loss": loss.item()})
                
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model and evaluate
    with torch.no_grad():
        step = int(modelConfig["batch_size"] // 10)
#         labelList = []
#         k = 0
#         for i in range(1, modelConfig["batch_size"] // 2 + 1):
#             labelList.append(torch.ones(size=[1]).long() * k)
#             if i % step == 0:
#                 if k < modelConfig["num_labels"] - 1:
#                     k += 1
                                    
#         labels = torch.cat(labelList, dim=0).long().to(device) + 1
#         labels = torch.cat((labels,labels))

        labelList = []
        k = 0
        for i in range(1, 80 + 1):
            labelList.append(torch.ones(size=[1]).long() * k)
            if i % 8 == 0:
                if k < 6 - 1:
                    k += 1

        labels = torch.cat(labelList, dim=0).long().to(device)+1
        labels = labels[:64]
        
        pattern = torch.tensor([1, 2, 3, 4])
        repeated_tensor = pattern.repeat(modelConfig["batch_size"] // pattern.size()[0]) 

        Condition = repeated_tensor.long().to(device)
        
        print("Condition: ", Condition)
        print("labels: ", labels)
        
        model = UNetModel(
                    T=modelConfig["T"], 
                    in_channels=3, 
                    out_channels=3, 
                    channels=modelConfig["channel"], 
                    channel_multipliers=modelConfig["channel_mult"], 
                    attention_levels=modelConfig["attention_levels"], 
                    n_heads=modelConfig["n_heads"],
                    n_res_blocks=modelConfig["num_res_blocks"],
                    num_labels=modelConfig["num_labels"],
                ).to(device)
        
        model = accelerator.prepare(model)
        ckpt = torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = DDIMSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"])
        sampler = accelerator.prepare(sampler)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, modelConfig["img_size"], modelConfig["img_size"]], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage, Condition, labels, steps = modelConfig["DDIM_steps"])
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        wandb.log({f"sample image": [wandb.Image(sampledImgs)]})
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])
