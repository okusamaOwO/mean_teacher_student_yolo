import argparse
import os
from pathlib import Path
import torch
import yaml
import numpy as np

from utils.dataloaders import create_dataloader
from utils.general import check_dataset, check_yaml, check_file, colorstr, init_seeds
from helpers.add_noise import add_random_noise
# --- configuration ----
DATA_YAML = './data/coco.yaml'           # <-- change to your dataset yaml
HYP_YAML  = './data/hyps/hyp.scratch-high.yaml'  # <-- change to your hyp yaml
BATCH_SIZE = 16
IMG_SIZE   = 640
WORKERS    = 8
# ----------------------

# single-GPU, no DDP
LOCAL_RANK = -1
WORLD_SIZE = 1

# Load hyp and data yaml
with open(check_file(check_yaml(HYP_YAML)), errors='ignore') as f:
    hyp = yaml.safe_load(f)
data_dict = check_dataset(check_file(DATA_YAML))

train_path = data_dict['train']
single_cls = False    # set True if you want single-class training

# Set random seed for reproducibility
init_seeds(0)

# Create train dataloader
train_loader, dataset = create_dataloader(
    path=train_path,
    imgsz=IMG_SIZE,
    batch_size=BATCH_SIZE // WORLD_SIZE,
    stride=32,                   # usually max stride, 32 is safe default
    single_cls=single_cls,
    hyp=hyp,
    augment=True,
    cache=None,
    rect=False,
    rank=LOCAL_RANK,
    workers=WORKERS,
    image_weights=False,
    close_mosaic=False,
    quad=False,
    prefix=colorstr('train: '),
    shuffle=True,
    min_items=0
)

import matplotlib.pyplot as plt

for imgs, targets, paths, _ in train_loader:
    imgs = add_random_noise(imgs)
    first_img = imgs[0]
    img = first_img.cpu().permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('sample_image.png')   # saved to the current working directory
    break

