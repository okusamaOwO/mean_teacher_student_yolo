from utils.torch_utils import select_device, torch_distributed_zero_first
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save, one_flat_cycle, non_max_suppression)
from utils.dataloaders import create_dataloader
from train_dual_mean_teacher import mixstyle_cross_domain
import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def get_dataloaders(opt):
    # Directories
    save_dir = Path(opt.save_dir)

    # Hyperparameters
    hyp = opt.hyp
    if isinstance(hyp, (str, Path)):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict

    # Config
    init_seeds(opt.seed + 1 + RANK, deterministic=True)

    # Data
    data_dict = check_dataset(opt.data)  # check if None
    train_path, val_path, unsupervised_data_path = data_dict[
        'train'], data_dict['val'], data_dict['foggy_zurich']

    # Grid size (Default to 32 as we are not loading model to check stride)
    gs = 32
    # verify imgsz is gs-multiple
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    workers = opt.workers
    batch_size = opt.batch_size
    single_cls = opt.single_cls

    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              close_mosaic=opt.close_mosaic != 0,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              min_items=opt.min_items)

    # Unsupervised Loader
    unsupervised_loader, unsupervised_dataset = create_dataloader(unsupervised_data_path,
                                                                  imgsz,
                                                                  batch_size // WORLD_SIZE,
                                                                  gs,
                                                                  single_cls,
                                                                  hyp=hyp,
                                                                  augment=True,
                                                                  cache=None if opt.cache == 'val' else opt.cache,
                                                                  rect=opt.rect,
                                                                  rank=LOCAL_RANK,
                                                                  workers=workers,
                                                                  image_weights=opt.image_weights,
                                                                  close_mosaic=opt.close_mosaic != 0,
                                                                  quad=opt.quad,
                                                                  prefix=colorstr(
                                                                      'unsupervised: '),
                                                                  shuffle=True,
                                                                  min_items=opt.min_items)

    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class

    # Process 0
    val_loader = None
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if opt.noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

    return train_loader, dataset, unsupervised_loader, unsupervised_dataset, val_loader


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # Including necessary arguments mostly from lmao.py
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--cfg', type=str,
                        default='yolo.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT /
                        'data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT /
                        'data/hyps/hyp.scratch-high.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100,
                        help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int,
                        default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true',
                        help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True,
                        default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true',
                        help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true',
                        help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true',
                        help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true',
                        help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300,
                        help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?',
                        const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true',
                        help='use weighted image selection for training')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true',
                        help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true',
                        help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str,
                        choices=['SGD', 'Adam', 'AdamW', 'LION'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true',
                        help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8,
                        help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT /
                        'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true',
                        help='cosine LR scheduler')
    parser.add_argument('--flat-cos-lr', action='store_true',
                        help='flat cosine LR scheduler')
    parser.add_argument('--fixed-lr', action='store_true',
                        help='fixed LR scheduler')
    parser.add_argument('--label-smoothing', type=float,
                        default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100,
                        help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int,
                        default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1,
                        help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--min-items', type=int,
                        default=0, help='Experimental')
    parser.add_argument('--close-mosaic', type=int,
                        default=0, help='Experimental')
    parser.add_argument('--weight-consistency-loss', type=int,
                        default=1, help='weight for consistency loss')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def visualize_styles(train_loader, unsupervised_loader, save_dir, num_samples=500):
    """
    Visualizes the style statistics (mean and std) of images from two domains using t-SNE.

    Args:
        train_loader: Dataloader for the clear domain (labeled).
        unsupervised_loader: Dataloader for the foggy domain (unlabeled).
        save_dir: Directory to save the plot.
        num_samples: Number of samples to collect from each domain.
    """
    print(
        f"Collecting {num_samples} samples from each domain for style visualization...")

    def get_style_stats(loader, label_name):
        styles = []
        count = 0
        pbar = tqdm(loader, desc=f"Extracting styles from {label_name}")
        for imgs, _, _, _ in pbar:
            # imgs is [B, C, H, W], values 0-255 (usually uint8 or float)
            # Normalize to 0-1 for standard stats calculation
            imgs = imgs.float() / 255.0

            # Compute Mean and Std per channel for each image in the batch
            # Mean: [B, C], Std: [B, C]
            mu = imgs.mean(dim=[2, 3])
            std = imgs.std(dim=[2, 3])

            # Concatenate to form style vector: [B, 2*C] (e.g., 6 dims for RGB)
            # R_mean, G_mean, B_mean, R_std, G_std, B_std
            batch_styles = torch.cat([mu, std], dim=1).cpu().numpy()

            styles.append(batch_styles)
            count += imgs.shape[0]
            if count >= num_samples:
                break
        return np.concatenate(styles, axis=0)[:num_samples]

    # 1. Extract Styles
    clear_styles = get_style_stats(train_loader, "Clear Domain")
    foggy_styles = get_style_stats(unsupervised_loader, "Foggy Domain")

    # 2. Prepare for t-SNE
    X = np.concatenate([clear_styles, foggy_styles], axis=0)
    # Labels: 0 for Clear, 1 for Foggy
    y = np.concatenate([np.zeros(len(clear_styles)),
                       np.ones(len(foggy_styles))], axis=0)

    print(
        f"Running t-SNE on {X.shape[0]} samples with feature dimension {X.shape[1]}...")
    tsne = TSNE(
        n_components=2, 
        verbose=1, 
        perplexity=50,        # Kept at 50 (Good for ~1500 points)
        n_iter=2000,          # Kept at 2000 (Ensures full convergence)
        learning_rate='auto', # Let sklearn optimize the step size
        init='pca',           # <--- CRITICAL: Preserves global structure better than random
        metric='cosine',      # <--- CRITICAL: Better for high-dim deep features than Euclidean
        random_state=42,
        n_jobs=-1             # Uses all cores for speed
    )
    X_embedded = tsne.fit_transform(X)

    # 3. Plot
    print("Plotting results...")
    plt.figure(figsize=(10, 8))

    # Plot Clear Domain (blue)
    plt.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1],
                c='blue', alpha=0.6, label='Clear Domain (Train)', s=20)
    
    # Plot Foggy Domain (red)
    plt.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1],
                c='red', alpha=0.6, label='Foggy Domain (Unsupervised)', s=20)

    plt.title("t-SNE Visualization of Domain Styles (Input Image Mean/Std)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    # Save logic
    save_path = Path(save_dir) / 'style_tsne_plot.png'
    plt.savefig(save_path, dpi=300)
    print(f"Style visualization saved to {save_path}")

if __name__ == "__main__":
    opt = parse_opt()
    # Set default save_dir if not present (since we removed the heavy main function logic)
    if not hasattr(opt, 'save_dir'):
        opt.save_dir = str(increment_path(
            Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # Ensure save directory exists
    Path(opt.save_dir).mkdir(parents=True, exist_ok=True)

    # Basic check for opt.resume could be added but simpler to just run

    print("Loading dataloaders...")
    train_loader, dataset, unsupervised_loader, unsupervised_dataset, val_loader = get_dataloaders(
        opt)

    print(f"Train Loader length: {len(train_loader)}")
    print(f"Unsupervised Loader length: {len(unsupervised_loader)}")
    if val_loader:
        print(f"Val Loader length: {len(val_loader)}")
    print("Dataloaders loaded successfully.")

    # Run Visualization
    visualize_styles(train_loader, unsupervised_loader, opt.save_dir)
