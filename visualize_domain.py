from utils.torch_utils import select_device, torch_distributed_zero_first
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save, one_flat_cycle, non_max_suppression)
from utils.dataloaders import create_dataloader
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


def mixstyle_cross_domain(x1, x2, alpha=0.1, eps=1e-6):
    """
    MixStyle implementation for explicit cross-domain mixing
    Args:
        x1: first batch [B, C, H, W] 
        x2: second batch [B, C, H, W] (different domain order)
        alpha: interpolation strength
        eps: small value to avoid division by zero
    """
    B = x1.size(0)
    if B < 2:
        return x1

    # Compute instance statistics for both batches
    mu1 = x1.mean(dim=[2, 3], keepdim=True)
    var1 = x1.var(dim=[2, 3], keepdim=True)
    sig1 = (var1 + eps).sqrt()

    mu2 = x2.mean(dim=[2, 3], keepdim=True)
    var2 = x2.var(dim=[2, 3], keepdim=True)
    sig2 = (var2 + eps).sqrt()

    # Normalize x1
    x1_normed = (x1 - mu1) / sig1

    # Generate mixing weight
    # Use torch.distributions.Beta if available or numpy
    # Ensure device compatibility
    lmda = torch.distributions.Beta(
        alpha, alpha).sample((B, 1, 1, 1)).to(x1.device)

    # Mix statistics between x1 and x2 (guaranteed cross-domain)
    mu_mix = lmda * mu1 + (1 - lmda) * mu2
    sig_mix = lmda * sig1 + (1 - lmda) * sig2

    # Apply mixed statistics to x1
    return x1_normed * sig_mix + mu_mix


def visualize_styles(train_loader, unsupervised_loader, save_dir, num_samples=500):
    """
    Visualizes the style statistics (mean and std) of images from 4 domains (Source, Target, Mixed Source, Mixed Target) using t-SNE.

    Args:
        train_loader: Dataloader for the clear domain (labeled).
        unsupervised_loader: Dataloader for the foggy domain (unlabeled).
        save_dir: Directory to save the plot.
        num_samples: Number of samples to collect from each domain.
    """
    print(
        f"Collecting {num_samples} samples from each domain for style visualization...")

    styles_clear = []
    styles_foggy = []
    styles_mixed_clear = []
    styles_mixed_foggy = []

    count = 0
    pbar = tqdm(total=num_samples, desc="Collecting samples")

    iter_clear = iter(train_loader)
    iter_foggy = iter(unsupervised_loader)

    def compute_stats(x):
        # Mean: [B, C], Std: [B, C]
        mu = x.mean(dim=[2, 3])
        std = x.std(dim=[2, 3])
        return torch.cat([mu, std], dim=1).detach().cpu().numpy()

    while count < num_samples:
        try:
            batch_c = next(iter_clear)
        except StopIteration:
            iter_clear = iter(train_loader)
            batch_c = next(iter_clear)

        try:
            batch_f = next(iter_foggy)
        except StopIteration:
            iter_foggy = iter(unsupervised_loader)
            batch_f = next(iter_foggy)

        # Unpack images (batch_c[0] is images)
        imgs_c = batch_c[0].float() / 255.0
        imgs_f = batch_f[0].float() / 255.0

        # Ensure sizes match for mixing
        min_b = min(imgs_c.shape[0], imgs_f.shape[0])
        if min_b < 2:
            continue  # skip small batches

        imgs_c = imgs_c[:min_b]
        imgs_f = imgs_f[:min_b]

        # X1 = torch.cat((source_imgs, imgs_student), 0)
        # X2 = torch.cat((imgs_student, source_imgs), 0)
        X1 = torch.cat((imgs_c, imgs_f), 0)
        X2 = torch.cat((imgs_f, imgs_c), 0)

        mixed_batch = mixstyle_cross_domain(X1, X2, alpha=0.3)

        # mixed_source_imgs = mixed_batch[:B_source]  -> mixed_c (Source content, Target style)
        # mixed_target_imgs = mixed_batch[B_source:]  -> mixed_f (Target content, Source style)
        mixed_c = mixed_batch[:min_b]
        mixed_f = mixed_batch[min_b:]

        # Collect stats
        styles_clear.append(compute_stats(imgs_c))
        styles_foggy.append(compute_stats(imgs_f))
        styles_mixed_clear.append(compute_stats(mixed_c))
        styles_mixed_foggy.append(compute_stats(mixed_f))

        count += min_b
        pbar.update(min_b)

    pbar.close()

    # Concatenate and crop
    s_clear = np.concatenate(styles_clear, axis=0)[:num_samples]
    s_foggy = np.concatenate(styles_foggy, axis=0)[:num_samples]
    s_mixed_c = np.concatenate(styles_mixed_clear, axis=0)[:num_samples]
    s_mixed_f = np.concatenate(styles_mixed_foggy, axis=0)[:num_samples]

    # Prepare for t-SNE
    X = np.concatenate([s_clear, s_foggy, s_mixed_c, s_mixed_f], axis=0)
    # Labels: 0:Clear, 1:Foggy, 2:Mixed-Clear, 3:Mixed-Foggy
    y = np.concatenate([
        np.zeros(len(s_clear)),
        np.ones(len(s_foggy)),
        np.full(len(s_mixed_c), 2),
        np.full(len(s_mixed_f), 3)
    ], axis=0)

    print(
        f"Running t-SNE on {X.shape[0]} samples with feature dimension {X.shape[1]}...")
    tsne = TSNE(
        n_components=2,
        verbose=1,
        perplexity=50,        # Kept at 50 (Good for ~1500 points)
        n_iter=2000,          # Kept at 2000 (Ensures full convergence)
        learning_rate='auto',  # Let sklearn optimize the step size
        init='pca',           # <--- CRITICAL: Preserves global structure better than random
        metric='cosine',      # <--- CRITICAL: Better for high-dim deep features than Euclidean
        random_state=42,
        n_jobs=-1             # Uses all cores for speed
    )
    X_embedded = tsne.fit_transform(X)

    # Plot
    print("Plotting results...")
    plt.figure(figsize=(10, 8))

    # Plot Clear Domain
    plt.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1],
                c='blue', alpha=0.6, label='Clear Domain (Source)', s=20)

    # Plot Foggy Domain
    plt.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1],
                c='red', alpha=0.6, label='Foggy Domain (Target)', s=20)

    # Plot Mixed Clear
    plt.scatter(X_embedded[y == 2, 0], X_embedded[y == 2, 1],
                c='cyan', alpha=0.6, label='Mixed Clear (Source->Target Style)', s=20)

    # Plot Mixed Foggy
    plt.scatter(X_embedded[y == 3, 0], X_embedded[y == 3, 1],
                c='orange', alpha=0.6, label='Mixed Foggy (Target->Source Style)', s=20)

    plt.title("t-SNE Visualization of Domain Styles (4 Domains)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    # Save logic
    save_path = Path(save_dir) / 'style_tsne_plot_4_domains.png'
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
