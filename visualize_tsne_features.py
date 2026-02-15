"""
t-SNE visualization of early-layer feature maps to verify MixStyle style exchange.

This script:
1. Loads a trained model (or initializes one from config)
2. Loads source (labeled) and target (unlabeled) images
3. Extracts feature maps from early backbone layers (layers 0, 1, 2) using hooks
4. Runs before/after MixStyle comparison
5. Visualizes using t-SNE to show that styles are exchanged

Usage:
    python visualize_tsne_features.py --weights yolov9-t-converted.pt --data data1.yaml --cfg models/detect/gelan-t.yaml --imgsz 640 --batch-size 8 --num-batches 10 --layers 0 1 2
"""

from itertools import cycle
from helpers.mixstyle_augment import apply_mixstyle_custom
from utils.torch_utils import select_device
from utils.general import check_dataset, check_img_size, check_yaml, check_file, intersect_dicts, colorstr
from utils.dataloaders import create_dataloader
from models.experimental import attempt_load
from models.yolo import Model
import matplotlib.pyplot as plt
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # non-interactive backend

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


class FeatureHook:
    """Forward hook to capture feature maps from a specific layer."""

    def __init__(self, name):
        self.name = name
        self.features = None

    def __call__(self, module, input, output):
        # output shape: (B, C, H, W)
        self.features = output.detach()

    def get_pooled(self):
        """Global average pool -> (B, C) vector representing style statistics."""
        if self.features is None:
            return None
        # Channel-wise mean and std as style representation
        feat = self.features
        mu = feat.mean(dim=[2, 3])   # (B, C)
        std = feat.std(dim=[2, 3])   # (B, C)
        # Concatenate mean and std as style descriptor
        return torch.cat([mu, std], dim=1).cpu().numpy()  # (B, 2*C)


def register_hooks(model, layer_indices):
    """Register forward hooks on specified layer indices of model.model."""
    hooks = {}
    handles = []
    for idx in layer_indices:
        hook = FeatureHook(name=f"layer_{idx}")
        handle = model.model[idx].register_forward_hook(hook)
        hooks[idx] = hook
        handles.append(handle)
    return hooks, handles


def collect_features(model, source_loader, target_loader, device, layer_indices,
                     num_batches=10, apply_mix=False, mix_alpha=1.3, mix_beta=6):
    """
    Run source and target images through the model, collect early-layer features.

    Returns:
        features_per_layer: dict {layer_idx: (N, 2*C) numpy array}
        labels: (N,) numpy array — 0=source, 1=target
    """
    model.eval()
    hooks, handles = register_hooks(model, layer_indices)

    all_features = {idx: [] for idx in layer_indices}
    all_labels = []

    target_iter = iter(cycle(target_loader))

    with torch.no_grad():
        for batch_i, (source_imgs, _, _, _) in enumerate(source_loader):
            if batch_i >= num_batches:
                break

            target_imgs, _, _, _ = next(target_iter)

            source_imgs = source_imgs.to(
                device, non_blocking=True).float() / 255
            target_imgs = target_imgs.to(
                device, non_blocking=True).float() / 255

            B = source_imgs.size(0)

            if apply_mix:
                mixed_output = apply_mixstyle_custom(
                    source_imgs, target_imgs, p=1, alpha=mix_alpha, beta=mix_beta)
                source_imgs_feed = mixed_output[0:B]
                target_imgs_feed = mixed_output[B:]
            else:
                source_imgs_feed = source_imgs
                target_imgs_feed = target_imgs

            # Forward source images
            _ = model(source_imgs_feed)
            for idx in layer_indices:
                feat = hooks[idx].get_pooled()  # (B, 2*C)
                if feat is not None:
                    all_features[idx].append(feat)
            all_labels.append(np.zeros(B))  # 0 = source

            # Forward target images
            _ = model(target_imgs_feed)
            for idx in layer_indices:
                feat = hooks[idx].get_pooled()  # (B, 2*C)
                if feat is not None:
                    all_features[idx].append(feat)
            all_labels.append(np.ones(target_imgs_feed.size(0)))  # 1 = target

    # Cleanup hooks
    for h in handles:
        h.remove()

    # Concatenate
    labels = np.concatenate(all_labels, axis=0)
    features_per_layer = {}
    for idx in layer_indices:
        features_per_layer[idx] = np.concatenate(all_features[idx], axis=0)

    return features_per_layer, labels


def plot_tsne(features_per_layer, labels, title_prefix, save_dir):
    """Run t-SNE on features from each layer and plot."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_layers = len(features_per_layer)
    fig, axes = plt.subplots(1, n_layers, figsize=(7 * n_layers, 6))
    if n_layers == 1:
        axes = [axes]

    for ax, (layer_idx, features) in zip(axes, sorted(features_per_layer.items())):
        print(f"  Running t-SNE for layer {layer_idx} with {features.shape[0]} samples, "
              f"feature dim={features.shape[1]}...")

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        perplexity = min(30, features.shape[0] - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                    n_iter=1000, learning_rate='auto', init='pca')
        embeddings = tsne.fit_transform(features)

        source_mask = labels == 0
        target_mask = labels == 1

        ax.scatter(embeddings[source_mask, 0], embeddings[source_mask, 1],
                   c='blue', alpha=0.6, s=20, label='Source', edgecolors='none')
        ax.scatter(embeddings[target_mask, 0], embeddings[target_mask, 1],
                   c='red', alpha=0.6, s=20, label='Target', edgecolors='none')
        ax.set_title(f'Layer {layer_idx}', fontsize=14)
        ax.legend(fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title_prefix, fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = save_dir / f"{title_prefix.replace(' ', '_').lower()}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='t-SNE visualization of early-layer features')
    parser.add_argument('--weights', type=str,
                        default='yolov9-t-converted.pt', help='model weights path')
    parser.add_argument(
        '--cfg', type=str, default='models/detect/gelan-t.yaml', help='model config yaml')
    parser.add_argument('--data', type=str,
                        default='data1.yaml', help='dataset yaml path')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--num-batches', type=int, default=10,
                        help='number of batches to collect')
    parser.add_argument('--device', default='', help='cuda device or cpu')
    parser.add_argument('--layers', nargs='+', type=int, default=[0, 1, 2],
                        help='backbone layer indices to hook (early layers for style)')
    parser.add_argument('--mix-alpha', type=float,
                        default=1.3, help='MixStyle Beta dist alpha')
    parser.add_argument('--mix-beta', type=float,
                        default=6.0, help='MixStyle Beta dist beta')
    parser.add_argument('--save-dir', type=str,
                        default='runs/tsne_vis', help='save directory')
    args = parser.parse_args()

    device = select_device(args.device)

    # Load data config
    data_dict = check_dataset(args.data)
    train_path = data_dict['train']
    target_path = data_dict['target_train']
    nc = int(data_dict['nc'])

    # Load model
    print(f"Loading model from {args.weights} with config {args.cfg}...")
    ckpt = torch.load(args.weights, map_location='cpu')
    model = Model(args.cfg, ch=3, nc=nc).to(device)
    if 'model' in ckpt:
        csd = ckpt['model'].float().state_dict()
    else:
        csd = ckpt
    csd = intersect_dicts(csd, model.state_dict())
    model.load_state_dict(csd, strict=False)
    model.eval()

    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(args.imgsz, gs, floor=gs * 2)

    # Minimal hyp dict needed by the dataloader (no augmentation)
    hyp = {
        'mosaic': 0.0, 'mixup': 0.0, 'copy_paste': 0.0,
        'degrees': 0.0, 'translate': 0.0, 'scale': 0.0, 'shear': 0.0,
        'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.0,
        'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0,
    }

    # Create dataloaders
    print("Creating dataloaders...")
    source_loader, _ = create_dataloader(
        train_path, imgsz, args.batch_size, gs,
        hyp=hyp, augment=True, cache=False, rect=False,
        rank=-1, workers=4, prefix=colorstr('source: '), shuffle=True)

    target_loader, _ = create_dataloader(
        target_path, imgsz, args.batch_size, gs,
        hyp=hyp, augment=True, cache=False, rect=False,
        rank=-1, workers=4, prefix=colorstr('target: '), shuffle=True)

    layer_indices = args.layers
    print(f"Will hook layers: {layer_indices}")
    print(
        f"Collecting {args.num_batches} batches x {args.batch_size} images each...")

    # ---- BEFORE MixStyle ----
    print("\n[1/2] Collecting features BEFORE MixStyle...")
    features_before, labels_before = collect_features(
        model, source_loader, target_loader, device, layer_indices,
        num_batches=args.num_batches, apply_mix=False)

    print("Plotting t-SNE (before)...")
    plot_tsne(features_before, labels_before,
              "Before MixStyle", args.save_dir)

    # ---- AFTER MixStyle ----
    print(
        f"\n[2/2] Collecting features AFTER MixStyle (alpha={args.mix_alpha}, beta={args.mix_beta})...")
    features_after, labels_after = collect_features(
        model, source_loader, target_loader, device, layer_indices,
        num_batches=args.num_batches, apply_mix=True,
        mix_alpha=args.mix_alpha, mix_beta=args.mix_beta)

    print("Plotting t-SNE (after)...")
    plot_tsne(features_after, labels_after,
              "After MixStyle", args.save_dir)

    # ---- COMBINED side-by-side plot ----
    print("\nCreating combined comparison plot...")
    n_layers = len(layer_indices)
    fig, axes = plt.subplots(2, n_layers, figsize=(7 * n_layers, 12))
    if n_layers == 1:
        axes = axes.reshape(2, 1)

    for col, layer_idx in enumerate(sorted(layer_indices)):
        for row, (features, labels, condition) in enumerate([
            (features_before, labels_before, "Before MixStyle"),
            (features_after, labels_after, "After MixStyle"),
        ]):
            feat = np.nan_to_num(
                features[layer_idx], nan=0.0, posinf=0.0, neginf=0.0)
            perplexity = min(30, feat.shape[0] - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                        n_iter=1000, learning_rate='auto', init='pca')
            emb = tsne.fit_transform(feat)

            ax = axes[row, col]
            src = labels == 0
            tgt = labels == 1
            ax.scatter(emb[src, 0], emb[src, 1], c='blue', alpha=0.6, s=20,
                       label='Source', edgecolors='none')
            ax.scatter(emb[tgt, 0], emb[tgt, 1], c='red', alpha=0.6, s=20,
                       label='Target', edgecolors='none')
            ax.set_title(f'{condition} — Layer {layer_idx}', fontsize=13)
            ax.legend(fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f'Style Feature t-SNE: MixStyle(alpha={args.mix_alpha}, beta={args.mix_beta})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = Path(args.save_dir) / 'comparison_before_after_mixstyle.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison: {save_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
