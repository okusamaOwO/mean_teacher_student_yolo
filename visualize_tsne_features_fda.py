"""
t-SNE visualization of early-layer feature maps to verify FDA (Fourier Domain Adaptation)
style exchange via low-frequency amplitude swapping.

This script:
1. Loads a trained model (or initializes one from config)
2. Loads source (labeled) and target (unlabeled) images
3. Applies FDA: swaps low-frequency amplitude between source and target images
4. Extracts feature maps from early backbone layers using hooks
5. Visualizes:
   - Source (original) vs Target (after FDA) — both should carry source style, expecting overlap
   - Source (after FDA) vs Target (original) — both should carry target style, expecting overlap
   - Source (original) vs Target (original) — different styles, expecting separation
   - All 4 style domains in one figure

Usage:
    python visualize_tsne_features_fda.py --weights yolov9-t-converted.pt --data data1.yaml --cfg models/detect/gelan-t.yaml --imgsz 640 --batch-size 8 --num-batches 10 --layers 0 1 2 --fda-beta 0.05
"""

from itertools import cycle
from helpers.fda_augment import fda_source_to_target
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


def collect_features_fda(model, source_loader, target_loader, device, layer_indices,
                         num_batches=10, fda_beta=0.01):
    """
    Collect features for FDA cross-style comparison:
      - source_original: source images WITHOUT FDA (original source style)
      - target_after_fda: target images AFTER FDA — target content + source low-freq amp
                          (should carry source style)
      - source_after_fda: source images AFTER FDA — source content + target low-freq amp
                          (should carry target style)
      - target_original: target images WITHOUT FDA (original target style)

    Returns:
        features dict with keys: 'source_orig', 'target_fda', 'source_fda', 'target_orig'
        Each is a dict {layer_idx: (N, 2*C) numpy array}
    """
    model.eval()
    hooks, handles = register_hooks(model, layer_indices)

    result = {
        'source_orig': {idx: [] for idx in layer_indices},
        'target_fda':  {idx: [] for idx in layer_indices},
        'source_fda':  {idx: [] for idx in layer_indices},
        'target_orig': {idx: [] for idx in layer_indices},
    }

    target_iter = iter(cycle(target_loader))

    def forward_and_collect(imgs, key):
        _ = model(imgs)
        for idx in layer_indices:
            feat = hooks[idx].get_pooled()
            if feat is not None:
                result[key][idx].append(feat)

    with torch.no_grad():
        for batch_i, (source_imgs, _, _, _) in enumerate(source_loader):
            if batch_i >= num_batches:
                break

            target_imgs, _, _, _ = next(target_iter)

            source_imgs = source_imgs.to(device, non_blocking=True).float() / 255
            target_imgs = target_imgs.to(device, non_blocking=True).float() / 255

            # 1) Original source features
            forward_and_collect(source_imgs, 'source_orig')

            # 2) Original target features
            forward_and_collect(target_imgs, 'target_orig')

            # 3) Apply FDA: swap low-frequency amplitude
            source_fda, target_fda = fda_source_to_target(
                source_imgs, target_imgs, beta=fda_beta)
            # source_fda: source content + target low-freq amp → target-styled source
            # target_fda: target content + source low-freq amp → source-styled target

            # 4) Source after FDA (carries target style)
            forward_and_collect(source_fda, 'source_fda')

            # 5) Target after FDA (carries source style)
            forward_and_collect(target_fda, 'target_fda')

    for h in handles:
        h.remove()

    # Concatenate
    for key in result:
        for idx in layer_indices:
            result[key][idx] = np.concatenate(result[key][idx], axis=0)

    return result


def save_sample_images(source_imgs, target_imgs, source_fda, target_fda, save_dir, batch_idx=0):
    """Save a grid of sample images showing original vs FDA-transformed for visual inspection."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_show = min(4, source_imgs.shape[0])
    fig, axes = plt.subplots(4, n_show, figsize=(4 * n_show, 16))
    if n_show == 1:
        axes = axes[:, None]

    row_labels = [
        'Source (original)',
        'Target (original)',
        'Source after FDA\n(target-styled)',
        'Target after FDA\n(source-styled)'
    ]
    imgs_list = [source_imgs, target_imgs, source_fda, target_fda]

    for row, (label, imgs) in enumerate(zip(row_labels, imgs_list)):
        for col in range(n_show):
            img = imgs[col].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            img = np.clip(img, 0, 1)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(label, fontsize=12, rotation=0,
                                          labelpad=120, va='center')

    fig.suptitle(f'FDA Sample Images (batch {batch_idx})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = save_dir / f'fda_sample_images_batch{batch_idx}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved sample images: {save_path}")


def plot_tsne_pair(features_a, features_b, label_a, label_b, layer_indices, title, save_dir):
    """Run t-SNE on combined features from two groups and plot per layer."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_layers = len(layer_indices)
    fig, axes = plt.subplots(1, n_layers, figsize=(7 * n_layers, 6))
    if n_layers == 1:
        axes = [axes]

    for ax, layer_idx in zip(axes, sorted(layer_indices)):
        fa = np.nan_to_num(features_a[layer_idx],
                           nan=0.0, posinf=0.0, neginf=0.0)
        fb = np.nan_to_num(features_b[layer_idx],
                           nan=0.0, posinf=0.0, neginf=0.0)
        combined = np.concatenate([fa, fb], axis=0)
        labels = np.array([0] * fa.shape[0] + [1] * fb.shape[0])

        print(f"  Running t-SNE for layer {layer_idx} with {combined.shape[0]} samples, "
              f"feature dim={combined.shape[1]}...")

        perplexity = min(30, combined.shape[0] - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                    n_iter=1000, learning_rate='auto', init='pca')
        emb = tsne.fit_transform(combined)

        mask_a = labels == 0
        mask_b = labels == 1

        ax.scatter(emb[mask_a, 0], emb[mask_a, 1],
                   c='blue', alpha=0.6, s=20, label=label_a, edgecolors='none')
        ax.scatter(emb[mask_b, 0], emb[mask_b, 1],
                   c='red', alpha=0.6, s=20, label=label_b, edgecolors='none')
        ax.set_title(f'Layer {layer_idx}', fontsize=14)
        ax.legend(fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = save_dir / f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tsne_four_domains(source_orig, target_orig, source_fda, target_fda,
                           layer_indices, title, save_dir):
    """Run t-SNE on all 4 style domains combined and plot per layer.

    Domains:
        1. Source (original)         - blue
        2. Target (original)         - red
        3. Source after FDA           - cyan   (target-styled source)
        4. Target after FDA           - orange (source-styled target)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_layers = len(layer_indices)
    fig, axes = plt.subplots(1, n_layers, figsize=(7 * n_layers, 6))
    if n_layers == 1:
        axes = [axes]

    domain_cfg = [
        ('source_orig', source_orig, 'Source (original)',      '#2176FF', 'o'),
        ('target_orig', target_orig, 'Target (original)',      '#E8333F', 's'),
        ('source_fda',  source_fda,  'Source after FDA',       '#00CFC1', '^'),
        ('target_fda',  target_fda,  'Target after FDA',       '#FF8C00', 'D'),
    ]

    for ax, layer_idx in zip(axes, sorted(layer_indices)):
        arrays = []
        labels = []
        for i, (key, feat_dict, label, color, marker) in enumerate(domain_cfg):
            f = np.nan_to_num(feat_dict[layer_idx],
                              nan=0.0, posinf=0.0, neginf=0.0)
            arrays.append(f)
            labels.extend([i] * f.shape[0])

        combined = np.concatenate(arrays, axis=0)
        labels = np.array(labels)

        print(f"  Running t-SNE (4 domains) for layer {layer_idx} with "
              f"{combined.shape[0]} samples, feature dim={combined.shape[1]}...")

        perplexity = min(30, combined.shape[0] - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                    n_iter=1000, learning_rate='auto', init='pca')
        emb = tsne.fit_transform(combined)

        for i, (key, _, label, color, marker) in enumerate(domain_cfg):
            mask = labels == i
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=color, alpha=0.6, s=25, label=label,
                       marker=marker, edgecolors='none')

        ax.set_title(f'Layer {layer_idx}', fontsize=14)
        ax.legend(fontsize=9, loc='best', markerscale=1.2)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = save_dir / f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_fda_amplitude_comparison(source_imgs, target_imgs, source_fda, target_fda,
                                  save_dir, batch_idx=0):
    """
    Visualize the amplitude spectra (log scale) of original vs FDA-transformed images
    to confirm that the low-frequency swap happened correctly.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Take the first image from the batch, average across channels
    imgs = {
        'Source (original)': source_imgs[0].mean(dim=0),
        'Target (original)': target_imgs[0].mean(dim=0),
        'Source after FDA': source_fda[0].mean(dim=0),
        'Target after FDA': target_fda[0].mean(dim=0),
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for col, (label, img) in enumerate(imgs.items()):
        # Spatial domain
        axes[0, col].imshow(img.cpu().numpy(), cmap='gray')
        axes[0, col].set_title(label, fontsize=11)
        axes[0, col].axis('off')

        # Frequency domain (log amplitude)
        fft = torch.fft.fft2(img)
        fft_shift = torch.fft.fftshift(fft)
        log_amp = torch.log1p(torch.abs(fft_shift)).cpu().numpy()
        axes[1, col].imshow(log_amp, cmap='inferno')
        axes[1, col].set_title(f'{label}\n(Log Amplitude Spectrum)', fontsize=10)
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('Spatial', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)

    fig.suptitle('FDA Amplitude Spectrum Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = save_dir / f'fda_amplitude_comparison_batch{batch_idx}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved amplitude comparison: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='t-SNE visualization of early-layer features with FDA style transfer')
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
    parser.add_argument('--fda-beta', type=float, default=0.05,
                        help='FDA low-frequency window fraction (0 < beta <= 1). '
                             'Typical: 0.01-0.15. Smaller=subtler, larger=stronger.')
    parser.add_argument('--save-dir', type=str,
                        default='runs/tsne_vis_fda', help='save directory')
    parser.add_argument('--save-samples', action='store_true',
                        help='save sample images showing original vs FDA-transformed')
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
    print(f"FDA beta (low-freq window): {args.fda_beta}")
    print(f"Collecting {args.num_batches} batches x {args.batch_size} images each...")

    # ---- Optionally save sample images showing FDA effect ----
    if args.save_samples:
        print("\nSaving sample FDA images for visual inspection...")
        target_iter = iter(cycle(target_loader))
        with torch.no_grad():
            for batch_i, (source_imgs, _, _, _) in enumerate(source_loader):
                if batch_i >= 2:  # save 2 batches of samples
                    break
                target_imgs, _, _, _ = next(target_iter)
                source_imgs = source_imgs.to(device, non_blocking=True).float() / 255
                target_imgs = target_imgs.to(device, non_blocking=True).float() / 255

                source_fda, target_fda = fda_source_to_target(
                    source_imgs, target_imgs, beta=args.fda_beta)

                save_sample_images(source_imgs, target_imgs, source_fda, target_fda,
                                   args.save_dir, batch_idx=batch_i)
                plot_fda_amplitude_comparison(source_imgs, target_imgs, source_fda, target_fda,
                                              args.save_dir, batch_idx=batch_i)

    # ---- Collect all features ----
    print(f"\nCollecting features (FDA beta={args.fda_beta})...")
    result = collect_features_fda(
        model, source_loader, target_loader, device, layer_indices,
        num_batches=args.num_batches, fda_beta=args.fda_beta)

    # ---- Plot 1: Source(original) vs Target(after FDA) ----
    # Target after FDA has source low-freq amp → source-like style
    # Both should carry SOURCE style => expect overlap
    print("\n[1/5] Source (original) vs Target (after FDA) — both carry source style, expect OVERLAP")
    plot_tsne_pair(
        result['source_orig'], result['target_fda'],
        label_a='Source (original)',
        label_b='Target (after FDA)',
        layer_indices=layer_indices,
        title='Source Style - Source(orig) vs Target(FDA)',
        save_dir=args.save_dir)

    # ---- Plot 2: Target(original) vs Source(after FDA) ----
    # Source after FDA has target low-freq amp → target-like style
    # Both should carry TARGET style => expect overlap
    print("\n[2/5] Target (original) vs Source (after FDA) — both carry target style, expect OVERLAP")
    plot_tsne_pair(
        result['target_orig'], result['source_fda'],
        label_a='Target (original)',
        label_b='Source (after FDA)',
        layer_indices=layer_indices,
        title='Target Style - Target(orig) vs Source(FDA)',
        save_dir=args.save_dir)

    # ---- Plot 3: Baseline - Source(original) vs Target(original) ----
    # Different styles → expect SEPARATION
    print("\n[3/5] Source (original) vs Target (original) — different styles, expect SEPARATION")
    plot_tsne_pair(
        result['source_orig'], result['target_orig'],
        label_a='Source (original)',
        label_b='Target (original)',
        layer_indices=layer_indices,
        title='Baseline - Source(orig) vs Target(orig)',
        save_dir=args.save_dir)

    # ---- Plot 4: Source(orig) vs Source(FDA) ----
    print("\n[4/5] Source (original) vs Source (FDA) — different styles, expect SEPARATION")
    plot_tsne_pair(
        result['source_orig'], result['source_fda'],
        label_a='Source (original)',
        label_b='Source (FDA)',
        layer_indices=layer_indices,
        title='Source(orig) vs Source(FDA)',
        save_dir=args.save_dir)

    # ---- Plot 5: All 4 style domains in one figure ----
    print("\n[5/5] All 4 style domains — Source, Target, Source(FDA), Target(FDA)")
    plot_tsne_four_domains(
        source_orig=result['source_orig'],
        target_orig=result['target_orig'],
        source_fda=result['source_fda'],
        target_fda=result['target_fda'],
        layer_indices=layer_indices,
        title='All 4 Style Domains (FDA)',
        save_dir=args.save_dir)

    print(f"\nDone! All plots saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
