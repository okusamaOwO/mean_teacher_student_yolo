"""
t-SNE visualization of early-layer feature maps to verify MixStyle style exchange.

This script:
1. Loads a trained model (or initializes one from config)
2. Loads source (labeled) and target (unlabeled) images
3. Extracts feature maps from early backbone layers (layers 0, 1, 2) using hooks
4. Visualizes:
   - Source (original) vs Target (after MixStyle) — both should carry source style, expecting overlap
   - Source (after MixStyle) vs Target (original) — both should carry target style, expecting overlap

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


def collect_features_cross_style(model, source_loader, target_loader, device, layer_indices,
                                 num_batches=10, mix_alpha=1.3, mix_beta=6):
    """
    Collect features for cross-style comparison:
      - source_original: source images WITHOUT MixStyle (original source style)
      - target_after_mix: target images AFTER MixStyle (should now carry source style)
      - source_after_mix: source images AFTER MixStyle (should now carry target style)
      - target_original: target images WITHOUT MixStyle (original target style)

    Returns:
        features dict with keys: 'source_orig', 'target_mixed', 'source_mixed', 'target_orig'
        Each is a dict {layer_idx: (N, 2*C) numpy array}
    """
    model.eval()
    hooks, handles = register_hooks(model, layer_indices)

    result = {
        'source_orig':   {idx: [] for idx in layer_indices},
        'target_mixed':  {idx: [] for idx in layer_indices},
        'source_mixed':  {idx: [] for idx in layer_indices},
        'target_orig':   {idx: [] for idx in layer_indices},
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

            source_imgs = source_imgs.to(
                device, non_blocking=True).float() / 255
            target_imgs = target_imgs.to(
                device, non_blocking=True).float() / 255

            B = source_imgs.size(0)

            # 1) Original source features
            forward_and_collect(source_imgs, 'source_orig')

            # 2) Original target features
            forward_and_collect(target_imgs, 'target_orig')

            # 3) Apply MixStyle
            mixed_output = apply_mixstyle_custom(
                source_imgs, target_imgs, p=1, alpha=mix_alpha, beta=mix_beta)
            source_mixed = mixed_output[0:B]    # source content + target style
            # target content + source style
            target_mixed = mixed_output[B:]

            # 4) Source after MixStyle (carries target style)
            forward_and_collect(source_mixed, 'source_mixed')

            # 5) Target after MixStyle (carries source style)
            forward_and_collect(target_mixed, 'target_mixed')

    for h in handles:
        h.remove()

    # Concatenate
    for key in result:
        for idx in layer_indices:
            result[key][idx] = np.concatenate(result[key][idx], axis=0)

    return result


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

    # ---- Collect all features ----
    print(
        f"\nCollecting features (alpha={args.mix_alpha}, beta={args.mix_beta})...")
    result = collect_features_cross_style(
        model, source_loader, target_loader, device, layer_indices,
        num_batches=args.num_batches, mix_alpha=args.mix_alpha, mix_beta=args.mix_beta)

    # ---- Plot 1: Source(original) vs Target(after MixStyle) ----
    # Both should carry SOURCE style => expect overlap
    print("\n[1/3] Source (original) vs Target (after MixStyle) — both carry source style, expect OVERLAP")
    plot_tsne_pair(
        result['source_orig'], result['target_mixed'],
        label_a='Source (original)',
        label_b='Target (after MixStyle)',
        layer_indices=layer_indices,
        title='Source Style - Source(orig) vs Target(mixed)',
        save_dir=args.save_dir)

    # ---- Plot 2: Target(original) vs Source(after MixStyle) ----
    # Both should carry TARGET style => expect overlap
    print("\n[2/3] Target (original) vs Source (after MixStyle) — both carry target style, expect OVERLAP")
    plot_tsne_pair(
        result['target_orig'], result['source_mixed'],
        label_a='Target (original)',
        label_b='Source (after MixStyle)',
        layer_indices=layer_indices,
        title='Target Style - Target(orig) vs Source(mixed)',
        save_dir=args.save_dir)

    # ---- Plot 3: Baseline - Source(original) vs Target(original) ----
    # Different styles => expect SEPARATION
    print("\n[3/3] Source (original) vs Target (original) — different styles, expect SEPARATION")
    plot_tsne_pair(
        result['source_orig'], result['target_orig'],
        label_a='Source (original)',
        label_b='Target (original)',
        layer_indices=layer_indices,
        title='Baseline - Source(orig) vs Target(orig)',
        save_dir=args.save_dir)   
     
    # ---- Plot 4: 
    # same images but with different aug 
    print("\n[3/3] Source (original) vs Source (mixed) — different styles, expect SEPARATION")
    plot_tsne_pair(
        result['source_orig'], result['source_mixed'],
        label_a='Source (original)',
        label_b='Source (mixed)',
        layer_indices=layer_indices,
        title='Baseline - Source(orig) vs Source(mixed)',
        save_dir=args.save_dir)


    print("\nDone! All plots saved to:", args.save_dir)


if __name__ == '__main__':
    main()
