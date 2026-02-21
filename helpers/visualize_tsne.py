from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend


def _register_hooks(model, layer_indices):
    """Register forward hooks on specified layer indices to capture feature maps."""
    features = {}
    hooks = []

    for idx in layer_indices:
        layer = model.model[idx]

        def hook_fn(module, input, output, idx=idx):
            features[idx] = output.detach()

        hooks.append(layer.register_forward_hook(hook_fn))

    return features, hooks


def _remove_hooks(hooks):
    """Remove all registered hooks."""
    for h in hooks:
        h.remove()


def extract_features(model, images, layer_indices):
    """
    Run a forward pass and extract feature maps from specified layers.

    Args:
        model: the YOLO model (student_model).
        images: (B, C, H, W) tensor on the correct device.
        layer_indices: list of int, which layers to extract (e.g. [0, 1, 2, 4]).

    Returns:
        dict mapping layer_index -> (B, D) flattened feature vectors (on CPU).
    """
    features, hooks = _register_hooks(model, layer_indices)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(images)
    if was_training:
        model.train()
    _remove_hooks(hooks)

    # Global Average Pool each feature map to (B, C), then move to CPU
    result = {}
    for idx, feat in features.items():
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        # feat shape: (B, C, H, W)
        pooled = feat.mean(dim=(-2, -1))  # (B, C)
        result[idx] = pooled.cpu().numpy()

    return result


def visualize_tsne(student_model, adapted_source_imgs, target_imgs, layer_indices,
                   save_dir, epoch, perplexity=30, n_iter=1000):
    """
    Generate t-SNE plots comparing early-layer features of adapted source
    images vs target images from the student model.

    Args:
        student_model: the student YOLO model.
        adapted_source_imgs: (B1, C, H, W) tensor – FDA-adapted source images.
        target_imgs: (B2, C, H, W) tensor – target domain images (imgs_teacher).
        layer_indices: list of int, which model layers to visualize.
        save_dir: Path or str, directory to save the plots.
        epoch: int, current epoch number (used in filename).
        perplexity: t-SNE perplexity (auto-clamped if samples are too few).
        n_iter: t-SNE iterations.
    """
    save_dir = str(save_dir)
    tsne_dir = os.path.join(save_dir, 'tsne')
    os.makedirs(tsne_dir, exist_ok=True)

    # Extract features for both domains
    src_feats = extract_features(
        student_model, adapted_source_imgs, layer_indices)
    tgt_feats = extract_features(student_model, target_imgs, layer_indices)

    n_layers = len(layer_indices)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    for ax, idx in zip(axes, layer_indices):
        src_vec = src_feats[idx]  # (B1, D)
        tgt_vec = tgt_feats[idx]  # (B2, D)

        n_src = src_vec.shape[0]
        n_tgt = tgt_vec.shape[0]

        combined = np.concatenate([src_vec, tgt_vec], axis=0)  # (B1+B2, D)
        # 0=adapted source, 1=target
        labels = np.array([0] * n_src + [1] * n_tgt)

        # Adjust perplexity if too few samples
        n_total = combined.shape[0]
        effective_perplexity = min(perplexity, max(5, n_total // 2 - 1))

        tsne = TSNE(n_components=2, perplexity=effective_perplexity,
                    n_iter=n_iter, random_state=42, init='pca')
        embedded = tsne.fit_transform(combined)

        # Plot
        src_emb = embedded[labels == 0]
        tgt_emb = embedded[labels == 1]

        ax.scatter(src_emb[:, 0], src_emb[:, 1],
                   c='dodgerblue', label='Adapted Source', alpha=0.7, s=30, edgecolors='k', linewidths=0.3)
        ax.scatter(tgt_emb[:, 0], tgt_emb[:, 1],
                   c='orangered', label='Target', alpha=0.7, s=30, edgecolors='k', linewidths=0.3)
        ax.set_title(f'Layer {idx}', fontsize=13)
        ax.legend(fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f't-SNE Feature Visualization — Epoch {epoch}', fontsize=15, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(tsne_dir, f'tsne_epoch_{epoch:04d}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return save_path
