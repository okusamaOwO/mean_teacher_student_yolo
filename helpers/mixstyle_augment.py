import torch
import numpy as np


def apply_mixstyle_custom(source_imgs, target_imgs, p=0.5, alpha=0.1, beta = 0.1, eps=1e-6):
    """
    Args:
        source_imgs: Batch of source images (B, C, H, W)
        target_imgs: Batch of target images (B, C, H, W)
        p: Probability of applying MixStyle (default 0.5)
        alpha: Beta distribution parameter (default 0.1)
        eps: Epsilon for numerical stability
    Returns:
        X_augmented: The combined batch [source, target] with MixStyle applied
    """

    # 1. Create X [source, target]
    # Shape: (2*B, C, H, W)
    X = torch.cat([source_imgs, target_imgs], dim=0)

    # 2. Check probability (p=0.5)
    # If not triggered, return the clean combined batch
    if np.random.random() > p:
        return X

    # 3. Create perm_X [target, source]
    # This ensures Source matches with Target, and Target matches with Source
    perm_X = torch.cat([target_imgs, source_imgs], dim=0)

    # --- START MIXSTYLE LOGIC ---

    # Calculate statistics for X (Content)
    # mu, sig shape: (2B, C, 1, 1)
    mu = X.mean(dim=[2, 3], keepdim=True)
    var = X.var(dim=[2, 3], keepdim=True)
    sig = (var + eps).sqrt()

    # Calculate statistics for perm_X (Style Reference)
    mu_perm = perm_X.mean(dim=[2, 3], keepdim=True)
    var_perm = perm_X.var(dim=[2, 3], keepdim=True)
    sig_perm = (var_perm + eps).sqrt()

    # Sample Lambda from Beta(0.1, 0.1)
    # Note: We sample a distinct lambda for each image in the batch (2*B)
    # to allow for diversity, or you can sample 1 scalar for the whole batch.
    # Standard MixStyle samples element-wise (N, C, 1, 1) or batch-wise (N, 1, 1, 1).
    N = X.size(0)
    beta_dist = torch.distributions.Beta(alpha, beta)
    lmda = beta_dist.sample((N, 1, 1, 1)).to(X.device)
    # print("-" * 150)
    # print("LAMBDA", lmda)
    # print("-" * 150)

    # Mix the statistics
    # This creates the "Source-like Target" and "Target-like Source" stats
    mu_mix = lmda * mu + (1 - lmda) * mu_perm
    sig_mix = lmda * sig + (1 - lmda) * sig_perm

    # Apply Style Transfer
    # 1. Normalize X (Strip original style)
    # 2. Denormalize with mixed stats (Apply new style)
    X_augmented = ((X - mu) / sig) * sig_mix + mu_mix

    return X_augmented


if __name__ == "__main__":
    # --- Usage Example ---
    # Batch size 4, 3 channels, 64x64 images
    B, C, H, W = 4, 3, 64, 64
    source_batch = torch.randn(B, C, H, W)
    target_batch = torch.randn(B, C, H, W)

    # Apply the pipeline
    mixed_output = apply_mixstyle_custom(
        source_batch, target_batch, p=0.5, alpha=0.1)

    # Result Breakdown:
    # mixed_output[0:B]  -> Source images (mixed with Target stats)
    # mixed_output[B:2B] -> Target images (mixed with Source stats)
    print(f"Output shape: {mixed_output.shape}")  # Should be (8, 3, 64, 64)
