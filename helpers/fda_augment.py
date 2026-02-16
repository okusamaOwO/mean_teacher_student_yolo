"""
Fourier Domain Adaptation (FDA) — low-frequency amplitude swap between domains.

Based on:
    "FDA: Fourier Domain Adaptation for Semantic Segmentation" (Yang & Soatto, CVPR 2020)

The core idea:
    1. Convert source and target images to the frequency domain via 2D FFT.
    2. Replace the low-frequency amplitude components of the source image with those
       of the target image (and vice versa).
    3. Convert back to the spatial domain via inverse FFT.

This produces:
    - "target-styled source":  source content + target low-freq style
    - "source-styled target":  target content + source low-freq style

The parameter `beta` (0 < beta <= 1) controls the size of the low-frequency window
that is swapped. Smaller beta → subtler style transfer; larger beta → stronger.
"""

import torch
import numpy as np


def _low_freq_mask(h, w, beta):
    """Create a centred binary mask selecting the low-frequency region.

    The mask covers a rectangle of size (2*b_h, 2*b_w) centred in the FFT,
    where b_h = beta * h and b_w = beta * w.  When used after fftshift the
    centre corresponds to the DC / low-frequency components.

    Returns:
        mask: (1, 1, H, W) boolean tensor
    """
    b_h = int(np.floor(beta * h))
    b_w = int(np.floor(beta * w))
    cy, cx = h // 2, w // 2
    mask = torch.zeros(1, 1, h, w, dtype=torch.bool)
    mask[:, :, cy - b_h:cy + b_h, cx - b_w:cx + b_w] = True
    return mask


def fda_source_to_target(source_imgs, target_imgs, beta=0.01):
    """
    Swap low-frequency amplitude from target into source images (pixel-level FDA).

    Args:
        source_imgs: (B, C, H, W) tensor, float, range [0, 1] or similar.
        target_imgs: (B, C, H, W) tensor, float, same spatial size as source.
        beta: float in (0, 1]. Controls the fraction of the spectrum that is swapped.
              Typical values: 0.01 – 0.15.

    Returns:
        source_in_target_style: (B, C, H, W) — source content with target low-freq style.
        target_in_source_style: (B, C, H, W) — target content with source low-freq style.
    """
    assert source_imgs.shape == target_imgs.shape, \
        f"Shape mismatch: source {source_imgs.shape} vs target {target_imgs.shape}"

    _, _, H, W = source_imgs.shape

    # 2D FFT per channel, then shift DC to centre
    fft_src = torch.fft.fft2(source_imgs, dim=(-2, -1))
    fft_trg = torch.fft.fft2(target_imgs, dim=(-2, -1))

    fft_src = torch.fft.fftshift(fft_src, dim=(-2, -1))
    fft_trg = torch.fft.fftshift(fft_trg, dim=(-2, -1))

    # Amplitude and phase
    amp_src = torch.abs(fft_src)
    amp_trg = torch.abs(fft_trg)
    pha_src = torch.angle(fft_src)
    pha_trg = torch.angle(fft_trg)

    # Low-frequency mask
    mask = _low_freq_mask(H, W, beta).to(source_imgs.device)

    # Swap low-frequency amplitude
    # source_in_target_style: source phase + (source high-freq amp, target low-freq amp)
    amp_src_new = amp_src.clone()
    amp_src_new[:, :, mask[0, 0]] = amp_trg[:, :, mask[0, 0]]

    # target_in_source_style: target phase + (target high-freq amp, source low-freq amp)
    amp_trg_new = amp_trg.clone()
    amp_trg_new[:, :, mask[0, 0]] = amp_src[:, :, mask[0, 0]]

    # Reconstruct complex spectrum and invert
    fft_src_new = amp_src_new * torch.exp(1j * pha_src)
    fft_trg_new = amp_trg_new * torch.exp(1j * pha_trg)

    fft_src_new = torch.fft.ifftshift(fft_src_new, dim=(-2, -1))
    fft_trg_new = torch.fft.ifftshift(fft_trg_new, dim=(-2, -1))

    source_in_target_style = torch.fft.ifft2(fft_src_new, dim=(-2, -1)).real
    target_in_source_style = torch.fft.ifft2(fft_trg_new, dim=(-2, -1)).real

    # Clamp to valid range (same as input range)
    source_in_target_style = torch.clamp(source_in_target_style, 0.0, 1.0)
    target_in_source_style = torch.clamp(target_in_source_style, 0.0, 1.0)

    return source_in_target_style, target_in_source_style


def apply_fda(source_imgs, target_imgs, p=0.5, beta=0.01):
    """
    Convenience wrapper matching MixStyle-like interface.

    Args:
        source_imgs: (B, C, H, W) source batch.
        target_imgs: (B, C, H, W) target batch.
        p: probability of applying FDA (default 0.5).
        beta: low-frequency window fraction (default 0.01).

    Returns:
        combined: (2*B, C, H, W) — [source_in_target_style, target_in_source_style]
                  If not triggered by probability, returns [source, target] unchanged.
    """
    if np.random.random() > p:
        return torch.cat([source_imgs, target_imgs], dim=0)

    src_styled, trg_styled = fda_source_to_target(source_imgs, target_imgs, beta=beta)
    return torch.cat([src_styled, trg_styled], dim=0)


if __name__ == "__main__":
    # Quick sanity check
    B, C, H, W = 4, 3, 64, 64
    source = torch.rand(B, C, H, W)
    target = torch.rand(B, C, H, W)

    src_styled, trg_styled = fda_source_to_target(source, target, beta=0.05)
    print(f"source_in_target_style shape: {src_styled.shape}")  # (4, 3, 64, 64)
    print(f"target_in_source_style shape: {trg_styled.shape}")  # (4, 3, 64, 64)
    print(f"Value range src_styled: [{src_styled.min():.3f}, {src_styled.max():.3f}]")
    print(f"Value range trg_styled: [{trg_styled.min():.3f}, {trg_styled.max():.3f}]")

    combined = apply_fda(source, target, p=1.0, beta=0.05)
    print(f"Combined output shape: {combined.shape}")  # (8, 3, 64, 64)
