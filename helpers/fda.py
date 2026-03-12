import torch


def fda_amplitude_swap(src: torch.Tensor, tgt: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """
    Fourier Domain Adaptation (FDA) via amplitude swapping.

    Replaces the low-frequency amplitude of source images with the corresponding
    low-frequency amplitude of target images, preserving the source phase.
    This shifts the source image style towards the target domain without
    altering the semantic structure (which is encoded in the phase).

    Reference: Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic Segmentation", CVPR 2020.

    Args:
        src  : source-domain images,  [B, C, H, W], float32 in [0, 1], on any device.
        tgt  : target-domain images,  [B, C, H, W], float32 in [0, 1], same device as src.
               If the batch sizes differ, target images are randomly sampled / tiled to match src.
        beta : controls the low-frequency window size relative to the spatial dimensions.
               The swapped region is a central square of size (2*beta*H) x (2*beta*W).
               Typical values: 0.05 – 0.2.  Default: 0.1.

    Returns:
        FDA-adapted source images, [B, C, H, W], float32 clipped to [0, 1].
    """
    assert src.ndim == 4 and tgt.ndim == 4, "Expected 4-D tensors [B, C, H, W]"
    assert 0.0 < beta < 0.5, "beta must be in (0, 0.5)"

    B, C, H, W = src.shape

    # --- align batch sizes ---------------------------------------------------
    B_tgt = tgt.shape[0]
    if B_tgt != B:
        # randomly sample B indices from the target batch (with replacement if needed)
        idx = torch.randint(0, B_tgt, (B,), device=tgt.device)
        tgt = tgt[idx]

    # --- FFT (DC at centre after fftshift) -----------------------------------
    src_fft = torch.fft.fftshift(
        torch.fft.fft2(src, norm='ortho'), dim=(-2, -1))
    tgt_fft = torch.fft.fftshift(
        torch.fft.fft2(tgt, norm='ortho'), dim=(-2, -1))

    src_amp = torch.abs(src_fft)
    src_pha = torch.angle(src_fft)
    tgt_amp = torch.abs(tgt_fft)

    # --- define the low-frequency central window -----------------------------
    h_c, w_c = H // 2, W // 2
    h_low = max(1, int(H * beta))
    w_low = max(1, int(W * beta))

    h1, h2 = h_c - h_low, h_c + h_low
    w1, w2 = w_c - w_low, w_c + w_low

    # --- swap amplitude in the low-frequency region --------------------------
    src_amp_new = src_amp.clone()
    src_amp_new[:, :, h1:h2, w1:w2] = tgt_amp[:, :, h1:h2, w1:w2]

    # --- reconstruct and inverse FFT -----------------------------------------
    # amp * e^{j*phase}
    src_fft_new = torch.polar(src_amp_new, src_pha)
    adapted = torch.fft.ifft2(
        torch.fft.ifftshift(src_fft_new, dim=(-2, -1)), norm='ortho'
    ).real

    return adapted.clamp(0.0, 1.0)
