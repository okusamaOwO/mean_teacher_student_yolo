import torch


def fourier_domain_adaptation(source_images, target_images, beta=0.01):
    """
    Fourier Domain Adaptation (FDA): transfer the style of target images to
    source images by swapping the low-frequency amplitude spectrum.

    Based on: "FDA: Fourier Domain Adaptation for Semantic Segmentation"
    (Yang & Soatto, CVPR 2020).

    Steps:
        1. Apply FFT to both source and target images.
        2. Extract amplitude and phase from both.
        3. Replace the center (low-frequency) region of source amplitude
           with the corresponding region from target amplitude.
        4. Inverse FFT to reconstruct the adapted source image.

    Args:
        source_images: (B, C, H, W) tensor in [0, 1], the labeled source batch.
        target_images: (B, C, H, W) tensor in [0, 1], the unlabeled target batch.
        beta: float in (0, 1], controls the size of the low-frequency window
              that is swapped.  A larger beta transfers more style.  Typical
              values are 0.01 – 0.09.

    Returns:
        adapted: (B, C, H, W) tensor in [0, 1], source images with target style.
    """
    # --- align target batch size to source batch size ---
    B_src = source_images.shape[0]
    B_tgt = target_images.shape[0]
    if B_tgt != B_src: # 
        # Randomly sample (with replacement if needed) B_src indices from target
        idx = torch.randint(0, B_tgt, (B_src, ), device=source_images.device)
        target_images = target_images[idx]

    # --- forward FFT (shift zero-frequency to center) ---
    src_fft = torch.fft.fft2(source_images, dim=(-2, -1))
    src_fft = torch.fft.fftshift(src_fft, dim=(-2, -1))

    tgt_fft = torch.fft.fft2(target_images, dim=(-2, -1))
    tgt_fft = torch.fft.fftshift(tgt_fft, dim=(-2, -1))

    # --- decompose into amplitude and phase ---
    src_amp = torch.abs(src_fft)
    src_phase = torch.angle(src_fft)

    tgt_amp = torch.abs(tgt_fft)

    # --- build the center mask based on beta ---
    _, _, H, W = source_images.shape
    h_center, w_center = H // 2, W // 2
    h_beta = int(H * beta)
    w_beta = int(W * beta)

    # Ensure at least 1 pixel is swapped
    h_beta = max(h_beta, 1)
    w_beta = max(w_beta, 1)

    # --- swap center amplitudes ---
    adapted_amp = src_amp.clone()
    adapted_amp[
        :, :,
        h_center - h_beta: h_center + h_beta,
        w_center - w_beta: w_center + w_beta,
    ] = tgt_amp[
        :, :,
        h_center - h_beta: h_center + h_beta,
        w_center - w_beta: w_center + w_beta,
    ]

    # --- reconstruct with adapted amplitude + original source phase ---
    adapted_fft = adapted_amp * torch.exp(1j * src_phase)

    # --- inverse FFT ---
    adapted_fft = torch.fft.ifftshift(adapted_fft, dim=(-2, -1))
    adapted = torch.fft.ifft2(adapted_fft, dim=(-2, -1)).real

    # --- clamp to valid range ---
    adapted = torch.clamp(adapted, 0.0, 1.0)

    return adapted