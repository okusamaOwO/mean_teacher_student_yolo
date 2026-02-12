# turn teacher images into source-like images with MixStyle
import torch


def mixstyle_for_teacher(target_imgs: torch.Tensor, source_imgs: torch.Tensor, p: float = 1, alpha: float = 1) -> torch.Tensor:
    if torch.rand(1).item() > p:
        return target_imgs

    # Compute channel-wise statistics for target images (content)
    target_mean = target_imgs.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
    target_std = target_imgs.std(
        dim=[2, 3], keepdim=True) + 1e-6  # [B, C, 1, 1]

    # Compute channel-wise statistics for source images (style)
    source_mean = source_imgs.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
    source_std = source_imgs.std(
        dim=[2, 3], keepdim=True) + 1e-6  # [B, C, 1, 1]

    # Mix the statistics: blend between target and source style
    mixed_mean = (1 - alpha) * target_mean + alpha * source_mean
    mixed_std = (1 - alpha) * target_std + alpha * source_std

    # Normalize target images and apply mixed statistics (style transfer)
    normalized = (target_imgs - target_mean) / target_std
    source_styles_target_imgs = normalized * mixed_std + mixed_mean

    return source_styles_target_imgs
