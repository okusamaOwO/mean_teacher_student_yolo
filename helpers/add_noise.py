import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
# appearance/noise augmentation – photometric only, no geometric changes
noise_aug_student = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    ToTensorV2()  # convert back to torch tensor, keep dtype=float32
])

# teacher can use the same pipeline – randomness is independent every call
noise_aug_teacher = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    ToTensorV2()
])

def add_random_noise(batch_imgs: torch.Tensor) -> torch.Tensor:
    """
    batch_imgs: torch tensor [B,3,H,W] in 0–255 range (uint8 or float)
    return: torch tensor [B,3,H,W], dtype=float32, values in 0–255
    """
    # Ensure on CPU for albumentations and in uint8 format
    imgs = batch_imgs.detach().cpu()
    if imgs.dtype != torch.uint8:
        imgs = imgs.to(torch.uint8)

    out = []
    for img in imgs:  # img shape: [3,H,W]
        # convert to HWC for albumentations
        img_np = img.permute(1, 2, 0).numpy()
        # albumentations expects dtype=uint8 in 0–255
        aug = noise_aug_student(image=img_np)['image']
        # ToTensorV2 outputs [C,H,W] float32 0–255
        out.append(aug)

    return torch.stack(out, dim=0)
