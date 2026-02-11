import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# ==========================================
# 1. Define The Pipelines
# ==========================================

# SHARED / TEACHER (Weak Augmentation)
# Goal: Geometric changes only.
# We use ShiftScaleRotate to cover both Rotation and "Zoom/Crop" effects.
aug_geometric = A.Compose([
    A.HorizontalFlip(p=0.5),
    # Shift (Translate), Scale (Zoom/Crop), and Rotate (+/- 15 degrees)
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15,
                       rotate_limit=15, p=0.75),
    # You can add RandomCrop here if you have a specific target size,
    # e.g., A.RandomCrop(height=224, width=224)
])

# STUDENT ONLY (Strong Augmentation)
# Goal: Appearance changes (Color & Pixel Distortion).
# We DO NOT add geometric changes here, because they are inherited from the teacher step.
aug_pixel_distortion = A.Compose([
    # Color Jitter: Brightness, Contrast, Saturation, Hue
    A.ColorJitter(brightness=0.4, contrast=0.4,
                  saturation=0.4, hue=0.1, p=0.8),
    # Pixel Level Distortions
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
    ], p=0.5),
])

# Helper to convert back to Tensor at the very end
to_tensor = ToTensorV2()

# ==========================================
# 2. Define The Functions
# ==========================================


def add_weak_augmentation(batch_imgs: torch.Tensor) -> torch.Tensor:
    """
    Applies WEAK (Geometric) augmentation.
    Input: [B, 3, H, W] float32 in [0,1]
    Output: [B, 3, H, W] float32 in [0,1]
    """
    imgs = batch_imgs.detach().clamp(0, 1)
    imgs = (imgs * 255).round().to(torch.uint8).cpu()

    out_tensors = []

    # We return the numpy versions too if you want to chain them purely in numpy,
    # but strictly following your request, we return Tensors.
    for img in imgs:
        img_np = img.permute(1, 2, 0).numpy()  # [H, W, 3]

        # Apply Weak/Geometric Augmentation
        augmented = aug_geometric(image=img_np)['image']

        # Convert to Tensor
        out_tensors.append(to_tensor(image=augmented)['image'].float() / 255.0)

    return torch.stack(out_tensors, dim=0)


def add_strong_augmentation(teacher_batch_imgs: torch.Tensor) -> torch.Tensor:
    """
    Applies STRONG (Pixel/Color) augmentation.

    CRITICAL NOTE: This function expects the input to ALREADY be the 
    geometrically transformed image (the output of augment_teacher_img).
    This ensures the Student and Teacher are looking at the same 'Crop/Flip'.
    Input: [B, 3, H, W] float32 in [0,1]
    Output: [B, 3, H, W] float32 in [0,1]
    """
    imgs = teacher_batch_imgs.detach().clamp(0, 1)
    imgs = (imgs * 255).round().to(torch.uint8).cpu()

    out_tensors = []
    for img in imgs:
        img_np = img.permute(1, 2, 0).numpy()  # [H, W, 3]

        # Apply Strong/Pixel Augmentation
        augmented = aug_pixel_distortion(image=img_np)['image']

        # Convert to Tensor
        out_tensors.append(to_tensor(image=augmented)['image'].float() / 255.0)

    return torch.stack(out_tensors, dim=0)
