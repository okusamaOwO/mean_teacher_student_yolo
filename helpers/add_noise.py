import albumentations as A
from albumentations.pytorch import ToTensorV2

# appearance/noise augmentation – photometric only, no geometric changes
noise_aug_student = A.Compose([
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    ToTensorV2()  # convert back to torch tensor, keep dtype=float32
])

# teacher can use the same pipeline – randomness is independent every call
noise_aug_teacher = A.Compose([
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    ToTensorV2()
])
