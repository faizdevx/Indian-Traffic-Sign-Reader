"""Image transforms.

Augmentation choices, each justified against class preservation for road signs:
  * RandomAffine (<=10 deg rotation, 5% shift, 0.9-1.1 scale): signs are photographed at
    tilted angles and varying distances; small geometric changes keep the sign identity.
  * ColorJitter (brightness/contrast 0.3, saturation 0.2): day/evening/night capture. Hue is
    NOT jittered because colour (red/blue/yellow border) carries class information.
  * GaussianBlur (applied with p=0.2): motion/defocus blur from handheld phones.
  * NO horizontal flip: many signs encode direction (turn left/right, keep left/right,
    merge, curve); a flip changes the label.
  * NO random-resized-crop: tight crops can cut off the pictogram that defines the class.
Validation/test transforms are deterministic (resize + normalise only).
Resizing is a direct resize to a square (aspect ratio is not preserved); signs are close to
square, but this does distort wide/tall images - noted as a limitation.
"""
from torchvision import transforms as T

from src.config import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE


def get_train_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(x):
    """Invert Normalize for plotting; x is (3,H,W)."""
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)
