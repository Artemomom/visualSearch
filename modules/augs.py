from torchvision import transforms
from math import ceil
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        s = max(w, h)
        lft = (s - w) // 2
        rgt = s - w - lft
        top = (s - h) // 2
        bot = s - h - top
        padding = (lft, top, rgt, bot)
        return transforms.functional.pad(image, padding, 0, 'constant')


class RandomLeftRightRotation:
    """Rotate by 90 degrees."""

    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            angle = random.choice([-90, 90])
            return transforms.functional.rotate(x, angle)
        else:
            return x

def prepare_data_transforms(size, kind):
    
    if kind == 'train':
        beforecropped = ceil(size / 0.8)
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomPerspective(0.2),
            transforms.RandomAffine(10),
            SquarePad(),
            transforms.Resize(beforecropped),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            RandomLeftRightRotation(p=0.2),
            transforms.RandomApply([transforms.RandomCrop((size, size))], p=0.5),
            transforms.Resize(size),
            transforms.RandomApply([transforms.GaussianBlur(9)], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
    if kind == 'train_soft': 
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0),
            SquarePad(),
            transforms.Resize(size),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
    if kind == 'val': 
        return transforms.Compose([
            SquarePad(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
    else:
        raise NotImplementedError(f"`{kind}` transform is not implemented. Implemented: " +
                                   "`train`, `train_soft`, `val`")