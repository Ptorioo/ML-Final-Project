from __future__ import annotations

from PIL import Image
from torchvision import transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvertRGB:
    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


def build_transform(image_size: int, normalization: str, augment: bool = False) -> transforms.Compose:
    if normalization == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = CIFAR10_MEAN, CIFAR10_STD

    steps: list[object] = [
        ConvertRGB(),
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
    ]
    if augment:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.25),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transforms.Compose(steps)
