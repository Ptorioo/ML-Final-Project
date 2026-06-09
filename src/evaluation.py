from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

try:
    from src.data_augmentation import CIFAR10_MEAN, CIFAR10_STD, IMAGENET_MEAN, IMAGENET_STD, ConvertRGB
    from src.train_baseline import (
        CifakeDataset,
        CombinedDataset,
        TinyGenImageDataset,
        build_model,
        evaluate_with_errors,
    )
except ModuleNotFoundError:
    from data_augmentation import CIFAR10_MEAN, CIFAR10_STD, IMAGENET_MEAN, IMAGENET_STD, ConvertRGB
    from train_baseline import (
        CifakeDataset,
        CombinedDataset,
        TinyGenImageDataset,
        build_model,
        evaluate_with_errors,
    )


LABEL_MAPPING = {"REAL": 0, "nature": 0, "FAKE": 1, "ai": 1}
DISTORTION_CHOICES = ["clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30", "blur", "resize", "noise"]


class JpegCompression:
    def __init__(self, quality: int) -> None:
        self.quality = quality

    def __call__(self, image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


class DownUpResize:
    def __init__(self, scale: float = 0.5) -> None:
        self.scale = scale

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        down_size = (max(1, int(width * self.scale)), max(1, int(height * self.scale)))
        downsampled = image.resize(down_size, resample=Image.Resampling.BILINEAR)
        return downsampled.resize((width, height), resample=Image.Resampling.BILINEAR)


class GaussianNoise:
    def __init__(self, std: float = 10.0) -> None:
        self.std = std

    def __call__(self, image: Image.Image) -> Image.Image:
        array = np.asarray(image).astype(np.float32)
        noise = np.random.normal(loc=0.0, scale=self.std, size=array.shape)
        array = np.clip(array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(array)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved baseline checkpoint on selected datasets.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a .pt checkpoint.")
    parser.add_argument(
        "--dataset",
        choices=["cifake", "tiny-genimage", "combined", "all"],
        required=True,
        help="Dataset to evaluate. Use 'all' to report CIFAKE and TinyGenImage separately.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--cifake-dir", type=Path, default=None, help="Direct path to CIFAKE root.")
    parser.add_argument("--tinygenimage-dir", type=Path, default=None, help="Direct path to TinyGenImage root.")
    parser.add_argument("--generators", nargs="*", default=None, help="TinyGenImage generator names to include.")
    parser.add_argument("--tinygenimage-split", choices=["train", "val"], default="val")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--model", choices=["simple_cnn", "resnet18", "resnet50"], default=None)
    parser.add_argument("--semantic-size", type=int, default=None)
    parser.add_argument("--normalization", choices=["cifar10", "imagenet"], default=None)
    parser.add_argument("--imagenet-resnet-stem", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-errors", type=int, default=25)
    parser.add_argument(
        "--distortions",
        nargs="+",
        choices=[*DISTORTION_CHOICES, "all"],
        default=["clean"],
        help="Test-time distortions to apply before resize/normalization. Use 'all' for robustness evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stochastic distortions such as noise.")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get("config", {})
    if isinstance(checkpoint, dict):
        return checkpoint, {}
    raise TypeError(f"Unsupported checkpoint format: {path}")


def config_value(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    if value is not None:
        return value
    return config.get(name, default)


def resolve_cifake_dir(args: argparse.Namespace) -> Path:
    if args.cifake_dir is not None:
        return args.cifake_dir
    candidates = [
        args.dataset_root / "cifake",
        args.dataset_root / "CIFAKE",
        args.dataset_root,
        Path.home() / "Desktop" / "CIFAKE",
    ]
    for candidate in candidates:
        if (candidate / "test").exists():
            return candidate
    return args.dataset_root / "cifake"


def resolve_tinygenimage_dir(args: argparse.Namespace) -> Path:
    if args.tinygenimage_dir is not None:
        return args.tinygenimage_dir
    candidates = [
        args.dataset_root / "tiny-genimage",
        args.dataset_root / "tinygenimage",
        Path.home() / "Desktop" / "tinygenimage",
        Path.home() / "Desktop" / "tiny-genimage",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return args.dataset_root / "tiny-genimage"


def build_eval_datasets(args: argparse.Namespace, transform: Any) -> dict[str, Dataset]:
    datasets: dict[str, Dataset] = {}

    if args.dataset in {"cifake", "combined", "all"}:
        cifake_root = resolve_cifake_dir(args)
        datasets["cifake_test"] = CifakeDataset(cifake_root / "test", transform=transform)

    if args.dataset in {"tiny-genimage", "combined", "all"}:
        tiny_root = resolve_tinygenimage_dir(args)
        datasets[f"tinygenimage_{args.tinygenimage_split}"] = TinyGenImageDataset(
            tiny_root,
            split=args.tinygenimage_split,
            transform=transform,
            generators=args.generators,
        )

    if args.dataset == "combined":
        datasets = {"combined": CombinedDataset(list(datasets.values()))}

    return datasets


def count_labels(dataset: Dataset) -> dict[str, int]:
    targets = list(getattr(dataset, "targets"))
    return {"real_0": targets.count(0), "fake_1": targets.count(1)}


def resolve_distortions(requested: list[str]) -> list[str]:
    if "all" in requested:
        return DISTORTION_CHOICES
    return list(dict.fromkeys(requested))


def build_distortion(name: str) -> object | None:
    if name == "clean":
        return None
    if name.startswith("jpeg_q"):
        return JpegCompression(int(name.replace("jpeg_q", "")))
    if name == "blur":
        return transforms.GaussianBlur(kernel_size=3, sigma=(0.8, 0.8))
    if name == "resize":
        return DownUpResize(scale=0.5)
    if name == "noise":
        return GaussianNoise(std=10.0)
    raise ValueError(f"Unsupported distortion: {name}")


def build_eval_transform(image_size: int, normalization: str, distortion: str) -> transforms.Compose:
    if normalization == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = CIFAR10_MEAN, CIFAR10_STD

    steps: list[object] = [ConvertRGB()]
    distortion_transform = build_distortion(distortion)
    if distortion_transform is not None:
        steps.append(distortion_transform)
    steps.extend(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transforms.Compose(steps)


def print_metrics_table(dataset_name: str, distortion_results: dict[str, Any]) -> None:
    print(f"\n{dataset_name}")
    print("Distortion\tAccuracy\tPrecision\tRecall\tF1\tAUROC")
    for distortion, payload in distortion_results.items():
        metrics = payload["metrics"]
        print(
            f"{distortion}\t"
            f"{metrics['accuracy']:.4f}\t"
            f"{metrics['precision']:.4f}\t"
            f"{metrics['recall']:.4f}\t"
            f"{metrics['f1']:.4f}\t"
            f"{metrics['auroc']:.4f}"
        )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    state_dict, checkpoint_config = load_checkpoint(args.checkpoint, device)

    model_args = SimpleNamespace(
        model=config_value(args, checkpoint_config, "model", "simple_cnn"),
        pretrained=False,
        imagenet_resnet_stem=args.imagenet_resnet_stem
        or not bool(checkpoint_config.get("cifar_resnet_stem", True)),
    )
    semantic_size = int(config_value(args, checkpoint_config, "semantic_size", 32))
    normalization = str(config_value(args, checkpoint_config, "normalization", "cifar10"))
    batch_size = int(config_value(args, checkpoint_config, "batch_size", 128))
    num_workers = int(config_value(args, checkpoint_config, "num_workers", 2))

    model = build_model(model_args).to(device)
    model.load_state_dict(state_dict)
    criterion = nn.BCEWithLogitsLoss()
    distortions = resolve_distortions(args.distortions)

    results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "label_mapping": LABEL_MAPPING,
        "model": model_args.model,
        "semantic_size": semantic_size,
        "normalization": normalization,
        "distortions": distortions,
        "datasets": {},
    }

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    for distortion in distortions:
        transform = build_eval_transform(semantic_size, normalization, distortion)
        datasets = build_eval_datasets(args, transform)
        for name, dataset in datasets.items():
            if name not in results["datasets"]:
                results["datasets"][name] = {
                    "split_summary": count_labels(dataset),
                    "distortions": {},
                }

            loader = DataLoader(dataset, shuffle=False, **loader_kwargs)
            metrics, confusion, errors = evaluate_with_errors(
                model=model,
                dataset=dataset,
                loader=loader,
                criterion=criterion,
                device=device,
                max_errors=args.max_errors,
            )
            results["datasets"][name]["distortions"][distortion] = {
                "metrics": metrics,
                "confusion_matrix_labels": ["REAL_0", "FAKE_1"],
                "confusion_matrix": confusion,
                "sample_errors": errors,
            }

    for name, payload in results["datasets"].items():
        print_metrics_table(name, payload["distortions"])

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
