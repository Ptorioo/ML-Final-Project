from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm

try:
    from src.data_augmentation import build_transform
except ModuleNotFoundError:
    from data_augmentation import build_transform

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class RunConfig:
    model: str
    dataset: str
    dataset_root: str
    generators: list[str] | None
    output_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    val_fraction: float
    seed: int
    num_workers: int
    semantic_size: int
    normalization: str
    augment: bool
    max_train_samples: int | None
    max_val_samples: int | None
    pretrained: bool
    cifar_resnet_stem: bool
    device: str


class BinaryImageDataset(Dataset):
    """Binary image dataset with explicit labels: real/nature=0, fake/ai=1."""

    def __init__(self, class_dirs: dict[Path, int], transform: transforms.Compose) -> None:
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        for class_dir, label in class_dirs.items():
            if not class_dir.exists():
                raise FileNotFoundError(f"Expected image folder not found: {class_dir}")
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((str(path), label))

        if not self.samples:
            raise ValueError(f"No image files found in: {[str(path) for path in class_dirs]}")
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


class CifakeDataset(BinaryImageDataset):
    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        super().__init__({root / "REAL": 0, root / "FAKE": 1}, transform)


class TinyGenImageDataset(BinaryImageDataset):
    def __init__(
        self,
        root: Path,
        split: str,
        transform: transforms.Compose,
        generators: list[str] | None = None,
    ) -> None:
        class_dirs: dict[Path, int] = {}
        generator_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        if generators:
            wanted = set(generators)
            generator_dirs = [
                path for path in generator_dirs if path.name in wanted or short_generator_name(path.name) in wanted
            ]
        for generator_dir in generator_dirs:
            split_dir = generator_dir / split
            if split_dir.exists():
                class_dirs[split_dir / "nature"] = 0
                class_dirs[split_dir / "ai"] = 1

        if not class_dirs:
            raise FileNotFoundError(f"No TinyGenImage generator folders with {split}/ found under {root}")
        super().__init__(class_dirs, transform)


class CombinedDataset(Dataset):
    def __init__(self, datasets_: list[Dataset]) -> None:
        self.datasets = datasets_
        self.cumulative_sizes = np.cumsum([len(dataset) for dataset in datasets_]).tolist()
        self.targets: list[int] = []
        self.samples: list[tuple[str | None, int]] = []
        for dataset in datasets_:
            self.targets.extend(getattr(dataset, "targets"))
            self.samples.extend(getattr(dataset, "samples", [(None, label) for label in getattr(dataset, "targets")]))

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        dataset_index = int(np.searchsorted(self.cumulative_sizes, index, side="right"))
        previous_size = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        sample_index = index - previous_size
        return self.datasets[dataset_index][sample_index]


class LabeledSubset(Subset):
    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        super().__init__(dataset, indices)
        parent_targets = getattr(dataset, "targets")
        parent_samples = getattr(dataset, "samples")
        self.targets = [parent_targets[index] for index in indices]
        self.samples = [parent_samples[index] for index in indices]


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            nn.MaxPool2d(2),
            self._block(32, 64),
            nn.MaxPool2d(2),
            self._block(64, 128),
            nn.MaxPool2d(2),
            self._block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN baselines on CIFAKE.")
    parser.add_argument("--model", choices=["simple_cnn", "resnet18", "resnet50"], default="simple_cnn")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--dataset", choices=["cifake", "tiny-genimage", "combined"], default="cifake")
    parser.add_argument("--generators", nargs="*", default=None, help="TinyGenImage generator names to include.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--semantic-size", type=int, default=32)
    parser.add_argument("--normalization", choices=["cifar10", "imagenet"], default="cifar10")
    parser.add_argument("--augment", action="store_true", help="Enable light training-only augmentation.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Useful for quick smoke tests.")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained ResNet weights.")
    parser.add_argument(
        "--imagenet-resnet-stem",
        action="store_true",
        help="Keep the standard ImageNet ResNet stem instead of the 32x32 CIFAR-style stem.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dataset-mode",
        dest="dataset",
        choices=["cifake", "tiny-genimage", "combined"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--data-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tinygenimage-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", dest="lr", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--image-size", dest="semantic_size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--tiny-overfit", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output_dir is None:
        output_name = args.model if args.dataset == "cifake" else f"{args.model}_{args.dataset}"
        args.output_dir = Path("outputs") / output_name
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stratified_train_val_indices(targets: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for label in sorted(set(targets)):
        label_indices = [i for i, target in enumerate(targets) if target == label]
        rng.shuffle(label_indices)
        val_count = max(1, int(round(len(label_indices) * val_fraction)))
        val_indices.extend(label_indices[:val_count])
        train_indices.extend(label_indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def count_labels(targets: Iterable[int]) -> dict[str, int]:
    targets = list(targets)
    return {"real_0": targets.count(0), "fake_1": targets.count(1)}


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    train_transform = build_transform(args.semantic_size, args.normalization, augment=args.augment)
    eval_transform = build_transform(args.semantic_size, args.normalization, augment=False)

    if args.dataset == "tiny-genimage":
        train_dataset, val_dataset, test_dataset, split_summary = make_tinygenimage_datasets(
            args=args,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
    else:
        train_dataset, val_dataset, test_dataset, split_summary = make_cifake_datasets(
            args=args,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

        if args.dataset == "combined":
            tinygenimage_dir = resolve_tinygenimage_dir(args)
            tiny_train = TinyGenImageDataset(
                tinygenimage_dir,
                split="train",
                transform=train_transform,
                generators=args.generators,
            )
            tiny_val = TinyGenImageDataset(
                tinygenimage_dir,
                split="val",
                transform=eval_transform,
                generators=args.generators,
            )
            train_dataset = CombinedDataset([train_dataset, tiny_train])
            val_dataset = CombinedDataset([val_dataset, tiny_val])
            split_summary["train"]["tinygenimage_train"] = count_labels(tiny_train.targets)
            split_summary["validation"]["tinygenimage_val"] = count_labels(tiny_val.targets)

    if args.max_train_samples:
        train_dataset = LabeledSubset(train_dataset, list(range(min(args.max_train_samples, len(train_dataset)))))
    if args.max_val_samples:
        val_dataset = LabeledSubset(val_dataset, list(range(min(args.max_val_samples, len(val_dataset)))))
    if args.tiny_overfit:
        tiny_indices = list(range(min(100, len(train_dataset))))
        train_dataset = LabeledSubset(train_dataset, tiny_indices)
        val_dataset = train_dataset

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    split_summary["train"]["total"] = count_labels(getattr(train_dataset, "targets"))
    split_summary["validation"]["total"] = count_labels(getattr(val_dataset, "targets"))
    split_summary["test"]["total"] = count_labels(getattr(test_dataset, "targets"))
    if args.tiny_overfit:
        split_summary["tiny_overfit"] = True
    return train_loader, val_loader, test_loader, split_summary


def make_cifake_datasets(
    args: argparse.Namespace,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> tuple[Dataset, Dataset, Dataset, dict]:
    data_dir = resolve_cifake_dir(args)
    train_root = data_dir / "train"
    test_root = data_dir / "test"
    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(
            "Expected CIFAKE layout with train/ and test/ under "
            f"{data_dir}. Use --dataset-root for dataset/cifake layout or --data-dir for a direct CIFAKE path."
        )

    full_train = CifakeDataset(train_root, transform=train_transform)
    full_train_eval = CifakeDataset(train_root, transform=eval_transform)
    train_indices, val_indices = stratified_train_val_indices(full_train.targets, args.val_fraction, args.seed)
    train_dataset: Dataset = LabeledSubset(full_train, train_indices)
    val_dataset: Dataset = LabeledSubset(full_train_eval, val_indices)
    test_dataset: Dataset = CifakeDataset(test_root, transform=eval_transform)
    split_summary = {
        "train": {"cifake": count_labels(getattr(train_dataset, "targets"))},
        "validation": {"cifake": count_labels(getattr(val_dataset, "targets"))},
        "test": {"cifake": count_labels(getattr(test_dataset, "targets"))},
    }
    return train_dataset, val_dataset, test_dataset, split_summary


def make_tinygenimage_datasets(
    args: argparse.Namespace,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
) -> tuple[Dataset, Dataset, Dataset, dict]:
    data_dir = resolve_tinygenimage_dir(args)
    train_dataset: Dataset = TinyGenImageDataset(
        data_dir,
        split="train",
        transform=train_transform,
        generators=args.generators,
    )
    val_dataset: Dataset = TinyGenImageDataset(
        data_dir,
        split="val",
        transform=eval_transform,
        generators=args.generators,
    )
    test_dataset = val_dataset
    split_summary = {
        "train": {"tinygenimage_train": count_labels(getattr(train_dataset, "targets"))},
        "validation": {"tinygenimage_val": count_labels(getattr(val_dataset, "targets"))},
        "test": {"tinygenimage_val": count_labels(getattr(test_dataset, "targets"))},
    }
    return train_dataset, val_dataset, test_dataset, split_summary


def resolve_cifake_dir(args: argparse.Namespace) -> Path:
    if args.data_dir is not None:
        return args.data_dir
    candidates = [
        args.dataset_root / "cifake",
        args.dataset_root / "CIFAKE",
        args.dataset_root,
        Path.home() / "Desktop" / "CIFAKE",
    ]
    for candidate in candidates:
        if (candidate / "train").exists() and (candidate / "test").exists():
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


def short_generator_name(name: str) -> str:
    return (
        name.replace("imagenet_ai_0419_", "")
        .replace("imagenet_ai_0424_", "")
        .replace("imagenet_ai_0508_", "")
        .replace("imagenet_", "")
    )


def build_resnet(name: str, pretrained: bool, cifar_resnet_stem: bool) -> nn.Module:
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported ResNet model: {name}")

    if cifar_resnet_stem:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "simple_cnn":
        return SimpleCNN()
    return build_resnet(
        name=args.model,
        pretrained=args.pretrained,
        cifar_resnet_stem=not args.imagenet_resnet_stem,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    labels: list[int] = []
    probs: list[float] = []

    progress = tqdm(loader, leave=False, desc="train" if is_train else "eval")
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images).view(-1)
            loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels.extend(targets.detach().cpu().numpy().astype(int).tolist())
        probs.extend(batch_probs.tolist())
        total_loss += loss.item() * images.size(0)

    return compute_metrics(labels, probs, total_loss / len(loader.dataset))


def compute_metrics(labels: list[int], probs: list[float], loss: float) -> dict[str, float]:
    predictions = [1 if prob >= 0.5 else 0 for prob in probs]
    metrics = {
        "loss": float(loss),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }
    try:
        metrics["auroc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auroc"] = float("nan")
    return {key: float(value) for key, value in metrics.items()}


@torch.no_grad()
def evaluate_with_errors(
    model: nn.Module,
    dataset: Dataset,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_errors: int = 25,
) -> tuple[dict[str, float], list[list[int]], list[dict[str, object]]]:
    model.eval()
    total_loss = 0.0
    labels: list[int] = []
    probs: list[float] = []
    errors: list[dict[str, object]] = []
    seen = 0

    for images, targets in tqdm(loader, leave=False, desc="test"):
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images).view(-1)
        loss = criterion(logits, targets)
        batch_probs = torch.sigmoid(logits).cpu().numpy()
        batch_labels = targets.cpu().numpy().astype(int)

        for offset, (label, prob) in enumerate(zip(batch_labels, batch_probs)):
            pred = int(prob >= 0.5)
            if pred != int(label) and len(errors) < max_errors:
                path = None
                if hasattr(dataset, "samples"):
                    path = dataset.samples[seen + offset][0]
                errors.append(
                    {
                        "path": path,
                        "label": int(label),
                        "prediction": pred,
                        "fake_probability": float(prob),
                    }
                )

        labels.extend(batch_labels.tolist())
        probs.extend(batch_probs.tolist())
        total_loss += loss.item() * images.size(0)
        seen += images.size(0)

    metrics = compute_metrics(labels, probs, total_loss / len(loader.dataset))
    cm = confusion_matrix(labels, [1 if prob >= 0.5 else 0 for prob in probs], labels=[0, 1]).tolist()
    return metrics, cm, errors


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        model=args.model,
        dataset=args.dataset,
        dataset_root=str(args.dataset_root),
        generators=args.generators,
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        semantic_size=args.semantic_size,
        normalization=args.normalization,
        augment=args.augment,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        pretrained=args.pretrained,
        cifar_resnet_stem=not args.imagenet_resnet_stem,
        device=str(device),
    )
    save_json(args.output_dir / "config.json", asdict(config))

    train_loader, val_loader, test_loader, split_summary = make_loaders(args)
    save_json(args.output_dir / "split_summary.json", split_summary)
    print(f"Split summary: {split_summary}")
    print(f"Using device: {device}")

    model = build_model(args).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    history: list[dict[str, object]] = []
    checkpoint_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        save_json(args.output_dir / "history.json", history)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "validation": val_metrics,
                    "label_mapping": {"REAL": 0, "nature": 0, "FAKE": 1, "ai": 1},
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_confusion_matrix, test_errors = evaluate_with_errors(
        model=model,
        dataset=test_loader.dataset,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )
    results = {
        "label_mapping": {"REAL": 0, "nature": 0, "FAKE": 1, "ai": 1},
        "best_checkpoint": str(checkpoint_path),
        "best_epoch": checkpoint["epoch"],
        "test": test_metrics,
        "confusion_matrix_labels": ["REAL_0", "FAKE_1"],
        "confusion_matrix": test_confusion_matrix,
        "sample_errors": test_errors,
    }
    save_json(args.output_dir / "test_results.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
