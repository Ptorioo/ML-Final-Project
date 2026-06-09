from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

try:
    from src.train_baseline import (
        RunConfig,
        build_model,
        evaluate_with_errors,
        make_loaders,
        run_epoch,
        save_json,
        seed_everything,
    )
except ModuleNotFoundError:
    from train_baseline import (
        RunConfig,
        build_model,
        evaluate_with_errors,
        make_loaders,
        run_epoch,
        save_json,
        seed_everything,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train real-vs-AI image classifiers.")
    parser.add_argument("--model", choices=["simple_cnn", "resnet18", "resnet50"], default="simple_cnn")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["cifake", "tiny-genimage"],
        default=None,
        help="Use one dataset or combine two datasets, e.g. --datasets cifake tiny-genimage.",
    )
    parser.add_argument(
        "--dataset",
        choices=["cifake", "tiny-genimage", "combined"],
        default=None,
        help="Legacy shortcut. Equivalent to --datasets cifake, --datasets tiny-genimage, or both.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--cifake-dir", type=Path, default=None, help="Direct path to CIFAKE root.")
    parser.add_argument("--tinygenimage-dir", type=Path, default=None, help="Direct path to TinyGenImage root.")
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
    parser.add_argument("--augment", action="store_true", help="Enable training-only augmentation.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Useful for quick smoke tests.")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained ResNet weights.")
    parser.add_argument(
        "--imagenet-resnet-stem",
        action="store_true",
        help="Keep the standard ImageNet ResNet stem instead of the 32x32 CIFAR-style stem.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tiny-overfit", action="store_true", help="Run a tiny overfit sanity check.")
    args = parser.parse_args()

    if args.dataset is not None and args.datasets is not None:
        parser.error("Use either --dataset or --datasets, not both.")

    args.dataset = resolve_dataset_mode(args, parser)
    args.data_dir = args.cifake_dir

    if args.output_dir is None:
        output_name = args.model if args.dataset == "cifake" else f"{args.model}_{args.dataset}"
        args.output_dir = Path("outputs") / output_name

    return args


def resolve_dataset_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.dataset is not None:
        return args.dataset

    datasets = args.datasets or ["cifake"]
    unique_datasets = list(dict.fromkeys(datasets))
    if len(unique_datasets) == 1:
        return unique_datasets[0]
    if set(unique_datasets) == {"cifake", "tiny-genimage"}:
        return "combined"

    parser.error("--datasets supports either one dataset or cifake + tiny-genimage.")
    raise AssertionError("unreachable")


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
