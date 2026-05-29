# ML Final Project

## Project

Generalizable AI-Generated Image Detection Using Baseline CNN Models

## Goal

Train reproducible baseline classifiers for real-vs-AI image detection before
merging in stronger multi-branch detector work.

The current local code keeps the baseline script as the main entry point and
only factors augmentation/preprocessing into a small helper module for easier
future merging.

## Files

- `src/train_baseline.py`: baseline training and evaluation script
- `src/data_augmentation.py`: RGB conversion, resizing, normalization, and optional light training augmentation
- `outputs/`: saved local experiment outputs and checkpoints

## Dataset Layout

Default dataset root:

```text
dataset/
```

Expected CIFAKE layout under `--dataset-root`:

```text
dataset/
  cifake/
    train/
      REAL/
      FAKE/
    test/
      REAL/
      FAKE/
```

TinyGenImage combined training expects this layout under `--dataset-root`:

```text
dataset/
  tiny-genimage/
    imagenet_ai_0419_biggan/
      train/
        ai/
        nature/
      val/
        ai/
        nature/
    ...
```

Labels:

- `REAL` / `nature` = `0`
- `FAKE` / `ai` = `1`

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Train

Simple CNN on CIFAKE:

```powershell
python src/train_baseline.py --dataset-root dataset --dataset cifake
```

ResNet-18 on CIFAKE:

```powershell
python src/train_baseline.py --dataset-root dataset --dataset cifake --model resnet18 --epochs 10 --batch-size 128
```

CIFAKE + TinyGenImage:

```powershell
python src/train_baseline.py --dataset-root dataset --dataset combined --model resnet18 --epochs 10 --batch-size 128
```

TinyGenImage only:

```powershell
python src/train_baseline.py --dataset-root dataset --dataset tiny-genimage --model resnet18 --epochs 10 --batch-size 64 --semantic-size 224 --normalization imagenet
```

TinyGenImage only, filtered to one generator:

```powershell
python src/train_baseline.py --dataset-root dataset --dataset tiny-genimage --generators sdv5 --model resnet18 --epochs 10 --batch-size 64 --semantic-size 224 --normalization imagenet
```

Training option names:

```powershell
python src/train_baseline.py --dataset-root dataset --dataset cifake --lr 0.001 --semantic-size 32 --max-train-samples 128 --max-val-samples 64
```

Enable light training-only augmentation:

```powershell
python src/train_baseline.py --dataset-root dataset --dataset cifake --augment
```

Run a tiny overfit sanity check:

```powershell
python src/train_baseline.py --dataset-root dataset --dataset cifake --tiny-overfit --epochs 20 --batch-size 32
```

## Individual Evaluation
### Evaluate Model with ONE DATASET only
Examples
```bash
python src/evaluation.py --checkpoint outputs/resnet18/best_model.pt --dataset cifake
```

```bash
python src/evaluation.py --checkpoint outputs/simple_cnn/best_model.pt --dataset tiny-genimage --tinygenimage-split val
```

### Combined Test
```bash
python src/evaluation.py --checkpoint outputs/resnet18/best_model.pt --dataset combined
```