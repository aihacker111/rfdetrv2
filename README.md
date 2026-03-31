# RF-DETR (rfdetrv2)

Object detection built on RF-DETR with DINOv3 backbones, supervised training, and optional prototype alignment.

## Layout

Entry CLIs live under **`scripts/`** and call **`rfdetrv2.runner.Pipeline`** with YAML from **`rfdetrv2/configs/`** (`train_from_scratch/`, `finetune/`, `inference/`).

## Quick start (supervised training)

From the repository root:

```bash
# Single GPU
./scripts/run_train.sh --dataset-dir /path/to/coco --output-dir ./output --variant base
python scripts/train.py --help

# Multi-GPU (torchrun)
./scripts/run_train_torchrun.sh --dataset-dir /path/to/coco --output-dir ./output --variant base
```

Fine-tune: `./scripts/run_finetune.sh` or `./scripts/run_finetune_single_gpu.sh` → `scripts/finetune.py`.

Weights resolve under `dinov3_pretrained/` or download automatically; DINOv3 hub source is ensured under `dinov3/` when needed.

## Other tools (inference, eval)

```bash
python scripts/inference.py --weights … --image … --variant base
python scripts/evaluate.py --weights … --dataset-dir … --variant base
./scripts/run_evaluate.sh --weights … --dataset-dir …
```

## Tests

```bash
pip install pytest
pytest
```

## Requirements

Install project dependencies for `rfdetrv2` (PyTorch, supervision, etc.) per your environment; `dinov3/requirements.txt` covers the vendored hub tree if you develop inside `dinov3/`.
