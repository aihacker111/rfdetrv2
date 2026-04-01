# RF-DETR (rfdetrv2)

Object detection built on RF-DETR with DINOv3 backbones, supervised training, and optional prototype alignment.

## Scripts (`scripts/`)

There are three entry points:

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | Supervised training (`model.train()`). |
| `scripts/finetune.py` | Fine-tuning from RF-DETR COCO weights (`model.finetune()`). |
| `scripts/inference.py` | Image (`--image`) or video (`--video`) inference. |
| `scripts/run_train.sh` | Single-GPU train (`python3 scripts/train.py`). |
| `scripts/run_finetune.sh` | Single-GPU finetune (`python3 scripts/finetune.py`). |
| `scripts/run_train_multigpu.sh` | Multi-GPU train (`torchrun`). |
| `scripts/run_finetune_multigpu.sh` | Multi-GPU finetune (`torchrun`). |

Examples:

```bash
python scripts/train.py --dataset-dir /path/to/COCO --output-dir ./out --dataset-file coco
python scripts/finetune.py --dataset-dir /path/to/data --output-dir ./out --freeze-encoder --unfreeze-at-epoch 5
python scripts/inference.py --weights checkpoint.pth --image photo.jpg --save out.jpg
python scripts/inference.py --weights checkpoint.pth --video clip.mp4 --output clip_det.mp4
```

Weights resolve under `dinov3_pretrained/` / `rfdetr_pretrained/` or download when configured.

Env shortcuts: `DATASET_DIR`, `OUTPUT_DIR`, `PRETRAINED_ENCODER`, `COCO_WEIGHTS` (finetune).

### Single GPU

```bash
./scripts/run_train.sh --dataset-dir /data/COCO --output-dir ./out
CUDA_VISIBLE_DEVICES=0 ./scripts/run_finetune.sh --dataset-dir /data/custom --output-dir ./out
```

### Multi-GPU (single machine)

```bash
# 4 GPUs (default NPROC=4); pick GPUs with CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/run_train_multigpu.sh --dataset-dir /data/COCO --output-dir ./out

NPROC=8 ./scripts/run_train_multigpu.sh --dataset-dir /data/COCO --output-dir ./out --batch-size 2

./scripts/run_finetune_multigpu.sh --dataset-dir /data/custom --output-dir ./out --freeze-encoder
```

Optional env: `NPROC`, `MASTER_PORT` (train defaults to `29500`, finetune to `29501` if both run on one host).

## Tests

```bash
pip install pytest
pytest
```

## Requirements

Install PyTorch, `supervision` (inference), and other deps for `rfdetrv2` for your environment.
