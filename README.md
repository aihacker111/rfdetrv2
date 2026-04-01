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

`run_train.sh` / `run_finetune.sh` **export `CUDA_VISIBLE_DEVICES`** with default **`0`** if unset (use one physical GPU). Override: `CUDA_VISIBLE_DEVICES=1 ./scripts/run_train.sh ...`.

```bash
./scripts/run_train.sh --dataset-dir /data/COCO --output-dir ./out
CUDA_VISIBLE_DEVICES=1 ./scripts/run_finetune.sh --dataset-dir /data/custom --output-dir ./out
```

### Multi-GPU (single machine)

`run_*_multigpu.sh` set **`CUDA_VISIBLE_DEVICES`** to **`0,1,...,NPROC-1`** when unset (default `NPROC=4` → `0,1,2,3`). Override with your own list, e.g. `CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC=4 ./scripts/run_train_multigpu.sh ...`.

```bash
./scripts/run_train_multigpu.sh --dataset-dir /data/COCO --output-dir ./out

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
