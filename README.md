# RF-DETR (rfdetrv2)

Object detection built on RF-DETR with DINOv3 backbones, supervised training, and optional prototype alignment.

## Scripts (`scripts/`)

There are three entry points:

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | Supervised training (`model.train()`). |
| `scripts/finetune.py` | Fine-tuning from RF-DETR COCO weights (`model.finetune()`). |
| `scripts/inference.py` | Image (`--image`) or video (`--video`) inference. |

Examples:

```bash
python scripts/train.py --dataset-dir /path/to/COCO --output-dir ./out --dataset-file coco
python scripts/finetune.py --dataset-dir /path/to/data --output-dir ./out --freeze-encoder --unfreeze-at-epoch 5
python scripts/inference.py --weights checkpoint.pth --image photo.jpg --save out.jpg
python scripts/inference.py --weights checkpoint.pth --video clip.mp4 --output clip_det.mp4
```

Weights resolve under `dinov3_pretrained/` / `rfdetr_pretrained/` or download when configured.

Env shortcuts: `DATASET_DIR`, `OUTPUT_DIR`, `PRETRAINED_ENCODER`, `COCO_WEIGHTS` (finetune).

## Tests

```bash
pip install pytest
pytest
```

## Requirements

Install PyTorch, `supervision` (inference), and other deps for `rfdetrv2` for your environment.
