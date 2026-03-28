#!/bin/bash
# autoresearch.sh — Bootstrap & verify environment trước khi start agent loop
# Usage: bash autoresearch.sh

set -e

echo "=== RF-DETRv2 AutoResearch Bootstrap ==="

# 1. Check COCO weights
COCO_WEIGHTS="${COCO_WEIGHTS:-weights/rfdetrv2_base_coco.pth}"
if [ ! -f "$COCO_WEIGHTS" ]; then
    echo "ERROR: COCO pretrained weights not found at $COCO_WEIGHTS"
    echo "  Set env var: export COCO_WEIGHTS=/path/to/your/weight.pth"
    exit 1
fi
echo "[OK] COCO weights: $COCO_WEIGHTS"

# 2. Check dataset
DATASET_DIR="${DATASET_DIR:-data/custom}"
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset not found at $DATASET_DIR"
    echo "  Set env var: export DATASET_DIR=/path/to/your/coco_format_data"
    exit 1
fi
echo "[OK] Dataset: $DATASET_DIR"

# 3. Check rfdetrv2 importable
python -c "from rfdetrv2 import RFDETRBase; print('[OK] rfdetrv2 importable')"

# 4. Init results.tsv if not exists
if [ ! -f results.tsv ]; then
    printf "commit\tval_mAP\tval_mAP50\tmemory_gb\tstatus\tdescription\n" > results.tsv
    echo "[OK] Initialized results.tsv"
fi

echo ""
echo "=== Setup complete ==="
echo "Now start the AI agent and tell it:"
echo "  'Read program.md and kick off a new experiment. Let's do the setup first.'"
echo ""
echo "The agent will:"
echo "  1. Create a branch autoresearch/<tag>"
echo "  2. Run baseline (finetune.py as-is)"  
echo "  3. Loop: modify finetune.py → train → eval → keep/discard"
echo ""
echo "Monitor progress: tail -f run.log"
echo "Check results:    cat results.tsv"