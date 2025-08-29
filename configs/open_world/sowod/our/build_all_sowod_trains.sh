#!/usr/bin/env bash
set -euo pipefail

# ====== EDIT HERE ======
COCO_JSON="/data/datasets/coco/annotations/instances_train2017.json"
FILE_PREFIX="JPEGImages/"
PY="python"  # or python3
# =======================

OUT_DIR_CUR="SOWOD_cur"
OUT_DIR_PREV="SOWOD_prevcur"

echo "[1/2] Generating CUR (stage-only) sets into ${OUT_DIR_CUR}"
mkdir -p "${OUT_DIR_CUR}"
$PY build_sowod_train_jsons.py \
  --coco_json "${COCO_JSON}" \
  --out_dir "${OUT_DIR_CUR}" \
  --file_prefix "${FILE_PREFIX}" \
  --mode cur --dense

echo "[2/2] Generating PREVCUR (cumulative) sets into ${OUT_DIR_PREV}"
mkdir -p "${OUT_DIR_PREV}"
$PY build_sowod_train_jsons.py \
  --coco_json "${COCO_JSON}" \
  --out_dir "${OUT_DIR_PREV}" \
  --file_prefix "${FILE_PREFIX}" \
  --mode prevcur --dense

echo "Done."
echo "  CUR jsons:     ${OUT_DIR_CUR}/t{1,2,3,4}_train.json"
echo "  PREVCUR jsons: ${OUT_DIR_PREV}/t{1,2,3,4}_train.json"
