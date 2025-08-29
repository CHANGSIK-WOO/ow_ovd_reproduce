#!/usr/bin/env bash
# Generate attribute & class embeddings for large_t1 ~ large_t4 (cumulative)
# Run from: ow_ovd/tools
#   chmod +x gen_all_embeds.sh
#   ./gen_all_embeds.sh

set -euo pipefail

# -------- paths & env --------
SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd)"
JSON_DIR="${SCRIPT_DIR}/../data/VOC2007/SOWOD"
OUT_DIR="${JSON_DIR}"

DEVICE="cuda"  # or "cpu"
MODEL_NAME="openai/clip-vit-base-patch32"

# -------- helpers --------
merge_jsons() {
  # Usage: merge_jsons OUT.json IN1.json IN2.json ...
  local OUT_JSON="$1"; shift
  if command -v jq >/dev/null 2>&1; then
    # merge objects: later keys override earlier; values are lists (we'll de-dup later in encode script)
    jq -s 'reduce .[] as $item ({}; . * $item)' "$@" > "${OUT_JSON}"
  else
    # python fallback merge
    python - "$OUT_JSON" "$@" <<'PY'
import json, sys, collections
out = sys.argv[1]
ins = sys.argv[2:]
merged = collections.OrderedDict()
for p in ins:
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    for k,v in d.items():
        if not isinstance(v, list): 
            continue
        merged.setdefault(k, [])
        merged[k].extend(v)
# keep order but light dedup (case-insensitive)
for k, lst in merged.items():
    seen=set(); uniq=[]
    for s in lst:
        s = str(s).strip()
        key=s.lower()
        if s and key not in seen:
            seen.add(key); uniq.append(s)
    merged[k]=uniq
with open(out, 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
print(f"[merge] wrote {out}")
PY
  fi
}

make_attr_embeds() {
  # Usage: make_attr_embeds TASK IN_JSON
  local TASK="$1"
  local IN_JSON="$2"
  local OUT_PTH="${OUT_DIR}/att_${TASK}_embeddings.pth"
  echo "[Attributes] ${IN_JSON} -> ${OUT_PTH}"
  python "${SCRIPT_DIR}/encode_attributes_from_json.py" \
    --in_json "${IN_JSON}" \
    --out_pth "${OUT_PTH}" \
    --model_name "${MODEL_NAME}" \
    --pattern "{category} is {attr}" \
    --device "${DEVICE}"
}

make_class_embeds_cumulative() {
  # Usage: make_class_embeds_cumulative TASK NUM_SPLITS
  # NUM_SPLITS: 1 => t1, 2 => t1+t2, 3 => t1+t2+t3, 4 => t1+t2+t3+t4
  local TASK="$1"
  local NUM="$2"
  local OUT_NPY="${OUT_DIR}/${TASK}_gt_embeddings.npy"
  echo "[Classes|cumulative ${NUM}] -> ${OUT_NPY}"
  python - <<PY
import os, numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModelWithProjection

CLASS_NAMES = {
    "t1" : ["airplane","bicycle","bird","boat","bus","car",
            "cat","cow","dog","horse","motorcycle","sheep",
            "train","elephant","bear","zebra","giraffe","truck","person"],
    "t2" : ["traffic light","fire hydrant","stop sign","parking meter","bench","chair","dining table",
            "potted plant","backpack","umbrella","handbag","tie","suitcase","microwave","oven","toaster",
            "sink","refrigerator","bed","toilet","couch"],
    "t3" : ["frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
            "tennis racket","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake"],
    "t4" : ["laptop","mouse","remote","keyboard","cell phone","book","clock","vase","scissors","teddy bear",
            "hair drier","toothbrush","wine glass","cup","fork","knife","spoon","bowl","tv","bottle"]
}

order = ["t1","t2","t3","t4"]
num = int(${NUM})
classes = []
for k in order[:num]:
    classes += CLASS_NAMES[k]

model_name = "${MODEL_NAME}"
device = "${DEVICE}"

tokenizer = AutoTokenizer.from_pretrained(model_name)
txt_model = CLIPTextModelWithProjection.from_pretrained(model_name).eval().to(device)

prompts = [f"a photo of a {c}" for c in classes]
enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
with torch.no_grad():
    embeds = txt_model(**enc).text_embeds
    embeds = F.normalize(embeds, dim=-1)
    arr = embeds.cpu().numpy().astype("float32")

out_path = "${OUT_NPY}"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.save(out_path, arr)
print(arr.shape, "-> saved to", out_path)
PY
}

# -------- run (large_t1..t4) --------
# t1 = t1
IN_T1="${JSON_DIR}/unique_attributes_t1.json"
make_attr_embeds "t1" "${IN_T1}"
make_class_embeds_cumulative "t1" 1

# t2 = t1 + t2
MERGE_T2="${JSON_DIR}/unique_attributes_large_t2.json"
merge_jsons "${MERGE_T2}" \
  "${JSON_DIR}/unique_attributes_t1.json" \
  "${JSON_DIR}/unique_attributes_t2.json"
make_attr_embeds "t2" "${MERGE_T2}"
make_class_embeds_cumulative "t2" 2

# t3 = t1 + t2 + t3
MERGE_T3="${JSON_DIR}/unique_attributes_large_t3.json"
merge_jsons "${MERGE_T3}" \
  "${JSON_DIR}/unique_attributes_t1.json" \
  "${JSON_DIR}/unique_attributes_t2.json" \
  "${JSON_DIR}/unique_attributes_t3.json"
make_attr_embeds "t3" "${MERGE_T3}"
make_class_embeds_cumulative "t3" 3

# t4 = t1 + t2 + t3 + t4
MERGE_T4="${JSON_DIR}/unique_attributes_large_t4.json"
merge_jsons "${MERGE_T4}" \
  "${JSON_DIR}/unique_attributes_t1.json" \
  "${JSON_DIR}/unique_attributes_t2.json" \
  "${JSON_DIR}/unique_attributes_t3.json" \
  "${JSON_DIR}/unique_attributes_t4.json"
make_attr_embeds "t4" "${MERGE_T4}"
make_class_embeds_cumulative "t4" 4

echo "✅ Done. Outputs at: ${OUT_DIR}"
