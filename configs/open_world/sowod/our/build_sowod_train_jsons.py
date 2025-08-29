# -*- coding: utf-8 -*-
"""
COCO instances_train2017.json -> SOWOD/t{stage}_train.json generator

- stages: T1(0..18), T2(19..39), T3(40..59), T4(60..79)  
  (class_names order as provided)

- --mode:
    - cur:     include only the current stage block (e.g., T2 → 19..39)
    - prevcur: include previous + current classes (e.g., T2 → 0..39)

- --dense / --no-dense:
    - dense (default): remap category_id into 0..K-1 and output categories in that order  
      => config's metainfo.classes must match this order
    - no-dense: keep the global ids (0..79)  
      (not recommended, can cause mismatch with num_train_classes)

Usage example:
python build_sowod_trains.py \
  --coco_json data/coco/annotations/instances_train2017.json \
  --out_dir SOWOD --mode cur --dense

→ Generates SOWOD/t1_train.json, t2_train.json, t3_train.json, t4_train.json  
Also generates per-stage class list text files (for metainfo alignment).
"""
import argparse, json, os
from collections import defaultdict

# ===== 1) Provided class_names (must exactly match) =====
CLASS_NAMES = [
    # t1 (0..18) = 19 classes
    "airplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorcycle","sheep",
    "train","elephant","bear","zebra","giraffe","truck","person",
    # t2 (19..39) = 21 classes
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","dining table",
    "potted plant","backpack","umbrella","handbag","tie",
    "suitcase","microwave","oven","toaster","sink","refrigerator","bed","toilet","couch",
    # t3 (40..59) = 20 classes
    "frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake",
    # t4 (60..79) = 20 classes
    "laptop","mouse","remote","keyboard","cell phone",
    "book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush","wine glass","cup","fork","knife","spoon","bowl","tv","bottle",
]
assert len(CLASS_NAMES) == 80

# COCO naming already matches (airplane, motorcycle, couch, dining table, potted plant, tv, etc.)
VOC2COCO_NAME = {}  # left empty for possible manual mapping

# ===== 2) Stage ranges (half-open intervals) =====
STAGE_SLICES = {
    "t1": (0, 19),   # 0..18
    "t2": (19, 40),  # 19..39
    "t3": (40, 60),  # 40..59
    "t4": (60, 80),  # 60..79
}

def argp():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_json", required=True, help="Path to COCO instances_train2017.json")
    parser.add_argument("--out_dir", required=True, help="Output directory (e.g., SOWOD)")
    parser.add_argument("--file_prefix", default="JPEGImages/",
                   help="Prefix added to file_name (should match config data_prefix.img)")
    parser.add_argument("--mode", choices=["cur", "prevcur"], default="cur",
                   help="cur: only current block / prevcur: include previous+current")
    parser.add_argument("--dense", dest="dense", action="store_true", default=True,
                   help="Remap labels into 0..K-1 (recommended)")
    parser.add_argument("--no-dense", dest="dense", action="store_false")
    parser.add_argument("--drop_empty", action="store_true", default=True,
                   help="Drop images without kept labels")
    return parser.parse_args()

def stage_classes(stage: str, mode: str):
    s, e = STAGE_SLICES[stage]
    if mode == "cur":
        idxs = list(range(s, e))
    else:  # prevcur
        # include previous + current
        idxs = list(range(0, e))
    names = [CLASS_NAMES[i] for i in idxs]
    return idxs, names

def main():
    args = argp()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Map COCO name -> cat_id
    coco_name2id = {c["name"]: c["id"] for c in coco["categories"]}

    # Map COCO image_id -> image dict
    image_by_id = {im["id"]: im for im in coco["images"]}

    for stage in ["t1", "t2", "t3", "t4"]:
        kept_global_ids, kept_names = stage_classes(stage, args.mode)

        # Validate mapping from our global ids to COCO category ids
        missing = []
        kept_coco_ids = []
        for gid in kept_global_ids:
            name = CLASS_NAMES[gid]
            coco_name = VOC2COCO_NAME.get(name, name)
            if coco_name not in coco_name2id:
                missing.append((name, coco_name))
            else:
                kept_coco_ids.append(coco_name2id[coco_name])
        if missing:
            print(f"[{stage}] Not found in COCO categories:")
            for a, b in missing:
                print(f"  ours='{a}' -> coco='{b}'  (NOT FOUND)")
            raise SystemExit(1)

        # Collect kept annotations per image
        anns_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            if ann["category_id"] in kept_coco_ids:
                anns_by_img[ann["image_id"]].append(ann)

        # Prepare output structures
        images_out = []
        annotations_out = []
        categories_out = []

        # Label remapping
        if args.dense:
            # kept_global_ids -> 0..K-1
            dense_map = {gid: i for i, gid in enumerate(kept_global_ids)}
            categories_out = [{"id": i, "name": CLASS_NAMES[gid]}
                              for i, gid in enumerate(kept_global_ids)]
        else:
            # keep global ids
            categories_out = [{"id": gid, "name": CLASS_NAMES[gid]}
                              for gid in kept_global_ids]

        # Re-index images and annotations
        next_img_id = 1
        next_ann_id = 1
        for old_img_id, kept_anns in anns_by_img.items():
            if args.drop_empty and len(kept_anns) == 0:
                continue

            im = image_by_id[old_img_id]
            file_name = im["file_name"]
            # add prefix (align with mmdet data_prefix)
            if args.file_prefix and not file_name.startswith(args.file_prefix):
                file_name = args.file_prefix + file_name.split("/")[-1]

            images_out.append({
                "id": next_img_id,
                "file_name": file_name,
                "width": im.get("width", 0),
                "height": im.get("height", 0),
            })

            for a in kept_anns:
                coco_cat = a["category_id"]
                # map back: coco_cat -> our global id
                gid = kept_global_ids[kept_coco_ids.index(coco_cat)]

                if args.dense:
                    cid = int(dense_map[gid])       # 0..K-1
                else:
                    cid = int(gid)                  # global 0..79

                bbox = a["bbox"]
                area = a.get("area", bbox[2]*bbox[3])
                annotations_out.append({
                    "id": next_ann_id,
                    "image_id": next_img_id,
                    "category_id": cid,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": a.get("iscrowd", 0),
                })
                next_ann_id += 1

            next_img_id += 1

        out_json = {
            "images": images_out,
            "annotations": annotations_out,
            "categories": categories_out,
        }

        out_path = os.path.join(args.out_dir, f"{stage}_train.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False)

        # Save stage class list (helpful for metainfo)
        with open(os.path.join(args.out_dir, f"{stage}_classes.txt"), "w", encoding="utf-8") as f:
            for n in kept_names:
                f.write(n + "\n")

        print(f"[{stage}] wrote {out_path}  "
              f"images={len(images_out)}  anns={len(annotations_out)}  "
              f"classes={len(kept_names)}  mode={args.mode}  dense={args.dense}")

        # Sanity checks
        used_ids = {a["category_id"] for a in annotations_out}
        if args.dense:
            assert used_ids.issubset(set(range(len(kept_names)))), used_ids
        else:
            assert used_ids.issubset(set(kept_global_ids)), used_ids
        assert all(im["file_name"].startswith(args.file_prefix) for im in images_out)
    print("All stages done.")

if __name__ == "__main__":
    main()
