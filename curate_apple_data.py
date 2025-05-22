#!/usr/bin/env python
# prepare_apples.py
"""
Curate the raw Blender renders into a deterministic manifest.

* Filters every apple instance by:
    -- visible-pixel ratio  <threshold>
    -- camera-depth outside (min,max)
* Writes one JSON line per **kept scene** to `out_dir/manifest.jsonl`

Each line has:
{
  "stem": "UUID-like-name",
  "boxes":   [[x1,y1,x2,y2], ...],      # int
  "centres": [[x,y,z], ...]             # float, camera space (Y restored!)
}
"""

import os, json, argparse, cv2, numpy as np
from blender import utils

PX_RATIO_THR = 0.15           #   ---- tune here -------------
DEPTH_MIN, DEPTH_MAX = 0.5, 2.5

total_samples = 0
total_kept_samples = 0

def curate_scene(stem: str, raw_dir: str) -> dict | None:
    """Return a metadata dict if at least one apple survives the filters."""
    pc_path  = os.path.join(raw_dir, f"{stem}_pc.npy")
    rgb_path = os.path.join(raw_dir, f"{stem}_rgb0000.png")
    id_path  = os.path.join(raw_dir, f"{stem}_id0000.exr")
    map_path = os.path.join(raw_dir, f"{stem}_id_map.json")
    data_js  = os.path.join(raw_dir, f"{stem}_apple_data.json")

    num_samples=0

    # -------------------- load json for every apple -----------------------
    with open(data_js) as jf:
        apple_meta = {k: json.loads(v) for k, v in json.load(jf).items()}

    # -------------------- visible-pixel counts per instance --------------
    vis, id_mask, _, _ = utils.get_visible_objects(
        exr_path=id_path,
        id_mapping_path=map_path,
        conditional=lambda _id, name: "apple" in name and "stem" not in name
    )
    px_counts = {name: (id_mask == _id).sum() for _id, name in vis}

    boxes, centres = [], []
    for item in apple_meta.values():
        num_samples += 1
        x1, y1, x2, y2 = map(int, item["apple_bbox"])
        box_area = abs(x2 - x1) * abs(y2 - y1)
        if box_area == 0:                    # degenerate bbox
            continue
        px_ratio = px_counts.get(item["apple_name"], 0) / box_area
        z_cam    = abs(item["apple_center"][-1])

        if px_ratio < PX_RATIO_THR or not (DEPTH_MIN <= z_cam <= DEPTH_MAX):
            continue

        # ----------- FIX Y-sign that was stored as -loc_cam.y -------------
        cx, cy, cz = item["apple_center"]
        centres.append([cx, -cy, cz])
        boxes.append([x1, y1, x2, y2])

    if not boxes:
        return None
    
    final_kept_samples = len(boxes)
    print(f"Kept {final_kept_samples} of {num_samples} samples in {stem}")
    global total_samples, total_kept_samples
    total_samples += num_samples
    total_kept_samples += final_kept_samples

    return {"stem": stem, "boxes": boxes, "centres": centres}

def main(raw_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")

    with open(manifest_path, "w") as mf:
        for f in os.listdir(raw_dir):
            if not f.endswith("apple_data.json"):
                continue
            stem = f.split("_apple_data.json")[0]
            record = curate_scene(stem, raw_dir)
            if record:
                mf.write(json.dumps(record) + "\n")

    print(f"Manifest written -> {manifest_path}")

    print(f"Total samples: {total_samples}")
    print(f"Total kept samples: {total_kept_samples}")
    print(f"Total kept ratio: {total_kept_samples / total_samples:.2%}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True,
                   help="folder that contains *_pc.npy, *_rgb0000.png ...")
    p.add_argument("--out_dir", required=True,
                   help="destination for manifest.jsonl")
    main(**vars(p.parse_args()))
