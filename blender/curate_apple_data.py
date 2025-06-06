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
  "centers": [[x,y,z], ...]             # float, camera space (Y restored!)
}
"""

import os, json, argparse, cv2, numpy as np
import random
import utils

APPLE_PIX_RATIO_THRESH = 0.15           ### tune
DEPTH_MIN, DEPTH_MAX = 0.5, 2.5
MIN_BOX_AREA = 900                # min area of bbox to keep
MIN_SIDE_RATIO = 0.7          # min ratio of min/max side length to keep
TRAIN_FRAC = 0.8        
RAND_SEED = 42             

total_samples = 0
total_kept_samples = 0

def curate_scene(stem: str, raw_dir: str) -> dict | None:
    """Return a metadata dict if at least one apple survives the filters."""
    pc_path  = os.path.join(raw_dir, f"{stem}_pc.npy")
    rgb_path = os.path.join(raw_dir, f"{stem}_rgb0000.png")
    id_path  = os.path.join(raw_dir, f"{stem}_id0000.exr")
    map_path = os.path.join(raw_dir, f"{stem}_id_map.json")
    data_js  = os.path.join(raw_dir, f"{stem}_apple_data.json")
    instance_data_path = os.path.join(raw_dir, f"{stem}_instance_data.npz")

    num_samples=0

    # -------------------- load json for every apple -----------------------
    with open(data_js) as jf:
        apple_meta = {k: json.loads(v) for k, v in json.load(jf).items()}

    # -------------------- visible-pixel counts per instance --------------
    instance_data = np.load(instance_data_path, allow_pickle=True)
    vis = instance_data['visible_apples']
    id_mask = instance_data['apple_id_mask']
    clusters = instance_data['clusters']
    if isinstance(clusters, np.ndarray):
        clusters = clusters.tolist()
    # visible_apples, id_mask2, _, _ = utils.get_visible_objects(
    #     exr_path=id_path,
    #     id_mapping_path=map_path,
    #     conditional=lambda _id, name: "apple" in name and "stem" not in name
    # )
    # assert np.allclose(id_mask, id_mask2), "ID mask mismatch!"

    px_counts = {name: (id_mask == int(_id)).sum() for _id, name in vis}

    boxes, centers, occ_rates = [], [], []
    valid_indices = []
    apple_keys = [cluster[0] for cluster in clusters]
    for i, apple_key in enumerate(apple_keys):
        item = apple_meta[apple_key]
        num_samples += 1
        x1_raw, y1_raw, x2_raw, y2_raw = map(int, item["apple_bbox"])
        # clip bbox to image size
        x1, y1, x2, y2 = max(0, x1_raw), max(0, y1_raw), max(0, x2_raw), max(0, y2_raw)
        x2, y2 = min(x2, id_mask.shape[1]), min(y2, id_mask.shape[0])
        box_area = abs(x2 - x1) * abs(y2 - y1)
        if box_area < MIN_BOX_AREA:                    # degenerate bbox
            continue

        # filter based on box height/width ratio. boxes that are too narrow should be avoided (too close to edge)
        height, width = abs(y2 - y1), abs(x2 - x1)
        side_ratio = min(height, width) / max(height, width)
        if side_ratio < MIN_SIDE_RATIO:
            continue
        px_ratio = px_counts[item["apple_name"]] / box_area
        z_cam    = abs(item["apple_center"][-1])

        if px_ratio < APPLE_PIX_RATIO_THRESH or not (DEPTH_MIN <= z_cam <= DEPTH_MAX):
            continue

        # ----------- FIX Y-sign that was stored as -loc_cam.y -------------
        cx, cy, cz = item["apple_center"]
        centers.append([cx, -cy, cz])
        boxes.append([x1, y1, x2, y2])  
        occ_rates.append(px_ratio)
        valid_indices.append(i)
    if not valid_indices:
        print(f"Warning: no apples kept in {stem}!")
        return None


    valid_names = [apple_keys[i] for i in valid_indices]
    valid_clusters = [clusters[i] for i in valid_indices]
    valid_clusters = [[apple for apple in cluster if apple in valid_names] for cluster in valid_clusters]
    # remove empty clusters
    valid_clusters = [cluster for cluster in valid_clusters if cluster]
    

    assert len(boxes) == len(centers) == len(occ_rates) == len(valid_clusters), \
        "Mismatch in number of kept apples!"
    final_kept_samples = len(boxes)
    print(f"Kept {final_kept_samples} of {num_samples} samples in {stem}")
    global total_samples, total_kept_samples
    total_samples += num_samples
    total_kept_samples += final_kept_samples

    return {"stem": stem, "boxes": boxes, "centers": centers, "occ_rates": occ_rates}

def build_manifest(raw_dir: str, out_dir: str) -> str:
    """Run curation and write manifest.jsonl → returns its path."""
    os.makedirs(out_dir, exist_ok=True)
    man_path = os.path.join(out_dir, "manifest.jsonl")

    with open(man_path, "w") as mf:
        for f in os.listdir(raw_dir):
            if not f.endswith("apple_data.json"):    # only scene files
                continue
            stem = f.split("_apple_data.json")[0]
            rec  = curate_scene(stem, raw_dir)
            if rec:
                mf.write(json.dumps(rec) + "\n")

    print(f"✓ Manifest written → {man_path}")
    print(f"Total apples kept {total_kept_samples}/{total_samples}"
          f"  ({total_kept_samples/total_samples:.1%})")
    return man_path

def split_manifest(manifest_path: str,
                   train_frac: float = 0.8,
                   seed: int = 42) -> tuple[str, str]:
    """
    Shuffle apples and split scenes into train/test by apple count.

    Ensures total number of apples respects `train_frac`.

    Returns:
        (train_path, test_path)
    """
    import json
    import random

    # ---------------- Load and flatten apples ----------------
    with open(manifest_path) as f:
        scenes = [json.loads(line) for line in f]

    all_apples = []  # [(scene_idx, apple_idx), ...]
    for i, scene in enumerate(scenes):
        for j in range(len(scene["boxes"])):
            all_apples.append((i, j))

    print(f"Total apples: {len(all_apples)} from {len(scenes)} scenes")

    # ---------------- Shuffle and split ----------------
    random.Random(seed).shuffle(all_apples)
    k = int(len(all_apples) * train_frac)
    train_apples = set(all_apples[:k])
    test_apples  = set(all_apples[k:])

    # ---------------- Rebuild per-scene data ----------------
    train_scenes, test_scenes = [], []

    for scene_idx, scene in enumerate(scenes):
        boxes   = scene["boxes"]
        centers = scene["centers"]
        occ_rates = scene["occ_rates"]

        train_bxs, train_ctrs, train_occ_rates = [], [], []
        test_bxs,  test_ctrs, test_occ_rates  = [], [], []

        for j in range(len(boxes)):
            if (scene_idx, j) in train_apples:
                train_bxs.append(boxes[j])
                train_ctrs.append(centers[j])
                train_occ_rates.append(occ_rates[j])
            elif (scene_idx, j) in test_apples:
                test_bxs.append(boxes[j])
                test_ctrs.append(centers[j])
                test_occ_rates.append(occ_rates[j])

        if train_bxs:
            train_scenes.append({
                "stem": scene["stem"],
                "boxes": train_bxs,
                "centers": train_ctrs,
                "occ_rates": train_occ_rates
            })
        if test_bxs:
            test_scenes.append({
                "stem": scene["stem"],
                "boxes": test_bxs,
                "centers": test_ctrs,
                "occ_rates": test_occ_rates
            })

    # ---------------- Write split files ----------------
    base = manifest_path.replace("manifest.jsonl", "")
    train_path = base + "train.jsonl"
    test_path  = base + "test.jsonl"

    with open(train_path, "w") as f:
        for scene in train_scenes:
            f.write(json.dumps(scene) + "\n")
    with open(test_path, "w") as f:
        for scene in test_scenes:
            f.write(json.dumps(scene) + "\n")

    print(f"✓ Split: {len(train_apples)} apples in train,"
          f" {len(test_apples)} in test")
    print(f"  ↳ {train_path}\n  ↳ {test_path}")

    return train_path, test_path


def main(raw_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    manifest = build_manifest(raw_dir, out_dir)
    split_manifest(manifest, TRAIN_FRAC, RAND_SEED)

    print(f"Total samples: {total_samples}")
    print(f"Total kept samples: {total_kept_samples}")
    print(f"Total kept ratio: {total_kept_samples / total_samples:.2%}")

if __name__ == "__main__":
    import os 
    import dotenv
    dotenv.load_dotenv(dotenv.find_dotenv())

    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=False,
                   default=os.path.join(PROJECT_ROOT, 'blender/dataset/raw/apple_orchard-test'),
                   help="folder that contains *_pc.npy, *_rgb0000.png ...")
    p.add_argument("--out_dir", required=False,
                   default=os.path.join(PROJECT_ROOT, 'blender/dataset/curated/test'),
                   help="destination for manifest.jsonl")
    main(**vars(p.parse_args()))
