import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

STORE_IN_RAM = False ## set to false if you have <64GB RAM, as the dataset is large

def augment_bounding_box(bounding_box: np.ndarray, 
                         x_extend_prop_range=(-.1, .1), 
                         y_extend_prop_range=(-.1, .1),
                         image_size=(1920, 1080)):
    """
    Augment the bounding box by extending its dimensions randomly within the specified ranges.
    Args:
        bounding_box (np.ndarray): The original bounding box with shape (,4) 
        x_extend_prop_range (tuple): The range for random extension in the x direction.
        y_extend_prop_range (tuple): The range for random extension in the y direction.
        image_size (tuple): The size of the image (width, height) to clip the bounding box.
    Returns:
        np.ndarray: The augmented bounding box with the same shape as the input.
    """
    # Ensure the bounding box is a numpy array
    bounding_box = np.array(bounding_box)

    x1, y1, x2, y2 = bounding_box
    x_len = max(x1, x2) - min(x1, x2)
    y_len = max(y1, y2) - min(y1, y2)
    
    x_extension = np.random.uniform(x_extend_prop_range[0] * x_len, x_extend_prop_range[1] * x_len)
    y_extension = np.random.uniform(y_extend_prop_range[0] * y_len, y_extend_prop_range[1] * y_len)


    # Extend the bounding box dimensions
    augmented_bounding_box = bounding_box.copy()
    augmented_bounding_box[0] += x_extension // 2
    augmented_bounding_box[1] += y_extension //2
    augmented_bounding_box[2] -= x_extension // 2
    augmented_bounding_box[3] -= y_extension // 2

    # clip the bounding box to the image size
    augmented_bounding_box[0] = np.clip(augmented_bounding_box[0], 0, image_size[0])
    augmented_bounding_box[1] = np.clip(augmented_bounding_box[1], 0, image_size[1])
    augmented_bounding_box[2] = np.clip(augmented_bounding_box[2], 0, image_size[0])
    augmented_bounding_box[3] = np.clip(augmented_bounding_box[3], 0, image_size[1])

    
    return augmented_bounding_box
def augment_rgb(image: np.ndarray):
    '''
    Augment the RGB image by applying random brightness, contrast, saturation, and hue adjustments.
    Args:
        image (np.ndarray): The original RGB image with shape (H, W, 3).
    Returns:
        np.ndarray: The augmented RGB image with the same shape as the input.
    '''

    # Ensure the image is a numpy array
    image = np.array(image)
    # Random brightness
    brightness_factor = np.random.uniform(0.8, 1.2)
    image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    # Random contrast
    contrast_factor = np.random.uniform(0.8, 1.2)
    image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
    # Random saturation
    saturation_factor = np.random.uniform(0.8, 1.2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:, :, 1] = cv2.convertScaleAbs(image[:, :, 1], alpha=saturation_factor, beta=0)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    # Random hue
    hue_factor = np.random.uniform(-10, 10)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:, :, 0] = (image[:, :, 0] + hue_factor) % 180
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    # Random noise
    noise = np.random.normal(0, 0.1, image.shape)
    image = cv2.add(image, noise.astype(np.uint8))
   
    return image


def rotate_point_cloud(point_cloud: np.ndarray,
                       rotation_range=(-np.pi, np.pi)
                       ):
    """
    Rotate the point cloud randomly within the specified range.
    Args:
        point_cloud (np.ndarray): The original point cloud with shape (N, 3) or (N, 6).
                                  Format: (x, y, z[, r, g, b])
        rotation_range (tuple): The range for random rotation in radians.
    Returns:
        np.ndarray: The randomly rotated point cloud with the same shape as the input.
    """
    point_cloud = np.array(point_cloud)
    
    # Split into position and color (if applicable)
    if point_cloud.shape[1] == 6:
        xyz = point_cloud[:, :3]
        rgb = point_cloud[:, 3:]
    elif point_cloud.shape[1] == 3:
        xyz = point_cloud
        rgb = None
    else:
        raise ValueError("Input point cloud must have shape (N, 3) or (N, 6)")

    # Random rotation angle
    theta = np.random.uniform(rotation_range[0], rotation_range[1])
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta),  np.cos(theta), 0],
                                 [0,              0,             1]])

    # Apply rotation
    rotated_xyz = xyz @ rotation_matrix.T

    # Concatenate color if present
    if rgb is not None:
        rotated_point_cloud = np.concatenate([rotated_xyz, rgb], axis=1)
    else:
        rotated_point_cloud = rotated_xyz

    return rotated_point_cloud, rotation_matrix
def random_dropout(point_cloud: np.ndarray, dropout_range=(0.6, 0.8)):
    """
    Randomly drop points from the point cloud based on a specified dropout range.
    Args:
        point_cloud (np.ndarray): The original point cloud with shape (N, 3) or (N, 6).
        dropout_range (tuple): The range for random dropout as a fraction of the total points.
    Returns:
        np.ndarray: The point cloud after random dropout.
    """
    point_cloud = np.array(point_cloud)
    
    # Calculate the number of points to drop
    num_points = point_cloud.shape[0]
    num_points_to_drop = int(num_points * np.random.uniform(*dropout_range))

    # Randomly select indices to drop
    indices_to_drop = np.random.choice(num_points, num_points_to_drop, replace=False)

    # Create a mask to keep the remaining points
    mask = np.ones(num_points, dtype=bool)
    mask[indices_to_drop] = False

    # Apply the mask to the point cloud
    dropped_point_cloud = point_cloud[mask]

    return dropped_point_cloud

def voxel_normalize(points, voxel_size=0.005, percentile=95):
    """Normalize using voxel grid to handle irregular density.
    NaNs are preserved and ignored in normalization."""
    
    # Create voxel grid with valid (non-NaN) points only
    voxel_grid = {}
    for i, point in enumerate(points):
        if np.any(np.isnan(point)):
            continue  # skip NaNs in voxel computation
        voxel_idx = tuple(np.floor(point / voxel_size).astype(int))
        if voxel_idx not in voxel_grid:
            voxel_grid[voxel_idx] = []
        voxel_grid[voxel_idx].append(i)

    # Compute voxel centers from valid points
    voxel_centers = []
    for point_indices in voxel_grid.values():
        voxel_points = points[point_indices]
        voxel_centers.append(np.mean(voxel_points, axis=0))

    voxel_centers = np.array(voxel_centers)
    
    # Center and scale using voxel centers 
    center = np.median(voxel_centers, axis=0)
    distances = np.linalg.norm(voxel_centers - center, axis=1)
    scale = np.percentile(distances, percentile)

    # Normalize all points 
    centered_points = points - center
    scaled_points = centered_points / scale

    return scaled_points, center, scale


TARGET_PTS = 4096                      # fixed length for every cloud
def pad_collate_fn(batch):
    """
    Collate variable-length point clouds to (B, TARGET_PTS, D).

    Returns
    -------
    batch_clouds : FloatTensor  (B, 4096, D)
    batch_centers: FloatTensor  (B, 3)
    batch_mask   : BoolTensor   (B, 4096)   1 = valid point, 0 = padding
    batch_aux    : dict         other stacked / listed fields
    """
    clouds, centers, aux_list = zip(*batch)

    # -------- convert to tensors -------------------------------------------------
    tensor_clouds = []
    for cloud in clouds:
        if isinstance(cloud, np.ndarray):
            cloud = torch.from_numpy(cloud).float()
        elif not torch.is_tensor(cloud):
            raise TypeError(f"Unsupported cloud type {type(cloud)}")
        tensor_clouds.append(cloud)                         # (Ni, D)

    # -------- pad / truncate -----------------------------------------------------
    fixed, masks = [], []
    for pc in tensor_clouds:
        n, d = pc.shape

        if n > TARGET_PTS:                                  # subsample
            idx  = torch.randperm(n, device=pc.device)[:TARGET_PTS]
            pc_f = pc[idx]
            mask = torch.ones(TARGET_PTS, dtype=torch.bool, device=pc.device)

        elif n < TARGET_PTS:                                # pad
            pad_len = TARGET_PTS - n
            pad     = torch.zeros((pad_len, d), dtype=pc.dtype, device=pc.device)
            pc_f    = torch.cat([pc, pad], dim=0)
            mask    = torch.cat([torch.ones(n, dtype=torch.bool, device=pc.device),
                                 torch.zeros(pad_len, dtype=torch.bool, device=pc.device)])
        else:                                               # already 4096
            pc_f = pc
            mask = torch.ones(TARGET_PTS, dtype=torch.bool, device=pc.device)

        fixed.append(pc_f)
        masks.append(mask)

    batch_clouds = torch.stack(fixed)           # (B, 4096, D)
    batch_mask   = torch.stack(masks)           # (B, 4096)

    # -------- centers ------------------------------------------------------------
    batch_centers = torch.stack([
        torch.as_tensor(c, dtype=torch.float32)
        if isinstance(c, np.ndarray) else c.float()
        for c in centers
    ])                                         # (B, 3)

    # -------- auxiliary fields ---------------------------------------------------
    batch_aux = {}
    for k in aux_list[0]:
        items = [aux[k] for aux in aux_list]

        if torch.is_tensor(items[0]):
            batch_aux[k] = torch.stack(items)
        elif isinstance(items[0], (np.number, float, int)):
            batch_aux[k] = torch.tensor(items, dtype=torch.float32)
        else:
            batch_aux[k] = items

    return batch_clouds, batch_centers, batch_mask, batch_aux

class ApplePointCloudDataset(Dataset):
    def __init__(self, data_root: str, manifest_path: str, config: dict={}, augment=True):
        self.root = data_root
        self.augment = augment
        self.records = []
        self.voxel_size = config.get("voxel_size", 0.003)  # default voxel size for normalization
        self.percentile = config.get("percentile", 95)    # default percentile for normalization
        self.subset_size = config.get("subset_size", 1.0)  


        with open(manifest_path) as f:
            scenes = [json.loads(line) for line in f]

        # randomly select a subset of scenes if subset_size < 1.0
        if self.subset_size < 1.0:
            np.random.seed(config['SEED'])
            np.random.shuffle(scenes)
            scenes = scenes[:int(len(scenes) * self.subset_size)]
        print(f"Loading {len(scenes)} scenes from {manifest_path} …")

        for scene_i, scene in enumerate(scenes):
            stem = scene["stem"]
            for apple_i, (bbox, center, occ_rate) in enumerate(zip(scene["boxes"], scene["centers"], scene["occ_rates"])):
                center[1] = -center[1]  # flip y-axis to match the point cloud
                self.records.append({
                    "stem": stem,
                    "bbox": bbox,
                    "occ_rate": occ_rate,
                    "center": center
                })

        if STORE_IN_RAM:
            print(f"Pre-loading {len(self.records)} apples into RAM …")
            stems_to_records = {}
            for r in self.records:
                if r["stem"] not in stems_to_records:
                    stems_to_records[r["stem"]] = []
                stems_to_records[r["stem"]].append(r)
            self.records = []
            for stem, recs in stems_to_records.items():
                xyz, rgb = self._load_scene_xyzrgb(stem)
                for r in recs:
                    r["sample"] = self._build_sample(r, xyz=xyz, rgb=rgb)
                    self.records.append(r)
                print("Loaded all samples for stem", stem)


    def __len__(self):
        return len(self.records)
    
    def _load_scene_xyzrgb(self, stem: str):
        """Load (or zip-cache) full scene xyzrgb array."""
        zipped = os.path.join(self.root, "zipped", f"{stem}.npz")
        try:
            with np.load(zipped) as data:
                return data["xyz"], data["rgb"]
        except Exception:
            xyz = np.load(os.path.join(self.root, f"{stem}_pc.npy"))
            rgb = cv2.cvtColor(
                cv2.imread(os.path.join(self.root, f"{stem}_rgb0000.png")),
                cv2.COLOR_BGR2RGB)
            os.makedirs(os.path.dirname(zipped), exist_ok=True)
            np.savez_compressed(zipped, xyz=xyz, rgb=rgb)
            return xyz, rgb
    def _build_sample(self, rec, xyz=None, rgb=None):
        """Creates (pc, center, meta) for one apple."""
        stem, bbox, center, occ = \
            rec["stem"], rec["bbox"], rec["center"], rec["occ_rate"]
        if xyz is None or rgb is None:
            xyz, rgb  = self._load_scene_xyzrgb(stem)
        xyzrgb    = np.concatenate((xyz, rgb), axis=2)

        if self.augment:
            bbox = augment_bounding_box(bbox)

        x1, y1, x2, y2 = map(int, bbox)
        crop = xyzrgb[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
        crop[:, :, 3:] = augment_rgb(crop[:, :, 3:]) if self.augment else crop[:, :, 3:]

        pc = crop.reshape(-1, 6)
        pc = pc[~((np.abs(pc[:,2]) < .45) | (np.abs(pc[:,2]) > 2.75))]
        pc = pc[~np.isnan(pc).any(1)]
        pc = pc[~np.isinf(pc).any(1)]
        if self.augment: pc = random_dropout(pc, (0.3, 0.7))

        norm_pc, norm_ctr, scale = voxel_normalize(
            pc[:, :3], voxel_size=self.voxel_size, percentile=self.percentile)
        pc[:, :3] = norm_pc
        # normalize rgb channels to [0, 1]
        pc[:, 3:6] = pc[:, 3:6] / 255.0
        center_t  = ((torch.tensor(center) - norm_ctr)/scale).float()

        meta = dict(stem=stem, bbox=bbox, occ_rate=occ,
                    norm_center=norm_ctr, norm_scale=scale)
        return pc.astype(np.float32), center_t, meta
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        if STORE_IN_RAM:
            return rec["sample"]          # already built & cached
        return self._build_sample(rec)


if __name__ == "__main__":

    import plotly.graph_objects as go
    import os
    import dotenv 

    dotenv.load_dotenv(dotenv.find_dotenv())
    PROJECT_ROOT = os.getenv("PROJECT_ROOT")

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_root = os.path.join(PROJECT_ROOT, "blender/dataset/")
    train_manifest = os.path.join(PROJECT_ROOT, "blender/dataset/curated/apple-orchard-v1/train.jsonl")
    test_manifest = os.path.join(PROJECT_ROOT, "blender/dataset/curated/apple-orchard-v1/test.jsonl")

    # dataset / loader (batch_size 1 is easiest for variable‑length clouds)
    config = {
        'voxel_size': 0.0045,  # default voxel size for normalization
        'percentile': 95,     # default percentile for normalization
        # 'subset_size': 0.01,   # use all data
        'SEED': SEED,  # for reproducibility
    }
    train_ds = ApplePointCloudDataset(
            data_root     = data_root,
            manifest_path = train_manifest,
            augment       = True,
            config        = config
            )
    # split into train/val
    train_size = int(len(train_ds) * 0.8)
    val_size = len(train_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

    test_ds = ApplePointCloudDataset(
            data_root     = data_root,
            manifest_path = test_manifest,
            augment       = False,   
            config        = config
            )
    
    print("train size", len(train_ds))
    print("val size", len(val_ds))
    print("test size", len(test_ds))

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=12, collate_fn=pad_collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=1, shuffle=True, num_workers=12)
    test_dl  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=12)
    # ------------------------------------------------------------------
    for scene_i, (clouds_batch, centers_batch,mask_batch,  aux) in enumerate(train_dl):
        pc  = clouds_batch[0]     # list[(N_i,6), …]
        assert pc.shape[1] == 6, f"Expected 6 channels, got {pc.shape[1]}"
        center  = centers_batch[0].numpy()  # (M,3)
        print(pc[:,3:6])
        break

        # fig = go.Figure()
        # fig.add_trace(go.Scatter3d(
        #     x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
        #     mode="markers",
        #     marker=dict(size=2,
        #                 color=pc[:, 3:6] / 255.0,   # RGB -> [0,1]
        #                 opacity=0.6)))

        # fig.add_trace(go.Scatter3d(
        #     x=[center[0]], y=[center[1]], z=[center[2]],
        #     mode="markers",
        #     marker=dict(size=8, color="red")))

        # fig.update_layout(scene_aspectmode="data",
        #                 width=700, height=700,
        #                 margin=dict(l=0, r=0, b=0, t=0))
        # fig.show()

        # if scene_i >= 2:        # stop after 2 scenes
        #     break
    for _ in val_dl:
        pass 
    for _ in test_dl:
        pass
    print("Dataloading worked")