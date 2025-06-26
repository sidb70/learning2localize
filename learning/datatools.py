import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import  matplotlib.pyplot as plt
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
def rotate_mask(mask, angle, center=None, scale=1.0):
    """
    Rotates a binary mask around its center (or given center), maintaining original size.
    """
    h, w = mask.shape
    if center is None:
        center = (w // 2, h // 2)
        
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return (rotated > 0).astype(np.uint8)


def scale_mask(mask, scale_x, scale_y):
    """
    Scales a binary mask and pads/crops it back to original size.
    """
    h, w = mask.shape
    new_w, new_h = int(w * scale_x), int(h * scale_y)
    scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Pad or crop to return to original size
    pad_x = max(w - new_w, 0)
    pad_y = max(h - new_h, 0)

    scaled = np.pad(scaled, ((pad_y // 2, pad_y - pad_y // 2),
                             (pad_x // 2, pad_x - pad_x // 2)), mode='constant', constant_values=0)

    # Crop if larger than original
    scaled = scaled[:h, :w]
    return (scaled > 0).astype(np.uint8)


def translate_mask(mask, shift_x, shift_y):
    """
    Translates a binary mask with same output size.
    """
    h, w = mask.shape
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return (translated > 0).astype(np.uint8)


def augment_mask(mask, angle_range=(-7, 7), scale_range=(0.9, 1.1), translate_range=(-4, 4)):
    """
    Applies random augmentations to a binary mask with shape preserved.
    """
    mask = mask.astype(np.uint8)  # Ensure mask is in uint8 format
    angle = np.random.uniform(*angle_range)
    scale_x = np.random.uniform(*scale_range)
    scale_y = np.random.uniform(*scale_range)
    shift_x = np.random.randint(*translate_range)
    shift_y = np.random.randint(*translate_range)
    
    mask = rotate_mask(mask, angle)
    mask = scale_mask(mask, scale_x, scale_y)
    mask = translate_mask(mask, shift_x, shift_y)
    
    return mask

def inverse_distance_to_center_map(H, W):
    """
    Returns a (H, W) array where each pixel contains an inverse-normalized
    Euclidean distance to the center of the bounding box.
    Values range from 1 (center) to 0 (farthest corner).

    Parameters
    ----------
    H : int
        Height of the bounding box.
    W : int
        Width of the bounding box.

    Returns
    -------
    dist_map : np.ndarray of shape (H, W)
        Inverse-normalized distance from each pixel to the center.
    """
    y = np.arange(H).reshape(-1, 1)    # shape (H, 1)
    x = np.arange(W).reshape(1, -1)    # shape (1, W)
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    dist_map = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt(max(cy, H - 1 - cy)**2 + max(cx, W - 1 - cx)**2)

    inverse_dist_map = 1.0 - (dist_map / max_dist)
    return inverse_dist_map
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
    
    num_non_nan_points = np.sum(~np.isnan(points), axis=1)
    assert  num_non_nan_points.sum() > 0,f"Number of non-NaN points: {num_non_nan_points.sum()}"

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
        self.voxel_size = config.get("voxel_size", 0.0045)  # default voxel size for normalization
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
                # print("Loaded", stem, "from zip cache")
                return data["xyz"], data["rgb"]
        except Exception:
            # print("Loading", stem, "from disk")
            xyz_path = os.path.join(self.root, f"{stem}_pc.npy")
            rgb_path = os.path.join(self.root, f"{stem}_rgb0000.png")
            if not os.path.exists(rgb_path):
                rgb_path = rgb_path.replace("0000", "0107")
            xyz = np.load(os.path.join(self.root, f"{stem}_pc.npy"))
            rgb = cv2.cvtColor(
                cv2.imread(rgb_path,),
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

class AppleInstancePointCloudDataset(ApplePointCloudDataset):
    """Dataset for apple instances"""
    def __init__(self, data_root: str, manifest_path: str, config: dict={}, augment=True):
        self.root = data_root
        self.augment = augment
        self.records = []
        self.voxel_size = config.get("voxel_size", 0.0045)  # default voxel size for normalization
        self.percentile = config.get("percentile", 95)    # default percentile for normalization
        self.subset_size = config.get("subset_size", 1.0)  
        self.mask_augment = config.get("mask_augment", True)  # whether to augment masks


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
            if stem == '3000d9ec-5b05-4af7-8961-f669ec75de2c':
                x=13
            for apple_i, (bbox, center, occ_rate) in enumerate(zip(scene["boxes"], scene["centers"], scene["occ_rates"])):
                center[1] = -center[1]  # flip y-axis to match the point cloud
                self.records.append({
                    "stem": stem,
                    "bbox": scene['boxes'][apple_i],
                    "occ_rate": scene['occ_rates'][apple_i],
                    "center": scene['centers'][apple_i],
                    # 'cluster': scene['clusters'][apple_i],
                    'apple_meta': scene['apple_meta'][apple_i],
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

    def _load_scene_data(self, stem: str):
        """Load (or zip-cache) full scene xyzrgb array."""
        zipped = os.path.realpath(os.path.join(self.root, "zipped", f"{stem}.npz")).replace('combined/', '')
        try:
            with np.load(zipped) as data:
                # print("Loaded", stem, "from zip cache")
                return data["xyz"], data["rgb"], data["id_mask"]
        except Exception:
            # print("Loading", stem, "from disk")
            xyz_path = os.path.realpath(os.path.join(self.root, f"{stem}_pc.npy")).replace('combined/', '')
            rgb_path = xyz_path.replace('_pc.npy', '_rgb0000.png')
            if not os.path.exists(rgb_path):
                rgb_path = rgb_path.replace('0000', '0107')
            if not os.path.exists(xyz_path) or not os.path.exists(rgb_path):
                print(f"Expected paths: {xyz_path}, {rgb_path}")
                raise FileNotFoundError(f"Data for {stem} not found in {self.root}. Please check the paths.")
            xyz = np.load(xyz_path)
            rgb = cv2.cvtColor(
                cv2.imread(rgb_path),
                cv2.COLOR_BGR2RGB)
            instance_data = np.load(os.path.realpath(os.path.join(self.root, f"{stem}_instance_data.npz")).replace('combined/',''), allow_pickle=True)
            id_mask = instance_data['apple_id_mask']
            os.makedirs(os.path.dirname(zipped), exist_ok=True)
            np.savez_compressed(zipped, xyz=xyz, rgb=rgb, id_mask=id_mask)
            return xyz, rgb, id_mask

    def _build_sample(self, rec, xyz=None, rgb=None, id_mask=None):
        """Creates (pc, center, meta) for one apple."""
        stem, bbox, center, occ, apple_meta = \
            rec["stem"], rec["bbox"], rec["center"], rec["occ_rate"], rec["apple_meta"]
        if xyz is None or rgb is None or id_mask is None:
            xyz, rgb, id_mask = self._load_scene_data(stem)
            
        xyzrgb    = np.concatenate((xyz, rgb), axis=2) #(720, 1280, 6)

        apple_id = int(apple_meta["apple_id"])
        assert apple_id in id_mask, f"Apple ID {apple_id} not found in id_mask for stem {stem}"
        x1, y1, x2, y2 = map(int, bbox)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # filter points by apple id
        mask = (id_mask == apple_id) 
        h, w = mask.shape
        x1 = np.clip(x1, 0, w)
        x2 = np.clip(x2, 0, w)
        y1 = np.clip(y1, 0, h)
        y2 = np.clip(y2, 0, h)
        if self.augment and self.mask_augment:
            # add noise to mask
            mask_crop = mask[y1:y2, x1:x2]  # crop mask to bbox
            mask_crop = augment_mask(mask_crop)
            # apply mask to the whole image
            mask[y1:y2, x1:x2] = mask_crop  # paste augmented mask back to the full mask
        # mask out xyzrgb. output should be (720, 1280, 6)
        xyzrgb = xyzrgb * mask[..., np.newaxis]  # apply mask to xyzrgb

        if xyzrgb.shape[0] == 0:
            raise ValueError(f"No points found for apple id {apple_id} in stem {stem}")
        

        if self.augment:
            bbox = augment_bounding_box(bbox)
        x1, y1, x2, y2 = map(int, bbox)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1 = np.clip(x1, 0, w)
        x2 = np.clip(x2, 0, w)
        y1 = np.clip(y1, 0, h)
        y2 = np.clip(y2, 0, h)


        crop = xyzrgb[y1:y2, x1:x2, :]
        assert crop.shape[0] > 0 and crop.shape[1] > 0, f"Crop is empty for bbox {bbox} in stem {stem}"
        crop[:, :, 3:] = augment_rgb(crop[:, :, 3:]) if self.augment else crop[:, :, 3:]

        # reshape to (N, C) where N is number of points and C is number of channels
        pc = crop.reshape(-1, crop.shape[2])
        pc = pc[~((np.abs(pc[:,2]) < .45) | (np.abs(pc[:,2]) > 2.75))]
        pc = pc[~np.isnan(pc).any(1)]
        pc = pc[~np.isinf(pc).any(1)]


        try:
            norm_pc, norm_ctr, scale = voxel_normalize(
                pc[:, :3], voxel_size=self.voxel_size, percentile=self.percentile)
        except AssertionError as e:
            print(f"Error normalizing point cloud for stem {stem}, bbox {bbox}: {e}")
            raise e
        pc[:, :3] = norm_pc
        # normalize rgb channels to [0, 1]
        pc[:, 3:6] = pc[:, 3:6] / 255.0
        center_t  = ((torch.tensor(center) - norm_ctr)/scale).float()

        if self.augment: pc = random_dropout(pc, (0.3, 0.7))


        meta = dict(stem=stem, bbox=bbox, occ_rate=occ,
                    norm_center=norm_ctr, norm_scale=scale)
        return pc.astype(np.float32), center_t, meta
class RealAppleInstancePointCloudDataset(ApplePointCloudDataset):
    """Dataset for real apple instances"""
    def __init__(self, data_root: str,  config: dict={}, manifest_path=None, augment=True):
        self.voxel_size = config.get("voxel_size", 0.0045)  # default voxel size for normalization
        self.percentile = config.get("percentile", 95)    # default percentile for normalization
        self.subset_size = config.get("subset_size", 1.0)  
        self.mask_augment = config.get("mask_augment", False)  # whether to augment masks
        self.records = [] 

        # load all instances from the data root
        instances = {}
        for file in os.listdir(data_root):
            spl = file.split("_")
            instance_id = spl[0] + '_' + spl[1]
            if instance_id not in instances:
                instances[instance_id] = {}
                gt_depth = float(spl[2])
                instances[instance_id]['gt_depth'] = gt_depth
                instances[instance_id]['instance_id'] = instance_id

            if 'pc_6d' in file:
                instances[instance_id]['xyzrgb'] = os.path.join(data_root, file)
            elif 'rgb' in file:
                instances[instance_id]['rgb'] = os.path.join(data_root, file)
            elif 'current_apple_mask' in file:
                instances[instance_id]['mask'] = os.path.join(data_root, file)
            elif 'bbox_u' in file:
                instances[instance_id]['bbox_u'] = os.path.join(data_root, file)
            elif 'bbox_o' in file:
                instances[instance_id]['bbox_o'] = os.path.join(data_root, file)
        print(f"Found {len(instances)} instances in {data_root} …")

        instances = list(instances.values())
        # randomly select a subset of instances if subset_size < 1.0
        if self.subset_size < 1.0:
            np.random.seed(config['SEED'])
            np.random.shuffle(instances)
            instances = instances[:int(len(instances) * self.subset_size)]  
        print(f"Loading {len(instances)} instances from {data_root} …")
        for instance in instances:
            pc_6d = np.load(instance['xyzrgb'])
            pc_6d[:,0:3] = pc_6d[:,0:3]/1000.0  # convert from mm to m
            rgb = cv2.cvtColor(
                cv2.imread(instance['rgb']),
                cv2.COLOR_BGR2RGB)
            mask = np.load(instance['mask'])
            bbox_u = np.load(instance['bbox_u'])
            self.records.append({
                "instance_id": instance['instance_id'],
                "pc_6d": pc_6d,
                "rgb": rgb,
                "mask": mask,
                "bbox_u": bbox_u,
                "gt_depth": instance['gt_depth']
            })

    def __len__(self):
        return len(self.records)
    
    def _build_sample(self, rec):
        """Creates (pc, center, meta) for one apple."""
        instance_id, pc_6d, rgb, mask, bbox_u, gt_depth = \
            rec["instance_id"], rec["pc_6d"], rec["rgb"], rec["mask"], rec["bbox_u"], rec["gt_depth"]
        
        # normalize point cloud
        try:
            norm_pc, norm_ctr, scale = voxel_normalize(
                pc_6d[:, :3], voxel_size=self.voxel_size, percentile=self.percentile)
        except AssertionError as e:
            print(f"Error normalizing point cloud for instance {instance_id}: {e}")
            raise e
        
        return (norm_pc.astype(np.float32),gt_depth, {
            "instance_id": instance_id,
            "bbox_u": bbox_u,
            "mask": mask.astype(np.float32),
            "rgb": rgb.astype(np.float32),
            "norm_center":  norm_ctr.astype(np.float32),
            "norm_scale":  scale.astype(np.float32),
        }
        )



if __name__ == "__main__":

    import plotly.graph_objects as go
    import os
    import dotenv 

    dotenv.load_dotenv(dotenv.find_dotenv())
    PROJECT_ROOT = os.getenv("PROJECT_ROOT")

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # real_ds = RealAppleInstancePointCloudDataset('/home/keyi/sid/learning2localize/realdata/crops')
    # print("Real dataset size:", len(real_ds))
    # real_dl = DataLoader(real_ds, batch_size=1, shuffle=True, num_workers=1)
    # for i, (pc, gt_depth, meta) in enumerate(real_dl):
    #     pass
    # print("Real dataset loaded successfully")


    # data_root = os.path.join(PROJECT_ROOT, "blender/dataset/raw/apple_orchard-5-20-fp-only")
    # train_manifest = os.path.join(PROJECT_ROOT, "blender/dataset/curated/apple-orchard-v2-fp-only/train.jsonl")
    # test_manifest = os.path.join(PROJECT_ROOT, "blender/dataset/curated/apple-orchard-v2-fp-only/test.jsonl")
    data_root = os.path.join(PROJECT_ROOT, "blender/dataset/raw/sample_rate16_processed")
    train_manifest = os.path.join(PROJECT_ROOT, "blender/dataset/curated/sample_rate16_processed/train.jsonl")
    test_manifest = os.path.join(PROJECT_ROOT, "blender/dataset/curated/sample_rate16_processed/test.jsonl")

    # dataset / loader (batch_size 1 is easiest for variable‑length clouds)
    config = {
        'voxel_size': 0.0045,  # default voxel size for normalization
        'percentile': 95,     # default percentile for normalization
        # 'subset_size': 0.01,   # use all data
        'SEED': SEED,  # for reproducibility
    }
    train_ds = AppleInstancePointCloudDataset(
            data_root     = data_root,
            manifest_path = train_manifest,
            augment       = True,
            config        = config
            )
    # split into train/val
    train_size = int(len(train_ds) * 0.8)
    val_size = len(train_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])
    val_ds.augment = False  # no augmentation for validation set

    test_ds = AppleInstancePointCloudDataset(
            data_root     = data_root,
            manifest_path = test_manifest,
            augment       = False,   
            config        = config
            )
    
    print("train size", len(train_ds))
    print("val size", len(val_ds))
    print("test size", len(test_ds))

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=12)#, collate_fn=pad_collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=1, shuffle=True, num_workers=12)
    test_dl  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=12)
    # ------------------------------------------------------------------
    for _ in enumerate(train_dl):
        pass
        

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