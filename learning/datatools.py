import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import random
import pyexr
import cv2
import sys 
sys.path.append("/home/siddhartha/RIVAL/learning2localize/blender/")
import utils



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


class ApplePointCloudDataset(Dataset):
    def __init__(self, data_root: str, manifest_path: str, augment=True):
        self.root = data_root
        self.augment = augment
        self.records = []

        with open(manifest_path) as f:
            scenes = [json.loads(line) for line in f]

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

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        stem, bbox, center, occ_rate = r["stem"], r["bbox"], r["center"], r["occ_rate"]

        xyz = np.load(os.path.join(self.root, f"{stem}_pc.npy"))
        rgb = cv2.cvtColor(cv2.imread(os.path.join(self.root, f"{stem}_rgb0000.png")), cv2.COLOR_BGR2RGB)
        xyzrgb = np.concatenate((xyz, rgb), axis=2)

        x1, y1, x2, y2 = map(int, bbox)
        crop = xyzrgb[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

        assert crop.shape[1] >0 and crop.shape[0] > 0 and crop.shape[2] == 6, \
            f"Invalid crop shape {crop.shape} for {stem} at index {idx} with bbox {bbox}"

        if self.augment:
            crop[:, :, 3:] = augment_rgb(crop[:, :, 3:])
 
        pc = crop.reshape(-1, 6)
        pc = pc[~np.isnan(pc).any(1)]
        pc = pc[~np.isinf(pc).any(1)]
        assert pc.shape[0] > 0, f"Empty point cloud for {stem} at index {idx}"
        if self.augment:
            pc = random_dropout(pc, dropout_range=(0.6, 0.8))
            # pc, rotation_matrix = rotate_point_cloud(pc)
        #     center = np.dot(rotation_matrix, center)
        #     print("Old center", r["center"])
        #     print("Rotated center", center)

        return pc, np.array(center, dtype=np.float32), {'stem': stem, 'bbox': bbox, 'occ_rate': occ_rate}


if __name__ == "__main__":

    import plotly.graph_objects as go

    data_root    = "/home/siddhartha/RIVAL/learning2localize/blender/dataset/apple_orchard-5-20-processed"
    train_manifest     = "/home/siddhartha/RIVAL/learning2localize/blender/curated/apple-orchard-v1/train.jsonl"
    test_manifest       = "/home/siddhartha/RIVAL/learning2localize/blender/curated/apple-orchard-v1/test.jsonl"

    # dataset / loader (batch_size 1 is easiest for variable‑length clouds)
    train_ds = ApplePointCloudDataset(
            data_root     = data_root,
            manifest_path = train_manifest,
            augment       = True,   
            )
    # split into train/val
    train_size = int(len(train_ds) * 0.8)
    val_size = len(train_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

    test_ds = ApplePointCloudDataset(
            data_root     = data_root,
            manifest_path = test_manifest,
            augment       = False,   
            )
    
    print("train size", len(train_ds))
    print("val size", len(val_ds))
    print("test size", len(test_ds))

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)    
    val_dl   = DataLoader(val_ds,   batch_size=1, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=1, shuffle=True)
    # ------------------------------------------------------------------
    for scene_i, (clouds_batch, centers_batch, (stem_batch, bbox_batch, occ_batch)) in enumerate(train_dl):
        pc  = clouds_batch[0]     # list[(N_i,6), …]
        assert pc.shape[1] == 6, f"Expected 6 channels, got {pc.shape[1]}"
        center  = centers_batch[0].numpy()  # (M,3)

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
            mode="markers",
            marker=dict(size=2,
                        color=pc[:, 3:6] / 255.0,   # RGB -> [0,1]
                        opacity=0.6)))

        fig.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode="markers",
            marker=dict(size=8, color="red")))

        fig.update_layout(scene_aspectmode="data",
                        width=700, height=700,
                        margin=dict(l=0, r=0, b=0, t=0))
        fig.show()

        if scene_i >= 2:        # stop after 2 scenes
            break
    for scene_i, (clouds_batch, centers_batch, (stem_batch, bbox_batch, occ_batch)) in enumerate(val_dl):
        pass 
    for scene_i, (clouds_batch, centers_batch, (stem_batch, bbox_batch, occ_batch)) in enumerate(test_dl):
        pass
    print("Dataloading worked")