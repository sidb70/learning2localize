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

def augment_point_cloud(point_cloud: np.ndarray,
                        rotation_range=(-np.pi, np.pi)
                        ):
    """
    Augment the point cloud by applying random rotation, translation.
    Args:
        point_cloud (np.ndarray): The original point cloud with shape (N, 3).
        rotation_range (tuple): The range for random rotation in radians.
        translation_range (tuple): The range for random translation.
    Returns:
        np.ndarray: The augmented point cloud with the same shape as the input.
    """
    # Ensure the point cloud is a numpy array
    point_cloud = np.array(point_cloud)

    # Random rotation
    theta = np.random.uniform(rotation_range[0], rotation_range[1])
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])

    # Random translation
    # translation = np.random.uniform(translation_range[0], translation_range[1], size=(3,))

    # Apply transformations
    augmented_point_cloud = point_cloud @ rotation_matrix.T #+ translation

    return augmented_point_cloud


class ApplePointCloudDataset(Dataset):
    """
    Pure reader + on the fly augmentations.
    """
    def __init__(self, data_root: str,
                 manifest_path: str,
                 augment: bool = True,
                 preload: bool = False):
        self.data_root = data_root
        self.augment   = augment

        # ---------------- read manifest once --------------------
        with open(manifest_path) as f:
            self.entries = [json.loads(l) for l in f]

        # optional RAM‑preload of xyzrgb (handy for tiny sets)
        self._cache = {}
        if preload:
            for e in self.entries:
                self._cache[e["stem"]] = self._load_xyzrgb(e["stem"])

    def _load_xyzrgb(self, stem):
        xyz = np.load(os.path.join(self.data_root, f"{stem}_pc.npy"))  # (H,W,3)
        rgb = cv2.cvtColor(cv2.imread(
                   os.path.join(self.data_root, f"{stem}_rgb0000.png")),
                   cv2.COLOR_BGR2RGB)
        return np.concatenate((xyz, rgb), axis=2)                      # (H,W,6)

    # ------------------------------------------------------------
    def __len__(self): return len(self.entries)

    def __getitem__(self, idx):
        rec      = self.entries[idx]
        stem     = rec["stem"]
        boxes    = np.asarray(rec["boxes"])
        centres  = np.asarray(rec["centres"], dtype=np.float32)

        xyzrgb   = (self._cache[stem] if stem in self._cache
                    else self._load_xyzrgb(stem))

        # list of variable‑length (N_i,6) clouds
        clouds = []
        for bb in boxes:
            bb = augment_bounding_box(bb) if self.augment else bb
            x1,y1,x2,y2 = map(int, bb)
            crop = xyzrgb[min(y1,y2):max(y1,y2),
                          min(x1,x2):max(x1,x2)]

            if self.augment:
                crop[:,:,3:] = augment_rgb(crop[:,:,3:])

            pc = crop.reshape(-1,6)
            pc = pc[~np.isnan(pc).any(1)]
            pc = pc[~np.isinf(pc).any(1)]

            if self.augment:
                pc = augment_point_cloud(pc)

            clouds.append(pc)

        return clouds, centres


if __name__ == "__main__":

    import plotly.graph_objects as go

    data_root    = "/home/siddhartha/RIVAL/learning2localize/blender/dataset/apple_orchard-5-20-processed"
    manifest     = "/home/siddhartha/RIVAL/learning2localize/blender/curated/apple-orchard-v1/manifest.jsonl"

    # dataset / loader (batch_size 1 is easiest for variable‑length clouds)
    ds = ApplePointCloudDataset(
            data_root     = data_root,
            manifest_path = manifest,
            augment       = False,   #  no random aug in visualisation
            preload       = False)

    dl = DataLoader(ds, batch_size=1, shuffle=True)    # default collate_ok

    # ------------------------------------------------------------------
    for scene_i, (clouds_batch, centres_batch) in enumerate(dl):
        # batch_size == 1  ⇒ strip the outer dimension added by DataLoader
        clouds_list  = clouds_batch[0]     # list[(N_i,6), …]
        centres_arr  = centres_batch[0].numpy()  # (M,3)

        for pc, centre in zip(clouds_list, centres_arr):
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
                mode="markers",
                marker=dict(size=2,
                            color=pc[:, 3:6] / 255.0,   # RGB -> [0,1]
                            opacity=0.6)))

            fig.add_trace(go.Scatter3d(
                x=[centre[0]], y=[centre[1]], z=[centre[2]],
                mode="markers",
                marker=dict(size=8, color="red")))

            fig.update_layout(scene_aspectmode="data",
                            width=700, height=700,
                            margin=dict(l=0, r=0, b=0, t=0))
            fig.show()

        if scene_i >= 3:        # stop after four scenes
            break