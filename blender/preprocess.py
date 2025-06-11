import pyexr
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import json
# import utils
import multiprocessing as mp
from functools import partial
import shutil
from pathlib import Path
from tqdm import tqdm        
import os
import argparse       
import dotenv 
import gc
import psutil
from collections import defaultdict

dotenv.load_dotenv()
PROJECT_ROOT = os.getenv('PROJECT_ROOT')



# -------------------- PATHS --------------------
ORIGINAL_DIR = Path(os.path.join(PROJECT_ROOT, 'blender/dataset/raw/apple_orchard-5-20'))
NEW_DIR      = Path(os.path.join(PROJECT_ROOT, 'blender/dataset/raw/apple_orchard-test'))
# NEW_DIR = Path('/media/siddhartha/games/apple_orchard-test')
NEW_DIR.mkdir(exist_ok=True)


def pc_img_to_pcd(pc_img, color_img=None):
    '''
    Convert a point cloud image to Open3D point cloud format.
    Parameters
    ----------
    pc_img : np.ndarray
        Point cloud image of shape (H, W, 3) where each pixel contains (x, y, z) coordinates.
    color_img : np.ndarray, optional
        Color image of shape (H, W, 3) where each pixel contains (R, G, B) values.
    Returns
    -------
    pcd : open3d.geometry.PointCloud
        Open3D point cloud object.
    '''
    pc_3d = pc_img.reshape(-1,3)
    if color_img is None:
        colors_3d = np.zeros_like(pc_3d)
    else:
        colors_3d = color_img.reshape(-1,3)
    #remove nan points
    colors_3d = colors_3d[~np.isnan(pc_3d).any(axis=1)]
    pc_3d = pc_3d[~np.isnan(pc_3d).any(axis=1)]
    assert pc_3d.shape == colors_3d.shape

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_3d / 255.0)
    return pcd



def find_shared_edges_and_ids(instance_mask):
    '''
    Find shared edges and their corresponding IDs in the instance mask.
    Parameters
    ----------
    instance_mask : np.ndarray
        Instance mask of shape (H, W) where each pixel value represents an instance ID.
    Returns
    -------
    edge_map : np.ndarray
        Binary edge map of shape (H, W) where edges are marked with 255.
    edge_to_id_pair : dict
        Dictionary where keys are pixel coordinates (y, x) and values are tuples of sorted instance IDs.
    '''
    H, W = instance_mask.shape
    edge_map = np.zeros((H, W), dtype=np.uint8)
    edge_to_id_pair = {}

    # Pad to avoid boundary issues
    padded = np.pad(instance_mask, ((1,1), (1,1)), mode='constant', constant_values=0)

    # Directions: top, bottom, left, right
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(1, H+1):
        for x in range(1, W+1):
            center_id = padded[y, x]
            if center_id == 0:
                continue  # Skip background

            for dy, dx in neighbors:
                neighbor_id = padded[y+dy, x+dx]
                if neighbor_id != center_id and neighbor_id != 0:
                    # Inter-instance edge
                    edge_map[y-1, x-1] = 255
                    id_pair = tuple(sorted((center_id, neighbor_id)))  # (smaller, larger)
                    edge_to_id_pair[(y-1, x-1)] = id_pair
                    break  # Already found a neighbor with a different ID

    return edge_map, edge_to_id_pair

def get_shared_contours(instance_mask):
    """
    Get shared contours and their corresponding IDs from the instance mask.

    Parameters
    ----------
    instance_mask : np.ndarray
        Instance mask of shape (H, W) where each pixel value represents an instance ID.

    Returns
    -------
    shared_contours : dict
        Dictionary where keys are tuples of sorted instance IDs and values are lists of contour points.
    """
    edge_map, edge_to_id_pair = find_shared_edges_and_ids(instance_mask)

    # Create a dictionary to store shared contours
    shared_contours = {}
    for xy, id_pair in edge_to_id_pair.items():
        id = '_'.join([str(int(id)) for id in sorted(list(id_pair))])
        if id not in shared_contours:
            shared_contours[id] = []
        shared_contours[id].append(xy)
    return shared_contours

def add_flying_pixels(pc_im, instance_mask, shared_contours, displacement_std=0.015):
    """
    Add flying pixels at depth discontinuities.
    Parameters
    ----------
    pc_im : np.ndarray
        Point cloud image of shape (H, W, 3).
    instance_mask : np.ndarray
        Instance mask of shape (H, W) where each pixel value represents an instance ID.
    shared_contours : dict
        Dictionary where keys are tuples of sorted instance IDs and values are lists of contour points.
    displacement_std : float
        Standard deviation of the noise to be added.
    Returns
    -------
    pc_img : np.ndarray
        Point cloud image with added noise of shape (H, W, 3).
    """
    pc_img = pc_im.copy()
    # Add noise to edge points
    mean_depths = {}
    for key in shared_contours:
        obj1, obj2 = key.split('_')
        obj1 = int(obj1)
        obj2 = int(obj2)

        obj1_mean_depth = mean_depths.get(obj1, np.nanmean(pc_img[instance_mask == obj1]))
        obj2_mean_depth = mean_depths.get(obj2, np.nanmean(pc_img[instance_mask == obj2]))
        mean_depths[obj1] = obj1_mean_depth
        mean_depths[obj2] = obj2_mean_depth
        
        for pt_idx in shared_contours[key][::5]:
            if np.isnan(pc_img[pt_idx[0], pt_idx[1]]).any():
                continue
            neighborhood = pc_img[pt_idx[0]-2:pt_idx[0]+3, pt_idx[1]-2:pt_idx[1]+3, 2]
            bigger_neighborhood = pc_img[pt_idx[0]-5:pt_idx[0]+6, pt_idx[1]-5:pt_idx[1]+6, 2]
            # noise = np.random.normal(0,displacement_std, neighborhood.shape)
            depth_diff = abs(obj1_mean_depth - obj2_mean_depth)

            # noise = -1*abs(np.random.normal(0, displacement_std, neighborhood.shape))
            noise  = np.random.uniform(-depth_diff, 0, neighborhood.shape)


            normal_noise = -1*abs(np.random.normal(0, displacement_std, bigger_neighborhood.shape))

            # only noise the pixels of the front object
            if obj1_mean_depth < obj2_mean_depth:
                noise[neighborhood < obj1_mean_depth] = 0
                normal_noise[bigger_neighborhood < obj1_mean_depth] = 0
            else:
                noise[neighborhood > obj1_mean_depth] = 0
                normal_noise[bigger_neighborhood > obj1_mean_depth] = 0
            
            new_depths_1 = pc_img[pt_idx[0]-2:pt_idx[0]+3, pt_idx[1]-2:pt_idx[1]+3,2]+ noise
            # project each point to the new depth
            pc_img[pt_idx[0]-2:pt_idx[0]+3, pt_idx[1]-2:pt_idx[1]+3,:2]*= (new_depths_1 / pc_img[pt_idx[0]-2:pt_idx[0]+3, pt_idx[1]-2:pt_idx[1]+3,2])[:, :, np.newaxis]
            new_depths_2 = pc_img[pt_idx[0]-5:pt_idx[0]+6, pt_idx[1]-5:pt_idx[1]+6,2]+ normal_noise
            # project each point to the new depth
            pc_img[pt_idx[0]-5:pt_idx[0]+6, pt_idx[1]-5:pt_idx[1]+6,:2]*= (new_depths_2 / pc_img[pt_idx[0]-5:pt_idx[0]+6, pt_idx[1]-5:pt_idx[1]+6,2])[:, :, np.newaxis]
            pc_img[pt_idx[0]-2:pt_idx[0]+3, pt_idx[1]-2:pt_idx[1]+3,2]+= noise
            pc_img[pt_idx[0]-5:pt_idx[0]+6, pt_idx[1]-5:pt_idx[1]+6,2]+= normal_noise
    return pc_img

def blur_img_edges(rgb_im, shared_contours, blur_radius=2):
    """
    Blur the edges of the RGB image where shared contours are present.
    Parameters
    ----------
    rgb_im : np.ndarray
        RGB image of shape (H, W, 3).
    shared_contours : dict
        Dictionary where keys are tuples of sorted instance IDs and values are lists of contour points.
    blur_radius : int
        Radius of the Gaussian blur to be applied.
    Returns
    -------
    rgb_img : np.ndarray
        RGB image with blurred edges of shape (H, W, 3).
    """
    rgb_img = rgb_im.copy()
    for key in shared_contours:
        obj1, obj2 = key.split('_')
        obj1 = int(obj1)
        obj2 = int(obj2)

        for pt_idx in shared_contours[key][::5]:
            if np.isnan(rgb_img[pt_idx[0], pt_idx[1]]).any():
                continue
            # Get the region around the edge point
            x, y = pt_idx
            x_start = max(0, x - blur_radius)
            x_end = min(rgb_img.shape[0], x + blur_radius)
            y_start = max(0, y - blur_radius)
            y_end = min(rgb_img.shape[1], y + blur_radius)

            # Apply Gaussian blur to the region
            rgb_img[x_start:x_end, y_start:y_end] = cv2.GaussianBlur(rgb_img[x_start:x_end, y_start:y_end], (3, 3), 0)
    return rgb_img

def add_distance_based_noise(pc_im, base_noise=0.0004, meter_noise_percent=0.0025):
    '''
    Add noise to the point cloud based on the z value.
    Parameters
    ----------
    pc_im : np.ndarray
        Point cloud image of shape (H, W, 3) where each pixel contains (x, y, z) coordinates.
    base_noise : float
        Base noise value to be added to the z coordinate.
    meter_noise_percent : float
        Percentage of the z value to be added as noise.
    Returns
    -------
    pc_img : np.ndarray
        Noisy point cloud image of shape (H, W, 3).
    '''
    pc_img = pc_im.copy()
    # add noise based on z val. higher z val, more noise
    ## ±4 mm + 0.25% of depth
    noise = np.random.normal(0, base_noise + meter_noise_percent * -1*pc_img[:, :, 2], pc_img[:,:,2].shape)
    new_depths = pc_img[:, :, 2] + noise
    # project each point to the new depth
    pc_img[:, :, :2] *= (new_depths / pc_img[:, :, 2])[:, :, np.newaxis]
    pc_img[:, :, 2] += noise
    return pc_img

def apply_noise(pc_im, rgb_im, instance_mask):
    """
    Apply noise to the point cloud and RGB image based on the instance mask.
    Parameters
    ----------
    pc_im : np.ndarray
        Point cloud image of shape (H, W, 3).
    rgb_im : np.ndarray
        RGB image of shape (H, W, 3).
    instance_mask : np.ndarray
        Instance mask of shape (H, W) where each pixel value represents an instance ID.
    Returns
    -------
    pc_im : np.ndarray
        Noisy point cloud image of shape (H, W, 3).
    rgb_im : np.ndarray
        Noisy RGB image of shape (H, W, 3).
    """
    cnts = get_shared_contours(instance_mask)
    pc_im = add_flying_pixels(pc_im, instance_mask, cnts)
    # pc_im = add_distance_based_noise(pc_im)
    rgb_im = blur_img_edges(rgb_im, cnts)

    return pc_im, rgb_im



def get_visible_objects(exr_path: str, id_mapping_path: str, conditional: callable = None):
    '''Load the object IDs from the EXR file and map them to object names.
    Args:
        exr_path (str): Path to the EXR file containing object IDs.
        id_mapping_path (str): Path to the JSON file mapping object IDs to names.
        conditional (callable, optional): A function that takes an ID and name and returns True if the object should be included.
    Returns:
        visible_objs (list): List of tuples containing object IDs and names.
        id_mask (np.ndarray): The id mask from the EXR file
        id_to_name (dict): Mapping from object IDs to names.
    '''
    with pyexr.open(exr_path) as exr_file:
        # print(exr_file.channel_map)
        object_id_channel = exr_file.get("V")  # Shape: (height, width, 1)
        id_mask = object_id_channel[:, :, 0].astype(int)  # Convert to 2D array

    # Load the mapping from pass indices to object names
    with open(id_mapping_path, "r") as f:
        id_to_name = json.load(f)

    # build apple instance mask
    visible_ids = np.unique(id_mask).astype(int)
    visible_ids = visible_ids[visible_ids != 0]
    visible_objs = []

    instance_mask = np.zeros_like(id_mask)
    unique_id = 1
    for id in visible_ids:
        name = id_to_name[str(id)]
        if conditional is None or conditional(id, name):
            visible_objs.append((id, name))
            instance_mask[id_mask == id] = unique_id
            unique_id+=1
    instance_mask = instance_mask.astype(np.uint8)

    return visible_objs, id_mask, instance_mask, id_to_name
def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Args:
        box1 (list): [x1, y1, x2, y2] coordinates of the first box.
        box2 (list): [x1, y1, x2, y2] coordinates of the second box.
    Returns:
        float: IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x != self.parent.setdefault(x, x):
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)
def process_sample(sample_id: str,
                   noise: bool = True,
                   overwrite: bool = False):
    """
    Full end-to-end pipeline for one sample-ID.
    Runs safely in a separate process.
    """
    # ---------- skip if already done ----------
    if not overwrite and (NEW_DIR / f'{sample_id}_pc.npy').exists():
        return f'skip:{sample_id}'

    sample_path = ORIGINAL_DIR / sample_id

    # ---------- load ----------
    pc_im  = np.load(str(sample_path) + "_pc.npy")
    rgb_im = cv2.imread(str(sample_path) + "_rgb0000.png")
    rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)

    visible_objs, id_mask, instance_mask, id_to_name = get_visible_objects(
        exr_path        = str(sample_path) + '_id0000.exr',
        id_mapping_path = str(sample_path) + '_id_map.json'
    )
    # ---------- optional processing ----------
    if noise:
        pc_im, rgb_im = apply_noise(pc_im, rgb_im, instance_mask)
    visible_apples, apple_id_mask, apple_instance_mask, _ = get_visible_objects(
                    exr_path        = str(sample_path) + '_id0000.exr',
                    id_mapping_path = str(sample_path) + '_id_map.json',
                    conditional=lambda id, name: 'apple' in name and 'stem' not in name
    ) 


    # --------- identify clustered apples ----------
    apple_data_path = ORIGINAL_DIR / f'{sample_id}_apple_data.json'
    assert apple_data_path.exists(), f"Apple data file not found: {apple_data_path}"
    with open(apple_data_path) as jf:
        apple_data = {k: json.loads(v) for k, v in json.load(jf).items()}
    uf = UnionFind()

    # Union apples with IoU > 0.1
    apple_keys = list(apple_data.keys())
    apple_boxes = {k: v['apple_bbox'] for k, v in apple_data.items()}
    # for i in range(len(apple_keys)):
    #     for j in range(i + 1, len(apple_keys)):
    #         k1, k2 = apple_keys[i], apple_keys[j]
    #         box1 = apple_boxes[k1]
    #         box2 = apple_boxes[k2]
    #         if iou(box1, box2) > 0.1:
    #             uf.union(k1, k2)

    # # Build clusters
    # cluster_map = defaultdict(list)
    # for k in apple_keys:
    #     root = uf.find(k)
    #     cluster_map[root].append(k)

    # clusters = list(cluster_map.values())
    # # reduce each cluster to only the apple names
    # clusters = [[apple for apple in cluster] for cluster in clusters]
    # print("Clusters found:", len(clusters))
    # print("Clusters:", clusters)
    clusters =[]



    # ---------- save ----------
    new_path = NEW_DIR / sample_id
    np.save(str(new_path) + '_pc.npy', pc_im)
    rgb_bgr = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(new_path) + '_rgb0000.png', rgb_bgr)
    
    # save visible apples and apple_id_mask
    np.savez_compressed(str(new_path) + '_instance_data.npz',
        visible_apples=visible_apples,
        apple_id_mask=apple_id_mask,
        clusters=np.array(clusters, dtype=object),
    )
    # copy auxiliary files (everything except pc/rgb)
    for fname in (f for f in os.listdir(ORIGINAL_DIR) if f.startswith(sample_id)):
        if any(k in fname for k in ('_pc.npy', '_rgb')):
            continue
        shutil.copy2(ORIGINAL_DIR / fname, NEW_DIR / fname)
    print(f'Processed {sample_id} with {len(visible_objs)} visible objects')
    return f'done:{sample_id}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply synthetic sensor noise and voxel normalisation to orchard dataset.'
    )
    parser.add_argument('--noise', type=bool, default=True,
                        help='Apply noise to the point cloud and RGB image.')
    parser.add_argument('--workers', type=int, default=mp.cpu_count(),
                        help='Number of parallel workers (1 = sequential, <=0 threads instead of processes).')
    parser.add_argument('--overwrite', action='store_true',  default=False,
                        help='Re-process samples even if output already exists.')
    args = parser.parse_args()

    # collect sample‑ids
    all_raw_files = os.listdir(ORIGINAL_DIR)
    sample_ids = {f.split('_')[0] for f in all_raw_files if f.endswith('apple_data.json')}
    # for i in range(len(sample_ids)):
    #     print(process_sample(list(sample_ids)[i], noise=True, overwrite=True))
    #     exit()

    if not args.overwrite:     # filter out already processed samples
        new_ids = [s for s in sample_ids if not (NEW_DIR / f'{s}_instance_data.npz').exists()]
        print(f"Processing {len(new_ids)} out of {len(sample_ids)} samples, skipping {len(sample_ids) - len(new_ids)} already processed samples.")
        sample_ids = new_ids

    from concurrent.futures import ProcessPoolExecutor as Executor
    max_workers = args.workers

    worker_fn = partial(process_sample,
                        noise=args.noise,
                        overwrite=args.overwrite)

    print(f"Processing {len(sample_ids)} samples with {max_workers} worker(s)…")
    # run the processing in parallel
    with Executor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_fn, sample_id) for sample_id in sample_ids]
        for future in tqdm(futures, desc='Processing samples'):
            result = future.result()
            if result.startswith('done:'):
                print(result)
            elif result.startswith('skip:'):
                print(result)
            else:
                print(f"Unexpected result: {result}")
                raise RuntimeError(f"Unexpected result: {result}")

    print('✅  All tasks complete.')