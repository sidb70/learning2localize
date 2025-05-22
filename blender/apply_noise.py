import pyexr
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import json

import utils


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
    pc_im = add_distance_based_noise(pc_im)
    rgb_im = blur_img_edges(rgb_im, cnts)

    return pc_im, rgb_im


if __name__ == "__main__":
    import os
    import shutil
    # Load data
    # pc_im = np.load('/home/siddhartha/RIVAL/learning2localize/blender/dataset/apple_orchard/c2083317-49fc-4530-9a23-447f6ca19da1_pc.npy')
    # rgb_im = cv2.imread('/home/siddhartha/RIVAL/learning2localize/blender/dataset/apple_orchard/c2083317-49fc-4530-9a23-447f6ca19da1_rgb0000.png')
    # rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
    
    # visible_objs, id_mask, instance_mask, id_to_name = utils.get_visible_objects(
    #     exr_path = '/home/siddhartha/RIVAL/learning2localize/blender/dataset/apple_orchard/c2083317-49fc-4530-9a23-447f6ca19da1_id0000.exr',
    #     id_mapping_path = '/home/siddhartha/RIVAL/learning2localize/blender/dataset/apple_orchard/c2083317-49fc-4530-9a23-447f6ca19da1_id_map.json'
    # )

    # pc_im, rgb_im = apply_noise(pc_im, rgb_im, instance_mask)

    # pcd = pc_img_to_pcd(pc_im, rgb_im)
    # o3d.visualization.draw_geometries([pcd])

    ORIGINAL_DIR = '/home/siddhartha/RIVAL/learning2localize/blender/dataset/apple_orchard-5-20'
    NEW_DIR = '/home/siddhartha/RIVAL/learning2localize/blender/dataset/apple_orchard-5-20-processed'
    os.makedirs(NEW_DIR, exist_ok=True)
    all_raw_files = os.listdir(ORIGINAL_DIR)
    sample_ids = set([f.split('_')[0] for f in all_raw_files if f.endswith('apple_data.json')])
    id_to_files = {
        sample_id: [f for f in all_raw_files if f.startswith(sample_id)] for sample_id in sample_ids
    }
    new_dir_ids = set([f.split('_')[0] for f in os.listdir(NEW_DIR) if f.endswith('pc.npy')])
    for sample_id in sample_ids:
        sample_path = os.path.join(ORIGINAL_DIR, sample_id)
        if sample_id in new_dir_ids:
            print("Skipping ", sample_id)
            continue 
        pc_im = np.load(sample_path + "_pc.npy") 
        rgb_im = cv2.imread(sample_path + "_rgb0000.png")
        rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
        visible_objs, id_mask, instance_mask, id_to_name = utils.get_visible_objects(
            exr_path = sample_path + '_id0000.exr',
            id_mapping_path = sample_path + '_id_map.json'
        )
        pc_im, rgb_im = apply_noise(pc_im, rgb_im, instance_mask)


        new_path = os.path.join(NEW_DIR, sample_id)
        np.save(new_path + '_pc.npy', pc_im)
        rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2BGR)
        cv2.imwrite(new_path + '_rgb0000.png', rgb_im)
        for fname in id_to_files[sample_id]:
            if 'pc' in fname or 'rgb' in fname:
                continue
            
            src = os.path.join(ORIGINAL_DIR, fname)
            dst = os.path.join(NEW_DIR, fname)
            shutil.copy2(src, dst)
        print("applied noise to ", sample_id)
     


