import bpy, os, math, numpy as np
import bmesh
import bpy_extras
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import json
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict
import pyexr
from uuid import uuid4


class AppleSample(BaseModel):
    apple_id: int = Field(..., description="ID of the apple")
    apple_name: str = Field(..., description="Name of the apple object")
    apple_bbox: List[float] = Field(..., description="Bounding box of the apple in camera pixel coordinates")
    apple_center: List[float] = Field(..., description="Center of the apple surface from the view of the camera in camera coordinates")
    def __repr__(self):
        return f"AppleSample(\napple_id={self.apple_id},\n" \
               f"apple_name='{self.apple_name}',\n" \
               f"apple_bbox={self.apple_bbox},\n" \
                f"apple_center={self.apple_center},\n"
    def __str__(self):
        return f"Apple ID: {self.apple_id}\n" \
               f"Apple Name: {self.apple_name}\n" \
               f"Apple BBox: {self.apple_bbox}\n" \
               f"Apple Center: {self.apple_center}\n"

def save_rgbd(res_x=1280, res_y=720,
                         path_stem="/home/siddhartha/RIVAL/learning2localize/blender/dataset/apple_orchard",
                         frame=None):
    """
    Render once and save
      • depth (single‑channel)      -> ‹stem›_depth####.{png|exr}
      • combined colour (RGB)       -> ‹stem›_rgb####.png
    Returns (depth_path, rgb_path).
    """
    scn   = bpy.context.scene
    scn.render.resolution_x = res_x # set user-defined resolution
    scn.render.resolution_y = res_y
    cam   = scn.camera or scn.objects['Camera']
    frame = frame or scn.frame_current

    # ---------- ensure GPU --------------------------------------
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'      # or OPTIX / HIP / METAL
    prefs.get_devices();  [setattr(d, "use", True) for d in prefs.devices if d.type != 'CPU']
    scn.cycles.device = 'GPU'              

    scn.render.engine        = 'CYCLES'
    scn.cycles.samples       = 1
    scn.cycles.max_bounces   = 0           # one‑sample, zero‑bounce path tracer 

    # ---------- compositor tree --------------------------------
    scn.use_nodes = True
    nt = scn.node_tree
    nt.nodes.clear()

    rlayer = nt.nodes.new("CompositorNodeRLayers")

    # depth ➜ EXR
    depth_out               = nt.nodes.new("CompositorNodeOutputFile")
    depth_out.base_path     = os.path.dirname(bpy.path.abspath(path_stem))
    depth_out.file_slots[0].path = os.path.basename(path_stem) + "_depth"
    depth_out.format.file_format  = 'OPEN_EXR'
    depth_out.format.color_depth  = '32'
    nt.links.new(rlayer.outputs['Depth'], depth_out.inputs[0])  # ← raw Z pass

    # RGB output node
    rgb_out = nt.nodes.new("CompositorNodeOutputFile")
    rgb_out.base_path          = depth_out.base_path
    rgb_out.file_slots[0].path = os.path.basename(path_stem) + "_rgb"
    rgb_out.format.file_format = 'PNG'        # tonemapped sRGB 
    rgb_out.format.color_mode  = 'RGB'
    rgb_out.format.color_depth = '8'
    nt.links.new(rlayer.outputs['Image'], rgb_out.inputs[0])       # Combined pass is 'Image' 

    # ---------- set up object indexing ----------------------
    for i, obj in enumerate(
        [o for o in bpy.context.scene.objects if o.type == 'MESH']):
        obj.pass_index = i + 1          # 0 is “background”, start at 1
        if 'test' in obj.name:
            obj.hide_render = True

    view_layer = bpy.context.view_layer          # <‑‑ active ViewLayer handle
    view_layer.use_pass_object_index = True      # enable the ID pass

    # Object index output node
    id_out = nt.nodes.new("CompositorNodeOutputFile")
    id_out.base_path = depth_out.base_path
    id_out.file_slots[0].path = os.path.basename(path_stem) + "_id"
    id_out.format.file_format  = 'OPEN_EXR'   # keeps integer indices intact
    id_out.format.color_mode   = 'BW'
    id_out.format.color_depth  = '32'

    ## write the object index mapping to a json file for later use
    mapping = {obj.pass_index: obj.name
           for obj in bpy.context.scene.objects
           if obj.type == 'MESH' and obj.pass_index != 0}
    with open(os.path.join(id_out.base_path, f"{path_stem}_id_map.json"), "w") as f:
        json.dump(mapping, f, indent=2)


    # connect the pass
    nt.links.new(rlayer.outputs['IndexOB'], id_out.inputs[0])

    # ---------- render one frame -------------------------------
    bpy.ops.render.render(write_still=True)

    depth_path = os.path.join(depth_out.base_path,
                              f"{depth_out.file_slots[0].path}{frame:04d}.exr")
    rgb_path   = os.path.join(rgb_out.base_path,
                              f"{rgb_out.file_slots[0].path}{frame:04d}.png")
    id_path = os.path.join(id_out.base_path,
                       f"{id_out.file_slots[0].path}{frame:04d}.exr")
    id_mapping_path = os.path.join(id_out.base_path,
                       f"{path_stem}_id_map.json")


    print("Depth saved →", depth_path)
    print("RGB   saved →", rgb_path)
    print("ID    saved →", id_path)
    print("ID mapping saved →", id_mapping_path)
    return depth_path, rgb_path, id_path, id_mapping_path


def depth_to_point_cloud(depth_path,
                         cam=None,
                         as_world=False):
    cam = cam or bpy.context.scene.camera
    if cam is None:
        raise RuntimeError("No active camera")

    img = bpy.data.images.load(depth_path, check_existing=False)

    # robust colours‑pace fallback
    enum = img.colorspace_settings.bl_rna.properties['name'].enum_items
    img.colorspace_settings.name = (
        'Linear'          if 'Linear'          in enum else
        'Linear Rec.709'  if 'Linear Rec.709'  in enum else
        'Non-Color')

    W, H = img.size
    depth = np.array(img.pixels[:], np.float32).reshape(H, W, img.channels)[..., 0]

    # apply scene‑unit scale ONCE
    depth *= (bpy.context.scene.unit_settings.scale_length or 1.0)  # :contentReference[oaicite:4]{index=4}

    # back‑project
    fx = (W/2)/math.tan(cam.data.angle_x/2)
    fy = (H/2)/math.tan(cam.data.angle_y/2)
    cx, cy = W/2, H/2
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    z  = depth
    xyz = np.stack([(uu-cx)*z/fx, -(vv-cy)*z/fy, -z], axis=-1).astype(np.float32)

    if as_world:
        R = cam.matrix_world
        flat = xyz.reshape(-1,3)
        xyz  = np.array([R @ Vector(p) for p in flat], np.float32).reshape(H, W, 3)

    bpy.data.images.remove(img)

    clip_far = cam.data.clip_end           # e.g. 100 m
    bad = depth >= clip_far * 0.999        # “almost far” or bigger ⇒ background
    depth[bad] = np.nan                    # or = clip_far if you prefer

    xyz = np.stack([(uu-cx)*depth/fx, -(vv-cy)*depth/fy, -depth], -1)
    xyz[~np.isfinite(xyz)] = np.nan        # drop NaNs/Infs that slipped through

    # optional 180° rot & flip to match your previous orientation
    xyz = np.rot90(xyz, 2)[:, ::-1, :]
    return xyz

def get_world_bounding_box(obj):
    # Local bounding box corners
    local_bbox_corners = [Vector(corner) for corner in obj.bound_box]
    # Convert to world space
    world_bbox_corners = [obj.matrix_world @ corner for corner in local_bbox_corners]
    return world_bbox_corners


def get_bbox(obj,
                   cam: bpy.types.Object = None,
                   scene: bpy.types.Scene = None,
                   z_clip: float = 0.0,
                   world: bool=False):
    """
    Return the 2D bounding box of the object from the camera's perspective.
    If world=True, return the 3D bounding box corners in world coordinates.
    The 2D bounding box is in normalized device coordinates (NDC) [0..1]².

    Returns:
        - bbox: (x_min, y_min, x_max, y_max) in either NDC or world coordinates

    """
    scene = scene or bpy.context.scene
    cam   = cam   or scene.camera
    deps  = bpy.context.evaluated_depsgraph_get()

    # --- get the evaluated mesh  ---------------
    obj_eval = obj.evaluated_get(deps)
    mesh_eval = obj_eval.to_mesh()          # temporary, must be freed later

    # --- project every vertex -------------------------------------
    xs, ys = [], []
    for v in mesh_eval.vertices:
        # world coordinate of the vertex
        p_world = obj_eval.matrix_world @ v.co
        # NDC = (x,y,z) in [0,1]² × [0,1]  (Blender: +Y is up, origin bottom‑left)
        p_ndc   = bpy_extras.object_utils.world_to_camera_view(scene, cam, p_world)
        if p_ndc.z >= z_clip:               # keep only points in front of camera
            xs.append(p_ndc.x)
            ys.append(p_ndc.y)

    # free the temp mesh to avoid leaks
    obj_eval.to_mesh_clear()

    if not xs:
        # everything is behind the camera: return an empty bbox
        box = (1.0, 1.0, 0.0, 0.0)
    else:
        box = (min(xs), min(ys), max(xs), max(ys))

    if world:
        box = [obj_eval.matrix_world @ Vector(c) for c in obj_eval.bound_box]
    return box

def ndc_to_pixel(ndc_xy, res_x, res_y):
    """NDC (0..1, bottom‑left) → integer pixel coordinates (top‑left origin)."""
    x_pix = ndc_xy[0] * res_x
    y_pix = (1.0 - ndc_xy[1]) * res_y   # flip Y to top‑left origin
    return round(x_pix), round(y_pix)

def ndc_center_of_bbox(obj, cam=None, scene=None):
    """Return (x_ndc, y_ndc) midpoint of obj's projected AABB."""
    (x_min, y_min, x_max, y_max) = get_bbox(obj, cam, scene)
    x_ndc = (x_min + x_max) * 0.5
    y_ndc = (y_min + y_max) * 0.5
    return x_ndc, y_ndc

def surface_point_at_bbox_center(obj, cam=None, scene=None):
    scene = scene or bpy.context.scene
    cam   = cam   or scene.camera
    if cam.type != 'CAMERA':
        raise ValueError("Need a CAMERA object")

    # 1. centre pixel of the 2‑D AABB ------------------------
    x_ndc, y_ndc = ndc_center_of_bbox(obj, cam, scene)
    sx = (x_ndc - 0.5) * 2.0
    sy = (y_ndc - 0.5) * 2.0
    hx = math.tan(cam.data.angle_x * 0.5)
    hy = math.tan(cam.data.angle_y * 0.5)
    d_cam   = Vector((sx * hx, sy * hy, -1.0)).normalized()

    origin_W = cam.matrix_world.translation
    dir_W    = cam.matrix_world.to_3x3() @ d_cam

    # 2. build BVH & transform ray to object space ----------
    deps      = bpy.context.evaluated_depsgraph_get()
    obj_eval  = obj.evaluated_get(deps)
    bvh       = BVHTree.FromObject(obj_eval, deps, epsilon=0.0001)

    M_w2l = obj_eval.matrix_world.inverted()
    origin_L = M_w2l @ origin_W
    dir_L    = (M_w2l.to_3x3() @ dir_W).normalized()

    loc_L, no_L, face_i, dist = bvh.ray_cast(origin_L, dir_L)
    if loc_L is None:
        return None                                 # miss -> occluded or off‑fruit

    # 3. back to camera space -------------------------------
    loc_W   = obj_eval.matrix_world @ loc_L
    loc_cam = cam.matrix_world.inverted() @ loc_W
    return loc_W, loc_cam



def get_obj_surface_center(obj_name) -> Tuple[Vector, Vector]:
    """
    Get the surface point of a specific object in the scene.
    Args:
        obj_name (str): The name of the object.
    Returns:
        Tuple[Vector, Vector]: The world and cam coordinates of the surface point.
        or None if the object is occluded or off-screen.
    """
    obj = bpy.data.objects[obj_name]
    cam = bpy.context.scene.camera
    
    # Loop through all objects in the scene
    for obj2 in bpy.data.objects:
        if obj2==obj:
            obj.hide_render = False  # Keep this object visible
        else:
            obj.hide_render = True   # Hide everything else from rendering
    res = surface_point_at_bbox_center(obj, cam)
    if res:
        pt_w, pt_cam = res
        
    else:
        return # Apple is occluded or off-screen
    for ob in bpy.data.objects:
        if 'test' in obj.name: 
            continue
        ob.hide_render = False  # make all objects visible in render again
    return pt_w, pt_cam
def get_apple_ground_truth(apple_obj_name: str,
                           scene=None,
                           res_x=None,
                           res_y=None) -> AppleSample:
    """
    Return pixel AABB and visible‑surface centre for a single apple,
    both expressed in the camera coordinate system defined by depth_to_point_cloud.
    """
    scene = scene or bpy.context.scene
    cam   = scene.camera
    apple = bpy.data.objects[apple_obj_name]

    # --- 1. 2‑D bounding box ----------------------------------
    ndc_bbox = get_bbox(apple, cam, scene)
    res_x = res_x or scene.render.resolution_x
    res_y = res_y or scene.render.resolution_y
    x0_pix, y0_pix = ndc_to_pixel((ndc_bbox[0], ndc_bbox[1]), res_x, res_y)
    x1_pix, y1_pix = ndc_to_pixel((ndc_bbox[2], ndc_bbox[3]), res_x, res_y)
    bbox_px = [int(x0_pix), int(y0_pix), int(x1_pix), int(y1_pix)]

    # --- 2. 3‑D centre on the apple surface -------------------
    res= get_obj_surface_center(apple_obj_name)
    if res is None:
        return 
    loc_W, loc_cam = res
    print("WOrld loc:", loc_W)
    # --- 3. Pack into the pydantic model ----------------------
    sample = AppleSample(
        apple_id=apple.name.split('apple')[1],
        apple_name=apple.name,
        apple_bbox=bbox_px,
        apple_center=[loc_cam.x, -1*loc_cam.y, loc_cam.z],
    )
    return sample

def get_visible_objects(exr_path: str, id_mapping_path: str, conditional: callable = None):
    '''Load the object IDs from the EXR file and map them to object names.
    Args:
        exr_path (str): Path to the EXR file containing object IDs.
        id_mapping_path (str): Path to the JSON file mapping object IDs to names.
        conditional (callable, optional): A function that takes an ID and name and returns True if the object should be included.
    Returns:
        list: A list of tuples containing the object ID and name for each visible object.
    '''
    with pyexr.open(exr_path) as exr_file:
        # print(exr_file.channel_map)
        object_id_channel = exr_file.get("V")  # Shape: (height, width, 1)
        object_ids = object_id_channel[:, :, 0]  # Convert to 2D array

    # Load the mapping from pass indices to object names
    with open(id_mapping_path, "r") as f:
        id_to_name = json.load(f)


    # build apple instance mask
    visible_ids = np.unique(object_ids).astype(int)
    visible_ids = visible_ids[visible_ids != 0]
    visible_objs = []
    for id in visible_ids:
        name = id_to_name.get(str(id), "Unknown")
        if conditional is None or conditional(id, name):
            visible_objs.append((id, name))
    return visible_objs


def collect_scene_data():
    """
    Collect data from the scene and saves it.
    """


    STEM = "/home/siddhartha/RIVAL/learning2localize/blender/dataset/apple_orchard"
    # print(get_apple_ground_truth('apple8'))
    os.makedirs(STEM, exist_ok=True)
    uid = str(uuid4())
    STEM = os.path.join(STEM, uid)

    depth_png, rgb_png, idx_png, id_mapping_json = save_rgbd(res_x=1280, res_y=720,
                                             path_stem=STEM)

    cloud = depth_to_point_cloud(depth_png, as_world=False)
    pc_path = STEM + "_pc.npy"
    np.save(pc_path, cloud)
    print("Saved point cloud to ", pc_path)
    print("cloud shape:", cloud.shape)

    visible_apples = get_visible_objects(idx_png, id_mapping_json, 
                                         conditional=lambda id, name: 'apple' in name and 'stem' not in name)
    print("Visible apples:", visible_apples)

    scene_apple_data = {}
    for _, apple_name in visible_apples:
        apple_sample = get_apple_ground_truth(apple_name)
        if apple_sample is None:
            print(f"Apple {apple_name} is occluded or off-screen.")
            continue
        scene_apple_data[apple_name] = apple_sample.model_dump_json()
        print(apple_sample)
    # Save the apple data to a JSON file
    json_path = STEM + "_apple_data.json"
    with open(json_path, "w") as f:
        json.dump(scene_apple_data, f)
    print("Saved apple data to ", json_path)

