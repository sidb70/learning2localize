import bpy, os, math, numpy as np
import bmesh
import bpy_extras
from mathutils import Vector
from mathutils.bvhtree import BVHTree




def save_depth_rgb_fast(res_x=1280, res_y=720,
                         max_distance=1.8,
                         depth_mode='PNG',          # or 'EXR'
                         path_stem="/home/siddhartha/RIVAL/blender/apple_orchard",
                         frame=None):
    """
    Render once and save
      • depth (single‑channel)      -> ‹stem›_depth####.{png|exr}
      • combined colour (RGB)       -> ‹stem›_rgb####.png
    Returns (depth_path, rgb_path).
    """
    scn   = bpy.context.scene
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

    rl   = nt.nodes.new("CompositorNodeRLayers")              # Render Layers 

    # Depth output node
    depth_out = nt.nodes.new("CompositorNodeOutputFile")      
    depth_out.base_path = os.path.dirname(bpy.path.abspath(path_stem))
    depth_out.file_slots[0].path = os.path.basename(path_stem) + "_depth"

    if depth_mode.upper() == 'EXR':
        depth_out.format.file_format = 'OPEN_EXR';  depth_out.format.color_depth = '32';  depth_ext = ".exr"
        nt.links.new(rl.outputs['Depth'], depth_out.inputs[0])                    # connect Z pass 
    else:                                                                         # 16‑bit greyscale PNG
        mapr = nt.nodes.new("CompositorNodeMapRange")
        mapr.inputs["From Max"].default_value = max_distance
        mapr.inputs["To Min"].default_value   = 0.0
        mapr.inputs["To Max"].default_value   = 1.0
        nt.links.new(rl.outputs['Depth'], mapr.inputs[0])
        depth_out.format.file_format = 'PNG'; depth_out.format.color_mode = 'BW'; depth_out.format.color_depth = '16'
        nt.links.new(mapr.outputs[0], depth_out.inputs[0]);  depth_ext = ".png"

    # RGB output node
    rgb_out = nt.nodes.new("CompositorNodeOutputFile")
    rgb_out.base_path          = depth_out.base_path
    rgb_out.file_slots[0].path = os.path.basename(path_stem) + "_rgb"
    rgb_out.format.file_format = 'PNG'        # tonemapped sRGB 
    rgb_out.format.color_mode  = 'RGB'
    rgb_out.format.color_depth = '8'
    nt.links.new(rl.outputs['Image'], rgb_out.inputs[0])       # Combined pass is 'Image' 

    # ---------- render one frame -------------------------------
    bpy.ops.render.render(write_still=True)

    depth_path = os.path.join(depth_out.base_path,
                              f"{depth_out.file_slots[0].path}{frame:04d}{depth_ext}")
    rgb_path   = os.path.join(rgb_out.base_path,
                              f"{rgb_out.file_slots[0].path}{frame:04d}.png")

    print("Depth saved →", depth_path)
    print("RGB   saved →", rgb_path)
    return depth_path, rgb_path


def depth_to_point_cloud(depth_path,
                         max_distance=1.8,
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

    W, H, C = *img.size, img.channels
    depth = np.array(img.pixels[:], np.float32).reshape(H, W, C)[..., 0]


    scale = bpy.context.scene.unit_settings.scale_length or 1.0 
    depth *= scale
    
    
    if depth_path.lower().endswith('.png') and C == 1:
        depth *= max_distance                           

    fx = (W/2)/math.tan(cam.data.angle_x/2);  fy = (H/2)/math.tan(cam.data.angle_y/2)
    cx, cy = W/2, H/2
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    z  = depth
    xyz = np.stack([(uu-cx)*z/fx, -(vv-cy)*z/fy, -z], axis=-1).astype(np.float32)

    if as_world:
        R = cam.matrix_world
        flat = xyz.reshape(-1,3)
        xyz  = np.array([R @ Vector(p) for p in flat], np.float32).reshape(H, W, 3)

    bpy.data.images.remove(img)
    
    # rotate 180 degrees and flip around y-axis
    transformed_pc = np.rot90(xyz, 2)
    transformed_pc = transformed_pc[:,::-1, :]
    return transformed_pc


def get_world_bounding_box(obj):
    # Local bounding box corners
    local_bbox_corners = [Vector(corner) for corner in obj.bound_box]
    # Convert to world space
    world_bbox_corners = [obj.matrix_world @ corner for corner in local_bbox_corners]
    return world_bbox_corners


def print_bboxes():
    # Loop through all mesh objects in the scene
    min_corner = Vector((float('inf'), float('inf'), float('inf')))
    max_corner = Vector((float('-inf'), float('-inf'), float('-inf')))
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and 'apple' in obj.name:
            bbox = get_world_bounding_box(obj)
            for corner in bbox:
                min_corner = Vector((min(min_corner.x, corner.x),
                                     min(min_corner.y, corner.y),
                                     min(min_corner.z, corner.z)))
                max_corner = Vector((max(max_corner.x, corner.x),
                                     max(max_corner.y, corner.y),
                                     max(max_corner.z, corner.z)))
            print(f"Object: {obj.name}")
            print(f"  Bounding Box corners: {bbox}")

    print(f"Global Min Corner: {min_corner}")
    print(f"Global Max Corner: {max_corner}")
    
def is_bbox_visible(obj,
                    cam=None,
                    scene=None,
                    margin: float = 0.0) -> bool:
    """
    True if *any* corner of the (evaluated) bounding box falls
    inside the camera’s 2‑D frame and between its near/far clips.

    Parameters
    ----------
    obj : bpy.types.Object
        The object to test (must be of type 'MESH').
    cam : bpy.types.Object, optional
        Camera to test against (defaults to scene.camera).
    scene : bpy.types.Scene, optional
        Scene providing render settings (defaults to bpy.context.scene).
    margin : float
        Grows/Shrinks the 2‑D test rectangle by ±margin in NDC
        (e.g. ‑0.02 lets the function accept boxes a smidge outside).

    Returns
    -------
    bool
    """
    scene = scene or bpy.context.scene
    cam   = cam   or scene.camera
    if cam is None or cam.type != 'CAMERA':
        raise ValueError("No valid camera specified")

    # Include modifiers / constraints
    deps = bpy.context.evaluated_depsgraph_get()
    obj_e = obj.evaluated_get(deps)

    # Clip‑plane distances in camera space
    clip_start, clip_end = cam.data.clip_start, cam.data.clip_end
    m_world_to_cam = cam.matrix_world.inverted()

    for corner in obj_e.bound_box:                      # local → world
        world_co = obj_e.matrix_world @ Vector(corner)
        cam_co   = m_world_to_cam @ world_co            # world → camera
        # test depth first (‑Z forward in camera space)
        depth = -cam_co.z
        if not (clip_start <= depth <= clip_end):
            continue

        ndc = bpy_extras.object_utils.world_to_camera_view(scene, cam, world_co)
        if (margin <= ndc.x <= 1 - margin and
            margin <= ndc.y <= 1 - margin and
            ndc.z >= 0.0):
            return True                                # at least one corner is in‑frame
    return False

def get_visible_apple_boxes(
                       cam=None,
                       scene=None,
                       margin=0.0):
    """
    Returns {object_name: [world‑space bbox corners]} for
    all visible mesh objects whose name contains *mask*.
    """
    scene = scene or bpy.context.scene
    boxes = {}
    for ob in scene.objects:
        if (ob.type == 'MESH' and 'apple' in ob.name and 'stem' not in ob.name and
                is_bbox_visible(ob, cam, scene, margin)):
            boxes[ob.name] = [ob.matrix_world @ Vector(c)
                              for c in ob.bound_box]
    return boxes

def ndc_center_of_bbox(obj, cam=None, scene=None):
    """Return (x_ndc, y_ndc) mid‑point of obj's projected AABB."""
    scene = scene or bpy.context.scene
    cam   = cam   or scene.camera
    deps  = bpy.context.evaluated_depsgraph_get()
    obj_e = obj.evaluated_get(deps)

    # project each corner to NDC ------------------------------
    ndc = [bpy_extras.object_utils.world_to_camera_view(
               scene, cam, obj_e.matrix_world @ Vector(c))
           for c in obj_e.bound_box]

    xs, ys = zip(*[(c.x, c.y) for c in ndc if c.z >= 0.0])
    return (min(xs)+max(xs))*0.5, (min(ys)+max(ys))*0.5


def surface_point_at_bbox_center(obj, cam=None, scene=None, max_dist=1e6):
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

    loc_L, no_L, face_i, dist = bvh.ray_cast(origin_L, dir_L, max_dist)
    if loc_L is None:
        return None                                 # miss -> occluded or off‑fruit

    # 3. back to camera space -------------------------------
    loc_W   = obj_eval.matrix_world @ loc_L
    loc_cam = cam.matrix_world.inverted() @ loc_W
    return loc_W, loc_cam



def get_apple_center(apple_id):
    obj_name = f'apple{apple_id}'
    apple = bpy.data.objects[obj_name]
    cam = bpy.context.scene.camera
    
    # Loop through all objects in the scene
    for obj in bpy.data.objects:
        if obj.name == obj_name:
            obj.hide_render = False  # Keep this apple visible
        else:
            obj.hide_render = True   # Hide everything else from rendering
    res = surface_point_at_bbox_center(apple, cam)
    if res:
        pt_w, pt_cam = res
        print("Surface centre in camera space:", pt_cam)
        print("Surface centre in world space:", pt_w)
        
    else:
        print("Apple is occluded or outside max_dist.")
    for obj in bpy.data.objects:
        if 'test' in obj.name: continue
        obj.hide_render = False  # make all objects visible in render again
if __name__ == "__main__":
    get_apple_center(8)
#    print(get_visible_apple_boxes())
#   print_bboxes()
    stem = "/home/siddhartha/RIVAL/blender/apple_orchard"
    
#    for collection in ['foliage', 'stems'=, 'branches']:
#        col = bpy.data.collections.get(collection)     # replace with your collection name
#        if col:                                     # safety check
#            col.hide_render  = True    # toggle camera icon
#            print("Hiding ", collection)
#        else:
#            print("No collection", collection)


    depth_png, rgb_png = save_depth_rgb_fast(res_x=1280, res_y=720,
                                             max_distance=1.8,
                                             depth_mode='PNG',
                                             path_stem=stem)

    cloud = depth_to_point_cloud(depth_png, max_distance=1.8, as_world=False)
    pc_path = stem + "_pc.npy"
    np.save(pc_path, cloud)
    print("Saved point cloud to ", pc_path)
    print("cloud shape:", cloud.shape)

