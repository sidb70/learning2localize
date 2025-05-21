import bpy, random, math
from mathutils import Vector, Euler

def move_sun_random(min_radius=10.0,
                    max_radius=20.0,
                    hemisphere="upper",
                    lamp_name="Sun"):
    """
    Move a Sun‑type lamp to a random point on a sphere shell of radius
    ∈ [min_radius, max_radius] and aim its –Z axis at the world origin.

    Parameters
    ----------
    min_radius, max_radius : float
        Inclusive range from which the radial distance is sampled.
    hemisphere : {"upper", "lower", "full"}
        Constrain the sun above the horizon (Y > 0), below (Y < 0),
        or anywhere on the sphere.
    lamp_name : str
        Name of the lamp object to move.  If no lamp of that name exists
        a new Sun lamp is created.

    Returns
    -------
    bpy.types.Object
        The lamp that was moved and oriented.
    """
    scn = bpy.context.scene

    # ------------------------------------------------------------------ 1 · get / create the Sun object
    sun = scn.objects.get(lamp_name)
    if sun is None or sun.type != 'LIGHT' or sun.data.type != 'SUN':
        if sun is not None:
            raise ValueError(f"'{lamp_name}' exists but is not a Sun lamp.")
        # create one at world origin
        light_data = bpy.data.lights.new(name=lamp_name, type='SUN')
        sun = bpy.data.objects.new(name=lamp_name, object_data=light_data)
        scn.collection.objects.link(sun)
    print("Found sun")
    # ------------------------------------------------------------------ 2 · sample a random position on shell
    r = random.uniform(min_radius, max_radius)

    # random direction on unit sphere
    theta = random.uniform(0.0, 2*math.pi)       # azimuth
    phi   = math.acos(random.uniform(-1.0, 1.0)) # polar (0..π)

    # optional hemisphere restriction
    if hemisphere == "upper":     # keep positive Z (Blender up‑axis)
        phi = random.uniform(0.0, math.pi/2)
    elif hemisphere == "lower":   # keep negative Z
        phi = random.uniform(math.pi/2, math.pi)

    x = r * math.sin(phi) * math.cos(theta)
    y = r * math.sin(phi) * math.sin(theta)
    z = r * math.cos(phi)
    sun.location = Vector((x, y, z))
    exp = random.uniform(4,8)
    sun.data.energy = 2**exp

    # ------------------------------------------------------------------ 3 · aim –Z axis at the origin
    # Sun points along its local –Z. Build quaternion that makes –Z → (origin - location)
    # aim Sun’s –Z at the origin
    direction = -sun.location.normalized()
    quat      = direction.to_track_quat('-Z', 'Y')   
    sun.rotation_euler = quat.to_euler()

    return sun


def recenter_object_origin(obj):
    """ Move object origin to the center of its mesh geometry. """
    if not obj.data or not hasattr(obj.data, "vertices"):
        return  # not a mesh, or no geometry

    verts = [v.co for v in obj.data.vertices]
    center = sum(verts, Vector()) / len(verts)

    # Move mesh so that center becomes the origin
    for v in obj.data.vertices:
        v.co -= center

    # Then move the object to compensate
    obj.location += obj.matrix_world.to_3x3() @ center
def clone_object_random(source_name: str,
                        name_suffix="_clone",
                        loc_range=((-2, 2), (-2, 2), (0, 2)),
                        rot_range=((-math.pi, math.pi),) * 3):
    """
    Clone an object by name and place it at a random location and orientation.

    Parameters
    ----------
    source_name : str
        Name of the object to duplicate.
    name_prefix : str
        Prefix to add to the new object’s name.
    loc_range : tuple of 3 (min, max) pairs
        Range of random positions in X, Y, Z.
    rot_range : tuple of 3 (min, max) pairs
        Range of random rotations (Euler XYZ) in radians.

    Returns
    -------
    bpy.types.Object
        The duplicated and transformed object.
    """
    src = bpy.data.objects.get(source_name)
    if src is None:
        raise ValueError(f"Object '{source_name}' not found.")

    # Duplicate object and link to current scene
    new_obj = src.copy()
    new_obj.data = src.data.copy() if src.data else None
    new_obj.name = src.name + name_suffix
    recenter_object_origin(new_obj)
    bpy.context.scene.collection.objects.link(new_obj)


    # Random location and rotation
    loc = [random.uniform(*r) for r in loc_range]
    rot = [random.uniform(*r) for r in rot_range]

    new_obj.location = Vector(loc)
    new_obj.rotation_euler = Euler(rot, 'XYZ')

    return new_obj

def move_camera_random(min_xyz=Vector((-1.7732, -1.6281, 0.6314)),
                      max_xyz=Vector((1.5449, 1.5844, 3.4049)),
                      look_at_target=None,
                      camera_name="Camera"):
    # Move a camera to a random point within a bounding box defined by min_xyz and max_xyz
    # coordinates, and aim it at a target apple object.
    # Also randomizes the distance from the camera to the target within a range of 0.6 to 2 meters.
    
    # Parameters
    # ----------
    # min_xyz : Vector
    #     Minimum XYZ coordinates defining the bounding box.
    # max_xyz : Vector
    #     Maximum XYZ coordinates defining the bounding box.
    # look_at_target : Vector or str or bpy.types.Object
    #     Target to point the camera at. Can be:
    #     - Vector coordinates of an apple
    #     - Name of an apple object in the scene
    #     - The apple object itself
    #     If None, the camera's rotation will remain unchanged.
    # camera_name : str
    #     Name of the camera object to move. If no camera of that name exists,
    #     a new camera is created.
    
    # Returns
    # -------
    # bpy.types.Object
    #     The camera that was moved and oriented.
    
    scn = bpy.context.scene
    
    # ---------------------------------------------------- 1 : get / create the Camera object
    camera = scn.objects.get(camera_name)
    if camera is None or camera.type != 'CAMERA':
        if camera is not None:
            raise ValueError(f"'{camera_name}' exists but is not a Camera.")
        # Create a new camera in the world origin
        camera_data = bpy.data.cameras.new(name=camera_name)
        camera = bpy.data.objects.new(name=camera_name, object_data=camera_data)
        scn.collection.objects.link(camera)
        print(f"Created new camera named '{camera_name}'")
    else:
        print(f"Found existing camera named '{camera_name}'")
    
    # ---------------------------------------------------- 2 : sample a random position within bounding box
    x = random.uniform(min_xyz.x, max_xyz.x)
    y = random.uniform(min_xyz.y, max_xyz.y)
    z = random.uniform(min_xyz.z, max_xyz.z)
    camera.location = Vector((x, y, z))
    
    # ---------------------------------------------------- 3 : aim camera at the target if provided
    if look_at_target is not None:
        target_location = None
        
        # handle different types of target input
        if isinstance(look_at_target, Vector):
            # If a Vector is provided, use it directly
            target_location = look_at_target
        elif isinstance(look_at_target, str):
            # If a string is provided, look for an object with that name
            target_obj = scn.objects.get(look_at_target)
            if target_obj is not None:
                target_location = target_obj.location
            else:
                print(f"Warning: Target object '{look_at_target}' not found. Camera rotation unchanged.")
        elif hasattr(look_at_target, "location"):
            # If an object with a location attribute is provided (like a Blender object)
            target_location = look_at_target.location
        
        if target_location is not None:
            # Calculate direction from camera to target
            look_dir = (target_location - camera.location).normalized()
            
            # -------------------------------------------- 4 : randomize distance from target (0.6 to 2 meters)
            random_distance = random.uniform(0.6, 2.0)
            # Adjust camera position to be at the random distance from target
            camera.location = target_location - look_dir * random_distance
            
            # Recalculate direction after position change
            look_dir = (target_location - camera.location).normalized()
            
            # Look down the -Z axis at the target, with Y up
            rot_quat = look_dir.to_track_quat('-Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()
            print(f"Camera positioned at {camera.location}, and aimed at {target_location}")
        else:
            print(f"Camera positioned at {camera.location}, rotation unchanged")
    else:
        print(f"Camera positioned at {camera.location}, rotation unchanged")
    
    # Optional: Set as active camera
    scn.camera = camera
    
    return camera

    """
    Example usage:
    move_camera_random(look_at_target="Apple_M01") # Look at an apple by name
    or
    apple = bpy.data.objects.get("Apple_M01")
    move_camera_random(look_at_target=apple) # Look at an apple object
    or
    apple_location = Vector((0.5, 0.3, 1.2))
    move_camera_random(look_at_target=apple_location) # Look at apple coordinates
    """

def position_camera():
    """
    Position a camera randomly around an imaginary ellipse.
    
    The ellipse is centered at (0, 0, 2) with dimensions:
    - x = 3
    - y = 3
    - z = 2.25
    
    The camera is positioned 0 to .5 meters away from the ellipse surface
    and always points at the ellipse center.
    """
    # Get the camera object (or create one if it doesn't exist)
    if 'Camera' in bpy.data.objects:
        camera = bpy.data.objects['Camera']
    else:
        camera_data = bpy.data.cameras.new(name='Camera')
        camera = bpy.data.objects.new('Camera', camera_data)
        bpy.context.collection.objects.link(camera)
    
    # Ellipse properties
    ellipse_center = Vector((0, 0, 2))
    ellipse_dimensions = Vector((3, 3, 2.25))
    
    # Generate a random point on a unit sphere
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0, math.pi)
    
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)
    
    # Convert the unit sphere point to ellipsoid surface
    point_on_ellipsoid = Vector((
        x * ellipse_dimensions.x,
        y * ellipse_dimensions.y,
        z * ellipse_dimensions.z
    ))
    
    # Calculate the distance from the center to this point on the ellipsoid
    ellipsoid_radius_in_this_direction = point_on_ellipsoid.length
    
    # Normalize the direction vector
    direction = point_on_ellipsoid.normalized()
    
    # Calculate a random distance from the ellipsoid (0 to .5 meters)
    distance_from_ellipsoid = random.uniform(0, .5)
    
    # Calculate the total distance from the center
    total_distance = ellipsoid_radius_in_this_direction + distance_from_ellipsoid
    
    # Calculate the final camera position
    camera_position = ellipse_center + direction * total_distance
    
    # Set camera position
    camera.location = camera_position
    
    # Point the camera at the ellipse center
    direction_to_center = ellipse_center - camera_position
    
    # Convert the direction to rotation (point the camera at the ellipse center)
    # The camera's -Z axis should point toward the target, and the Y axis should be up
    rot_quat = direction_to_center.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    
    # Update the scene
    bpy.context.view_layer.update()
    
    return camera


if __name__ == "__main__":
    move_sun_random()
    
