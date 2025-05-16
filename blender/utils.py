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

if __name__ == "__main__":
    move_sun_random()
    
