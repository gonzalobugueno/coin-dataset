##
# This file will render the dataset
# The keys inside param_grid dict are the hyperparameters
# Tune at will but be mindful, each new addition will grow the size of the dataset exponentially
##
import bpy
import itertools
import math
from mathutils import Vector
import os
import random
import json

param_grid = {
    'coin_x_metres': [-0.5, -0.25, 0, 0.25, 0.5],
    'coin_y_metres': [-0.1, 0, 0.25, 0.5],
    'coin_yaw': [math.radians(i) for i in range(0, 360, 45)],
    'sun_angle': [math.radians(0.53)], # 0.53 degrees
    'sun_energy': [5,10],
    'sun_rotation': [(15, 90), (75, 180), (15, 270)], # elevation, azimuth,
    'render_samples': [1],
    'res_factor' : [2] # for 1080p, 2k, etc
}

def grid_search(params):
    keys = list(params.keys())
    for values in itertools.product(*(params[key] for key in keys)):
        yield dict(zip(keys, values))


def get_object_render_bbox(obj):
    """
    Returns the top-left and bottom-right pixel coordinates of an object's bounding box
    in the final render.

    Args:
        obj (bpy.types.Object): The object to calculate bounds for

    Returns:
        tuple: (top_left_x, top_left_y, bottom_right_x, bottom_right_y) in pixel coordinates
               or None if object is not visible
    """
    # Get render dimensions
    render = bpy.context.scene.render
    scale = render.resolution_percentage / 100
    width = int(render.resolution_x * scale)
    height = int(render.resolution_y * scale)

    # Get camera and matrices
    camera = bpy.context.scene.camera
    view_matrix = camera.matrix_world.inverted()
    projection_matrix = camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(),
        x=width,
        y=height
    )

    # Initialize min/max values
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = -float('inf'), -float('inf')

    # Process each corner of the bounding box
    for corner in obj.bound_box:
        # Convert to world space
        world_corner = obj.matrix_world @ Vector(corner)

        # Convert to camera space
        camera_corner = view_matrix @ world_corner

        # Project to clip space
        clip_corner = projection_matrix @ camera_corner.to_4d()

        # Skip points behind camera
        if clip_corner.w <= 0:
            continue

        # Perspective division
        ndc = clip_corner / clip_corner.w

        # Convert to pixel coordinates
        # Note: In render space, y=0 is at the bottom
        x = (ndc.x * 0.5 + 0.5) * width
        y = (1 - (ndc.y * 0.5 + 0.5)) * height  # Flip Y to match top-left origin

        # Update bounds
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    # Return None if no valid points
    if min_x == float('inf'):
        return None

    return (min_x, min_y, max_x, max_y)


def box_to_yolo(x1, y1, x2, y2, img_width, img_height):
    """
    Converts absolute box coordinates (x1, y1, x2, y2) to normalized YOLO format.
    """
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height



bpy.ops.wm.open_mainfile(filepath="my.blend")

# write metadata

def thedirs(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

    assert os.path.isdir(path), f'couldnt create {path} directory'

thedirs('ds')
thedirs('ds/images')
thedirs('ds/labels')
thedirs('ds/meta')
thedirs('ds/images/test')
thedirs('ds/images/train')
thedirs('ds/labels/test')
thedirs('ds/labels/train')
thedirs('ds/meta/test')
thedirs('ds/meta/train')

with open('ds/dataset.yml', 'w') as m:
    m.write(f"""\
path: {os.getcwd()}/ds
train: images/train
test: images/test
names:
  0: 50cent
""")



for idx, grid in enumerate(grid_search(param_grid)):

    if random.random() < 0.2:
        destiny = 'train'
    else:
        destiny = 'test'

    img_output_path = f'ds/images/{destiny}/{idx}.png'
    label_output_path = f'ds/labels/{destiny}/{idx}.txt'
    meta_output_path = f'ds/meta/{destiny}/{idx}.json'

    with open(meta_output_path, 'w') as meta:
        json.dump(grid, meta, indent=4)

    print("Output: ", img_output_path)
    print("Rendering for settings", grid)

    obj = bpy.data.objects["50cent"]
    obj.location.x = grid['coin_x_metres']
    obj.location.y = grid['coin_y_metres']
    obj.rotation_euler[2] = grid['coin_yaw']  # Z-axis yaw

    sun = bpy.context.scene.objects["Sun"]
    elevation, azimuth = grid['sun_rotation']
    sun.rotation_euler = (elevation,0, azimuth)

    sun.data.angle = grid['sun_angle']
    sun.data.energy = grid['sun_energy']

    bpy.context.scene.camera = bpy.data.objects["Camera"]


    w = 1920 * grid['res_factor']
    h = 1080 * grid['res_factor']

    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.resolution_x = 1920 * grid['res_factor']
    bpy.context.scene.render.resolution_y = 1080 * grid['res_factor']
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = img_output_path

    # Disable anti-aliasing in EEVEE
    bpy.context.scene.eevee.taa_render_samples = 1

    # For Cycles (disable adaptive sampling and set fixed samples)
    bpy.context.scene.cycles.use_adaptive_sampling = False
    bpy.context.scene.cycles.samples = 1


    bpy.ops.render.render(write_still=True)

    with open(label_output_path, 'a') as l:
        #ulx, uly, lrx, lry = get_object_render_bbox(obj)
        x_center, y_center, width, height = box_to_yolo(*get_object_render_bbox(obj), w, h)
        l.write(f'0 {x_center} {y_center} {width} {height}')