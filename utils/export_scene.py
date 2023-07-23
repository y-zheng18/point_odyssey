import bpy

import sys
import argparse
import os
import json

argv = sys.argv

if "--" not in argv:
    argv = []
else:
    argv = argv[argv.index("--") + 1:]

print("argsv:{0}".format(argv))
parser = argparse.ArgumentParser(description='Export obj data')

parser.add_argument('--scene_root', type=str, default='')
parser.add_argument('--output_dir', type=str, metavar='PATH', default='',
                    help='obj save dir')
parser.add_argument('--export_character', type=bool, default=True)
args = parser.parse_args(argv)
print("args:{0}".format(args))

bpy.ops.wm.open_mainfile(filepath=args.scene_root)

collection_set = ['Furniture', 'Wall', 'Floor', 'Ceiling']
assets_keys = []
for collection_name in collection_set:
    if not collection_name in bpy.data.collections:
        continue
    collection = bpy.data.collections[collection_name]
    assets_keys += [obj.name for obj in collection.objects if obj.type == 'MESH' and not obj.hide_render and not 'Fire' in obj.name and not 'Smoke' in obj.name]

scene_info = json.load(open(os.path.join(args.output_dir, 'scene_info.json'), 'r'))
scene_info['assets'] = ['background'] + assets_keys
scene_info['character'] = [s for s in bpy.data.collections.keys() if s not in collection_set and s not in ['Camera', 'Light', 'fog', 'Fog']]
obj_save_dir = os.path.join(args.output_dir, 'obj')
if not os.path.exists(obj_save_dir):
    os.makedirs(obj_save_dir)
print('assets_keys', assets_keys)

# select obj
bpy.ops.object.select_all(action='DESELECT')
for asset in assets_keys:
    # set visible
    bpy.data.objects[asset].hide_viewport = False
    bpy.data.objects[asset].select_set(True)
# export obj
bpy.ops.export_scene.obj(filepath=os.path.join(obj_save_dir, f'scene.obj'),
                         use_selection=True, use_mesh_modifiers=True,
                         use_normals=False, use_uvs=False, use_triangles=False,
                         keep_vertex_order=True, use_materials=False)

# export character
if args.export_character:
    bpy.ops.object.select_all(action='DESELECT')
    character_set = []
    for character in scene_info['character']:
        for obj in bpy.data.collections[character].objects:
            if not obj.type == 'MESH' or obj.hide_render:
                continue
            obj.hide_viewport = False
            obj.select_set(True)
            character_set.append(obj.name)
    frames = range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1)
    for frame_nr in frames:
        bpy.context.scene.frame_set(frame_nr)
        bpy.ops.export_scene.obj(filepath=os.path.join(obj_save_dir, f'character_{frame_nr:04d}.obj'),
                                 use_selection=True, use_mesh_modifiers=True,
                                 use_normals=False, use_uvs=False, use_triangles=False,
                                 keep_vertex_order=True, use_materials=False)


# save json
json.dump(scene_info, open(os.path.join(args.output_dir, 'scene_info.json'), 'w'), indent=4)