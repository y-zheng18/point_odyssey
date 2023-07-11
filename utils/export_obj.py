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

parser.add_argument('--scene_root', type=str,
                    default='/Users/yangzheng/Documents/Blender/Assets/scene/hdri_plane.blend')
parser.add_argument('--output_dir', type=str, metavar='PATH', default='./',
                    help='img save dir')
args = parser.parse_args(argv)
print("args:{0}".format(args))

bpy.ops.wm.open_mainfile(filepath=args.scene_root)
frames = range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1)

assets_keys = bpy.data.objects.keys()
assets_keys = [key for key in assets_keys if
               bpy.data.objects[key].type == 'MESH' and key != 'Plane' and 'Smoke' not in key and not bpy.data.objects[
                   key].hide_render]
scene_info = json.load(open(os.path.join(args.output_dir, 'scene_info.json'), 'r'))
scene_info['assets'] = ['background'] + assets_keys
obj_save_dir = os.path.join(args.output_dir, 'obj')
if not os.path.exists(obj_save_dir):
    os.makedirs(obj_save_dir)
print('assets_keys', assets_keys)
for frame_nr in frames:
    bpy.context.scene.frame_set(frame_nr)
    for asset in assets_keys:
        bpy.data.objects[asset].select_set(True)
        save_asset_name = asset.replace('.', '_')
        bpy.ops.export_scene.obj(filepath=os.path.join(obj_save_dir, f'{save_asset_name}_{frame_nr:04d}.obj'),
                                 use_selection=True, use_mesh_modifiers=True,
                                 use_normals=False, use_uvs=False, use_triangles=False,
                                 keep_vertex_order=True, use_materials=False)
        bpy.data.objects[asset].select_set(False)
# save json
json.dump(scene_info, open(os.path.join(args.output_dir, 'scene_info.json'), 'w'), indent=4)
