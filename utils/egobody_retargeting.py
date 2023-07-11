import numpy as np
import json
import os
import math
import argparse
from typing import Any, Dict, Optional, Sequence, Union
import mathutils
import bpy
import time
import sys
import glob
import json
from scipy.spatial.transform import Rotation


FOCAL_LENGTH = 30
SENSOR_WIDTH = 50
RESULOUTION_X = 960
RESULOUTION_Y = 540

# randomize np seed using time
np.random.seed(int(time.time()))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class Blender_render():
    def __init__(self,
                 scratch_dir=None,
                 partnet_path=None,
                 GSO_path=None,
                 character_path=None,
                 egobody_path=None,
                 camera_path=None,
                 render_engine='BLENDER_EEVEE',
                 seq=None
                 ):
        self.blender_scene = bpy.context.scene
        self.render_engine = render_engine

        self.scratch_dir = scratch_dir

        self.character_path = character_path

        self.egobody_path = egobody_path
        self.seq = seq
        # delete the cube
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

        self.load_assets()
        # save blend file
        os.makedirs(scratch_dir, exist_ok=True)
        absolute_path = os.path.abspath(scratch_dir)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(absolute_path, 'scene.blend'))



    def load_assets(self):
        # load character
        character_list = os.listdir(self.character_path)
        character_list = [c for c in character_list if '.blend' in c]
        character_list = np.random.choice(character_list, 2, replace=False)
        # character_list = ['man3.blend', 'man7.blend']
        print(character_list)
        motion_path = os.path.join(self.egobody_path, 'results_amass')
        motion_list = [os.path.join(motion_path, 'interactee_{}.npz'.format(self.seq)),
                       os.path.join(motion_path, 'interactor_{}.npz'.format(self.seq))]

        for i, character in enumerate(character_list):
            character_collection_path = os.path.join(self.character_path, character, 'Collection')
            bpy.ops.wm.append(directory=character_collection_path, filename=character[:-6])
            skeleton = bpy.data.objects[character[:-6]]

            self.retarget_smplx2skeleton(skeleton, motion_list[i])



    def clear_scene(self):
        for k in bpy.data.objects.keys():
            bpy.data.objects[k].select_set(False)

    def retarget_smplx2skeleton(self, skeleton, motion_path):

        # load smplx motion using smplx addon
        bpy.ops.object.smplx_add_animation(filepath=motion_path)
        smplx = bpy.context.selected_objects[0]
        smplx_skeleton = smplx.parent
        print(smplx_skeleton)
        # deselect all
        bpy.ops.object.select_all(action='DESELECT')
        smplx.select_set(False)
        # select smplx skeleton
        smplx_skeleton.select_set(True)
        bpy.context.view_layer.objects.active = smplx_skeleton


        # go to the pose mode
        bpy.ops.object.mode_set(mode='POSE')

        smplx_skeleton.data.bones['root'].select = True
        bpy.context.object.data.bones.active = smplx_skeleton.data.bones['root']

        # bpy.context.object.rotation_euler[0] = -np.pi / 2
        bpy.context.active_pose_bone.rotation_euler[0] = -np.pi / 2
        # bpy.context.active_pose_bone.keyframe_insert(data_path='rotation_euler', frame=0)
        bpy.ops.object.mode_set(mode='OBJECT')


        # rotate the root for -90 degree around x axis


        bpy.context.scene.rsl_retargeting_armature_source = smplx_skeleton
        bpy.context.scene.rsl_retargeting_armature_target = skeleton

        print('mapping skeleton')
        bpy.ops.rsl.build_bone_list()

        mapping = json.load(open(os.path.join(self.character_path, 'bone_mapping.json')))

        print('mapping skeleton')
        for bone in bpy.context.scene.rsl_retargeting_bone_list:
            if bone.bone_name_source in mapping.keys():
                if mapping[bone.bone_name_source] in skeleton.data.bones.keys():
                    bone.bone_name_target = mapping[bone.bone_name_source]
            else:
                continue
        # retarget motion
        print('retargeting')
        bpy.ops.rsl.retarget_animation()
        print('retargeting done')

        calibration_path = os.path.join('{}/calibrations/{}/cal_trans/kinect12_to_world/'.format(self.egobody_path, self.seq))
        calibration_file = glob.glob(calibration_path + '*.json')[0]
        print(calibration_file)
        Rt = np.array(json.load(open(calibration_file, 'r'))['trans'])
        R = Rt[:3, :3]
        t = Rt[:3, 3]
        print(R, t)
        euler = Rotation.from_matrix(R).as_euler('xyz', degrees=False)

        # convert R to euler angles

        # add transformation to the skeleton
        skeleton.select_set(True)
        bpy.context.view_layer.objects.active = skeleton
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        # transform the skeleton
        skeleton.location = t
        skeleton.rotation_euler = euler
        # apply transformation to skeleton
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        # rotate x by 90 degree
        skeleton.rotation_euler[0] = np.pi / 2


        # add transformation to smplx
        smplx_skeleton.select_set(True)
        bpy.context.view_layer.objects.active = smplx_skeleton
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        # transform the skeleton
        smplx_skeleton.location = t
        smplx_skeleton.rotation_euler = euler
        # apply transformation to skeleton
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        # rotate x by 90 degree
        smplx_skeleton.rotation_euler[0] = np.pi / 2



if __name__ == "__main__":
    import sys

    argv = sys.argv

    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]

    print("argsv:{0}".format(argv))
    parser = argparse.ArgumentParser(description='Render Motion in 3D Environment for HuMoR Generation.')
    parser.add_argument('--scratch_dir', type=str, default='./data/egobody/scenes', help='scratch directory')
    parser.add_argument('--character_path', type=str, default='./data/characters', help='character path')
    parser.add_argument('--egobody_path', type=str, default='./data/egobody', help='egobody path')
    parser.add_argument('--seq', type=str, default='recording_20210918_S05_S06_01', help='sequence path')

    args = parser.parse_args(argv)

    blender_render = Blender_render(scratch_dir=args.scratch_dir,
                                    character_path=args.character_path,
                                    egobody_path=args.egobody_path,
                                    seq=args.seq)
