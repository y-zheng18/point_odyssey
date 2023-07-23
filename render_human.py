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
                 use_character=None,
                 motion_path=None,
                 camera_path=None,
                 render_engine='BLENDER_EEVEE',
                 adaptive_sampling=False,
                 use_denoising=True,
                 samples_per_pixel=128,
                 num_assets=2,
                 background_hdr_path=None,
                 custom_scene=None,
                 use_gpu: bool = False,
                 use_indoor_cam: bool = False,
                 add_force: bool = False,
                 force_step: int = 3,
                 force_num: int = 3,
                 force_interval: int = 200
                 ):
        self.blender_scene = bpy.context.scene
        self.render_engine = render_engine
        self.use_gpu = use_gpu
        self.scale_factor = 10 if not use_indoor_cam else 1

        self.set_render_engine()
        hdr_list = os.listdir(background_hdr_path)
        hdr_list = [os.path.join(background_hdr_path, x) for x in hdr_list if '.hdr' in x or '.exr' in x]
        self.scratch_dir = scratch_dir
        self.background_hdr_path = np.random.choice(hdr_list)
        self.GSO_path = GSO_path
        self.partnet_path = partnet_path
        self.character_path = character_path
        self.use_character = use_character
        self.motion_path = motion_path
        self.GSO_path = GSO_path
        self.camera_path = camera_path
        self.motion_dataset = ['CMU'] # ['TotalCapture', 'DanceDB', 'CMU', 'MoSh', 'SFU']
        self.motion_speed = {'TotalCapture': 1 / 1.5, 'DanceDB': 1.0, 'CMU': 1.0, 'MoSh': 1.0 / 1.2, 'SFU': 1.0 / 1.2}

        self.force_step = force_step
        self.force_num = force_num
        self.force_interval = force_interval
        self.add_force = add_force

        self.num_assets = num_assets
        # set scene
        assert custom_scene is not None
        print("Loading scene from '%s'" % custom_scene)
        bpy.ops.wm.open_mainfile(filepath=custom_scene)

        self.obj_set = set(bpy.context.scene.objects)
        self.assets_set = []
        self.gso_force = []
        self.setup_scene()

        self.activate_render_passes(normal=True, optical_flow=True, segmentation=True, uv=True)

        self.adaptive_sampling = adaptive_sampling  # speeds up rendering
        self.use_denoising = use_denoising  # improves the output quality
        self.samples_per_pixel = samples_per_pixel

        self.exr_output_node = self.set_up_exr_output_node()

        # self.blender_scene.render.resolution_percentage = 100
        if background_hdr_path:
            print('loading hdr from:', self.background_hdr_path)
            self.load_background_hdr(self.background_hdr_path)

        # save blend file
        os.makedirs(scratch_dir, exist_ok=True)
        absolute_path = os.path.abspath(scratch_dir)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(absolute_path, 'scene.blend'))

    def set_render_engine(self):
        bpy.context.scene.render.engine = self.render_engine
        print("Using render engine: {}".format(self.render_engine))
        if self.use_gpu:
            print("----------------------------------------------")
            print('setting up gpu ......')

            bpy.context.scene.cycles.device = "GPU"
            for scene in bpy.data.scenes:
                print(scene.name)
                scene.cycles.device = 'GPU'

            # if cuda arch use cuda, else use metal
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)

            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            for d in bpy.context.preferences.addons["cycles"].preferences.devices:
                d.use = True
                print("Device '{}' type {} : {}".format(d.name, d.type, d.use))
            print('setting up gpu done')
            print("----------------------------------------------")

    def setup_scene(self):
        bpy.ops.object.camera_add()
        self.camera = bpy.data.objects["Camera"]

        # adjust gravity
        bpy.context.scene.gravity *= self.scale_factor

        # setup camera
        self.cam_loc = mathutils.Vector((np.random.uniform(-3, -3.5) * np.random.choice((-1, 1)), np.random.uniform(-3, -3.5) * np.random.choice((-1, 1)),
                                    np.random.uniform(1, 2.5))) * self.scale_factor
        self.cam_lookat = mathutils.Vector((0, 0, 0.5)) * self.scale_factor
        self.set_cam(self.cam_loc, self.cam_lookat)
        self.camera.data.lens = FOCAL_LENGTH
        self.camera.data.clip_end = 10000
        self.camera.data.sensor_width = SENSOR_WIDTH

        # setup scene
        bpy.context.scene.render.resolution_x = RESULOUTION_X
        bpy.context.scene.render.resolution_y = RESULOUTION_Y
        bpy.context.scene.render.resolution_percentage = 100
        # setup render sampling
        bpy.context.scene.cycles.samples = 64
        # setup framerate
        bpy.context.scene.render.fps = 30

        # scale boundingbox object
        if 'Cube' in bpy.data.objects.keys():
            bpy.data.objects['Cube'].location *= self.scale_factor
            bpy.data.objects['Cube'].scale *= self.scale_factor
            # apply scale
            bpy.context.view_layer.objects.active = bpy.data.objects['Cube']
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # setup area light
        bpy.ops.object.light_add(type='AREA', align='WORLD',
                                 location=mathutils.Vector((np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(4, 5))) * self.scale_factor)
        self.light = bpy.data.objects["Area"]
        self.light.data.energy = 1000 * self.scale_factor

        # add camera to scene
        bpy.context.scene.camera = self.camera

        # disable gravity
        # bpy.context.scene.gravity = (0, 0, 0)

        # load assets
        self.load_assets()


    def set_cam(self, cam_loc, point):
        self.camera.location = self.cam_loc
        direction = point - cam_loc
        rot_quat = direction.to_track_quat('-Z', 'Y')
        self.camera.rotation_euler = rot_quat.to_euler()

    def activate_render_passes(self,
            normal: bool = True,
            optical_flow: bool = True,
            segmentation: bool = True,
            uv: bool = True,
            depth: bool = True):
            # We use two separate view layers
            # 1) the default view layer renders the image and uses many samples per pixel
            # 2) the aux view layer uses only 1 sample per pixel to avoid anti-aliasing

            # Starting in Blender 3.0 the depth-pass must be activated separately
        if depth:
            default_view_layer = bpy.context.scene.view_layers[0]
            default_view_layer.use_pass_z = True

        aux_view_layer = bpy.context.scene.view_layers.new("AuxOutputs")
        aux_view_layer.samples = 1  # only use 1 ray per pixel to disable anti-aliasing
        aux_view_layer.use_pass_z = False  # no need for a separate z-pass
        if hasattr(aux_view_layer, 'aovs'):
            object_coords_aov = aux_view_layer.aovs.add()
        else:
            # seems that some versions of blender use this form instead
            object_coords_aov = aux_view_layer.cycles.aovs.add()

        object_coords_aov.name = "ObjectCoordinates"
        aux_view_layer.cycles.use_denoising = False

        # For optical flow, uv, and normals we use the aux view layer
        aux_view_layer.use_pass_vector = optical_flow
        aux_view_layer.use_pass_uv = uv
        aux_view_layer.use_pass_normal = normal  # surface normals
        # We use the default view layer for segmentation, so that we can get
        # anti-aliased crypto-matte
        if bpy.app.version >= (2, 93, 0):
            aux_view_layer.use_pass_cryptomatte_object = segmentation
            if segmentation:
                aux_view_layer.pass_cryptomatte_depth = 2
        else:
            aux_view_layer.cycles.use_pass_crypto_object = segmentation
            if segmentation:
                aux_view_layer.cycles.pass_crypto_depth = 2

    def load_assets(self):
        # load character
        character_list = os.listdir(self.character_path)
        character_list = [c for c in character_list if os.path.isdir(os.path.join(self.character_path, c))]
        character = np.random.choice(character_list) if self.use_character is None else self.use_character
        self.character_name = character
        character_collection_path = os.path.join(self.character_path, character, '{}.blend'.format(character),
                                                 'Collection')

        bpy.ops.wm.append(directory=character_collection_path, filename=character)
        character_collection = bpy.data.collections[character]
        self.character = character_collection
        self.skeleton = bpy.data.objects['skeleton_' + character]

        bone_mapping_path = os.path.join(self.character_path, character, 'bone_mapping.json')
        bone_mapping = json.load(open(bone_mapping_path, 'r'))
        # add visible mesh objs in character collection to assets_set
        for obj in character_collection.objects:
            if not obj.hide_render and obj.name in bpy.data.objects.keys() and bpy.data.objects[obj.name].type == 'MESH':
                self.assets_set.append(obj)

        self.retarget_smplx2skeleton(bone_mapping)

        bpy.ops.object.select_all(action='DESELECT')
        # select all objects in the collection
        for obj in character_collection.objects:
            obj.select_set(True)
        # add scale
        bpy.ops.transform.resize(value=(self.scale_factor * 1.2, self.scale_factor * 1.2, self.scale_factor * 1.2),
                                 orient_type='GLOBAL',
                                 orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH',
                                 proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False)

        # add objects
        GSO_assets = os.listdir(self.GSO_path)
        GSO_assets = [os.path.join(self.GSO_path, asset) for asset in GSO_assets]
        GSO_assets = [asset for asset in GSO_assets if os.path.isdir(asset)]
        GSO_assets_path = np.random.choice(GSO_assets, size=self.num_assets // 2, replace=False)
        print(GSO_assets_path, self.GSO_path)

        partnet_assets = os.listdir(self.partnet_path)
        partnet_assets = [os.path.join(self.partnet_path, asset) for asset in partnet_assets]
        partnet_assets = [asset for asset in partnet_assets if os.path.isdir(asset) and len(os.listdir(os.path.join(asset, 'objs'))) < 15]
        partnet_assets = np.random.choice(partnet_assets, size=self.num_assets - len(GSO_assets_path), replace=False)

        # generating location lists for assets, and remove the center area
        location_list = np.random.uniform(np.array([-2.5, -2.5, 0.8]), np.array([-1, -1, 2]), size=(self.num_assets * 50, 3)) * self.scale_factor
        location_list = location_list * np.sign(np.random.uniform(-1, 1, size=(self.num_assets * 50, 3)))
        location_list[:, 2] = np.abs(location_list[:, 2])
        location_list = self.farthest_point_sampling(location_list, self.num_assets + 1)
        for i, asset_path in enumerate(GSO_assets_path):
            bpy.ops.import_scene.obj(filepath=os.path.join(asset_path, 'meshes', 'model.obj'))
            imported_object = bpy.context.selected_objects[0]
            self.assets_set.append(imported_object)
            self.load_asset_texture(imported_object, mat_name=imported_object.data.name+'mat',
                                    texture_path=os.path.join(asset_path, 'materials', 'textures', 'texture.png'))
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            # randomize location and translation
            imported_object.location = location_list[i]
            imported_object.rotation_euler = (np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi))

            # set scale
            dimension = np.max(imported_object.dimensions)
            scale = np.random.uniform(1, 6) * self.scale_factor
            if scale * dimension > 0.8 * self.scale_factor:         # max 0.8m
                scale = 0.8 * self.scale_factor / dimension
            elif scale * dimension < 0.1 * self.scale_factor:
                scale = 0.1 * self.scale_factor / dimension
            imported_object.scale = (scale, scale, scale)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


            # set obj active
            bpy.context.view_layer.objects.active = imported_object
            # add rigid body
            bpy.ops.rigidbody.object_add()
            imported_object.rigid_body.type = 'ACTIVE'
            # imported_object.rigid_body.collision_shape = 'MESH'
            imported_object.rigid_body.collision_shape = 'CONVEX_HULL'

            imported_object.rigid_body.mass = 0.5 * scale / self.scale_factor
            # bpy.ops.object.modifier_add(type='COLLISION')
        print('GSO assets loaded')
        print('loading partnet assets')
        print(partnet_assets)
        for j, obj_path in enumerate(partnet_assets):
            parts = os.listdir(os.path.join(obj_path, 'objs'))
            part_objs = []
            for p in parts:
                if not 'obj' in p:
                    continue
                bpy.ops.import_scene.obj(filepath=os.path.join(obj_path, 'objs', p))
                imported_object = bpy.context.selected_objects[0]
                part_objs.append(imported_object)

                # unwrap obj
                bpy.ops.object.select_all(action='DESELECT')
                imported_object.select_set(True)
                bpy.context.view_layer.objects.active = imported_object
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                # smart uv project the entire object
                bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)
                # finish the edit mode
                bpy.ops.mesh.select_all(action='DESELECT')
                bpy.ops.object.mode_set(mode='OBJECT')

                # load random texture from gso
                gso_random_index = np.random.choice(range(len(GSO_assets)))
                self.load_asset_texture(imported_object, mat_name=imported_object.data.name+'mat',
                                        texture_path=os.path.join(GSO_assets[gso_random_index], 'materials', 'textures', 'texture.png'))
            # merge parts into one obj
            bpy.ops.object.select_all(action='DESELECT')
            for part in part_objs:
                part.select_set(True)
            bpy.ops.object.join()
            imported_object = bpy.context.selected_objects[0]
            # randomize location and translation
            imported_object.location = location_list[len(GSO_assets_path) + j]
            imported_object.rotation_euler = (np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi))
            # random scale
            dimension = np.max(imported_object.dimensions)
            scale = np.random.uniform(1, 6) * self.scale_factor
            if scale * dimension > 0.8 * self.scale_factor:
                scale = 0.8 * self.scale_factor / dimension
            elif scale * dimension < 0.1 * self.scale_factor:
                scale = 0.1 * self.scale_factor / dimension
            imported_object.scale = (scale, scale, scale)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

            # set obj active
            bpy.context.view_layer.objects.active = imported_object
            # add rigid body
            bpy.ops.rigidbody.object_add()
            imported_object.rigid_body.type = 'ACTIVE'
            imported_object.rigid_body.collision_shape = 'CONVEX_HULL'
            imported_object.rigid_body.mass = 1 * scale / self.scale_factor
            self.assets_set.append(imported_object)

        # add force
        if self.add_force:
            for i in range(self.force_num):
                dxyz = np.random.uniform(-4, 4, size=3) * self.scale_factor
                dxyz[2] = -abs(dxyz[2]) * 5
                bpy.ops.object.empty_add(type='PLAIN_AXES', location=dxyz)
                obj_axis = bpy.context.selected_objects[0]
                self.gso_force.append(obj_axis)
                # add force filed to axis
                bpy.ops.object.forcefield_toggle()
                bpy.context.object.field.shape = 'POINT'
                bpy.context.object.field.type = 'FORCE'
                # set min and max range
                bpy.context.object.field.use_min_distance = True
                bpy.context.object.field.use_max_distance = True
                bpy.context.object.field.distance_max = 1000
                bpy.context.object.field.strength = np.random.uniform(1000, 200)
        print('len of assets_set:', len(self.assets_set))
        print('len of forces:', len(self.gso_force))

    @staticmethod
    def farthest_point_sampling(p, K):
        """
        greedy farthest point sampling
        p: point cloud
        K: number of points to sample
        """

        farthest_point = np.zeros((K, 3))
        max_idx = np.random.randint(0, p.shape[0] -1)
        farthest_point[0] = p[max_idx]
        for i in range(1, K):
            pairwise_distance = np.linalg.norm(p[:, None, :] - farthest_point[None, :i, :], axis=2)
            distance = np.min(pairwise_distance, axis=1, keepdims=True)
            max_idx = np.argmax(distance)
            farthest_point[i] = p[max_idx]
        return farthest_point

    def load_background_hdr(self, background_hdr_path):
        world = bpy.context.scene.world

        node_env = world.node_tree.nodes['Environment Texture']
        node_env.image = bpy.data.images.load(background_hdr_path) # bpy.data.images[os.path.basename(background_hdr_path)]

    def load_asset_texture(self, obj, mat_name, texture_path, normal_path=None, roughness_path=None):
        mat = bpy.data.materials.new(name=mat_name)

        mat.use_nodes = True

        mat_nodes = mat.node_tree.nodes
        mat_links = mat.node_tree.links

        img_tex_node = mat_nodes.new(type='ShaderNodeTexImage')
        img_tex_node.image = bpy.data.images.load(texture_path)

        mat_links.new(img_tex_node.outputs['Color'], mat_nodes['Principled BSDF'].inputs['Base Color'])

        if normal_path:
            diffuse_tex_node = mat_nodes.new(type='ShaderNodeTexImage')
            diffuse_tex_node.image = bpy.data.images.load(normal_path)
            img_name = diffuse_tex_node.image.name
            bpy.data.images[img_name].colorspace_settings.name = 'Raw'
            mat_links.new(diffuse_tex_node.outputs['Color'], mat_nodes['Principled BSDF'].inputs['Normal'])
        if roughness_path:
            roughness_tex_node = mat_nodes.new(type='ShaderNodeTexImage')
            roughness_tex_node.image = bpy.data.images.load(roughness_path)
            bpy.data.images[roughness_tex_node.image.name].colorspace_settings.name = 'Raw'
            mat_links.new(roughness_tex_node.outputs['Color'], mat_nodes['Principled BSDF'].inputs['Roughness'])

        # clear all materials
        obj.data.materials.clear()

        # assign to 1st material slot
        obj.data.materials.append(mat)

    def set_up_exr_output_node(self):
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links

        # set exr output
        default_layers = ("Image", "Depth")
        aux_layers = ("UV", "Normal", "CryptoObject00", "ObjectCoordinates")

        # clear existing nodes
        for node in tree.nodes:
            tree.nodes.remove(node)

        # the render node has outputs for all the rendered layers
        render_node = tree.nodes.new(type="CompositorNodeRLayers")
        render_node_aux = tree.nodes.new(type="CompositorNodeRLayers")
        render_node_aux.name = "Render Layers Aux"
        render_node_aux.layer = "AuxOutputs"

        # create a new FileOutput node
        out_node = tree.nodes.new(type="CompositorNodeOutputFile")
        # set the format to EXR (multilayer)
        out_node.format.file_format = "OPEN_EXR_MULTILAYER"

        out_node.file_slots.clear()
        for layer_name in default_layers:
            out_node.file_slots.new(layer_name)
            links.new(render_node.outputs.get(layer_name), out_node.inputs.get(layer_name))

        for layer_name in aux_layers:
            out_node.file_slots.new(layer_name)
            links.new(render_node_aux.outputs.get(layer_name), out_node.inputs.get(layer_name))

        # manually convert to RGBA. See:
        # https://blender.stackexchange.com/questions/175621/incorrect-vector-pass-output-no-alpha-zero-values/175646#175646
        split_rgba = tree.nodes.new(type="CompositorNodeSepRGBA")
        combine_rgba = tree.nodes.new(type="CompositorNodeCombRGBA")
        for channel in "RGBA":
            links.new(split_rgba.outputs.get(channel), combine_rgba.inputs.get(channel))
        out_node.file_slots.new("Vector")
        links.new(render_node_aux.outputs.get("Vector"), split_rgba.inputs.get("Image"))
        links.new(combine_rgba.outputs.get("Image"), out_node.inputs.get("Vector"))
        return out_node

    def set_exr_output_path(self, path_prefix: Optional[str]):
        """Set the target path prefix for EXR output.

        The final filename for a frame will be "{path_prefix}{frame_nr:04d}.exr".
        If path_prefix is None then EXR output is disabled.
        """
        if path_prefix is None:
            self.exr_output_node.mute = True
        else:
            self.exr_output_node.mute = False
            self.exr_output_node.base_path = str(path_prefix)

    def clear_scene(self):
        for k in bpy.data.objects.keys():
            bpy.data.objects[k].select_set(False)

    def retarget_smplx2skeleton(self, mapping):
        # get source motion
        motion_dataset = np.random.choice(self.motion_dataset)
        # find all the npz file in the folder recursively
        motion_files = glob.glob(os.path.join(self.motion_path, motion_dataset, '**/*.npz'), recursive=True)
        motion_files = [f for f in motion_files]
        # filter out too small motion
        motion_files = [f for f in motion_files if os.path.getsize(f) > 2e7]
        motion = np.random.choice(motion_files)

        # load smplx motion using smplx addon
        bpy.ops.object.smplx_add_animation(filepath=motion)
        smplx = bpy.context.selected_objects[0]
        smplx_skeleton = smplx.parent

        # slow down the motion
        speed_scale = self.motion_speed[motion_dataset]
        bpy.context.scene.frame_end = int(bpy.context.scene.frame_end / speed_scale)
        for fcurve in smplx_skeleton.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.co[0] = keyframe.co[0] / speed_scale


        bpy.context.scene.rsl_retargeting_armature_source = smplx_skeleton
        bpy.context.scene.rsl_retargeting_armature_target = self.skeleton
        bpy.ops.rsl.build_bone_list()

        print('mapping skeleton')
        for bone in bpy.context.scene.rsl_retargeting_bone_list:
            if bone.bone_name_source in mapping.keys():
                if mapping[bone.bone_name_source] in self.skeleton.data.bones.keys():
                    bone.bone_name_target = mapping[bone.bone_name_source]
            else:
                bone.bone_name_target = ''

        # retarget motion
        print('retargeting')
        bpy.ops.rsl.retarget_animation()
        print('retargeting done')

        # delete smplx
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if 'SMPLX' in obj.name:
                obj.select_set(True)
        smplx.select_set(True)
        bpy.ops.object.delete()

        # post processing the motion to make it stand on the ground without interpenetration
        bpy.ops.object.select_all(action='DESELECT')
        # find root bone
        root_bone = None
        for bone in self.skeleton.data.bones:
            if bone.parent is None:
                root_bone = bone
                break
        z_pre = 0
        for f in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
            bpy.context.scene.frame_set(f)
            # get min z of the character
            print('resolve interpenetration: frame {}'.format(f))
            min_z = 0
            for character_part in self.assets_set:
                character_part.select_set(True)
                bounding_box = [character_part.matrix_world @ mathutils.Vector(corner) for corner in character_part.bound_box]
                min_z = min(min_z, min([v[2] for v in bounding_box]))
                character_part.select_set(False)
            delta_z = z_pre
            if min_z < 0:
                delta_z = min_z
                z_pre = min_z

            bpy.ops.object.select_all(action='DESELECT')
            self.skeleton.select_set(True)
            bpy.ops.object.mode_set(mode='POSE')
            root_bone.select = True
            bpy.context.object.data.bones.active = root_bone
            bpy.context.active_pose_bone.location[1] -= delta_z
            # replace keyframe
            bpy.context.active_pose_bone.keyframe_insert(data_path='location', frame=f)
            bpy.ops.object.mode_set(mode='OBJECT')
        self.skeleton.select_set(False)



    @staticmethod
    def bake_to_keyframes(frame_start, frame_end, step):
        bake = []
        objects = []
        context = bpy.context
        scene = bpy.context.scene
        frame_orig = scene.frame_current
        frames_step = range(frame_start, frame_end + 1, step)
        frames_full = range(frame_start, frame_end + 1)

        # filter objects selection
        for obj in context.selected_objects:
            if not obj.rigid_body or obj.rigid_body.type != 'ACTIVE':
                obj.select_set(False)

        objects = context.selected_objects

        if objects:
            # store transformation data
            # need to start at scene start frame so simulation is run from the beginning
            for f in frames_full:
                scene.frame_set(f)
                print('saving transform data for frame ', f)
                if f in frames_step:
                    mat = {}
                    for i, obj in enumerate(objects):
                        mat[i] = obj.matrix_world.copy()
                    bake.append(mat)

            # apply transformations as keyframes
            for i, f in enumerate(frames_step):
                scene.frame_set(f)
                for j, obj in enumerate(objects):
                    mat = bake[i][j]
                    # Convert world space transform to parent space, so parented objects don't get offset after baking.
                    if obj.parent:
                        mat = obj.matrix_parent_inverse.inverted() @ obj.parent.matrix_world.inverted() @ mat

                    obj.location = mat.to_translation()

                    rot_mode = obj.rotation_mode
                    if rot_mode == 'QUATERNION':
                        q1 = obj.rotation_quaternion
                        q2 = mat.to_quaternion()
                        # make quaternion compatible with the previous one
                        if q1.dot(q2) < 0.0:
                            obj.rotation_quaternion = -q2
                        else:
                            obj.rotation_quaternion = q2
                        obj.keyframe_insert(data_path="rotation_quaternion", frame=f)
                    elif rot_mode == 'AXIS_ANGLE':
                        # this is a little roundabout but there's no better way right now
                        aa = mat.to_quaternion().to_axis_angle()
                        obj.rotation_axis_angle = (aa[1], *aa[0])
                        obj.keyframe_insert(data_path="rotation_axis_angle", frame=f)
                    else:  # euler
                        # make sure euler rotation is compatible to previous frame
                        # NOTE: assume that on first frame, the starting rotation is appropriate
                        obj.rotation_euler = mat.to_euler(rot_mode, obj.rotation_euler)
                        obj.keyframe_insert(data_path="rotation_euler", frame=f)
                    # bake to keyframe
                    obj.keyframe_insert(data_path="location", frame=f)

                print("Baking frame %d" % f)

            # remove baked objects from simulation
            for obj in objects:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.object_remove()

            # clean up keyframes
            for obj in objects:
                action = obj.animation_data.action
                for fcu in action.fcurves:
                    keyframe_points = fcu.keyframe_points
                    i = 1
                    # remove unneeded keyframes
                    while i < len(keyframe_points) - 1:
                        val_prev = keyframe_points[i - 1].co[1]
                        val_next = keyframe_points[i + 1].co[1]
                        val = keyframe_points[i].co[1]

                        if abs(val - val_prev) + abs(val - val_next) < 0.0001:
                            keyframe_points.remove(keyframe_points[i])
                        else:
                            i += 1
                    # use linear interpolation for better visual results
                    for keyframe in keyframe_points:
                        keyframe.interpolation = 'LINEAR'

    def render(self):
        """Renders all frames (or a subset) of the animation.
        """
        print("Using scratch rendering folder: '%s'" % self.scratch_dir)
        # setup rigid world cache
        bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
        bpy.context.scene.rigidbody_world.point_cache.frame_end = bpy.context.scene.frame_end + 1
        bpy.context.view_layer.objects.active = self.assets_set[0]

        self.set_exr_output_path(os.path.join(self.scratch_dir, "exr", "frame_"))
        # --- starts rendering
        camera_save_dir = os.path.join(self.scratch_dir, 'cam')
        obj_save_dir = os.path.join(self.scratch_dir, 'obj')
        os.makedirs(camera_save_dir, exist_ok=True)
        os.makedirs(obj_save_dir, exist_ok=True)

        camdata = self.camera.data
        focal = camdata.lens  # mm
        sensor_width = camdata.sensor_width  # mm
        sensor_height = camdata.sensor_height  # mm
        scene_info = {'sensor_width': sensor_width, 'sensor_height': sensor_height, 'focal_length': focal,
                      'assets': ['background']}
        scene_info['assets'] += [x.data.name for x in self.assets_set]
        scene_info['character'] = self.character_name
        json.dump(scene_info, open(os.path.join(self.scratch_dir, 'scene_info.json'), 'w'))

        self.set_render_engine()
        self.clear_scene()

        frames = range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1)

        absolute_path = os.path.abspath(self.scratch_dir)

        # add forces
        for frame_nr in frames:
            if frame_nr % self.force_interval == 1 and self.add_force:
                # add keyframe to force strength
                bpy.context.scene.frame_set(frame_nr)
                force_loc_list = np.random.uniform(np.array([-4, -4, -5]), np.array([4, 4, -3]),
                                                size=(self.num_assets * 50, 3)) * self.scale_factor
                force_loc_list = self.farthest_point_sampling(force_loc_list, self.force_num)
                print('force_loc_list', force_loc_list)
                for i in range(len(self.gso_force)):
                    force_source = self.gso_force[i]
                    # select obj
                    force_source.field.strength = np.random.uniform(500, 1000) * self.scale_factor
                    force_source.field.distance_max = 1000
                    force_loc_list[i][2] *= 5
                    force_source.location = force_loc_list[i]
                    force_source.keyframe_insert(data_path='location', frame=frame_nr)
                    force_source.keyframe_insert(data_path='location', frame=frame_nr + self.force_interval - 1)
                    force_source.keyframe_insert(data_path='field.strength', frame=frame_nr)
                    force_source.keyframe_insert(data_path='field.strength', frame=frame_nr + self.force_step - 1)
                    force_source.keyframe_insert(data_path='field.distance_max', frame=frame_nr)
                    force_source.keyframe_insert(data_path='field.distance_max', frame=frame_nr + self.force_step - 1)
                    force_source.field.strength *= 0  # disable force
                    force_source.field.distance_max *= 0
                    force_source.keyframe_insert(data_path='field.strength', frame=frame_nr + self.force_step)
                    force_source.keyframe_insert(data_path='field.strength', frame=frame_nr + self.force_interval - 1)
                    force_source.keyframe_insert(data_path='field.distance_max', frame=frame_nr + self.force_step)
                    force_source.keyframe_insert(data_path='field.distance_max',
                                                 frame=frame_nr + self.force_interval - 1)

        # bpy.ops.wm.save_as_mainfile(filepath=os.path.join(absolute_path, 'scene1.blend'))
        # bake rigid body simulation

        bpy.ops.object.select_all(action='SELECT')
        bpy.context.view_layer.objects.active = self.assets_set[0]
        print('start baking')
        self.bake_to_keyframes(frames[0], frames[-1], 1)
        print('baking done')
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(absolute_path, 'scene.blend'))

        # set camera poses from real camera trajectory


        camera_files = glob.glob(os.path.join(self.camera_path, '*/*.txt'))
        # filter out small files
        camera_files = [c for c in camera_files if os.path.getsize(c) > 5000]
        camera_file = np.random.choice(camera_files)
        print('camera file: ', camera_file)
        camera_rt = np.loadtxt(camera_file, skiprows=1)[:, 7:].reshape(-1, 3, 4)
        self.camera_T = -camera_rt[:, :3, :3].transpose((0, 2, 1)) @ camera_rt[:, :3, 3:]


        # normalize camera trajectory to capture the character
        # self.camera_T /= np.std(self.camera_T, axis=1, keepdims=True)
        # self.camera_T /= 2\
        xy_min = np.min(self.camera_T[:, :2], axis=0)
        xy_max = np.max(self.camera_T[:, :2], axis=0)
        xy_length = np.max(np.abs(xy_max - xy_min))
        scale = 1.5
        if xy_length < 8:
            scale = 8 / xy_length
        elif xy_length > 10:
            scale = 10 / xy_length
        self.camera_T[:, :2] *= scale
        trajectory_vec = (self.camera_T[-1] - self.camera_T[0]).reshape(-1)
        cam_vec = np.array(self.cam_loc)
        cam_sign = np.sign(cam_vec * trajectory_vec)
        cam_sign *= -1
        cam_sign[2] = 1
        self.cam_sign = cam_sign.reshape(-1)
        time_ratio = self.camera_T.shape[0] / (frames[-1] - frames[0])
        initial_r = np.linalg.norm(cam_vec[:2])

        # set camera poses
        for cam_idx in range(0, self.camera_T.shape[0] - 1):
            frame_nr = frames[0] + int(cam_idx / time_ratio)
            bpy.context.scene.frame_set(frame_nr)
            bpy.context.scene.camera.keyframe_insert(data_path="location", frame=frame_nr)
            bpy.context.scene.camera.keyframe_insert(data_path="rotation_euler", frame=frame_nr)

            frame_next = frames[0] + int((cam_idx + 1) / time_ratio)
            bpy.context.scene.frame_set(frame_next)
            delta_T = self.camera_T[cam_idx + 1] - self.camera_T[cam_idx]
            delta_T = delta_T.reshape(-1)
            delta_T = delta_T * self.cam_sign
            delta_T *= self.scale_factor
            print('delta_T', delta_T)
            # delta_T = np.clip(delta_T, -0.2 * 1 / time_ratio * self.scale_factor, 0.2 * 1 / time_ratio * self.scale_factor).reshape(3)
            mean_location = np.mean([obj.matrix_world.translation for obj in self.assets_set if
                                     np.max(np.abs(obj.matrix_world.translation)) < 3 * self.scale_factor], axis=0)

            # mean_location[:2] *= 0
            self.cam_lookat = self.cam_lookat * 0.95 + mathutils.Vector(mean_location) * 0.05

            self.cam_loc = self.cam_loc + mathutils.Vector([delta_T[0], delta_T[1], delta_T[2]])
            if np.linalg.norm(np.array(self.cam_loc[:2])) < initial_r * 0.75:
                self.cam_loc[:2] = self.cam_loc[:2] / np.linalg.norm(np.array(self.cam_loc[:2])) * initial_r * 0.75
            cam_height_max = 3 * self.scale_factor
            if self.cam_loc[2] > cam_height_max:
                self.cam_loc[2] = cam_height_max
                self.cam_sign[2] *= -1
            if self.cam_loc[2] < 0.5 * self.scale_factor:
                self.cam_loc[2] = 0.5 * self.scale_factor
                self.cam_sign[2] *= -1
            self.set_cam(self.cam_loc, self.cam_lookat)
            print('inserting camera keyframe {}, delta_T:{}'.format(frame_next, delta_T))
            # add camera lcoation and rotation keyframe
            bpy.context.scene.camera.keyframe_insert(data_path="location", frame=frame_next)
            bpy.context.scene.camera.keyframe_insert(data_path="rotation_euler", frame=frame_next)

        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(absolute_path, 'scene.blend'))
        for frame_nr in frames:
            bpy.context.scene.frame_set(frame_nr)

            bpy.context.scene.render.filepath = os.path.join(
                self.scratch_dir, "images", f"frame_{frame_nr:04d}.png")

            bpy.ops.render.render(animation=False, write_still=True)

            modelview_matrix = bpy.context.scene.camera.matrix_world.inverted()
            K = get_calibration_matrix_K_from_blender(bpy.context.scene, mode='simple')

            np.savetxt(os.path.join(camera_save_dir, f"RT_{frame_nr:04d}.txt"), modelview_matrix)
            np.savetxt(os.path.join(camera_save_dir, f"K_{frame_nr:04d}.txt"), K)
            print("Rendered frame '%s'" % bpy.context.scene.render.filepath)



def get_calibration_matrix_K_from_blender(scene, mode='simple'):
    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale  # px
    height = scene.render.resolution_y * scale  # px

    camdata = scene.camera.data
    K = np.zeros((3, 3), dtype=np.float32)

    if mode == 'simple':
        aspect_ratio = width / height
        K = np.zeros((3, 3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(camdata.angle / 2)
        K[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
        K[0][2] = width / 2.
        K[1][2] = height / 2.
        K[2][2] = 1.
        K.transpose()

    if mode == 'complete':

        focal = camdata.lens  # mm
        sensor_width = camdata.sensor_width  # mm
        sensor_height = camdata.sensor_height  # mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if (camdata.sensor_fit == 'VERTICAL'):
            # the sensor height is fixed (sensor fit is horizontal),
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = width / sensor_width / pixel_aspect_ratio
            s_v = height / sensor_height
        else:  # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal),
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        # parameters of intrinsic calibration matrix K
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0  # only use rectangular pixels

        K = np.array([
            [alpha_u, skew, u_0],
            [0, alpha_v, v_0],
            [0, 0, 1]
        ], dtype=np.float32)

    return K


if __name__ == "__main__":
    import sys

    argv = sys.argv

    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]

    print("argsv:{0}".format(argv))
    parser = argparse.ArgumentParser(description='Render Motion in 3D Environment for HuMoR Generation.')
    parser.add_argument('--character_root', type=str, metavar='PATH', default='./data/robots/')
    parser.add_argument('--use_character', type=str, metavar='PATH', default=None)
    parser.add_argument('--camera_root', type=str, metavar='PATH', default='./data/camera_trajectory/MannequinChallenge')
    parser.add_argument('--motion_root', type=str, metavar='PATH', default='./data/motions/')
    parser.add_argument('--partnet_root', type=str, metavar='PATH', default='./data/partnet/')
    parser.add_argument('--gso_root', type=str, metavar='PATH', default='./data/GSO/')
    parser.add_argument('--background_hdr_path', type=str, default='./data/hdri/')
    parser.add_argument('--scene_root', type=str, default='./data/blender_assets/hdri_plane.blend')
    parser.add_argument('--output_dir', type=str, metavar='PATH', default='./',
                        help='img save dir')
    parser.add_argument('--output_name', type=str, metavar='PATH',
                        help='img save name',
                        default='test')
    # parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--force_step', type=int, default=3)
    parser.add_argument('--force_interval', type=int, default=120)
    parser.add_argument('--force_num', type=int, default=3)
    parser.add_argument('--add_force', action='store_true', default=False)
    parser.add_argument('--num_assets', type=int, default=15)
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--indoor', action='store_true', default=False)
    parser.add_argument('--render_engine', type=str, default='CYCLES', choices=['BLENDER_EEVEE', 'CYCLES'])
    args = parser.parse_args(argv)
    print("args:{0}".format(args))

    ## Load the world
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    renderer = Blender_render(scratch_dir=output_dir, render_engine=args.render_engine, use_gpu=args.use_gpu, character_path=args.character_root, motion_path=args.motion_root,
                              camera_path=args.camera_root, background_hdr_path=args.background_hdr_path, GSO_path=args.gso_root, num_assets=args.num_assets,
                              custom_scene=args.scene_root, use_indoor_cam=args.indoor, partnet_path=args.partnet_root, use_character=args.use_character,
                              add_force=args.add_force, force_step=args.force_step, force_interval=args.force_interval, force_num=args.force_num)

    renderer.render()


