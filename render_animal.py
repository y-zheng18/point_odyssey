import numpy as np
import json
import os
import math
import argparse
from typing import Any, Dict, Optional, Sequence, Union
import shutil
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


def read_obj_file(obj_file_path):
    '''
    Load .obj file, return vertices, faces.
    return: vertices: N_v X 3, faces: N_f X 3
    '''
    obj_f = open(obj_file_path, 'r')
    lines = obj_f.readlines()
    vertices = []
    faces = []
    vt = []
    vt_f = []
    for ori_line in lines:
        line = ori_line.split()
        if line[0] == 'v':
            vertices.append([float(line[1]), float(line[2]), float(line[3])])  # x, y, z
        elif line[0] == 'f':  # Need to consider / case, // case, etc.
            faces.append([int(line[3].split('/')[0]),
                          int(line[2].split('/')[0]),
                          int(line[1].split('/')[0]) \
                          ])  # Notice! Need to reverse back when using the face since here it would be clock-wise!
            # Convert face order from clockwise to counter-clockwise direction.
            if len(line[1].split('/')) > 1:
                vt_f.append([int(line[3].split('/')[1]),
                           int(line[2].split('/')[1]),
                           int(line[1].split('/')[1]) \
                           ])
        elif line[0] == 'vt':
            vt.append([float(line[1]), float(line[2])])
        obj_f.close()

    return np.asarray(vertices), np.asarray(faces), np.asarray(vt), np.asarray(vt_f)

def save_obj_file(obj_file_path, vertices, faces, f_idx_offset=0, vt=None, vt_f=None):
    with open(obj_file_path, 'w') as f:
        for v in vertices:
            f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        # adding uv coordinates
        if vt is not None:
            for v in vt:
                f.write('vt %f %f\n' % (v[0], v[1]))
        for i, face in enumerate(faces):
            if vt_f is not None and i < vt_f.shape[0]:
                f.write('f %d/%d %d/%d %d/%d\n' % (face[0] + f_idx_offset, vt_f[i][0], face[1] + f_idx_offset, vt_f[i][1], face[2] + f_idx_offset, vt_f[i][2]))
            else:
                f.write('f %d %d %d\n' % (face[0] + f_idx_offset, face[1] + f_idx_offset, face[2] + f_idx_offset))


def copy_obj(data_root, animal_name, num_seq, save_path):
    animal_sequences = [p for p in os.listdir(data_root) if animal_name in p]
    animal_sequences = np.random.choice(animal_sequences, num_seq, replace=False if num_seq < len(animal_sequences) else True)

    os.makedirs(save_path, exist_ok=True)

    idx = 0
    vt = None
    vt_f = None
    for i, animal_sequence in enumerate(animal_sequences):
        print(animal_sequence)
        obj_list = [p for p in os.listdir(os.path.join(data_root, animal_sequence, 'mesh_seq')) if '.obj' in p]
        obj_list = sorted(obj_list)
        # copy obj from the left to the right timeline
        if idx == 0:
            shutil.copy(os.path.join(data_root, animal_sequence, 'mesh_seq', obj_list[0]), os.path.join(save_path, str(idx).zfill(5) + '.obj'))
            # using blender to unwrap the first obj

            # load obj
            bpy.ops.import_scene.obj(filepath=os.path.join(save_path, str(idx).zfill(5) + '.obj'),
                                     use_groups_as_vgroups=True, split_mode='OFF')
            # select the object
            imported_object = bpy.context.selected_objects[0]
            bpy.ops.object.select_all(action='DESELECT')
            imported_object.select_set(True)
            bpy.context.view_layer.objects.active = imported_object

            # edit mode
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            # smart uv project the entire object
            bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)

            # scale the uv


            # finish the edit mode
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')

            # save the obj
            bpy.ops.export_scene.obj(filepath=os.path.join(save_path, str(idx).zfill(5) + '.obj'), use_selection=True,
                                     use_materials=False, use_normals=False, use_uvs=True, use_triangles=False,
                                     keep_vertex_order=True)
            # delete the object
            bpy.ops.object.delete(use_global=False)

            v, f, vt, vt_f = read_obj_file(os.path.join(save_path, str(idx).zfill(5) + '.obj'))
            # scale vt

            vt -= 0.5
            vt *= 5
            vt += 0.5

        for obj in obj_list:
            v, f, _, _ = read_obj_file(os.path.join(data_root, animal_sequence, 'mesh_seq', obj))

            save_obj_file(os.path.join(save_path, str(idx).zfill(5) + '.obj'), v, f, vt=vt, vt_f=vt_f)
            idx += 1
        # copy obj from the right to the left timeline
        for obj in obj_list[::-1]:
            v, f, _, _ = read_obj_file(os.path.join(data_root, animal_sequence, 'mesh_seq', obj))

            save_obj_file(os.path.join(save_path, str(idx).zfill(5) + '.obj'), v, f, vt=vt, vt_f=vt_f)
            idx += 1

        if i < num_seq - 1:
            # interpolate obj
            obj_0 = read_obj_file(os.path.join(data_root, animal_sequence, 'mesh_seq', obj_list[0]))
            obj_list1 = [p for p in os.listdir(os.path.join(data_root, animal_sequences[i+1], 'mesh_seq')) if '.obj' in p]
            obj_list1 = sorted(obj_list1)
            obj_1 = read_obj_file(os.path.join(data_root, animal_sequences[i+1], 'mesh_seq', obj_list1[0]))

            for j in range(0, 10):
                obj_v = (obj_0[0] * (10 - j) + obj_1[0] * j) / 10
                obj_f = obj_0[1]
                save_obj_file(os.path.join(save_path, str(idx).zfill(5) + '.obj'), obj_v, obj_f, vt=vt, vt_f=vt_f)
                idx += 1

def anime2obj(anime_path, save_path):
    f = open(anime_path, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    '''check data consistency'''
    if len(offset_data) != (nf - 1) * nv * 3:
        raise ("data inconsistent error!", anime_path)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))

    obj_v = vert_data
    v_list = []
    for i in range(nf):
        if i == 0:
            obj_v = vert_data
        else:
            obj_v = vert_data + offset_data[i - 1]

        # check if the obj is under the ground
        z_min = np.min(obj_v[:, 2])
        z_max = np.max(obj_v[:, 2])
        z_diff = z_max - z_min
        if z_min < -z_diff * 0.2:
            return
        v_list.append(obj_v.copy())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(nf):
        obj_v = v_list[i]
        obj_f = face_data
        save_obj_file(os.path.join(save_path, str(i).zfill(5) + '.obj'), obj_v, obj_f, f_idx_offset=1)


class Blender_render():
    def __init__(self,
                 scratch_dir=None,
                 partnet_path=None,
                 GSO_path=None,
                 animal_path=None,
                 animal_name=None,
                 material_path=None,
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
                 add_smoke: bool = False,
                 force_step: int = 3,
                 force_num: int = 3,
                 force_interval: int = 200,
                 views=1,
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
        self.animal_path = animal_path
        self.animal_name = animal_name
        self.material_path = material_path

        self.GSO_path = GSO_path
        self.camera_path = camera_path

        self.force_step = force_step
        self.force_num = force_num
        self.force_interval = force_interval
        self.add_force = add_force

        self.add_smoke = add_smoke

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
        self.views = views

        # self.blender_scene.render.resolution_percentage = 100
        if background_hdr_path:
            print('loading hdr from:', self.background_hdr_path)
            self.load_background_hdr(self.background_hdr_path)

        # clear unused data recursively
        bpy.ops.outliner.orphans_purge()
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_recursive=True)

        # pack external data
        bpy.ops.file.pack_all()
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
        # load animal
        animal_list = os.listdir(self.animal_path)
        if self.animal_name is None:
            animal = np.random.choice(animal_list)
            self.animal_name = animal.split('_')[0]
        animal_list = [c for c in animal_list if self.animal_name in c]
        # sort animal_list by file size

        animal_list = sorted(animal_list, key=lambda x: os.path.getsize(os.path.join(self.animal_path, x)))[:50]
        print(animal_list, self.animal_name)
        animal_list = np.random.choice(animal_list, 30, replace=False if len(animal_list) > 30 else True)

        animal_obj_savedir = os.path.join(self.scratch_dir, 'tmp')
        for animal_seq in animal_list:
            anime2obj(os.path.join(self.animal_path, animal_seq, animal_seq + '.anime'), os.path.join(animal_obj_savedir, animal_seq, 'mesh_seq'))
        copy_obj(animal_obj_savedir, self.animal_name, 18, os.path.join(self.scratch_dir, 'tmp', 'animal_obj'))

        # set the end frame according to the number of frames in the sequence
        bpy.context.scene.frame_end = len(os.listdir(os.path.join(self.scratch_dir, 'tmp', 'animal_obj')))

        # load mesh sequence
        seq_imp_settings = bpy.types.PropertyGroup.bl_rna_get_subclass_py("SequenceImportSettings")
        seq_imp_settings.fileNamePrefix = bpy.props.StringProperty(name='File Name', default='0')
        print('importing mesh sequence')
        bpy.ops.ms.import_sequence(directory=os.path.join(self.scratch_dir, 'tmp', 'animal_obj'))
        print('importing mesh sequence done!')
        self.animal = bpy.context.selected_objects[0]
        # scale and rotate the animal
        dimension = np.max(self.animal.dimensions)
        animal_scale = np.random.uniform(1, 1.4) * self.scale_factor * np.random.uniform(1.5, 2) / dimension
        self.animal.scale = (animal_scale, animal_scale, animal_scale)
        self.animal.rotation_euler = (0, 0, 0)

        # make the animal stand on the ground without penetrating
        z_min = np.min([b[2] for b in self.animal.bound_box], axis=0)
        self.animal.location = (0, 0, -z_min * animal_scale)

        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

        # add random texture to animal
        print('adding texture')
        # append materials
        bpy.ops.wm.append(
            directory=os.path.join(self.material_path, "Object"),
            filename="Cube"
        )

        furry_material = [f for f in bpy.data.materials if 'Animal' in f.name]
        furry_material = np.random.choice(furry_material)

        self.animal.data.materials.clear()
        self.animal.data.materials.append(furry_material)

        # add physics
        bpy.context.view_layer.objects.active = self.animal
        bpy.ops.rigidbody.object_add()
        self.animal.rigid_body.collision_shape = 'MESH'
        self.animal.rigid_body.type = 'PASSIVE'
        # enable animated
        self.animal.rigid_body.kinematic = True

        # adding smoke
        if self.add_smoke:
            # add a cube
            bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD', location=(np.random.uniform(-1.5, 1.5) * self.scale_factor, np.random.uniform(-1.5, 1.5) * self.scale_factor, 1),
                                            scale=(self.scale_factor / 5, self.scale_factor / 5, self.scale_factor / 5))
            # change the name of the cube
            bpy.context.object.name = 'Smoke_cube'
            # make it not rendered
            bpy.context.object.hide_render = True
            # add smoke
            bpy.ops.object.quick_smoke()
            # scale the smoke
            bpy.context.object.scale = (self.scale_factor / 2, self.scale_factor / 2, self.scale_factor * 3)
            # move z axis
            z_min = bpy.context.object.bound_box[0][2]
            bpy.context.object.location[2] -= -z_min * self.scale_factor / 5
            # change the resulution of the smoke
            bpy.context.object.modifiers['Fluid'].domain_settings.resolution_max = 128

            # enable the adptive domain
            bpy.context.object.modifiers['Fluid'].domain_settings.use_adaptive_domain = True
            bpy.context.object.modifiers['Fluid'].domain_settings.cache_frame_start = 1
            bpy.context.object.modifiers['Fluid'].domain_settings.cache_frame_end = bpy.context.scene.frame_end

        # add objects
        GSO_assets = os.listdir(self.GSO_path)
        GSO_assets = [os.path.join(self.GSO_path, asset) for asset in GSO_assets]
        GSO_assets = [asset for asset in GSO_assets if os.path.isdir(asset)]
        GSO_assets_path = np.random.choice(GSO_assets, size=self.num_assets // 2, replace=False)

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

    def bake_camera(self, camera_rt, frames):
        self.camera_T = -camera_rt[:, :3, :3].transpose((0, 2, 1)) @ camera_rt[:, :3, 3:]

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

    def render(self):
        """Renders all frames (or a subset) of the animation.
        """
        print("Using scratch rendering folder: '%s'" % self.scratch_dir)
        # setup rigid world cache
        bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
        bpy.context.scene.rigidbody_world.point_cache.frame_end = bpy.context.scene.frame_end + 1
        bpy.context.view_layer.objects.active = self.assets_set[0]

        self.set_exr_output_path(os.path.join(self.scratch_dir, "exr", "frame_"))

        self.set_render_engine()
        self.clear_scene()

        absolute_path = os.path.abspath(self.scratch_dir)

        camdata = self.camera.data
        focal = camdata.lens  # mm
        sensor_width = camdata.sensor_width  # mm
        sensor_height = camdata.sensor_height  # mm
        scene_info = {'sensor_width': sensor_width, 'sensor_height': sensor_height, 'focal_length': focal,
                      'assets': ['background']}
        scene_info['assets'] += [x.data.name for x in self.assets_set]
        scene_info['character'] = self.animal.name
        json.dump(scene_info, open(os.path.join(self.scratch_dir, 'scene_info.json'), 'w'))

        frames = range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1)

        # add forces
        for frame_nr in frames:
            if frame_nr % self.force_interval == 1 and self.add_force:
                # add keyframe to force strength
                bpy.context.scene.frame_set(frame_nr)
                force_loc_list = np.random.uniform(np.array([-16, -16, -3]), np.array([16, 16, 0]),
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
        bpy.ops.object.select_all(action='SELECT')
        bpy.context.view_layer.objects.active = self.assets_set[0]
        print('start baking')
        self.bake_to_keyframes(frames[0], frames[-1], 1)
        print('baking done')
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(absolute_path, 'scene.blend'))

        # --- starts rendering
        camera_save_dir = os.path.join(self.scratch_dir, 'cam')
        obj_save_dir = os.path.join(self.scratch_dir, 'obj')
        os.makedirs(camera_save_dir, exist_ok=True)
        os.makedirs(obj_save_dir, exist_ok=True)

        use_multiview = self.views > 1

        if not use_multiview:
            # set camera poses from real camera trajectory
            camera_files = glob.glob(os.path.join(self.camera_path, '*/*.txt'))
            # filter out small files
            camera_files = [c for c in camera_files if os.path.getsize(c) > 5000]
            camera_file = np.random.choice(camera_files)
            print('camera file: ', camera_file)
            camera_rt = np.loadtxt(camera_file, skiprows=1)[:, 7:].reshape(-1, 3, 4)
            self.bake_camera(camera_rt, frames)

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
        else:
            # set camera poses from real camera trajectory
            camera_files = glob.glob(os.path.join(self.camera_path, '*/*.txt'))
            # filter out small files
            camera_files = [c for c in camera_files if os.path.getsize(c) > 5000]
            camera_files = np.random.choice(camera_files, self.views, replace=False)
            print('camera files: ', camera_files)

            self.camera_list = []
            for i in range(self.views):
                # create new cameras
                bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 0), rotation=(0, 0, 0))
                self.camera_list.append(bpy.context.object)

                self.camera = self.camera_list[i]
                bpy.context.scene.camera = self.camera

                # setup camera
                self.cam_loc = mathutils.Vector((np.random.uniform(-3, -3.5) * np.random.choice((-1, 1)),
                                                 np.random.uniform(-3, -3.5) * np.random.choice((-1, 1)),
                                                 np.random.uniform(1, 2.5))) * self.scale_factor
                self.cam_lookat = mathutils.Vector((0, 0, 0.5)) * self.scale_factor
                self.set_cam(self.cam_loc, self.cam_lookat)
                self.camera.data.lens = FOCAL_LENGTH
                self.camera.data.clip_end = 10000
                self.camera.data.sensor_width = SENSOR_WIDTH

                camera_file = camera_files[i]
                camera_rt = np.loadtxt(camera_file, skiprows=1)[:, 7:].reshape(-1, 3, 4)
                self.bake_camera(camera_rt, frames)

            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(absolute_path, 'scene.blend'))
            for frame_nr in frames:
                bpy.context.scene.frame_set(frame_nr)

                for i in range(len(self.camera_list)):
                    camera_save_dir = os.path.join(self.scratch_dir, 'cam', "view{}".format(i))
                    if not os.path.exists(camera_save_dir):
                        os.makedirs(camera_save_dir)
                    bpy.context.scene.camera = self.camera_list[i]
                    self.set_exr_output_path(os.path.join(self.scratch_dir, "view{}".format(i), "exr", "frame_"))
                    bpy.context.scene.render.filepath = os.path.join(
                        self.scratch_dir, "view{}".format(i), "images", f"frame_{frame_nr:04d}.png")

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
    parser.add_argument('--material_root', type=str, metavar='PATH',
                        help='path to specific folder which include folders containing .obj files',
                        default='/Users/yangzheng/Documents/Blender/Assets/materials')
    parser.add_argument('--animal_root', type=str, metavar='PATH', default='./data/deformingthings4d')
    parser.add_argument('--animal_name', type=str, metavar='PATH', default=None)
    parser.add_argument('--camera_root', type=str, metavar='PATH', default='./data/camera_trajectory/MannequinChallenge')
    parser.add_argument('--partnet_root', type=str, metavar='PATH', default='./data/partnet/')
    parser.add_argument('--gso_root', type=str, metavar='PATH', default='./data/GSO/')
    parser.add_argument('--background_hdr_path', type=str, default='./data/hdri/')
    parser.add_argument('--material_path', type=str, default='./data/blender_assets/animal_material.blend')
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
    parser.add_argument('--num_assets', type=int, default=5)
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--indoor', action='store_true', default=False)
    parser.add_argument('--render_engine', type=str, default='CYCLES', choices=['BLENDER_EEVEE', 'CYCLES'])
    parser.add_argument('--add_smoke', action='store_true', default=False)
    parser.add_argument('--views', default=1, type=int)
    args = parser.parse_args(argv)
    print("args:{0}".format(args))

    ## Load the world
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    renderer = Blender_render(scratch_dir=output_dir, render_engine=args.render_engine, use_gpu=args.use_gpu, material_path=args.material_path,
                              animal_path=args.animal_root, animal_name=args.animal_name,
                              camera_path=args.camera_root, background_hdr_path=args.background_hdr_path, GSO_path=args.gso_root, num_assets=args.num_assets,
                              custom_scene=args.scene_root, use_indoor_cam=args.indoor, partnet_path=args.partnet_root,
                              add_force=args.add_force, force_step=args.force_step, force_interval=args.force_interval, force_num=args.force_num,
                              add_smoke=args.add_smoke, views=args.views)

    renderer.render()


