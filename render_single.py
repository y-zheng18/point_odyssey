import numpy as np
import json
import os
import math
import argparse
from typing import Any, Dict, Optional, Sequence, Union
import random

import bpy
import mathutils

class Blender_render():
    def __init__(self,
                 scratch_dir=None,
                 render_engine='BLENDER_EEVEE',
                 adaptive_sampling=False,
                 use_denoising=True,
                 samples_per_pixel=128,
                 background_transparency=False,
                 background_hdr_path=None,
                 use_gpu: bool = False,
                 add_fog = False,
                 fog_path = None,
                 custom_scene: Optional[str] = None,
                 randomize=False,
                 material_path=None,
                 views: int=1
                 ):
        self.blender_scene = bpy.context.scene
        self.render_engine = render_engine
        self.use_gpu = use_gpu
        self.add_fog = add_fog
        self.fog_path = fog_path
        self.randomize = randomize
        self.material_path = material_path
        self.samples_per_pixel = samples_per_pixel

        self.views = views

        self.set_render_engine()


        # set scene
        bpy.ops.wm.read_factory_settings(use_empty=True)


        if custom_scene is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")
        else:
            print("Loading scene from '%s'" % custom_scene)
            bpy.ops.wm.open_mainfile(filepath=custom_scene)

        self.scratch_dir = scratch_dir

        self.activate_render_passes(normal=True, optical_flow=True, segmentation=True, uv=True)
        self.exr_output_node = self.set_up_exr_output_node()

        if background_hdr_path and os.path.exists(background_hdr_path):
            print('loading hdr from:', background_hdr_path)
            self.load_background_hdr(background_hdr_path)

        if randomize and os.path.exists(self.material_path):
            self.randomize_scene()
        if self.add_fog and os.path.exists(self.fog_path):
            self.load_fog()
        # # save blend file
        os.makedirs(scratch_dir, exist_ok=True)
        absolute_path = os.path.abspath(scratch_dir)
        try:
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(absolute_path, 'scene.blend'))
        except:
            print('error saving blend file, skipping')

    def set_render_engine(self):
        bpy.context.scene.render.resolution_x = 960
        bpy.context.scene.render.resolution_y = 540
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.render.engine = self.render_engine
        print("Using render engine: {}".format(self.render_engine))
        if self.use_gpu:
            print("----------------------------------------------")
            print('setting up gpu ......')

            bpy.context.scene.cycles.device = "GPU"
            for scene in bpy.data.scenes:
                print(scene.name)
                scene.cycles.device = 'GPU'

            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)

            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            for d in bpy.context.preferences.addons["cycles"].preferences.devices:
                d.use = True
                print("Device '{}' type {} : {}".format(d.name, d.type, d.use))
            print('setting up gpu done')
            print("----------------------------------------------")

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

    def load_background_hdr(self, background_hdr_path):
        # bpy.ops.image.open(filepath=background_hdr_path)
        world = bpy.context.scene.world
        for node in world.node_tree.nodes:
            world.node_tree.nodes.remove(node)
        # world.node_tree.clear()

        node_background = world.node_tree.nodes.new(type='ShaderNodeBackground')
        node_env = world.node_tree.nodes.new(type='ShaderNodeTexEnvironment')
        node_output = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')

        # world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
        # world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
        # world.node_tree.nodes["Background"].inputs[2].default_value = 1.0

        node_env.image = bpy.data.images.load(background_hdr_path) # bpy.data.images[os.path.basename(background_hdr_path)]

        world.node_tree.links.new(node_env.outputs["Color"], node_background.inputs["Color"])
        world.node_tree.links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

    def load_fog(self):
        print("Loading fog")
        # append the fod file
        bpy.ops.wm.append(
            directory=os.path.join(self.fog_path, "Collection"),
            filename="fog"
        )

        # addjust the fog
        fog_material = bpy.data.materials["fog"]
        # randomize the colorRamp
        fog_material.node_tree.nodes["ColorRamp"].color_ramp.elements[0].position = np.random.uniform(0.45, 0.55)
        fog_material.node_tree.nodes["ColorRamp"].color_ramp.elements[1].position = np.random.uniform(0.6, 1.0)

        # randomize the noise texture
        fog_material.node_tree.nodes["Noise Texture"].inputs[3].default_value = np.random.uniform(500, 4000)
        fog_material.node_tree.nodes["Noise Texture"].inputs[4].default_value = np.random.uniform(0.25, 1.0)

        # add keyframes of the noise texture
        mapping = fog_material.node_tree.nodes["Mapping"]
        for i in range(0, bpy.context.scene.frame_end // 200):
            bpy.context.scene.frame_set(i * 200)
            mapping.inputs[1].default_value[0] = np.random.uniform(-3, 3)
            mapping.inputs[1].default_value[1] = np.random.uniform(-3, 3)
            mapping.inputs[1].default_value[2] = np.random.uniform(-3, 3)
            mapping.inputs[2].default_value[0] = np.random.uniform(-np.pi, np.pi)
            mapping.inputs[2].default_value[1] = np.random.uniform(-np.pi, np.pi)
            mapping.inputs[2].default_value[2] = np.random.uniform(-np.pi, np.pi)

            # add keyframes of the mapping
            mapping.inputs[1].keyframe_insert(data_path="default_value", frame=i * 200)
            mapping.inputs[2].keyframe_insert(data_path="default_value", frame=i * 200)


        print("Loading fog done")

    def randomize_scene(self):
        '''
            Randomize the scene: textures of floors, walls, ceilings, and strength of light
        '''
        print("Randomizing scene ...")
        # randomize light strength
        for light in bpy.data.lights:
            light.energy *= np.random.uniform(0.7, 1.3)

        # append materials
        bpy.ops.wm.append(
            directory=os.path.join(self.material_path, "Object"),
            filename="Material"
        )

        # randomize floor material
        if "Floor" in bpy.data.collections:
            floor_collection = bpy.data.collections["Floor"]
            floor_materials = [m for m in bpy.data.materials if "floor" in m.name or "Floor" in m.name]
            for obj in floor_collection.objects:
                if len(obj.data.materials) == 0:
                    # create a new material
                    obj.data.materials.append(np.random.choice(floor_materials))
                else:
                    obj.data.materials[0] = np.random.choice(floor_materials)

        # randomize wall material
        if "Wall" in bpy.data.collections:
            wall_collection = bpy.data.collections["Wall"]
            wall_materials = [m for m in bpy.data.materials if "wall" in m.name or "Wall" in m.name]
            # randomize each 2 walls with the same material
            for i in range(0, len(wall_collection.objects), 2):
                wall_material = np.random.choice(wall_materials)
                for j in range(2):
                    if i+j < len(wall_collection.objects):
                        wall_collection.objects[i+j].data.materials.append(wall_material)
                        wall_collection.objects[i+j].data.materials[0] = wall_material

        # randomize ceiling material
        if "Ceiling" in bpy.data.collections:
            ceiling_collection = bpy.data.collections["Ceiling"]
            ceiling_materials = [m for m in bpy.data.materials if "ceiling" in m.name or "Ceiling" in m.name]
            for obj in ceiling_collection.objects:
                obj.data.materials.append(np.random.choice(ceiling_materials))
                obj.data.materials[0] = np.random.choice(ceiling_materials)

        print("Scene randomized")



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

    def clear_scene(self):
        for k in bpy.data.objects.keys():
            bpy.data.objects[k].select_set(False)

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

    def render(self,
               frames: Optional[Sequence[int]] = None,
               ignore_missing_textures: bool = False,
               return_layers: Sequence[str] = ("rgba", "backward_flow",
                                               "forward_flow", "depth",
                                               "normal", "object_coordinates",
                                               "segmentation"),
               ) -> Dict[str, np.ndarray]:
        """Renders all frames (or a subset) of the animation and returns images as a dict of arrays.

        Args:
          frames: list of frames to render (defaults to range(scene.frame_start, scene.frame_end+1)).
          ignore_missing_textures: if False then raise a RuntimeError when missing textures are
            detected. Otherwise, proceed to render (with purple color instead of missing texture).
          return_layers: list of layers to return. For possible values refer to
            the Blender.post_processors dict. Defaults to ("backward_flow",
            "forward_flow", "depth", "normal", "object_coordinates", "segmentation").

        Returns:
          A dictionary with one entry for each return layer. By default:
            - "rgba": shape = (nr_frames, height, width, 4)
            - "segmentation": shape = (nr_frames, height, width, 1) (int)
            - "backward_flow": shape = (nr_frames, height, width, 2)
            - "forward_flow": shape = (nr_frames, height, width, 2)
            - "depth": shape = (nr_frames, height, width, 1)
            - "object_coordinates": shape = (nr_frames, height, width, 3) (uint16)
            - "normal": shape = (nr_frames, height, width, 3) (uint16)
        """
        print("Using scratch rendering folder: '%s'" % self.scratch_dir)


        # --- starts rendering
        self.set_exr_output_path(os.path.join(self.scratch_dir, "exr", "frame_"))
        # --- starts rendering
        camera_save_dir = os.path.join(self.scratch_dir, 'cam')
        os.makedirs(camera_save_dir, exist_ok=True)

        self.set_render_engine()
        self.clear_scene()

        camdata = bpy.context.scene.camera.data
        focal = camdata.lens  # mm
        sensor_width = camdata.sensor_width  # mm
        sensor_height = camdata.sensor_height  # mm
        scene_info = {'sensor_width': sensor_width, 'sensor_height': sensor_height, 'focal_length': focal,
                      'assets': ['background']}
        assets_name = bpy.context.scene.objects.keys()

        # only mesh objects
        assets_name = [name for name in assets_name if bpy.data.objects[name].type == 'MESH']

        scene_info['assets'] += assets_name
        json.dump(scene_info, open(os.path.join(self.scratch_dir, 'scene_info.json'), 'w'))

        # set png output
        bpy.context.scene.render.image_settings.file_format = 'PNG'

        # set samples per pixel
        bpy.context.scene.cycles.samples = self.samples_per_pixel
        frames = range(frames[0], bpy.context.scene.frame_end + 1)

        use_multiview = self.views > 1
        if not use_multiview:
            for frame_nr in frames:
                bpy.context.scene.frame_set(frame_nr)
                # When writing still images Blender doesn't append the frame number to the png path.
                # (but for exr it does, so we only adjust the png path)
                bpy.context.scene.render.filepath = os.path.join(
                    self.scratch_dir, "images", f"frame_{frame_nr:04d}.png")
                bpy.ops.render.render(animation=False, write_still=True)

                modelview_matrix = bpy.context.scene.camera.matrix_world.inverted()
                K = get_calibration_matrix_K_from_blender(bpy.context.scene, mode='simple')
                # K = get_intrinsics(bpy.context.scene)
                np.savetxt(os.path.join(camera_save_dir, f"RT_{frame_nr:04d}.txt"), modelview_matrix)
                np.savetxt(os.path.join(camera_save_dir, f"K_{frame_nr:04d}.txt"), K)

                print("Rendered frame '%s'" % bpy.context.scene.render.filepath)
        else:
            self.camera_list = []
            self.camera_list.append(bpy.context.scene.camera)

            # find the bounding box of the wall collection
            wall_collection = bpy.data.collections['Wall']

            # find character collections
            static_keys = ['Floor', 'Ceiling', 'Wall', 'Furniture']
            character_collections = [bpy.data.collections[key] for key in bpy.data.collections.keys() if key not in static_keys]
            print('Found characters:', character_collections)


            for i in range(self.views):
                bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 0), rotation=(0, 0, 0))
                self.camera_list.append(bpy.context.object)

                self.camera = self.camera_list[i + 1]
                bpy.context.scene.camera = self.camera

                self.camera.data.lens = focal
                self.camera.data.clip_end = 10000
                self.camera.data.sensor_width = sensor_width
                self.camera.data.sensor_height = sensor_height

                wall_objects = [obj for obj in wall_collection.objects if obj.type == 'MESH']

                random_wall = np.random.choice(wall_objects)
                print('random wall:', random_wall)
                # apply all transformations to the chosen wall
                bpy.context.view_layer.objects.active = random_wall
                random_wall.select_set(True)
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

                # get the vertices of the wall
                vertices = [vert.co for vert in random_wall.data.vertices]
                vertices = np.stack(vertices, axis=0)

                v_min = np.min(vertices, axis=0)
                v_max = np.max(vertices, axis=0)
                print('wall min:', v_min)
                print('wall max:', v_max)
                cam_position = mathutils.Vector((np.random.uniform(v_min[0], v_max[0]),
                                                 np.random.uniform(v_min[1], v_max[1]),
                                                 np.random.uniform(v_max[2] * 0.8, v_max[2])))
                cam_position = cam_position * np.random.uniform(0.6, 0.7)

                self.camera.location = cam_position
                print('baking camera %d' % i)
                for f in frames:
                    bpy.context.scene.frame_set(f)

                    if not f % 50 == 0:
                        continue

                    # find the center of the characters
                    bbox_corners = []

                    # Loop over all mesh objects in each collection
                    for c in character_collections:
                        for obj in c.objects:
                            if obj.type == 'MESH':
                                # Compute bounding box (in world coordinates)
                                bbox_corners += [obj.matrix_world @ mathutils.Vector(v) for v in obj.bound_box]

                    # Convert to numpy array
                    bbox_corners = np.array(bbox_corners)
                    bbox_corners = np.reshape(bbox_corners, (-1, 3))

                    bbox_center = np.mean(bbox_corners, axis=0)
                    cam_lookat = mathutils.Vector(bbox_center)

                    direction = cam_lookat - cam_position
                    rot_quat = direction.to_track_quat('-Z', 'Y')
                    self.camera.rotation_euler = rot_quat.to_euler()

                    # add camera lcoation and rotation keyframe
                    bpy.context.scene.camera.keyframe_insert(data_path="location", frame=f)
                    bpy.context.scene.camera.keyframe_insert(data_path="rotation_euler", frame=f)

            absolute_path = os.path.abspath(self.scratch_dir)
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(absolute_path, 'scene.blend'))
            for frame_nr in frames:
                bpy.context.scene.frame_set(frame_nr)

                for i in range(len(self.camera_list)):
                    camera_save_dir = os.path.join(self.scratch_dir, 'cam', "surveil_{}".format(i) if i else "handcrafted")
                    if not os.path.exists(camera_save_dir):
                        os.makedirs(camera_save_dir)
                    bpy.context.scene.camera = self.camera_list[i]
                    self.set_exr_output_path(os.path.join(self.scratch_dir, "surveil_{}".format(i) if i else "handcrafted", "exr", "frame_"))
                    bpy.context.scene.render.filepath = os.path.join(
                        self.scratch_dir, "surveil_{}".format(i) if i else "handcrafted", "images", f"frame_{frame_nr:04d}.png")

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

def get_intrinsics(scene):

    camdata = scene.camera.data
    focal_length = camdata.lens  # mm
    sensor_width = camdata.sensor_width  # mm
    sensor_height = camdata.sensor_height  # mm
    width, height = scene.render.resolution_x, scene.render.resolution_y
    f_x = focal_length / sensor_width * width
    f_y = focal_length / sensor_height * height
    p_x = width / 2.
    p_y = height / 2.
    return np.array([
        [f_x, 0, -p_x],
        [0, -f_y, -p_y],
        [0,   0,   -1],
    ])



if __name__ == "__main__":
    import sys

    argv = sys.argv

    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]

    print("argsv:{0}".format(argv))
    parser = argparse.ArgumentParser(description='Render Motion in 3D Environment for HuMoR Generation.')
    parser.add_argument('--scene', type=str, metavar='PATH',
                        help='path to specific .ply path for 3D scene',
                        default='')
    parser.add_argument('--background_hdr_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, metavar='PATH', default='../results/human_in_scene',
                        help='img save dir')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=1800)
    parser.add_argument('--samples_per_pixel', type=int, default=128)
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--render_engine', type=str, default='CYCLES', choices=['BLENDER_EEVEE', 'CYCLES'])
    parser.add_argument('--add_fog', default=False, action='store_true')
    parser.add_argument('--fog_path', default=None, type=str)
    parser.add_argument('--randomize', default=False, action='store_true')
    parser.add_argument('--material_path', default=None, type=str)
    parser.add_argument('--views', default=1, type=int)
    args = parser.parse_args(argv)
    print("args:{0}".format(args))

    ## Load the world
    blender_file = args.scene
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    renderer = Blender_render(scratch_dir=output_dir, render_engine=args.render_engine, custom_scene=blender_file, use_gpu=args.use_gpu,
                              background_hdr_path=args.background_hdr_path, samples_per_pixel=args.samples_per_pixel, add_fog=args.add_fog,
                              fog_path=args.fog_path, randomize=args.randomize, material_path=args.material_path,
                              views=args.views)

    frames = range(args.start_frame, args.end_frame)
    renderer.render(frames)


