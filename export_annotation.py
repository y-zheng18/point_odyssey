import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=str, default='./data/demo_scene/robot.blend')
    parser.add_argument('--save_dir', type=str, default='./results/robot_demo')

    # rendering settings
    parser.add_argument('--rendering',  default=False, action='store_true')
    parser.add_argument('--background_hdr_path', type=str,
                        default='./data/hdri/OutdoorHDRI028_4K-HDR.exr')
    parser.add_argument('--start_frame', type=int, default=0)

    parser.add_argument('--add_fog', default=False, action='store_true')
    parser.add_argument('--fog_path', default='./data/blender_assets/fog.blend', type=str)
    parser.add_argument('--end_frame', type=int, default=1100)
    parser.add_argument('--samples_per_pixel', type=int, default=1024)
    parser.add_argument('--use_gpu',  default=False, action='store_true')
    parser.add_argument('--randomize', default=False, action='store_true')
    parser.add_argument('--material_path', default='./data/blender_assets/materials.blend', type=str)

    # exr settings
    parser.add_argument('--exr',  default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--frame_idx', type=int, default=1)

    # export obj settings
    parser.add_argument('--export_obj',  default=False, action='store_true')
    parser.add_argument('--ignore_character',  default=False, action='store_true')

    # export tracking settings
    parser.add_argument('--export_tracking',  default=False, action='store_true')
    parser.add_argument('--vis_num', type=int, default=0)
    parser.add_argument('--sampling_scene_num', type=int, default=20000)
    parser.add_argument('--sampling_character_num', type=int, default=5000)
    args = parser.parse_args()


    current_path = os.path.dirname(os.path.realpath(__file__))

    if args.rendering:
        rendering_script = 'blender --background --python {}/render_single.py \
        -- --output_dir {} --scene {} --render_engine CYCLES --start_frame {} --end_frame {} --samples_per_pixel {} --background_hdr_path {}'.format(
            current_path, args.save_dir, args.scene_dir, args.start_frame, args.end_frame, args.samples_per_pixel, args.background_hdr_path)
        if args.use_gpu:
            rendering_script += ' --use_gpu'
        if args.add_fog:
            rendering_script += ' --add_fog'
            rendering_script += ' --fog_path {}'.format(args.fog_path)
        if args.randomize:
            rendering_script += ' --randomize'
        if args.material_path is not None:
            rendering_script += ' --material_path {}'.format(args.material_path)
        os.system(rendering_script)
    if args.exr:
        exr_script = 'python -m utils.openexr_utils --data_dir {} --output_dir {} --batch_size {} --frame_idx {}'.format(
            args.save_dir, args.save_dir + '/exr_img', args.batch_size, args.frame_idx)
        os.system(exr_script)
    if args.export_obj:
        obj_script = 'blender --background --python {}/utils/export_scene.py \
         -- --scene_root {} --output_dir {} --export_character {}'.format(
            current_path, args.scene_dir, args.save_dir, not args.ignore_character)
        os.system(obj_script)
    if args.export_tracking:
        tracking_script = 'python -m utils.gen_tracking_indoor --data_root {} --sampling_scene_num {} --sampling_character_num {} --visualize_num {}'.format(
            args.save_dir, args.sampling_scene_num, args.sampling_character_num, args.vis_num)
        os.system(tracking_script)