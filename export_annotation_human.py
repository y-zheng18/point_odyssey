import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='./results/outdoor')
    parser.add_argument('--character_root', type=str, metavar='PATH', default='./data/robots/')
    parser.add_argument('--use_character', type=str, metavar='PATH', default=None)
    parser.add_argument('--camera_root', type=str, metavar='PATH',
                        default='./data/camera_trajectory/MannequinChallenge')
    parser.add_argument('--motion_root', type=str, metavar='PATH', default='./data/motions/')
    parser.add_argument('--partnet_root', type=str, metavar='PATH', default='./data/partnet/')
    parser.add_argument('--gso_root', type=str, metavar='PATH', default='./data/GSO/')
    parser.add_argument('--background_hdr_path', type=str, default='./data/hdri/')
    parser.add_argument('--scene_root', type=str, default='./data/blender_assets/hdri_plane.blend')
    parser.add_argument('--num_assets', type=int, default=5)
    parser.add_argument('--use_gpu', default=False, action='store_true')
    parser.add_argument('--render_engine', type=str, default='CYCLES')
    parser.add_argument('--force_num', type=int, default=5)
    parser.add_argument('--add_force', default=False, action='store_true')
    parser.add_argument('--force_step', type=int, default=3)
    parser.add_argument('--force_interval', type=int, default=120)
    parser.add_argument('--indoor_scale', action='store_true', default=False)

    # rendering settings
    parser.add_argument('--rendering',  default=False, action='store_true')


    # exr settings
    parser.add_argument('--exr',  default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--frame_idx', type=int, default=1)

    # export obj settings
    parser.add_argument('--export_obj',  default=False, action='store_true')

    # export tracking settings
    parser.add_argument('--export_tracking',  default=False, action='store_true')
    parser.add_argument('--sampling_points', type=int, default=5000)
    parser.add_argument('--sampling_scene_points', type=int, default=1000)
    args = parser.parse_args()


    current_path = os.path.dirname(os.path.realpath(__file__))

    if args.rendering:
        rendering_script = 'blender --background --python {}/render_human.py \
        -- --output_dir {} --character_root {} --partnet_root {} --gso_root {} --background_hdr_path {} --scene_root {} --camera_root {} ' \
                           '--num_assets {} --render_engine {} --force_num {} ' \
                           '--force_step {} --force_interval {} '.format(
            current_path, args.output_dir, args.character_root, args.partnet_root, args.gso_root,
            args.background_hdr_path, args.scene_root, args.camera_root, args.num_assets,
            args.render_engine, args.force_num, args.force_step, args.force_interval)
        if args.use_gpu:
            rendering_script += ' --use_gpu'
        if args.indoor_scale:
            rendering_script += ' --indoor'
        os.system(rendering_script)
    if args.export_obj:
        obj_script = 'blender --background --python {}/utils/export_obj.py \
         -- --scene_root {} --output_dir {}'.format(
            current_path, os.path.join(args.output_dir, 'scene.blend'), args.output_dir)
        os.system(obj_script)
    if args.exr:
        exr_script = 'python -m utils.openexr_utils --data_dir {} --output_dir {} --batch_size {} --frame_idx {}'.format(
            args.output_dir, args.output_dir + '/exr_img', args.batch_size, args.frame_idx)
        os.system(exr_script)

    if args.export_tracking:
        tracking_script = 'python -m utils.gen_tracking --data_root {} --cp_root {} --sampling_points {} --sampling_scene_points {}'.format(
            args.output_dir, args.output_dir, args.sampling_points, args.sampling_scene_points)
        os.system(tracking_script)
