import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='../results/animal')
    parser.add_argument('--partnet_root', type=str, metavar='PATH', default='./data/partnet/')
    parser.add_argument('--gso_root', type=str, metavar='PATH', default='./data/GSO/')
    parser.add_argument('--background_hdr_path', type=str, default='./data/hdri/')
    parser.add_argument('--animal_root', type=str, default='./data/deformingthings4d')
    parser.add_argument('--camera_root', type=str, metavar='PATH',
                        default='./data/camera_trajectory/MannequinChallenge')
    parser.add_argument('--num_assets', type=int, default=5)
    parser.add_argument('--use_gpu', default=False, action='store_true')
    parser.add_argument('--render_engine', type=str, default='CYCLES')
    parser.add_argument('--force_num', type=int, default=5)
    parser.add_argument('--add_force', default=False, action='store_true')
    parser.add_argument('--force_step', type=int, default=3)
    parser.add_argument('--force_interval', type=int, default=120)
    parser.add_argument('--material_path', type=str, default='./data/blender_assets/animal_material.blend')
    parser.add_argument('--add_smoke', default=False, action='store_true')
    parser.add_argument('--animal_name', type=str, metavar='PATH', default=None)

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
    parser.add_argument('--vis_num', type=int, default=1000)
    parser.add_argument('--sampling_points', type=int, default=5000)
    parser.add_argument('--sampling_scene_points', type=int, default=1000)
    args = parser.parse_args()


    current_path = os.path.dirname(os.path.realpath(__file__))

    if args.rendering:
        rendering_script = 'blender --background --python {}/render_animal.py \
        -- --output_dir {} --partnet_root {} --gso_root {} --background_hdr_path {} --animal_root {} --camera_root {} ' \
                           '--num_assets {} --render_engine {} --force_num {} ' \
                           '--force_step {} --force_interval {} --material_path {}'.format(
            current_path, args.output_dir, args.partnet_root, args.gso_root,
            args.background_hdr_path, args.animal_root, args.camera_root, args.num_assets,
            args.render_engine, args.force_num, args.force_step, args.force_interval, args.material_path)
        if args.use_gpu:
            rendering_script += ' --use_gpu'
        if args.add_force:
            rendering_script += ' --add_force'
        if args.add_smoke:
            rendering_script += ' --add_smoke'
        if args.animal_name is not None:
            rendering_script += ' --animal_name {}'.format(args.animal_name)
        os.system(rendering_script)
    if args.exr:
        exr_script = 'python -m utils.openexr_utils --data_dir {} --output_dir {} --batch_size {} --frame_idx {}'.format(
            args.output_dir, args.output_dir + '/exr_img', args.batch_size, args.frame_idx)
        os.system(exr_script)
    if args.export_obj:
        obj_script = 'blender --background --python {}/utils/export_obj.py \
         -- --scene_root {} --output_dir {}'.format(
            current_path, os.path.join(args.output_dir, 'scene.blend'), args.output_dir)
        os.system(obj_script)
    if args.export_tracking:
        tracking_script = 'python -m utils.gen_tracking --data_root {} --sampling_points {} --visualize_points {}'.format(
            args.output_dir, args.sampling_points, args.vis_num)
        os.system(tracking_script)
        tracking_script = 'python -m utils.gen_tracking_ground --data_root {} --sampling_points {} --visualize_points {}'.format(
            args.output_dir, args.sampling_scene_points, args.vis_num)
        os.system(tracking_script)

    # rm redundant files
    current_path = os.path.dirname(os.path.realpath(__file__))
    output_dir_abs = os.path.abspath(args.output_dir)
    os.system('cd {} && rm -rf ./*.blend1 && rm -r exr && rm -r tmp'.format(output_dir_abs))
    os.system('cd {}'.format(current_path))