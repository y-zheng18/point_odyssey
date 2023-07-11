import numpy as np
import cv2
import os
import glob
import matplotlib
from utils.file_io import read_tiff
from tqdm import tqdm
import json
import utils.plotting as plotting


def read_obj_file(obj_path:str):
    '''
        Load .obj file, return vertices, faces.
        return: vertices: N_v X 3, faces: N_f X 3
        '''
    obj_f = open(obj_path, 'r')
    lines = obj_f.readlines()
    vertices = []
    faces = []
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
    obj_f.close()

    return np.asarray(vertices), np.asarray(faces)


def reprojection(points: np.ndarray, K: np.ndarray, RT: np.ndarray, h: int, w: int):

    v = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    model_R = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    XYZ = (RT @ model_R @ v.T).T[:, :3]
    Z = -XYZ[:, 2:]
    depth = np.linalg.norm(XYZ, axis=1)
    XYZ = XYZ / XYZ[:, 2:]
    xyz = (K @ XYZ.T).T
    uv = xyz[:, :2]
    return uv, Z


def farthest_point_sampling(p, K):
    """
    greedy farthest point sampling
    p: point cloud
    K: number of points to sample
    """

    farthest_point = np.zeros((K, 3))
    idx = []
    max_idx = np.random.randint(0, p.shape[0] -1)
    farthest_point[0] = p[max_idx]
    idx.append(max_idx)
    print('farthest point sampling')
    for i in range(1, K):
        pairwise_distance = np.linalg.norm(p[:, None, :] - farthest_point[None, :i, :], axis=2)
        distance = np.min(pairwise_distance, axis=1, keepdims=True)
        max_idx = np.argmax(distance)
        farthest_point[i] = p[max_idx]
        idx.append(max_idx)
    print('farthest point sampling done')
    return farthest_point, idx



def tracking(data_root: str, save_dir: str, tracking_index_list: list, pallette: np.array, visualize_num=2, depth_max=500, scale=1, temporal_window=10):
    obj_root = os.path.join(data_root, 'obj')
    img_root = os.path.join(data_root, 'images')
    exr_root = os.path.join(args.data_root, 'exr_img')
    cam_root = os.path.join(args.data_root, 'cam')

    os.makedirs(save_dir, exist_ok=True)
    tracking_results = []
    tracking_results_vis_id = []
    frames = sorted(glob.glob(os.path.join(img_root, '*.png')))

    search_list = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])
    d_threshold = 0.2 * scale
    for i in tqdm(range(0, len(frames) - 1)):
        tracking_results.append([])
        tracking_results_vis_id.append([])
        K = np.loadtxt(os.path.join(cam_root, 'K_{}.txt'.format(str(i + 2).zfill(4))))
        RT = np.loadtxt(os.path.join(cam_root, 'RT_{}.txt'.format(str(i + 2).zfill(4))))
        depth = read_tiff(os.path.join(exr_root, 'depth_{}.tiff'.format(str(i + 1).zfill(5))))
        mask = cv2.imread(os.path.join(exr_root, 'segmentation_{}.png'.format(str(i + 1).zfill(5))))
        img = cv2.imread(os.path.join(exr_root, 'rgb_{}.png'.format(str(i + 1).zfill(5))))
        h, w, _ = img.shape
        for idx in range(len(assets)):
            asset = assets[idx]
            tracking_idx = tracking_index_list[idx]
            if not len(tracking_idx):
                continue
            asset_name = asset.replace('.', '_')
            obj_path = os.path.join(obj_root, '{}_{}.obj'.format(asset_name, str(i + 2).zfill(4)))
            obj_v, obj_f = read_obj_file(obj_path)
            # print(obj_v.shape, tracking_idx, obj_path)
            uv, z = reprojection(obj_v[tracking_idx], K, RT, h, w)
            uv[:, 0] = w - uv[:, 0]
            # uv = np.round(uv).astype(np.int32)
            visibility = np.zeros((uv.shape[0], 1))
            # find nearest depth
            asset_mask = np.logical_and(mask[:, :, 2] == pallette[idx + 1, 0],
                                        mask[:, :, 1] == pallette[idx + 1, 1])
            asset_mask = np.logical_and(asset_mask, mask[:, :, 0] == pallette[idx + 1, 2])
            # masked depth
            # depth_masked = depth * asset_mask[:, :, None].astype(np.float32)
            #print(np.sum(asset_mask))

            for j in range(len(uv)):
                u, v = uv[j]
                if u < 0 or u >= w or v < 0 or v >= h:
                    visibility[j] = 0
                    #print('out of range')
                    continue
                else:
                    ## sampling on the neighborhood
                    # blocked = False
                    # changed_flag = False
                    for delta_uv in search_list:
                        u_, v_ = np.floor(uv[j]).astype(np.int32) + delta_uv
                        if u_ < 0 or u_ >= w or v_ < 0 or v_ >= h:
                            continue
                        if asset_mask[int(v_), int(u_)]:
                            if not asset_mask[int(v), int(u)]:
                                uv[j] = np.array([u_, v_])
                                break
                    #             changed_flag = True
                    #
                    #         continue
                    #     else: #if mask[int(v_), int(u_), 0] > 0 and mask[int(v_), int(u_), 1] > 0 and mask[int(v_), int(u_), 2] > 0:
                    #         blocked = True
                    # if blocked:
                    #     visibility[j] = 0
                    #     uv[j] = np.round(uv[j]).astype(np.int32)
                    #     confidence[j] = 1
                    #     continue
                    v_low = np.floor(uv[j, 1]).astype(np.int32)
                    v_high = np.min([np.ceil(uv[j, 1]).astype(np.int32), h - 1])
                    u_low = np.floor(uv[j, 0]).astype(np.int32)
                    u_high = np.min([np.ceil(uv[j, 0]).astype(np.int32), w - 1])
                    # find nearest depth
                    d_max = np.max(depth[v_low:v_high + 1, u_low:u_high + 1])
                    d_median = np.median(depth[v_low:v_high + 1, u_low:u_high + 1])
                    # delta_min = np.max((np.abs(depth_masked[v_low:v_high, u_low:u_high] - z[j])).reshape(-1))
                    # uv[j] = np.round(uv[j]).astype(np.int32)
                    if z[j] < 0 or z[j] > depth_max:
                        visibility[j] = 2
                        # print('invalid depth')
                        continue
                    # d_rendered = depth[valid_v, valid_u]
                    if d_max >= 0.97 * z[j] and z[j] > 0.95 * d_median and z[j] < 1.05 * d_median: #and d_max - z[j] * 0.99 < d_threshold * scale:
                        visibility[j] = 1

            tracking_results[i].append(np.concatenate((uv, visibility), axis=1))
            tracking_results_vis_id[i].append(np.arange(visualize_num))
            for pre_idx in range(len(tracking_results[i]) - 1):
                tracking_results_vis_id[i][-1] += len(tracking_results[i][pre_idx]) - 1


        tracking_results[i] = np.concatenate(tracking_results[i], axis=0)
        tracking_results_vis_id[i] = np.concatenate(tracking_results_vis_id[i], axis=0)

        tracking_vis_id = np.linspace(0, len(tracking_results[i]) - 1, visualize_num).astype(np.int32)
        draw_uv = tracking_results[i][tracking_vis_id]
        visible_uv = draw_uv[draw_uv[:, 2] == 1][:, :2]
        invisible_uv = draw_uv[draw_uv[:, 2] == 0][:, :2]
        if len(visible_uv):
            # img_back = img.copy()
            for u, v in visible_uv:
                if u < 0 or u >= w - 1 or v < 0 or v >= h - 1:
                    continue
                cv2.circle(img, (int(u), int(v)), 2, (0, 0, 216), -1, lineType=cv2.LINE_AA)
                # img[round(v), round(u), :] = (0, 0, 216)
            # img = cv2.addWeighted(img_back, 0.1, img, 0.9, 0)
        if len(invisible_uv):
            # img_back = img.copy()
            for u, v in invisible_uv:
                if u < 0 or u >= w - 1 or v < 0 or v >= h - 1:
                    continue
                cv2.circle(img, (int(u), int(v)), 2, (0, 216, 0), -1, lineType=cv2.LINE_AA)
                # img[round(v), round(u), :] = (216, 0, 0)
            # img = cv2.addWeighted(img_back, 0.5, img, 0.5, 0)
        cv2.imwrite(os.path.join(save_dir, 'tracking_{}.png'.format(str(i + 1).zfill(4))), img)
    tracking_results = np.stack(tracking_results, axis=0)

    return tracking_results



if __name__ == '__main__':
    import argparse

    np.random.seed(128)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./results/robot_dome')
    parser.add_argument('--sampling_points', type=int, default=5000)
    parser.add_argument('--visualize_points', type=int, default=2)
    parser.add_argument('--depth_max', type=float, default=500)
    parser.add_argument('--scale', type=float, default=10)
    parser.add_argument('--temporal_window', type=int, default=10)
    parser.add_argument('--farthest_sampling', action='store_true', default=False)

    args = parser.parse_args()
    exr_root = os.path.join(args.data_root, 'exr_img')
    obj_root = os.path.join(args.data_root, 'obj')
    cam_root = os.path.join(args.data_root, 'cam')

    save_dir = os.path.join(args.data_root, 'tracking')
    scene_info = json.load(open(os.path.join(args.data_root, 'scene_info.json'), 'r'))
    #assets = ['background'] + [i for i in scene_info['assets'] if  'robot' in i or 'Rock' in i or 'Cube' in i] + ['Sphere']  # Sphere is the dome
    assets = scene_info['assets']
    character_name = scene_info['character'] if 'character' in scene_info.keys() else None
    character_assets = [i for i in assets if character_name in i] if character_name else []
    obj_assets = [i for i in assets if i not in character_assets and i != 'background']
    character_sampling_num = args.sampling_points // len(character_assets) if character_name else 0
    obj_sampling_num = args.sampling_points // len(obj_assets)
    print('character sampling num: {}, object sampling num: {}'.format(character_sampling_num, obj_sampling_num))
    print('character assets: ', character_assets)
    print('object assets: ', obj_assets)
    print(args.sampling_points // len(obj_assets))
    pallette = plotting.hls_palette(len(assets) + 2)


    frames = sorted(glob.glob(os.path.join(args.data_root, 'images', '*.png')))

    tracking_index_list = []
    tracking_results = []
    for idx in range(len(assets)):
        asset = assets[idx]
        print('tracking asset: {}'.format(asset))
        asset_name = asset.replace('.', '_')
        tracking_index = []
        max_points = 0
        for i in range(0, len(frames) - 2):
            if not os.path.exists(os.path.join(obj_root, '{}_{}.obj'.format(asset_name, str(i + 2).zfill(4)))):
                continue
            ## only sample points when they are visible in the first frame
            initial_depth = read_tiff(os.path.join(exr_root, 'depth_{}.tiff'.format(str(i + 1).zfill(5))))
            initial_mask = cv2.imread(os.path.join(exr_root, 'segmentation_{}.png'.format(str(i + 1).zfill(5))))
            initial_rgb = cv2.imread(os.path.join(exr_root, 'rgb_{}.png'.format(str(i + 1).zfill(5))))

            h, w, _ = initial_depth.shape

            inital_points, _ = read_obj_file(os.path.join(obj_root, '{}_{}.obj'.format(asset_name, str(i + 2).zfill(4))))
            inital_K = np.loadtxt(os.path.join(cam_root, 'K_{}.txt'.format(str(i + 2).zfill(4))))
            inital_RT = np.loadtxt(os.path.join(cam_root, 'RT_{}.txt'.format(str(i + 2).zfill(4))))
            uv, z = reprojection(inital_points, inital_K, inital_RT, h, w)
            uv[:, 0] = w - uv[:, 0]
            uv = np.round(uv).astype(np.int32)

            asset_mask = np.logical_and(initial_mask[:, :, 2] == pallette[idx + 1, 0],
                                        initial_mask[:, :, 1] == pallette[idx + 1, 1])
            asset_mask = np.logical_and(asset_mask, initial_mask[:, :, 0] == pallette[idx + 1, 2])

            initial_depth = initial_depth #* asset_mask[:, :, None]

            uv_mask_0 = np.logical_and(np.logical_and(uv[:, 0] >= 2, uv[:, 0] < w - 3), np.logical_and(uv[:, 1] >= 2, uv[:, 1] < h - 3))[:, None]
            uv_idx_0 = np.where(uv_mask_0)[0]
            if len(uv_idx_0) == 0:
                continue
            uv = uv[uv_idx_0] 
            uv_mask = z[uv_idx_0] <= initial_depth[uv[:, 1], uv[:, 0]]

            # print('uv_mask: ', uv_mask[:args.visualize_points])

            # to make sure points are not on the boundary
            for delta_u in range(-2, 3):
                for delta_v in range(-2, 3):
                    uv_mask = np.logical_and(uv_mask, 0.995 * z[uv_idx_0] <= initial_depth[uv[:, 1] + delta_v, uv[:, 0] + delta_u])
                    # print(delta_v, delta_u, asset_mask[uv[:, 1] + delta_v, uv[:, 0] + delta_u][:args.visualize_points])
            # print('uv_mask: ', uv_mask[:args.visualize_points])

            uv_idx = np.where(uv_mask)[0]
            tracking_num = character_sampling_num if asset in character_assets else obj_sampling_num

            if len(uv_idx) and max_points < len(uv_idx):
                max_points = len(uv_idx)

                # print('pointnum:{}/{}, enough points in frame {}'.format(len(uv_idx), len(z), i + 2))
                # print(uv_idx)
                if len(uv_idx) <= tracking_num:
                    sample_idx = np.arange(len(uv_idx))
                else:
                    sample_idx = np.random.choice(np.arange(len(uv_idx)), tracking_num, replace=False)
                if args.farthest_sampling:
                    _, sample_idx = farthest_point_sampling(inital_points[uv_idx_0][uv_idx], args.sampling_points)
                # sample_idx = np.random.choice(len(uv_idx), args.sampling_points, replace=False if args.sampling_points < len(uv_idx) else True)

                tracking_index = uv_idx_0[uv_idx[sample_idx]]
            if (len(uv_idx) >= 0.05 * len(inital_points) and i > len(frames) // 3) or len(uv_idx) > tracking_num:
                # print('pointnum:{}/{}, not enough points in frame {}'.format(len(uv_idx), len(z), i + 2))
                print('assets: {}, frame: {}, tracking points: {}'.format(asset, i + 2, len(tracking_index)))
                break
        print('assets: {}, tracking points: {}'.format(asset, len(tracking_index)))
        tracking_index_list.append(tracking_index)

        # print(tracking_index)
    # print(tracking_index_list)
    tracking_results = tracking(args.data_root, save_dir, tracking_index_list, pallette, args.visualize_points, args.depth_max, args.scale, temporal_window=args.temporal_window)
    np.save(os.path.join(args.data_root, 'tracking_results.npy'), tracking_results)
