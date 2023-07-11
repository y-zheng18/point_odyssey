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



def tracking(data_root: str, save_dir: str, sampling_num=1000, visualize_num=2, depth_max=500, scale=1):
    obj_root = os.path.join(data_root, 'obj')
    img_root = os.path.join(data_root, 'images')
    exr_root = os.path.join(args.data_root, 'exr_img')
    cam_root = os.path.join(args.data_root, 'cam')

    os.makedirs(save_dir, exist_ok=True)
    tracking_results = []
    tracking_results_vis_id = []
    frames = sorted(glob.glob(os.path.join(img_root, '*.png')))


    # sample random 3d points on the ground from (-3, -3, 0) to (3, 3, 0)
    tracking_points = np.random.uniform(-4, 4, (sampling_num, 3))
    tracking_points[:, 1] *= 0
    # tracking_points[sampling_num // 2:, [0, 2]] *= 10
    tracking_points *= scale
    search_list = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])

    d_threshold = 0.2 * scale
    for i in tqdm(range(0, len(frames) - 1)):
        K = np.loadtxt(os.path.join(cam_root, 'K_{}.txt'.format(str(i + 2).zfill(4))))
        RT = np.loadtxt(os.path.join(cam_root, 'RT_{}.txt'.format(str(i + 2).zfill(4))))
        depth = read_tiff(os.path.join(exr_root, 'depth_{}.tiff'.format(str(i + 1).zfill(5))))
        mask = cv2.imread(os.path.join(exr_root, 'segmentation_{}.png'.format(str(i + 1).zfill(5))))
        img = cv2.imread(os.path.join(exr_root, 'rgb_{}.png'.format(str(i + 1).zfill(5))))
        h, w, _ = img.shape


        uv, z = reprojection(tracking_points, K, RT, h, w)
        uv[:, 0] = w - uv[:, 0]
        # uv = np.round(uv).astype(np.int32)
        visibility = np.zeros((uv.shape[0], 1))
        # find nearest depth
        asset_mask = np.logical_and(mask[:, :, 2] == 0,
                                    mask[:, :, 1] == 0)
        asset_mask = np.logical_and(asset_mask, mask[:, :, 0] == 0)
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

        tracking_results.append(np.concatenate((uv, visibility), axis=1))
        tracking_results_vis_id.append(np.linspace(0, len(uv) - 1, visualize_num).astype(np.int32))

        # draw_uv = tracking_results[i][tracking_results_vis_id[i]]
        # visible_uv = draw_uv[draw_uv[:, 2] == 1][:, :2]
        # invisible_uv = draw_uv[draw_uv[:, 2] == 0][:, :2]
        # if len(visible_uv):
        #     # img_back = img.copy()
        #     for u, v in visible_uv:
        #         if u < 0 or u >= w - 1 or v < 0 or v >= h - 1:
        #             continue
        #         cv2.circle(img, (int(u), int(v)), 2, (0, 0, 216), -1, lineType=cv2.LINE_AA)
        #         # img[round(v), round(u), :] = (0, 0, 216)
        #     # img = cv2.addWeighted(img_back, 0.1, img, 0.9, 0)
        # if len(invisible_uv):
        #     # img_back = img.copy()
        #     for u, v in invisible_uv:
        #         if u < 0 or u >= w - 1 or v < 0 or v >= h - 1:
        #             continue
        #         cv2.circle(img, (int(u), int(v)), 2, (0, 216, 0), -1, lineType=cv2.LINE_AA)
        #         # img[round(v), round(u), :] = (216, 0, 0)
            # img = cv2.addWeighted(img_back, 0.5, img, 0.5, 0)
        # cv2.imwrite(os.path.join(save_dir, 'tracking_ground_{}.png'.format(str(i + 1).zfill(4))), img)
    tracking_results = np.stack(tracking_results, axis=0)

    return tracking_results



if __name__ == '__main__':
    import argparse

    np.random.seed(128)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./results/robot_dome')
    parser.add_argument('--sampling_points', type=int, default=1000)
    parser.add_argument('--visualize_points', type=int, default=2)
    parser.add_argument('--depth_max', type=float, default=500)
    parser.add_argument('--scale', type=float, default=10)
    parser.add_argument('--temporal_window', type=int, default=10)
    parser.add_argument('--farthest_sampling', action='store_true', default=False)

    args = parser.parse_args()
    exr_root = os.path.join(args.data_root, 'exr_img')
    obj_root = os.path.join(args.data_root, 'obj')
    cam_root = os.path.join(args.data_root, 'cam')

    save_dir = os.path.join(args.data_root, 'tracking_ground')

    frames = sorted(glob.glob(os.path.join(args.data_root, 'images', '*.png')))


    tracking_results = tracking(args.data_root, save_dir, args.sampling_points, args.visualize_points, args.depth_max, args.scale)
    np.save(os.path.join(args.data_root, 'tracking_ground_results.npy'), tracking_results)
