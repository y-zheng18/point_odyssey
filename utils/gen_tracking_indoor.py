import numpy as np
import cv2
import os
import glob
import matplotlib
from utils.file_io import read_tiff, write_png
from tqdm import tqdm
import trimesh
import shutil


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
    XYZ = (RT @ v.T).T[:, :3]
    Z = XYZ[:, 2:]
    XYZ = XYZ / XYZ[:, 2:]
    xyz = (K @ XYZ.T).T
    uv = xyz[:, :2]
    return uv, Z


def check_visibility(uv, z, depth, h, w):
    visibility = np.zeros((uv.shape[0], 1))
    for j in range(len(uv)):
        u, v = uv[j]
        if u < 0 or u >= w or v < 0 or v >= h:
            visibility[j] = 0
            # print('out of range')
            continue
        else:
            v_low = np.floor(uv[j, 1]).astype(np.int32)
            v_high = np.min([np.ceil(uv[j, 1]).astype(np.int32), h - 1])
            u_low = np.floor(uv[j, 0]).astype(np.int32)
            u_high = np.min([np.ceil(uv[j, 0]).astype(np.int32), w - 1])
            # find nearest depth
            d_max = np.max(depth[v_low:v_high + 1, u_low:u_high + 1])
            d_min = np.min(depth[v_low:v_high + 1, u_low:u_high + 1])
            d_median = np.median(depth[v_low:v_high + 1, u_low:u_high + 1])
            if z[j] < 0 or z[j] > 1000:
                visibility[j] = 2
                # print('invalid depth')
                continue
            if d_max >= 0.97 * z[j] and d_min <= 1.05 * z[j] and z[j] > 0.95 * d_median and z[j] < 1.05 * d_median:
                visibility[j] = 1
    return visibility


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


def tracking(cp_root: str, data_root: str, sampling_scene_num=100000, sampling_character_num=5000):
    img_root = os.path.join(cp_root, 'images')
    exr_root = os.path.join(cp_root, 'exr_img')
    cam_root = os.path.join(cp_root, 'cam')
    obj_root = os.path.join(cp_root, 'obj')


    print('copying exr data ...')

    save_rgbs_root = os.path.join(data_root, 'rgbs')
    save_depths_root = os.path.join(data_root, 'depths')
    save_masks_root = os.path.join(data_root, 'masks')
    save_normals_root = os.path.join(data_root, 'normals')

    os.makedirs(save_rgbs_root, exist_ok=True)
    os.makedirs(save_depths_root, exist_ok=True)
    os.makedirs(save_masks_root, exist_ok=True)
    os.makedirs(save_normals_root, exist_ok=True)

    frames = sorted(glob.glob(os.path.join(img_root, '*.png')))

    # save_vis_dir = os.path.join(data_root, 'tracking')

    tracking_results = []
    tracking_results_3d = []
    K_data = []
    RT_data = []

    R1 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    R2 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    R3 = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    scene_mesh = trimesh.load(os.path.join(obj_root, 'scene.obj'))
    scene_points = trimesh.sample.sample_surface(scene_mesh, sampling_scene_num)[0]
    print('scene points shape', scene_points.shape)

    # filter out points that are invisible in the most of the frames
    print('filtering...')
    mask_s = np.zeros((scene_points.shape[0], 1), dtype=np.bool)
    for i in tqdm(range(len(frames) // 50)):
        K = np.loadtxt(os.path.join(cam_root, 'K_{}.txt'.format(str(i * 50 + 1).zfill(4))))
        RT = np.loadtxt(os.path.join(cam_root, 'RT_{}.txt'.format(str(i * 50 + 1).zfill(4))))
        RT = R3 @ R2 @ RT @ R1
        depth = read_tiff(os.path.join(exr_root, 'depth_{}.tiff'.format(str(i * 50 + 1).zfill(5))))
        h, w, _ = depth.shape
        uv, z = reprojection(scene_points, K, RT, h, w)
        visibility = check_visibility(uv, z, depth, h, w)
        mask_s = mask_s | (visibility == 1)
    print('filtering done')
    mask_idx = np.where(mask_s == 1)[0]
    scene_points = scene_points[mask_idx]
    print('scene points shape', scene_points.shape)

    if sampling_character_num > 0:
        c_obj, _ = read_obj_file(os.path.join(obj_root, 'character_0001.obj'))
        sampling_idx = np.random.choice(len(c_obj), sampling_character_num, replace=False if len(c_obj) > sampling_character_num else True)
    else:
        sampling_idx = None

    for i in tqdm(range(0, len(frames) - 1)):
        K = np.loadtxt(os.path.join(cam_root, 'K_{}.txt'.format(str(i + 1).zfill(4))))
        RT = np.loadtxt(os.path.join(cam_root, 'RT_{}.txt'.format(str(i + 1).zfill(4))))
        RT = R3 @ R2 @ RT @ R1
        depth = read_tiff(os.path.join(exr_root, 'depth_{}.tiff'.format(str(i + 1).zfill(5))))
        img = cv2.imread(os.path.join(exr_root, 'rgb_{}.png'.format(str(i + 1).zfill(5))))
        h, w, _ = img.shape

        # convert img to jpg
        save_img_path = os.path.join(save_rgbs_root, 'rgb_{}.jpg'.format(str(i).zfill(5)))
        cv2.imwrite(save_img_path, img)

        # convert depth to 16 bit png
        save_depth_path = os.path.join(save_depths_root, 'depth_{}.png'.format(str(i).zfill(5)))
        max_value = 1000
        min_value = 0
        data = depth.copy()
        data[data > max_value] = max_value
        data[data < min_value] = min_value
        data = (data - min_value) * 65535 / (max_value - min_value)
        data = data.astype(np.uint16)
        write_png(data, save_depth_path)

        # cp normals and masks
        save_normal_path = os.path.join(save_normals_root, 'normal_{}.jpg'.format(str(i).zfill(5)))
        save_mask_path = os.path.join(save_masks_root, 'mask_{}.png'.format(str(i).zfill(5)))
        save_normal = cv2.imread(os.path.join(exr_root, 'normal_{}.png'.format(str(i + 1).zfill(5))))
        cv2.imwrite(save_normal_path, save_normal)

        if os.path.exists(os.path.join(exr_root, 'segmentation_{}.png'.format(str(i + 1).zfill(5)))):
            shutil.copy(os.path.join(exr_root, 'segmentation_{}.png'.format(str(i + 1).zfill(5))), save_mask_path)

        uv, z = reprojection(scene_points, K, RT, h, w)
        visibility = check_visibility(uv, z, depth, h, w)
        save_trajs_3d = scene_points
        if sampling_character_num > 0:
            c_obj, _ = read_obj_file(os.path.join(obj_root, 'character_{}.obj'.format(str(i + 1).zfill(4))))
            c_obj = np.array(c_obj)[sampling_idx]
            uv_, z_ = reprojection(c_obj, K, RT, h, w)
            visibility_ = check_visibility(uv_, z_, depth, h, w)
            uv = np.concatenate([uv, uv_], axis=0)
            visibility = np.concatenate([visibility, visibility_], axis=0)
            save_trajs_3d = np.concatenate([save_trajs_3d, c_obj], axis=0)

        tracking_results.append(np.concatenate((uv, visibility), axis=1).astype(np.float16))
        tracking_results_3d.append(save_trajs_3d.astype(np.float16))
        K_data.append(K.astype(np.float16))
        RT_data.append(RT.astype(np.float16))
    tracking_results = np.stack(tracking_results, axis=0)
    tracking_results = tracking_results.astype(np.float16)

    tracking_results_3d = np.stack(tracking_results_3d, axis=0)
    tracking_results_3d = tracking_results_3d.astype(np.float16)

    K_data = np.stack(K_data, axis=0)
    K_data = K_data.astype(np.float16)

    RT_data = np.stack(RT_data, axis=0)
    RT_data = RT_data.astype(np.float16)

    # save annotations as npz
    np.savez(os.path.join(data_root, 'annotations.npz'),
             trajs_2d=tracking_results[:, :, :2],
             trajs_3d=tracking_results_3d,
             visibilities=tracking_results[:, :, 2],
             intrinsics=K_data,
             extrinsics=RT_data)

    return tracking_results



if __name__ == '__main__':
    import argparse

    np.random.seed(128)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/Users/yangzheng/code/project/long-term-tracking/data/scenes/render0')
    parser.add_argument('--cp_root', type=str, default='/Users/yangzheng/code/project/long-term-tracking/data/scenes/render0')
    parser.add_argument('--sampling_scene_num', type=int, default=20000)
    parser.add_argument('--sampling_character_num', type=int, default=5000)
    args = parser.parse_args()

    tracking(args.cp_root, args.data_root, args.sampling_scene_num, args.sampling_character_num)