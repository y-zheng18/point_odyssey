import cv2
import numpy as np
from matplotlib import cm
import os
import glob
from tqdm import tqdm

def writing_video(rgb_list,
                  save_path: str, frame_rate: int = 24):

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    h, w, _ = rgb_list[0].shape
    out = cv2.VideoWriter(save_path, fourcc, frame_rate, (w, h))

    for img in rgb_list:
        out.write(img)

    out.release()
    return

def farthest_point_sampling(p, K):
    """
    greedy farthest point sampling
    p: point cloud
    K: number of points to sample
    """

    farthest_point = np.zeros((K, 2))
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

def draw_traj_on_image_py(rgb, traj, S=50, linewidth=1, show_dots=False, cmap='coolwarm', maxdist=None):
    # all inputs are numpy tensors
    # rgb is 3 x H x W
    # traj is S x 2

    H, W, C = rgb.shape
    assert (C == 3)

    rgb_back = rgb.astype(np.uint8).copy()
    S1, D = traj.shape

    color_map = cm.get_cmap(cmap)
    S1, D = traj.shape

    for s in range(S1 - 1):
        if traj[s, 2] > 1 or traj[s + 1, 2] > 1:
            # invalid
            continue
        if traj[s, 0] < 0 or traj[s, 0] >= W or traj[s, 1] < 0 or traj[s, 1] >= H:
            continue
        if traj[s + 1, 0] < 0 or traj[s + 1, 0] >= W or traj[s + 1, 1] < 0 or traj[s + 1, 1] >= H:
            continue
        color = np.array(color_map((s) / max(1, float(S - 2)))[:3]) * 255 # rgb
        cv2.line(rgb_back,
                 (int(traj[s, 0]), int(traj[s, 1])),
                 (int(traj[s + 1, 0]), int(traj[s + 1, 1])),
                 color,
                 linewidth,
                 cv2.LINE_AA)
        if show_dots:
            cv2.circle(rgb_back, (traj[s, 0], traj[s, 1]), linewidth, color, -1)

    if maxdist is not None:
        val = (np.sqrt(np.sum((traj[-1] - traj[0]) ** 2)) / maxdist).clip(0, 1)
        color = np.array(color_map(val)[:3]) * 255  # rgb
    else:
        # draw the endpoint of traj, using the next color (which may be the last color)
        color = np.array(color_map((S1 - 1) / max(1, float(S - 2)))[:3]) * 255  # rgb

    # color = np.array(color_map(1.0)[:3]) * 255
    rgb = rgb_back.astype(np.uint8)
    if traj[-1, 2] < 2 and traj[-1, 0] >= 0 and traj[-1, 0] < W and traj[-1, 1] >= 0 and traj[-1, 1] < H:
        cv2.circle(rgb, (traj[-1, 0], traj[-1, 1]), linewidth, color, -1)
    # rgb = cv2.addWeighted(rgb_back, 0.5, rgb, 0.5, 0)

    return rgb

def summ_traj2ds_on_rgbs(data_root, sample_points=1000, S=16, linewidth=1, show_dots=False, cmap='spring', maxdist=None, frame_idx=0, frame_end=1000):
    # trajs is S, N, 2
    # rgbs is S, C, H, W
    img_root = os.path.join(data_root, 'images')
    save_root = os.path.join(data_root, 'traj1')

    os.makedirs(save_root, exist_ok=True)
    # trajs_path = os.path.join(data_root, 'ground_truth')
    # trajs = []
    # for i in range(frame_idx, frame_end):
    #     traj = np.load(os.path.join(trajs_path, '%05d.npy' % (i + 1)))
    #     trajs.append(traj)
    # trajs = np.stack(trajs, axis=0)
    trajs = np.load(os.path.join(data_root, 'tracking_results.npy'))[frame_idx:frame_end]
    if os.path.exists(os.path.join(data_root, 'tracking_ground_results.npy')):
        print('adding ground points')
        trajs_ = np.load(os.path.join(data_root, 'tracking_ground_results.npy'))[frame_idx:frame_end]
        trajs = np.concatenate([trajs, trajs_], axis=1)
    trajs = trajs.astype(np.float64)
    trajs = np.round(trajs).astype(np.int32)
    print('trajs', trajs.shape)

    img_list = sorted(glob.glob(os.path.join(img_root, '*.png')))[1 + frame_idx:frame_end + 1]

    vis0 = trajs[0, :, 2] == 1
    valids = trajs[:, :, 2] < 2
    vis_idx = np.where(vis0)[0]
    trajs = trajs[:, vis_idx, :]
    S2, N, D = trajs.shape
    print('vis points on frame 0', trajs.shape)

    # sample_idx = np.linspace(0, N - 1, sample_points).astype(np.int32)
    _, sample_idx = farthest_point_sampling(trajs[0, :, :2], sample_points)


    # print('valids', valids.shape)

    rgbs_color = []

    print('loading images from %s' % img_root)
    for s in tqdm(range(S2)):
        img = cv2.imread(img_list[s])
        H, W, C = img.shape
        assert (C == 3)
        img = img.astype(np.uint8).copy()
        rgbs_color.append(img)
    H, W, _ = rgbs_color[0].shape

    writing_video(rgbs_color, os.path.join(save_root, 'video.mp4'), frame_rate=30)

    for i, point_id in tqdm(enumerate(sample_idx)):


        traj = trajs[:, point_id].astype(np.int32)  # S, 2
        valid = valids[:, point_id]  # S,

        # print('traj', traj.shape)
        # print('valid', valid.shape)
        print('saving traj %d' % point_id)
        for t in range(S2):
            if valid[t]:
                min_idx = 0 if t - S < 0 else t - S
                rgbs_color[t] = draw_traj_on_image_py(rgbs_color[t], traj[min_idx:t + 1], S=S, show_dots=show_dots,
                                                      cmap=cmap, linewidth=linewidth)

    print('saving images to %s' % save_root)
    for s in tqdm(range(S2)):
        cv2.imwrite(os.path.join(save_root, 'tracking_%04d.png' % s), rgbs_color[s])
    writing_video(rgbs_color, os.path.join(save_root, 'tracking.mp4'), frame_rate=30)
    return rgbs_color

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/results/robot')
    parser.add_argument('--sample_points', type=int, default=12)
    parser.add_argument('--show_dots', action='store_true', default=False)
    parser.add_argument('--cmap', type=str, default='spring')
    parser.add_argument('--linewidth', type=int, default=2)
    parser.add_argument('--S', type=int, default=128)
    parser.add_argument('--maxdist', type=float, default=None)
    parser.add_argument('--frame_idx', type=int, default=0)
    parser.add_argument('--frame_end', type=int, default=1200)
    args = parser.parse_args()

    summ_traj2ds_on_rgbs(args.data_root, sample_points=args.sample_points, show_dots=args.show_dots, cmap=args.cmap,
                         linewidth=args.linewidth, S=args.S,
                         maxdist=args.maxdist, frame_idx=args.frame_idx, frame_end=args.frame_end)