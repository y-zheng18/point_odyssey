import numpy as np
import cv2
import os


def save_to_ply(data, filename):
    with open(filename, 'w') as ply:
        # PLY header
        ply.write("ply\n")
        ply.write("format ascii 1.0\n")
        ply.write("element vertex {}\n".format(data.shape[0]))  # only non-zero values
        ply.write("comment vertices\n")
        ply.write("property float x\n")
        ply.write("property float y\n")
        ply.write("property float z\n")
        # ply.write("property float value\n")  # or use "uchar red", "uchar green", "uchar blue" for RGB colors
        # Adding properties for RGB colors

        ply.write("end_header\n")

        # PLY data
        for (x, y, z) in data:
            ply.write("{} {} {} \n".format(x, y, z))


def reprojection(points, K, RT):
    v = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    XYZ = (RT @ v.T).T[:, :3]
    Z = XYZ[:, 2:]
    XYZ = XYZ / XYZ[:, 2:]
    xyz = (K @ XYZ.T).T
    uv = xyz[:, :2]
    return uv, Z

def inverse_projection(depth, K, RT):
    h, w = depth.shape

    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    u = w - u - 1
    uv_homogeneous = np.vstack((u.flatten(), v.flatten(), np.ones_like(u.flatten())))

    K_inv = np.linalg.inv(K)

    # use max depth as 10m for visualization
    depth = depth.flatten()
    mask = depth < 10

    XYZ = K_inv @ uv_homogeneous * depth

    XYZ = np.vstack((XYZ, np.ones(XYZ.shape[1])))
    world_coordinates = np.linalg.inv(RT) @ XYZ
    world_coordinates = world_coordinates[:3, :].T
    world_coordinates = world_coordinates[mask]

    return world_coordinates




if __name__ == '__main__':
    data_path = './data/point_odyssey/train'
    annotations = np.load('{}/dancing/annotations.npz'.format(data_path))
    trajs_3d = annotations['trajs_3d'].astype(np.float32)
    cam_ints = annotations['intrinsics'].astype(np.float32)
    cam_exts = annotations['extrinsics'].astype(np.float32)

    depth_16bit = cv2.imread('{}/dancing/depths/depth_00244.png'.format(data_path), cv2.IMREAD_ANYDEPTH)
    img = cv2.imread('{}/dancing/rgbs/rgb_00244.jpg'.format(data_path))
    h, w = depth_16bit.shape
    print(depth_16bit.shape, np.max(depth_16bit), np.min(depth_16bit))

    print(trajs_3d.shape, cam_ints.shape, cam_exts.shape)

    trajs = trajs_3d[243]
    cam_intrinsic = cam_ints[243]
    cam_extrinsic = cam_exts[243]

    R1 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    R2 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    cam_extrinsic = R2 @ cam_extrinsic @ R1

    depth = depth_16bit.astype(np.float32) / 65535.0 * 1000.0
    depth_inv = inverse_projection(depth, cam_intrinsic, cam_extrinsic)

    save_to_ply(depth_inv, '{}/dancing/depth_inv.ply'.format(data_path))
    save_to_ply(trajs, '{}/dancing/trajs.ply'.format(data_path))

    uv, Z = reprojection(trajs, cam_intrinsic, cam_extrinsic)

    uv = np.round(uv).astype(np.int32)
    for i in range(len(uv)):
        u, v = uv[i]
        z = Z[i]
        if 0 < u < w and 0 < v < h:
            d = depth[int(v), w - int(u)]

            if d > z - 0.15 and d < z + 0.15:
                img[int(v), w - int(u), :] = np.array([255, 255, 255])

    cv2.imshow('img', img)
    cv2.waitKey(0)