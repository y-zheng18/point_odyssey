import numpy as np
import cv2
import os
import glob
from utils.file_io import read_tiff


def writing_video(data_dir: str, prefix: str,
                  save_path: str, frame_rate: int = 60):

    frames = sorted(glob.glob(os.path.join(data_dir, prefix)))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    img = cv2.imread(frames[0]) if not 'depth' in prefix else read_tiff(frames[0])
    h, w, _ = img.shape
    out = cv2.VideoWriter(save_path, fourcc, frame_rate, (w, h))

    for f in frames:
        img = cv2.imread(f) if not 'depth' in prefix else read_tiff(f)
        if 'depth' in prefix:
            img = np.round(img).astype(np.uint8)
            img = np.repeat(img, 3, axis=2)
        out.write(img)

    out.release()
    return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./results/robot/exr_img')
    parser.add_argument('--img_type', type=str, default='depth', choices=['frame', 'depth', 'segmentation', 'rgb', 'normal', 'tracking'])
    parser.add_argument('--save_dir', type=str, default='./results/robot/video')
    parser.add_argument('--frame_rate', type=int, default=60)
    args = parser.parse_args()

    prefix = args.img_type + '*tiff' if args.img_type == 'depth' else args.img_type + '*png'
    save_path = os.path.join(args.save_dir, args.img_type + '.mp4')
    os.makedirs(args.save_dir, exist_ok=True)

    fs = args.frame_rate
    writing_video(args.data_dir, prefix, save_path, fs)
