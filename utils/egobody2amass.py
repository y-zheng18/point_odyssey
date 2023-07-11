import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import json

amass_data_root = './data/motions/CMU/01/01_01_stageii.npz'

amass_data = np.load(amass_data_root, allow_pickle=True)

for k in amass_data.keys():
    print(k, amass_data[k].shape if type(amass_data[k]) is np.ndarray else amass_data[k])


Rx = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

def egobody2amass(egobody_sequence_name, cal_path, save_dir):
    egobody_data_dir = os.path.join(egobody_sequence_name, 'body_idx_0/results')
    if not os.path.exists(egobody_data_dir):
        egobody_data_dir = os.path.join(egobody_sequence_name, 'body_idx_1/results')
    os.makedirs(save_dir, exist_ok=True)
    frames = os.listdir(egobody_data_dir)
    frames = [f for f in frames if os.path.isdir(os.path.join(egobody_data_dir, f))]
    frames.sort()
    save_npz = {}
    for k in amass_data.keys():
        save_npz[k] = []
    save_npz['betas'] = np.zeros(16)
    save_npz['mocap_frame_rate'] = 30

    Rt = np.array(json.load(open(cal_path, 'r'))['trans'])

    for i, f in tqdm(enumerate(frames)):
        data = np.load(os.path.join(egobody_data_dir, f, '000.pkl'), allow_pickle=True)
        bodypose = data['body_pose']
        global_orient = data['global_orient']
        left_hand_pose = data['left_hand_pose']
        right_hand_pose = data['right_hand_pose']
        jaw_pose = data['jaw_pose']
        leye_pose = data['leye_pose']
        reye_pose = data['reye_pose']

        # convert 12 hand pose to 45 hand pose
        left_hand_pose = np.concatenate([left_hand_pose, np.zeros((1, 33))], axis=1) * 0
        right_hand_pose = np.concatenate([right_hand_pose, np.zeros((1, 33))], axis=1) * 0

        transl = data['transl']

        pose = np.concatenate([global_orient, bodypose, left_hand_pose * 0, right_hand_pose * 0, jaw_pose * 0, leye_pose * 0, reye_pose * 0], axis=1)

        save_npz['poses'].append(pose)
        save_npz['pose_body'].append(bodypose)
        save_npz['root_orient'].append(global_orient)
        save_npz['pose_hand'].append(np.concatenate([left_hand_pose, right_hand_pose], axis=1) * 0)
        save_npz['pose_jaw'].append(jaw_pose * 0)
        save_npz['pose_eye'].append(np.concatenate([leye_pose, reye_pose], axis=1) * 0)
        save_npz['trans'].append(transl)
        if i == 0:
            save_npz['gender'] = data['gender']

    save_npz['poses'] = np.concatenate(save_npz['poses'], axis=0)
    save_npz['pose_body'] = np.concatenate(save_npz['pose_body'], axis=0)
    save_npz['root_orient'] = np.concatenate(save_npz['root_orient'], axis=0)
    save_npz['pose_hand'] = np.concatenate(save_npz['pose_hand'], axis=0)
    save_npz['pose_jaw'] = np.concatenate(save_npz['pose_jaw'], axis=0)
    save_npz['pose_eye'] = np.concatenate(save_npz['pose_eye'], axis=0)
    save_npz['trans'] = np.concatenate(save_npz['trans'], axis=0)
    save_name = 'interactee_' + os.path.basename(egobody_sequence_name) + '.npz' if 'interactee' in egobody_sequence_name else 'interactor_' + os.path.basename(egobody_sequence_name) + '.npz'
    np.savez(os.path.join(save_dir, save_name), **save_npz)

if __name__ == '__main__':
    egobody_interactee_data_dir = './data/egobody/smplx_interactee_train/'
    calibration_dir = './data/egobody/calibrations/'
    save_dir = './data/egobody/results_amass'

    for s in os.listdir(egobody_interactee_data_dir):
        if not os.path.isdir(os.path.join(egobody_interactee_data_dir, s)):
            continue
        calibration_path = os.path.join(calibration_dir, s, 'cal_trans/kinect12_to_world')
        calibration_path = os.path.join(calibration_path, os.listdir(calibration_path)[0])
        egobody2amass(os.path.join(egobody_interactee_data_dir, s), calibration_path, save_dir)
    egobody_interactor_data_dir = './data/egobody/smplx_camera_wearer_train/'
    for s in os.listdir(egobody_interactor_data_dir):
        if not os.path.isdir(os.path.join(egobody_interactor_data_dir, s)):
            continue

        calibration_path = os.path.join(calibration_dir, s, 'cal_trans/kinect12_to_world')
        calibration_path = os.path.join(calibration_path, os.listdir(calibration_path)[0])
        egobody2amass(os.path.join(egobody_interactor_data_dir, s), calibration_path, save_dir)





