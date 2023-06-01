import sys, os

ROOT_DIR = os.path.abspath(os.path.join('', 'src'))
sys.path.insert(0, os.path.join(ROOT_DIR))

import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from data import load_poses, load_intrinsic


def main(cfg, instance_id):
    data_dir = (
        Path(cfg['data']['data_dir'])
        / cfg['data']['cat'] / cfg['data']['splits'] / instance_id
    )

    focal, H, W = load_intrinsic(data_dir / 'intrinsics.txt')
    cx, cy = W / 2.0, H / 2.0
    K = np.asarray([focal, 0, cx, 0, focal, cy, 0, 0, 1]).reshape((3, 3))
    K_norm = np.eye(4)
    K_norm[0, 0] = 2.0 / W
    K_norm[1, 1] = 2.0 / H
    K_norm[:2, 2] = -1

    poses = load_poses(data_dir / 'pose', np.arange(50))
    srn_coords_trans = np.diag(np.array([1, -1, -1, 1])) # SRN dataset
    poses = [i @ srn_coords_trans for i in poses]
    projection_matrices = [
        get_projection_matrix(K, T)
        for T in poses
    ]

    scale_matrix = np.eye(4)
    cameras = [
        [
            (f'world_mat_{i}', projection_matrices[i]),
            (f'camera_mat_{i}', K_norm),
            (f'scale_mat_{i}', scale_matrix),
        ]
        for i in range(50)
    ]
    cameras = dict([j for i in cameras for j in i])
    np.savez(data_dir / 'cameras.npz', **cameras)

    if not (data_dir / 'image').is_dir():
        os.symlink(data_dir / 'rgb', data_dir / 'image')

    mask_dir = data_dir / 'mask'
    mask_dir.mkdir(exist_ok=True)
    for i in range(50):
        img = cv2.imread(str(data_dir / 'rgb' / f'{i:06d}.png'))
        mask = np.all(img == 255, axis=-1, keepdims=True)
        mask = ~mask
        mask = np.repeat(mask, 3, axis=-1)
        mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(str(mask_dir / f'{i:03d}.png'), mask)

    print(focal, H, W)
    print(poses[0])
    print(projection_matrices[0])


def get_projection_matrix(K, T):
    proj_mat = K @ np.linalg.inv(T)[:3, :]
    # IDR convention adds [0, 0, 0, 1] row
    proj_mat = np.concatenate((proj_mat, [[0, 0, 0, 1]]), axis=0)
    return proj_mat


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="CodeNeRF")
    arg_parser.add_argument("--instance-id", dest="instance_id", required=True)
    arg_parser.add_argument("--jsonfile", dest="jsonfile", default="jsonfiles/srnchair.json")
    args = arg_parser.parse_args()

    with open(args.jsonfile) as f:
        cfg = json.load(f)

    main(cfg, args.instance_id)
