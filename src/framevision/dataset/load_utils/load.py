import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def load_images(act_folder: Path, cam_names: list[str], frame_indices: tuple[int]):
    return np.stack([load_image(act_folder, cam_names, idx) for idx in frame_indices], axis=0)


def load_multi_poses(act_folder: Path, pose_names: list[str], frame_indices: tuple[int], folder_name: str = "poses"):
    return np.stack([load_multi_pose(act_folder, idx, pose_names, folder_name) for idx in frame_indices], axis=0)


def load_keypoints(act_folder: Path, cam_names: list[str], frame_indices: tuple[int], keypoint_indices: list[int]):
    keypoint_dicts = [load_keypoint(act_folder, cam_names, idx, keypoint_indices) for idx in frame_indices]
    return {k: np.stack([d[k] for d in keypoint_dicts], axis=0) for k in keypoint_dicts[0].keys()}


def load_image(act_folder: Path, cam_names: list[str], frame_index: int):
    imgs = []
    for video_name in cam_names:
        video_folder = act_folder / "videos" / video_name
        img_path = video_folder / f"frame_{frame_index:05d}.jpg"
        imgs.append(img_path)

    imgs = [cv2.imread(str(img)) for img in imgs]
    imgs_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]

    return np.stack(imgs_rgb, axis=0)


def load_multi_pose(act_folder: Path, frame_index: int, pose_names: list[str], folder_name: str = "poses"):
    return np.stack([load_pose(act_folder, frame_index, k, folder_name) for k in pose_names])


def load_pose(act_folder: Path, frame_index: int, pose_name: str, folder_name: str = "poses"):
    head_pose = load_pose_file(act_folder, pose_name, folder_name)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = head_pose["rotations"][frame_index].astype(np.float32)
    T[:3, 3] = head_pose["translations"][frame_index].squeeze().astype(np.float32)

    return T


def load_keypoint(act_folder: Path, cam_names: list[str], frame_index: int, keypoint_indices: list[int]):
    joints_3D = load_joints_3D(act_folder, keypoint_indices)[frame_index]

    joints_2Ds = []
    joints_3D_ccs = []
    visibility_masks = []

    for video_name in cam_names:
        joints2D, joints3D_cc, mask = load_joints_cam(act_folder, video_name, keypoint_indices)

        joints_2Ds.append(joints2D[frame_index])
        joints_3D_ccs.append(joints3D_cc[frame_index])
        visibility_masks.append(mask[frame_index])

    joints_2Ds = np.stack(joints_2Ds, axis=0)
    joints_3D_ccs = np.stack(joints_3D_ccs, axis=0)
    visibility_masks = np.stack(visibility_masks, axis=0)

    # Convert everything to float32
    joints_3D = joints_3D.astype(np.float32)
    joints_2Ds = joints_2Ds.astype(np.float32)
    joints_3D_ccs = joints_3D_ccs.astype(np.float32)
    visibility_masks = visibility_masks.astype(np.float32)

    return dict(joints_3D=joints_3D, joints_2D=joints_2Ds, joints_3D_cc=joints_3D_ccs, visibility_mask=visibility_masks)


@lru_cache(maxsize=128)
def load_transforms(seq_folder: Path):
    transforms_folder = seq_folder / "meta" / "transforms"
    transform_files = list(transforms_folder.glob("*.npz"))

    transforms = {}
    for transform_file in transform_files:
        data = np.load(transform_file)

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = data["rotations"].squeeze().astype(np.float32)
        T[:3, 3] = data["translations"].squeeze().astype(np.float32)

        transforms[transform_file.stem] = T

    return transforms


def load_cache_data(cache_folder: Path, cache_name: str, frame_indexes: tuple[int]):
    cache_data = load_cache_file(cache_folder, cache_name)
    return cache_data[frame_indexes, ...]


@lru_cache(maxsize=128)
def load_skeleton(seq_folder: Path, keypoint_indices: tuple[int]) -> np.ndarray:
    """
    Load the skeleton data and index it based on the specified keypoints.

    Args:
    seq_folder (Path): Path to the sequence folder.
    keypoint_indices (list[int]): Indices of keypoints to keep.

    Returns:
    np.ndarray: Indexed skeleton tree.
    """
    skeleton_folder = seq_folder / "meta" / "skeleton"
    keypoint_indices = list(keypoint_indices)

    with open(skeleton_folder / "index_tree.json", "r") as f:
        index_tree_data = json.load(f)

    # Convert None to -1 in the loaded data
    index_tree_data = [[-1 if x is None else x for x in y] for y in index_tree_data]
    tree = np.array(index_tree_data, dtype=np.int32)  # Shape: (J, 2)

    return index_tree(tree, keypoint_indices)


@lru_cache(maxsize=4096)
def load_cache_file(cache_folder: Path, cache_name: str):
    cache_file = cache_folder / f"{cache_name}.npz"
    cache_data = np.load(cache_file)["motions"]
    return cache_data


@lru_cache(maxsize=4096)
def load_pose_file(act_folder: Path, pose_name: str, folder_name: str):
    pose_file = act_folder / folder_name / f"{pose_name}.npz"
    pose = np.load(pose_file)
    return dict(rotations=pose["rotations"], translations=pose["translations"])


@lru_cache(maxsize=128)
def load_intrinsics(seq_folder: Path, cam_names: list[str]):
    intrinsics_folder = seq_folder / "meta" / "intrinsics"
    Ks, ds = [], []
    for video_name in cam_names:
        intrinsics_file = intrinsics_folder / f"{video_name}.json"
        with open(intrinsics_file, "r") as f:
            calibration = json.load(f)

        K = np.array(calibration["K"], dtype=np.float32)
        d = np.array(calibration["d"], dtype=np.float32)

        Ks.append(K)
        ds.append(d)

    Ks = np.stack(Ks, axis=0)
    ds = np.stack(ds, axis=0)

    return dict(K=Ks, d=ds)


def index_tree(tree: np.ndarray, keypoint_indices: list[int]) -> np.ndarray:
    """Index the tree to include only the specified keypoints and update parent relationships"""
    # Create a mapping from old indices to new indices
    index_map = np.full(tree.shape[0], -1, dtype=int)
    index_map[keypoint_indices] = np.arange(len(keypoint_indices))

    # Filter the tree
    filtered_tree = tree[keypoint_indices].copy()

    # Update parent indices
    for i in range(len(filtered_tree)):
        current_parent = filtered_tree[i, 0]
        while current_parent != -1:
            if current_parent in keypoint_indices:
                filtered_tree[i, 0] = index_map[current_parent]
                break
            current_parent = tree[current_parent, 0]
        else:
            filtered_tree[i, 0] = -1

    # Update the joint indices (second column)
    filtered_tree[:, 1] = np.arange(len(keypoint_indices))

    return filtered_tree


@lru_cache(maxsize=4096)
def load_joints_3D(act_folder: Path, keypoint_indices: tuple[int]):
    joints_3D_file = act_folder / "body_tracking" / "joints_3D.npz"
    joints_3D = np.load(joints_3D_file)["translations"][:, keypoint_indices]
    return joints_3D


@lru_cache(maxsize=4096)
def load_joints_cam(act_folder: Path, cam_name: str, keypoint_indices: tuple[int]):
    joints_file = act_folder / "body_tracking" / f"joints_{cam_name}.npz"
    joints_data = np.load(joints_file)

    joints_2D = joints_data["joints2D"][:, keypoint_indices]
    joints_3D_cc = joints_data["joints3D_cc"][:, keypoint_indices]
    visibility_mask = joints_data["masks"][:, keypoint_indices]

    return joints_2D, joints_3D_cc, visibility_mask


def load_meta(seq_folder: Path, act_folder: Path, frame_indexes: tuple[int], keypoint_indices: tuple[int]):
    meta = {}
    meta["idx"] = np.array(frame_indexes, dtype=np.int32)
    meta["sequence"] = seq_folder.name
    meta["action"] = act_folder.name
    meta["skeleton_idxes"] = np.array(keypoint_indices, dtype=np.int32)
    return meta
