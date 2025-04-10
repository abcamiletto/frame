import warnings
from pathlib import Path
from typing import Callable, Union

import humanize
import rich
from torch.utils.data import Dataset

from . import load_utils as lu
from . import split_utils as su
from .keypoint_map import KEYPOINT_NAMES, KEYPOINT_SETS

EGO_CAM_NAMES = ["egocam_left", "egocam_right"]
CONSOLE = rich.get_console()


class FrameDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        seq2actions: dict[str, list[str]] = None,
        keypoint_set: Union[str, list[str]] = "mo2cap2",
        processing: Callable = None,
        skip_images: bool = False,
        cache_name: str = None,
        sequence_length: int = 1,
        undersampling_factor: int = 1,
        stride_factor: int = 1,
    ):
        """
        Initializes the SelfDataset.

        Args:
            root_dir: Root directory containing the dataset sequences.
            seq2actions: Dictionary mapping sequence names to lists of action names.
            keypoint_set: Keypoint set to use. Defaults to "mo2cap2".
            processing: Callable to apply to the data. Usually preprocessing/augmentation. Defaults to None.
            skip_images: Whether to skip loading images. Defaults to False.
            cache_name: Name of the cache file to load. Defaults to None.
            sequence_length: Number of frames to load per sample. Defaults to 1.
            undersampling_factor: Factor by which to undersample each sequence. Defaults to 1.
            stride_factor: Factor by which to stride two consecutive samples. Defaults to 1.

        """
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.undersampling_factor = undersampling_factor
        self.stride_factor = stride_factor
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Directory {root_dir} does not exist.")

        self.ego_cam_names = tuple(EGO_CAM_NAMES)
        self.seq2actions = seq2actions if seq2actions is not None else self._get_default_mapping(self.root_dir)

        self.index_map: list[tuple[str, str, tuple[int]]] = []  # list of (sequence, action, indexes) tuples
        self.generate_global_index()
        self.keypoint_indices = self.process_keypoint_set(keypoint_set)
        self.processing = processing
        self.skip_images = skip_images
        self.cache_name = cache_name
        self.current_epoch = 0

        # Print some information about the dataset
        CONSOLE.print(f"Loaded dataset with {len(self.seq2actions)} sequences and {humanize.intcomma(len(self.index_map))} steps.")
        CONSOLE.print("The following sequences are available:")
        for seq, actions in self.seq2actions.items():
            CONSOLE.print(f"  - {seq}: {len(actions)} actions")

    def __getitem__(self, index: int):
        """Return a sample from the dataset.
        In the followings, T is the sequence length, V is the number of cameras and J is the number of keypoints.

        Args:
            index: Index of the sample to return.

        Returns:
            A dictionary containing the following keys:
                - "images": Numpy array of shape (T, V, H, W, C) containing the images.
                - "body_tracking": A dictionary containing the following keys:
                    - "joints_3D": Numpy array of shape (T, J, 3) containing the 3D joint positions.
                    - "joints_2D": Numpy array of shape (T, V, J, 2) containing the 2D joint positions.
                    - "visibility_mask": Numpy array of shape (T, V, J) containing the joint visibility mask.
                - "cam_poses": A dictionary containing the following keys:
                    - "vr": Numpy array of shape (T, V, 4, 4) containing the camera poses estimated on the fly by the VR system.
                - "poses": A dictionary containing the following keys:
                    - "vr": A dictionary containing the following keys:
                        - "pose_name": Numpy array of shape (T, 4, 4) containing the VR poses.
                - "transforms": A dictionary containing the following keys:
                    - "transform_name": Numpy array of shape (4, 4) containing the transformation matrix.
                - "intrinsics": A dictionary containing the following keys:
                    - "K": Numpy array of shape (V, 3, 3) containing the camera intrinsics.
                    - "d": Numpy array of shape (V, 4) containing the camera distortion coefficients.
                - "skeleton": Numpy array of shape (J, 2) containing the skeleton tree in terms of parent-child relationships.
                - "meta": A dictionary containing the following keys:
                    - "sequence": Name of the sequence.
                    - "action": Name of the action.
                    - "indexes": Numpy array of shape (T,) containing the indexes of the frames in the action.
                    - "skeleton_idxes": Numpy array of shape (J,) containing the indices of the keypoints in the global skeleton.
        """
        sequence, action, indexes = self.index_map[index]

        seq_folder = self.root_dir / sequence
        act_folder = seq_folder / "actions" / action

        sample = self.load_sample_data(seq_folder, act_folder, indexes)
        if self.processing:
            sample = self.processing(sample)

        return sample

    def load_sample_data(self, seq_folder: Path, act_folder: Path, indexes: tuple[int]) -> dict:
        sample = {}

        if not self.skip_images:
            sample["images"] = self.load_images(act_folder, indexes, self.ego_cam_names)

        sample["body_tracking"] = self.load_keypoints(act_folder, indexes, self.ego_cam_names)
        sample["intrinsics"] = self.load_intrinsics(seq_folder, self.ego_cam_names)
        sample["skeleton"] = self.load_skeleton(seq_folder)
        sample["meta"] = self.load_meta(seq_folder, act_folder, indexes)
        sample["meta"]["epoch"] = self.current_epoch

        sample["cam_poses"] = {}
        sample["cam_poses"]["vr"] = self.load_vr_camera_poses(act_folder, indexes, self.ego_cam_names)

        sample["poses"] = {}
        sample["poses"]["vr"] = self.load_vr_additional_pose(act_folder, indexes, "egocam_middle")

        sample["transforms"] = self.load_transforms(seq_folder)

        if self.cache_name:
            sample["cache"] = self.load_cache_data(act_folder, indexes)

        return sample

    def load_images(self, act_folder: Path, indexes: tuple[int], cam_names: list[str]):
        return lu.load_images(act_folder, cam_names, indexes)

    def load_gt_camera_poses(self, act_folder: Path, indexes: tuple[int], cam_names: list[str]):
        return lu.load_multi_poses(act_folder, cam_names, indexes, folder_name="poses")

    def load_gt_additional_pose(self, act_folder: Path, indexes: tuple[int], pose_name: str):
        return {pose_name: lu.load_multi_poses(act_folder, [pose_name], indexes, folder_name="poses").squeeze(1)}

    def load_vr_camera_poses(self, act_folder: Path, indexes: tuple[int], cam_names: list[str]):
        return lu.load_multi_poses(act_folder, cam_names, indexes, folder_name="on_device_poses")

    def load_vr_additional_pose(self, act_folder: Path, indexes: tuple[int], pose_name: str):
        return {pose_name: lu.load_multi_poses(act_folder, [pose_name], indexes, folder_name="on_device_poses").squeeze(1)}

    def load_keypoints(self, act_folder: Path, indexes: tuple[int], cam_names: list[str]):
        return lu.load_keypoints(act_folder, cam_names, indexes, self.keypoint_indices)

    def load_intrinsics(self, seq_folder: Path, cam_names: list[str]):
        return lu.load_intrinsics(seq_folder, cam_names)

    def load_skeleton(self, seq_folder: Path):
        return lu.load_skeleton(seq_folder, self.keypoint_indices)

    def load_meta(self, seq_folder: Path, act_folder: Path, indexes: tuple[int]):
        return lu.load_meta(seq_folder, act_folder, indexes, self.keypoint_indices)

    def load_transforms(self, seq_folder: Path):
        return lu.load_transforms(seq_folder)

    def load_cache_data(self, act_folder: Path, indexes: tuple[int]):
        cache_folder = act_folder / "cache" / self.cache_name
        files = [f.stem for f in (cache_folder).glob("*.npz")]
        return {f: lu.load_cache_data(cache_folder, f, indexes) for f in files}

    def process_keypoint_set(self, keypoint_set):
        if isinstance(keypoint_set, str):
            keypoints = KEYPOINT_SETS[keypoint_set.lower()]
        elif isinstance(keypoint_set, list) and all(isinstance(k, str) for k in keypoint_set):
            keypoints = keypoint_set
        else:
            raise ValueError("Invalid keypoint set specification. Use a string for predefined sets or a list of keypoint names.")

        return tuple([KEYPOINT_NAMES.index(k) for k in keypoints if k in KEYPOINT_NAMES])

    def _get_default_mapping(self, root_dir: Path):
        all_sequences = su.get_all_sequences(root_dir)
        return {seq: su.get_actions(root_dir, seq) for seq in all_sequences}

    def generate_global_index(self):
        for seq, actions in self.seq2actions.items():
            for action in actions:
                total_frames = self.get_n_frames(seq, action)

                offset = self.sequence_length - 1
                sample_count = (total_frames - offset * self.undersampling_factor) // self.stride_factor

                for i in range(sample_count):
                    start = i * self.stride_factor
                    end = start + self.sequence_length * self.undersampling_factor
                    indexes = tuple(range(start, end, self.undersampling_factor))

                    self.index_map.append((seq, action, indexes))

    def load_action_data(self, sequence: str, action: str) -> dict:
        seq_folder = self.root_dir / sequence
        act_folder = seq_folder / "actions" / action

        total_frames = self.get_n_frames(sequence, action)
        indexes = tuple(range(total_frames))

        return self.load_sample_data(seq_folder, act_folder, indexes)

    def get_n_frames(self, sequence: str, action: str) -> int:
        folder = self.root_dir / sequence / "actions" / action
        video_folder = folder / "videos" / self.ego_cam_names[0]

        total_frames = len(list(video_folder.glob("*.jpg")))
        if total_frames == 0:
            warnings.warn(f"No frames found in {video_folder}. Skipping sequence {sequence} action {action}.")
            return 0

        return total_frames

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def __len__(self):
        return len(self.index_map)

    def __repr__(self):
        return f"FrameDataset(sequences={len(self.seq2actions)}, frames={len(self.index_map)})"
