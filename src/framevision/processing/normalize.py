import numpy as np


class NormalizeImages:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean).reshape(1, 1, 3, 1, 1).astype(np.float32)
        self.std = np.array(std).reshape(1, 1, 3, 1, 1).astype(np.float32)

    def __call__(self, sample):
        images = sample["images"].transpose(0, 1, 4, 2, 3)

        images = images.astype(np.float32) / 255.0
        images = (images - self.mean) / self.std

        sample["images"] = images
        return sample


class NormalizeJoints2D:
    """Normalize 2D joints to be in the range [0, 1]. To be used after ResizeImages."""

    def __call__(self, sample):
        return sample
        # Parse the image W and H
        H, W = sample["images"].shape[-2:]
        # Normalize joints
        j2D = sample["body_tracking"]["joints_2D"].copy()

        j2D[..., 0] = j2D[..., 0] / W
        j2D[..., 1] = j2D[..., 1] / H

        sample["body_tracking"]["joints_2D_norm"] = j2D

        return sample


class NormalizeIntrinsics:
    """Normalize intrinsics to represent an image of size 1x1. To be used after ResizeImages."""

    def __call__(self, sample):
        # Parse the image W and H
        H, W = sample["images"].shape[-2:]

        # Normalize intrinsics
        K = sample["intrinsics"]["K"].copy()
        d = sample["intrinsics"]["d"].copy()

        # Divide the first row by W and the second row by H
        K[..., 0, :] = K[..., 0, :] / W
        K[..., 1, :] = K[..., 1, :] / H

        normalized_intrinsics = dict(K=K, d=d)

        sample["intrinsics_norm"] = normalized_intrinsics

        return sample
