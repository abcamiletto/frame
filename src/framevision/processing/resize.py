from typing import Tuple

import cv2
import numpy as np


class Resize:
    def __init__(self, size: Tuple[int, int] = None, scale: float = None):
        self.size = size
        self.scale = scale

        if size is None and scale is None:
            raise ValueError("Either size or scale must be provided")
        if size is not None and scale is not None:
            raise ValueError("Only one of size or scale should be provided")

    def __call__(self, sample):
        images = sample["images"]  # Shape: (T, N, H, W, C)
        # joints_2D = sample["body_tracking"]["joints_2D"]  # Shape: (T, N, J, 2)
        intrinsics = sample["intrinsics"]  # Dict with keys: "K", "d"

        H, W, _ = images.shape[-3:]

        if self.size:
            new_H, new_W = self.size
            scale_h, scale_w = new_H / H, new_W / W
        elif self.scale:
            new_H, new_W = int(H * self.scale), int(W * self.scale)
            scale_h = scale_w = self.scale

        resized_images = self.resize_images(images, new_H, new_W)
        # scaled_joints_2D = self.scale_joints(joints_2D, scale_w, scale_h)
        scaled_intrinsics = self.adjust_intrinsics(intrinsics, scale_w, scale_h)

        sample["images"] = resized_images
        # sample["body_tracking"]["joints_2D"] = scaled_joints_2D
        sample["intrinsics"] = scaled_intrinsics

        return sample

    def resize_images(self, images: np.ndarray, new_H: int, new_W: int) -> np.ndarray:
        T, N, H, W, C = images.shape
        resized_images = np.zeros((T, N, new_H, new_W, C), dtype=images.dtype)
        for t in range(T):
            for i in range(N):
                resized_images[t, i] = cv2.resize(images[t, i], (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        return resized_images

    def scale_joints(self, joints_2D: np.ndarray, scale_w: float, scale_h: float) -> np.ndarray:
        scaled_joints_2D = joints_2D.copy()
        scaled_joints_2D[..., 0] *= scale_w
        scaled_joints_2D[..., 1] *= scale_h
        return scaled_joints_2D

    def adjust_intrinsics(self, intrinsics: np.ndarray, scale_w: float, scale_h: float) -> np.ndarray:
        K = intrinsics["K"].copy()

        K[:, 0, 0] *= scale_w  # fx
        K[:, 1, 1] *= scale_h  # fy
        K[:, 0, 2] *= scale_w  # cx
        K[:, 1, 2] *= scale_h  # cy

        scaled_intrinsics = {"K": K, "d": intrinsics["d"]}
        return scaled_intrinsics
