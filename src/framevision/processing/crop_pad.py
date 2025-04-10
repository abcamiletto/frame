import numpy as np


class Crop:
    """
    Crop images and adjust joints and intrinsics accordingly.

    Args:
        top, bottom, left, right: If int, number of pixels to crop. If float, percentage of image dimension to crop.
    """

    def __init__(self, top=0, bottom=0, left=0, right=0):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def __call__(self, sample):
        if "intrinsics_norm" in sample or "joints_2D_norm" in sample["body_tracking"]:
            raise ValueError("Normalized intrinsics or joints already exist. Crop should be applied before normalization.")

        images = sample["images"]
        joints_2D = sample["body_tracking"]["joints_2D"]
        intrinsics = sample["intrinsics"]

        H, W, _ = images.shape[-3:]

        top_pixels = int(H * self.top) if isinstance(self.top, float) else self.top
        bottom_pixels = int(H * self.bottom) if isinstance(self.bottom, float) else self.bottom
        left_pixels = int(W * self.left) if isinstance(self.left, float) else self.left
        right_pixels = int(W * self.right) if isinstance(self.right, float) else self.right

        # Crop images
        cropped_images = images[..., top_pixels : H - bottom_pixels, left_pixels : W - right_pixels, :]

        # Adjust joints
        adjusted_joints_2D = joints_2D.copy()
        adjusted_joints_2D[..., 0] -= left_pixels
        adjusted_joints_2D[..., 1] -= top_pixels

        # Adjust intrinsics
        K = intrinsics["K"].copy()
        K[..., 0, 2] -= left_pixels  # cx
        K[..., 1, 2] -= top_pixels  # cy

        # Update sample
        sample["images"] = cropped_images
        sample["body_tracking"]["joints_2D"] = adjusted_joints_2D
        sample["intrinsics"]["K"] = K

        return sample


class Pad:
    """
    Pad images and adjust joints and intrinsics accordingly.

    Args:
        top, bottom, left, right: If int, number of pixels to pad. If float, percentage of image dimension to pad.
        pad_value: Value used for padding (default: 0)
    """

    def __init__(self, top=0, bottom=0, left=0, right=0, pad_value=0):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.pad_value = pad_value

    def __call__(self, sample):
        if "intrinsics_norm" in sample or "joints_2D_norm" in sample["body_tracking"]:
            raise ValueError("Normalized intrinsics or joints already exist. Pad should be applied before normalization.")

        images = sample["images"]
        joints_2D = sample["body_tracking"]["joints_2D"]
        intrinsics = sample["intrinsics"]

        H, W, _ = images.shape[-3:]

        top_pixels = int(H * self.top) if isinstance(self.top, float) else self.top
        bottom_pixels = int(H * self.bottom) if isinstance(self.bottom, float) else self.bottom
        left_pixels = int(W * self.left) if isinstance(self.left, float) else self.left
        right_pixels = int(W * self.right) if isinstance(self.right, float) else self.right

        # Pad images
        pad_width = ((0, 0), (0, 0), (top_pixels, bottom_pixels), (left_pixels, right_pixels), (0, 0))
        padded_images = np.pad(images, pad_width, mode="constant", constant_values=self.pad_value)

        # Adjust joints
        adjusted_joints_2D = joints_2D.copy()
        adjusted_joints_2D[..., 0] += left_pixels
        adjusted_joints_2D[..., 1] += top_pixels

        # Adjust intrinsics
        K = intrinsics["K"].copy()
        K[..., 0, 2] += left_pixels  # cx
        K[..., 1, 2] += top_pixels  # cy

        # Update sample
        sample["images"] = padded_images
        sample["body_tracking"]["joints_2D"] = adjusted_joints_2D
        sample["intrinsics"]["K"] = K

        return sample
