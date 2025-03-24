from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from ..dataset import FrameDataset
from ..dataset import split_utils as su


class FrameDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 4,
        split: Optional[Dict[str, Union[List[str], str]]] = None,
        split_by: str = "actions",
        keypoint_set: Union[str, List[str]] = "mo2cap2",
        train_processing: Optional[Callable] = None,
        test_processing: Optional[Callable] = None,
        **dataset_kwargs,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.split_by = split_by
        self.keypoint_set = keypoint_set
        self.dataset_kwargs = dataset_kwargs

        self.train_processing = train_processing
        self.test_processing = test_processing

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if self.train_data is None:
            self._split_data()

        if stage == "fit" or stage is None:
            self.train_dataset = self._create_dataset(self.train_data, self.train_processing)
            if self.val_data:
                self.val_dataset = self._create_dataset(self.val_data, self.test_processing)

        if stage == "test":
            if self.test_data:
                self.test_dataset = self._create_dataset(self.test_data, self.test_processing)

    def _split_data(self):
        if self.split is None:
            return

        if self.split_by == "sequences":
            split_data = su.split_by_sequences(self.root_dir, self.split)
        elif self.split_by == "actions":
            split_data = su.split_by_actions(self.root_dir, self.split)
        elif self.split_by == "both":
            split_data = su.split_by_both(self.root_dir, self.split)
        else:
            raise ValueError(f"Invalid value for split_by: {self.split_by}")

        self.train_data = split_data["train"]
        self.val_data = split_data["val"]
        self.test_data = split_data["test"]

    def _create_dataset(self, data, processing=None):
        return FrameDataset(
            root_dir=self.root_dir,
            seq2actions=data,
            keypoint_set=self.keypoint_set,
            processing=processing,
            **self.dataset_kwargs,
        )

    def train_dataloader(self, drop_last=True):
        self.train_dataset.set_epoch(self.current_epoch)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=drop_last,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 1 else False,
        )

    def val_dataloader(self):
        self.val_dataset.set_epoch(self.current_epoch)
        if self.val_dataset:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
            )

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
            )

    @property
    def current_epoch(self):
        return self.trainer.current_epoch if self.trainer else 0
