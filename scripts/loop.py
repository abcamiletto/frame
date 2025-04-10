import pprint
from itertools import chain
from pathlib import Path

import lightning as L
import lovely_tensors as lt
import torchvision
import typer
from tqdm import tqdm

import framevision
from framevision.pl_wrappers import FrameDataModule

lt.monkey_patch()


def main(
    data: Path = typer.Option(..., help="Path to the dataset root folder"),
    batch_size: int = typer.Option(32, help="Batch size for the dataloader"),
    num_workers: int = typer.Option(4, help="Number of workers for the dataloader"),
    seq_len: int = typer.Option(1, help="Length of the sequence"),
    only_val: bool = typer.Option(False, help="Only cache the validation set"),
    print_batch: bool = typer.Option(False, help="Print the first batch"),
):
    L.seed_everything(42)

    processing = torchvision.transforms.Compose(
        [
            framevision.processing.Resize((256, 256)),
            framevision.processing.NormalizeImages(),
            framevision.processing.NormalizeJoints2D(),
            framevision.processing.NormalizeIntrinsics(),
        ]
    )

    datamodule = FrameDataModule(
        root_dir=data,
        batch_size=batch_size,
        num_workers=num_workers,
        split=dict(train="others", val=["test_actor00_seq1", "test_actor00_seq2", "test_actor01_seq1", "test_actor01_seq2"]),
        split_by="sequences",
        sequence_length=seq_len,
        train_processing=processing,
        test_processing=processing,
    )

    # Setup the data
    datamodule.setup()

    # Get the dataloader from the datamodule
    train_dataloader = datamodule.train_dataloader(drop_last=False)
    val_dataloader = datamodule.val_dataloader()
    all_iterator = chain(train_dataloader, val_dataloader) if not only_val else val_dataloader
    iterator_len = len(train_dataloader) + len(val_dataloader) if not only_val else len(val_dataloader)

    # Manually loop through the batches
    for batch_idx, batch in tqdm(enumerate(all_iterator), total=iterator_len, desc="Processing batches"):
        if batch_idx == 0 and print_batch:
            pprint.pp(batch)


if __name__ == "__main__":
    typer.run(main)
