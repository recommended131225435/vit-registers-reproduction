"""Stage 1 of the pipeline: load and prepare the images.

Dataset: Tiny-ImageNet
    - 200 classes, 64x64 colour images
    - train: 100,000 images (500 per class)
    - val:    10,000 images (50 per class)

We use the official val split as our test set, because the official test
split has no public labels.

Folder layout after unzipping:
    tiny-imagenet-200/
        wnids.txt                       list of the 200 class IDs
        train/<class_id>/images/*.JPEG  training images, one folder per class
        val/images/*.JPEG               validation images, all in one folder
        val/val_annotations.txt         which class each validation image belongs to
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"  # about 237 MB
NUM_CLASSES = 200
IMG_SIZE = 64

# Average colour and spread of ImageNet images (Tiny-ImageNet is a subset of ImageNet).
# Used to normalise pixels so the model sees values centred around 0.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------
def download_tiny_imagenet(root: str = "data") -> Path:
    """Download and unzip Tiny-ImageNet into root/tiny-imagenet-200 (skipped if already there)."""
    path = Path(root) / "tiny-imagenet-200"
    if (path / "train").is_dir() and (path / "val").is_dir():
        print(f"Tiny-ImageNet already present at {path}")
        return path
    Path(root).mkdir(parents=True, exist_ok=True)
    download_and_extract_archive(TINY_IMAGENET_URL, download_root=root, remove_finished=True)
    return path


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class TinyImageNet(Dataset):
    """Returns (image, label) pairs from the official Tiny-ImageNet folder."""

    def __init__(self, path: Path, split: str = "train", transform=None):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        path = Path(path)
        self.transform = transform

        # Map each class ID (e.g. "n01443537") to a number 0..199. Sorted, so the mapping is always the same.
        class_ids = sorted(line.strip() for line in (path / "wnids.txt").read_text().splitlines() if line.strip())
        self.class_to_idx = {cid: i for i, cid in enumerate(class_ids)}

        self.samples = self._list_train(path, class_ids) if split == "train" else self._list_val(path)
        if not self.samples:
            raise RuntimeError(f"No images found for split '{split}' in {path}")

    def _list_train(self, path: Path, class_ids: list[str]) -> list[tuple[Path, int]]:
        samples = []
        for cid in class_ids:
            image_dir = path / "train" / cid / "images"
            for img_path in sorted(image_dir.iterdir()):
                if img_path.suffix.lower() == ".jpeg":
                    samples.append((img_path, self.class_to_idx[cid]))
        return samples

    def _list_val(self, path: Path) -> list[tuple[Path, int]]:
        # Each line of val_annotations.txt: "<file name> <class id> <box coordinates...>" (tab separated).
        samples = []
        for line in (path / "val" / "val_annotations.txt").read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                file_name, cid = parts[0], parts[1]
                samples.append((path / "val" / "images" / file_name, self.class_to_idx[cid]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # a few images are greyscale; make them 3-channel
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def get_transforms(train: bool) -> transforms.Compose:
    """Image preprocessing.

    Training images get random crops and flips (data augmentation). This shows
    the model slightly different versions of each image, which reduces
    overfitting. Validation images are left unchanged so the score is fair.
    """
    to_normalised_tensor = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.35, 1.0)),
            transforms.RandomHorizontalFlip(),
            *to_normalised_tensor,
        ])
    return transforms.Compose(to_normalised_tensor)


def _random_subset(dataset: Dataset, size: int | None, seed: int) -> Dataset:
    """Keep a random `size` images (same ones every time for a given seed). None keeps everything."""
    if size is None or size >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, sorted(indices[:size]))


# ----------------------------------------------------------------------------
# DataLoaders
# ----------------------------------------------------------------------------
def get_dataloaders(
    root: str = "data",
    batch_size: int = 128,
    num_workers: int = 2,
    train_subset: int | None = None,
    val_subset: int | None = None,
    download: bool = True,
    fake: bool = False,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build the train and validation DataLoaders (they hand the model images in batches).

    Args:
        train_subset / val_subset: use only this many images (for quick test runs).
        fake: use random noise images of the right shape instead of the real data.
              Only for testing the code without downloading anything.
    """
    if fake:
        train_set = datasets.FakeData(512, (3, IMG_SIZE, IMG_SIZE), NUM_CLASSES, get_transforms(train=True))
        val_set = datasets.FakeData(256, (3, IMG_SIZE, IMG_SIZE), NUM_CLASSES, get_transforms(train=False),
                                    random_offset=10_000)
    else:
        path = download_tiny_imagenet(root) if download else Path(root) / "tiny-imagenet-200"
        train_set = TinyImageNet(path, "train", get_transforms(train=True))
        val_set = TinyImageNet(path, "val", get_transforms(train=False))

    train_set = _random_subset(train_set, train_subset, seed)
    val_set = _random_subset(val_set, val_subset, seed)

    use_pinned_memory = torch.cuda.is_available()  # speeds up copying batches to the GPU
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=use_pinned_memory,
        drop_last=len(train_set) > batch_size,  # drop the last incomplete batch, unless it's the only one
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=use_pinned_memory,
    )
    return train_loader, val_loader
