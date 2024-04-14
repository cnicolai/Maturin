import functools
import torch
import torchvision
import pandas as pd
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
from image_utils import get_mean, get_std

preprocess = transforms.Compose(
    [
        # convert to float32
        transforms.Lambda(lambda x: x.to(torch.float32)),
        transforms.Resize((512, 512)),
        # transforms.Normalize(get_mean("data"), get_std("data")),
        transforms.Lambda(lambda x: x.to(torch.device("mps"))),
    ]
)


@functools.lru_cache(1)
def _files() -> List[Tuple[str, float]]:
    df = pd.read_csv("data/data.csv", delimiter=";")
    return list(zip(df["File"], df["Angle"]))


class CrookedScanDataset(Dataset):
    def __init__(self, transform=preprocess):
        self.transform = transform
        self.files = _files()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file, angle = self.files[idx]
        image = torchvision.io.read_image(file, mode=torchvision.io.ImageReadMode.GRAY)
        image = self.transform(image)
        return image, torch.tensor(angle, dtype=torch.float32, device=torch.device("mps"))


def get_crooked_scan_dataloaders(
    batch_size: int = 16,
    val_split: float = 0.2,
    shuffle: bool = True,
    num_workers: int = 0,
):
    # raise ValueError if val_split is not between 0 and 1
    if val_split <= 0 or val_split >= 1:
        raise ValueError("val_split must be between 0 and 1")

    train_set, val_set = random_split(
        CrookedScanDataset(),
        [1 - val_split, val_split],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
