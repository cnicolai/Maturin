import numpy as np
import cv2

from pathlib import Path
import numpy as np
from PIL import Image
from pathlib import Path


def get_mean(folder: str):
    imageFilesDir = Path(folder)
    files = list(imageFilesDir.rglob("*.jpg")) + list(imageFilesDir.rglob("*.png"))
    mean = np.array([0.0, 0.0, 0.0])
    for file in files:
        img = Image.open(str(file))
        img = img.convert("RGB")
        img = np.array(img, dtype=float) / 255.0
        mean += np.mean(img, axis=(0, 1))
    mean /= len(files)
    return mean


def get_std(folder: str):
    mean = get_mean(folder)
    files = list(Path(folder).rglob("*.jpg")) + list(Path(folder).rglob("*.png"))
    std = np.array([0.0, 0.0, 0.0])
    for file in files:
        img = Image.open(str(file))
        img = img.convert("RGB")
        img = np.array(img, dtype=float) / 255.0
        std += np.mean((img - mean) ** 2, axis=(0, 1))
    std = np.sqrt(std / len(files))
    return std


if __name__ == "__main__":
    folder = "/Users/christoph/Developer/Maturin/data"
    print(get_mean(folder))
    print(get_std(folder))
