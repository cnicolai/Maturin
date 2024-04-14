import numpy as np
import cv2

from pathlib import Path


def get_mean(folder: str):
    imageFilesDir = Path(folder)
    files = list(imageFilesDir.rglob("*.jpg")) + list(imageFilesDir.rglob("*.png"))

    mean = np.array([0.0, 0.0, 0.0])

    for file in files:
        img = cv2.imread(str(file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(float) / 255.0

        mean += np.mean(img, axis=(0, 1))

    mean /= len(files)

    return mean


def get_std(folder: str):
    mean = get_mean(folder)
    files = list(Path(folder).rglob("*.jpg")) + list(Path(folder).rglob("*.png"))

    std = np.array([0.0, 0.0, 0.0])

    for file in files:
        img = cv2.imread(str(file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(float) / 255.0

        std += np.mean((img - mean) ** 2, axis=(0, 1))

    std = np.sqrt(std / len(files))

    return std
