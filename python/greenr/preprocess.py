"""
Image loading, resizing, and contrast enhancement.

Python equivalent of the preprocessing steps in Detect.m and DetectDisease_GUI.m.
Original MATLAB: imread → imresize([256,256]) → imadjust(stretchlim())
"""

import cv2
import numpy as np


TARGET_SIZE = (256, 256)


def load_image(path: str) -> np.ndarray:
    """Load an image and return it as an RGB uint8 array (H, W, 3)."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize(image: np.ndarray, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Resize image to (height, width). Equivalent to MATLAB imresize."""
    return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


def enhance_contrast(image: np.ndarray, pct_low: float = 1.0, pct_high: float = 99.0) -> np.ndarray:
    """
    Contrast enhancement via percentile-based intensity stretching.

    Equivalent to MATLAB: imadjust(I, stretchlim(I))
    Stretches each channel so that pct_low% of pixels are mapped to 0
    and pct_high% are mapped to 255.
    """
    result = np.empty_like(image)
    for c in range(image.shape[2]):
        lo, hi = np.percentile(image[:, :, c], [pct_low, pct_high])
        if hi > lo:
            result[:, :, c] = np.clip(
                (image[:, :, c].astype(np.float32) - lo) * 255.0 / (hi - lo),
                0, 255,
            ).astype(np.uint8)
        else:
            result[:, :, c] = image[:, :, c]
    return result


def preprocess(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline: load → resize → enhance contrast.

    Returns
    -------
    original : np.ndarray  RGB uint8 resized image
    enhanced : np.ndarray  RGB uint8 contrast-enhanced image
    """
    image = load_image(path)
    image = resize(image)
    enhanced = enhance_contrast(image)
    return image, enhanced
