"""
Extract 13 handcrafted texture and statistical features from a segmented leaf ROI.

Python equivalent of EvaluateFeatures.m.

Feature vector layout (indices 0–12):
    0  Contrast      GLCM
    1  Correlation   GLCM
    2  Energy        GLCM
    3  Homogeneity   GLCM
    4  Mean          Statistical
    5  Std Dev       Statistical
    6  Entropy       Statistical
    7  RMS           Statistical
    8  Variance      Statistical
    9  Smoothness    Statistical
    10 Kurtosis      Statistical
    11 Skewness      Statistical
    12 IDM           Spatial (Inverse Difference Moment)

Original MATLAB: graycomatrix / graycoprops + custom formulas.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from skimage.feature import graycomatrix, graycoprops


FEATURE_NAMES = [
    "Contrast",
    "Correlation",
    "Energy",
    "Homogeneity",
    "Mean",
    "Std Dev",
    "Entropy",
    "RMS",
    "Variance",
    "Smoothness",
    "Kurtosis",
    "Skewness",
    "IDM",
]


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 to grayscale uint8."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.astype(np.uint8)


def _disease_area_ratio(seg_img: np.ndarray, leaf_img: np.ndarray) -> float:
    """
    Compute the fraction of the leaf area that is diseased.

    Equivalent to the connected-component area calculation in EvaluateFeatures.m.
    A +0.15 offset is applied when the ratio is < 0.1 (to avoid near-zero values).
    """
    def largest_component_area(gray: np.ndarray) -> int:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        labeled, n = ndimage.label(binary)
        if n == 0:
            return 0
        sizes = ndimage.sum(binary, labeled, range(1, n + 1))
        return int(np.max(sizes))

    area_disease = largest_component_area(_to_gray(seg_img))
    area_leaf = largest_component_area(_to_gray(leaf_img))

    ratio = area_disease / area_leaf if area_leaf > 0 else 0.0
    if ratio < 0.1:
        ratio += 0.15
    return ratio


def _glcm_features(gray: np.ndarray) -> dict[str, float]:
    """GLCM texture features using scikit-image (matches MATLAB graycomatrix defaults)."""
    # MATLAB default: offset [0,1] (horizontal adjacency), symmetric=true
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    props = {}
    for prop in ("contrast", "correlation", "energy", "homogeneity"):
        props[prop] = float(graycoprops(glcm, prop)[0, 0])
    return props


def _statistical_features(seg_img: np.ndarray, gray: np.ndarray) -> dict[str, float]:
    """
    First- and second-order statistical features.

    Matches the formulas in EvaluateFeatures.m line by line.
    """
    flat_seg = seg_img.flatten().astype(np.float64)
    flat_gray = gray.flatten().astype(np.float64)

    mean_val = float(np.mean(flat_seg))
    std_val = float(np.std(flat_seg))

    # Entropy: -sum(p * log2(p)) over histogram bins (MATLAB uses entropy() on uint8)
    counts, _ = np.histogram(gray, bins=256, range=(0, 255))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy_val = float(-np.sum(probs * np.log2(probs)))

    rms_val = float(np.sqrt(np.mean(flat_gray ** 2)))
    variance_val = float(np.var(flat_seg))

    # Smoothness: 1 - 1/(1 + sum_of_pixel_values)  (MATLAB formula)
    pixel_sum = float(np.sum(flat_seg))
    smoothness_val = 1.0 - 1.0 / (1.0 + pixel_sum)

    # Higher-order moments (scipy uses Fisher's definition; MATLAB uses Pearson's)
    kurtosis_val = float(scipy_kurtosis(flat_seg, fisher=False))  # Pearson (matches MATLAB)
    skewness_val = float(scipy_skew(flat_seg))

    return {
        "mean": mean_val,
        "std": std_val,
        "entropy": entropy_val,
        "rms": rms_val,
        "variance": variance_val,
        "smoothness": smoothness_val,
        "kurtosis": kurtosis_val,
        "skewness": skewness_val,
    }


def _idm(seg_img: np.ndarray) -> float:
    """
    Inverse Difference Moment (IDM) — spatial locality measure.

    MATLAB: sum over all pixels of pixel_value / (1 + (row - col)^2)
    Vectorised with broadcasting to avoid a Python-level double loop.
    """
    if seg_img.ndim == 3:
        values = seg_img.mean(axis=2)  # average RGB → scalar per pixel
    else:
        values = seg_img.astype(np.float64)

    h, w = values.shape
    rows = np.arange(h, dtype=np.float64).reshape(-1, 1)   # (H, 1)
    cols = np.arange(w, dtype=np.float64).reshape(1, -1)   # (1, W)
    weights = 1.0 / (1.0 + (rows - cols) ** 2)             # (H, W)
    return float(np.sum(values * weights))


def extract_features(
    seg_img: np.ndarray,
    leaf_img: np.ndarray | None = None,
) -> np.ndarray:
    """
    Extract the 13-dimensional feature vector from a segmented ROI image.

    Parameters
    ----------
    seg_img : np.ndarray
        RGB uint8 image of the segmented disease region (background = 0).
    leaf_img : np.ndarray or None
        Full RGB leaf image (256×256) used to compute the affected-area ratio.
        If None, the ratio step is skipped and 0.0 is used.

    Returns
    -------
    features : np.ndarray  shape (13,), dtype float64
    """
    gray = _to_gray(seg_img)

    glcm = _glcm_features(gray)
    stats = _statistical_features(seg_img, gray)
    idm = _idm(seg_img)

    features = np.array([
        glcm["contrast"],
        glcm["correlation"],
        glcm["energy"],
        glcm["homogeneity"],
        stats["mean"],
        stats["std"],
        stats["entropy"],
        stats["rms"],
        stats["variance"],
        stats["smoothness"],
        stats["kurtosis"],
        stats["skewness"],
        idm,
    ], dtype=np.float64)

    return features
