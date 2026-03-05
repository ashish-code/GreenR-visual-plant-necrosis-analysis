"""
Color-based leaf disease segmentation using K-means clustering in L*a*b* space.

Python equivalent of the segmentation block in EvaluateFeatures.m / Detect.m.

Original MATLAB:
    cform = makecform('srgb2lab');
    I_lab = applycform(I, cform);
    ab = double(I_lab(:,:,2:3));
    [cluster_idx, cluster_center] = kmeans(ab, 3, 'distance', 'sqEuclidean', 'Replicates', 3);
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


N_CLUSTERS = 3


def segment_clusters(
    image: np.ndarray,
    n_clusters: int = N_CLUSTERS,
    n_init: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Segment a leaf image into `n_clusters` colour regions using K-means in L*a*b*.

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 image (H, W, 3).
    n_clusters : int
        Number of K-means clusters (3 in original MATLAB).
    n_init : int
        Number of random restarts (3 replicates in original MATLAB).
    random_state : int
        Seed for reproducibility (MATLAB was non-deterministic).

    Returns
    -------
    cluster_map : np.ndarray  shape (H, W), dtype int, cluster label per pixel
    cluster_images : list[np.ndarray]  one masked RGB image per cluster
    """
    # RGB → L*a*b*  (OpenCV scales L to 0-255, a/b to 0-255 offset by 128)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    h, w = image.shape[:2]

    # Use only the a* and b* channels (chrominance), as in the MATLAB code
    ab = lab[:, :, 1:3].reshape(-1, 2).astype(np.float32)

    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(ab).reshape(h, w)

    cluster_images = []
    for k in range(n_clusters):
        mask = labels == k
        seg = image.copy()
        seg[~mask] = 0
        cluster_images.append(seg)

    return labels, cluster_images


def auto_select_disease_cluster(
    cluster_images: list[np.ndarray],
    cluster_map: np.ndarray,
) -> tuple[int, np.ndarray]:
    """
    Automatically select the cluster most likely to represent diseased tissue.

    Strategy: the disease region typically deviates most from pure green.
    We pick the cluster whose mean pixel colour has the lowest green-channel
    dominance relative to the red channel (browning / yellowing / darkening).

    This replaces the interactive inputdlg() prompt in the original MATLAB GUI.
    The cluster index is returned alongside the masked image so callers can
    override the selection if needed.

    Parameters
    ----------
    cluster_images : list[np.ndarray]
        Masked RGB images from segment_clusters().
    cluster_map : np.ndarray
        Per-pixel cluster label array.

    Returns
    -------
    best_idx : int   cluster index most likely to be diseased
    seg_img  : np.ndarray  corresponding masked RGB image
    """
    scores = []
    for k, seg in enumerate(cluster_images):
        mask = cluster_map == k
        if mask.sum() == 0:
            scores.append(-np.inf)
            continue
        pixels = seg[mask].astype(np.float32)  # shape (N, 3)
        r_mean = pixels[:, 0].mean()
        g_mean = pixels[:, 1].mean()
        # High R relative to G → browning/yellowing (disease symptom)
        # Low overall brightness can also indicate necrosis
        score = r_mean - g_mean
        scores.append(score)

    best_idx = int(np.argmax(scores))
    return best_idx, cluster_images[best_idx]
