"""
GreenR — Plant Disease Prediction CLI

Python replacement for Detect.m (standalone script) and DetectDisease_GUI.m.

Usage
-----
    python predict.py path/to/leaf.jpg
    python predict.py path/to/leaf.jpg --cluster 1   # force cluster selection
    python predict.py path/to/leaf.jpg --show         # display intermediate images

The script:
    1. Loads and preprocesses the image
    2. Segments it into 3 colour clusters (K-means in L*a*b*)
    3. Auto-selects the cluster most likely to represent diseased tissue
    4. Extracts 13 texture/statistical features from that cluster
    5. Loads the pre-trained SVM (or trains from Training_Data.mat if absent)
    6. Prints the predicted disease class + confidence + feature values
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from greenr.classifier import CLASS_NAMES, DiseaseClassifier
from greenr.features import FEATURE_NAMES, extract_features
from greenr.preprocess import preprocess
from greenr.segmentation import auto_select_disease_cluster, segment_clusters

MODEL_PATH = Path(__file__).parent / "models" / "svm_greenr.pkl"
MAT_PATH = Path(__file__).parent.parent / "Code" / "GreenR" / "Training_Data.mat"


def _load_or_train_classifier() -> DiseaseClassifier:
    if MODEL_PATH.exists():
        print(f"Loading model from {MODEL_PATH}")
        return DiseaseClassifier.load(MODEL_PATH)
    elif MAT_PATH.exists():
        print(f"Training SVM from {MAT_PATH} ...")
        clf = DiseaseClassifier.from_mat(MAT_PATH)
        clf.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        return clf
    else:
        raise FileNotFoundError(
            "No pre-trained model found and Training_Data.mat is not accessible.\n"
            f"Expected model at: {MODEL_PATH}\n"
            f"Expected .mat at:  {MAT_PATH}"
        )


def predict(image_path: str, cluster_idx: int | None = None, show: bool = False) -> None:
    # 1. Preprocess
    original, enhanced = preprocess(image_path)

    # 2. Segment
    cluster_map, cluster_images = segment_clusters(enhanced)

    # 3. Select ROI cluster
    if cluster_idx is not None:
        if not 0 <= cluster_idx < len(cluster_images):
            raise ValueError(f"--cluster must be 0, 1, or 2 (got {cluster_idx})")
        selected_idx = cluster_idx
        seg_img = cluster_images[cluster_idx]
    else:
        selected_idx, seg_img = auto_select_disease_cluster(cluster_images, cluster_map)

    # 4. Extract features
    features = extract_features(seg_img, leaf_img=enhanced)

    # 5. Classify
    clf = _load_or_train_classifier()
    label, name, proba = clf.predict(features)

    # 6. Report
    print("\n" + "=" * 50)
    print(f"Image      : {image_path}")
    print(f"Prediction : {name} (class {label})")
    print(f"Confidence : {proba[label] * 100:.1f}%")
    print(f"Cluster    : {selected_idx} (auto-selected)" if cluster_idx is None else f"Cluster    : {cluster_idx} (manual)")
    print()
    print("Probabilities per class:")
    for cls_id, cls_name in CLASS_NAMES.items():
        bar = "█" * int(proba[cls_id] * 20)
        print(f"  {cls_id} {cls_name:<30} {proba[cls_id]*100:5.1f}%  {bar}")
    print()
    print("Feature values:")
    for name_f, val in zip(FEATURE_NAMES, features):
        print(f"  {name_f:<15} {val:.4f}")
    print("=" * 50)

    # 7. Optional display
    if show:
        fig, axes = plt.subplots(1, 2 + len(cluster_images), figsize=(14, 4))
        axes[0].imshow(original);        axes[0].set_title("Original")
        axes[1].imshow(enhanced);        axes[1].set_title("Enhanced")
        for i, cimg in enumerate(cluster_images):
            axes[2 + i].imshow(cimg)
            title = f"Cluster {i}"
            if i == selected_idx:
                title += " ← selected"
            axes[2 + i].set_title(title)
        for ax in axes:
            ax.axis("off")
        plt.suptitle(f"Prediction: {CLASS_NAMES[label]}  ({proba[label]*100:.1f}%)")
        plt.tight_layout()
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="GreenR plant disease classifier")
    parser.add_argument("image", help="Path to leaf image (JPG / PNG / BMP)")
    parser.add_argument("--cluster", type=int, default=None,
                        help="Force a specific K-means cluster (0, 1, or 2). "
                             "Default: auto-select most diseased-looking cluster.")
    parser.add_argument("--show", action="store_true",
                        help="Display intermediate images using matplotlib")
    args = parser.parse_args()
    predict(args.image, cluster_idx=args.cluster, show=args.show)


if __name__ == "__main__":
    main()
