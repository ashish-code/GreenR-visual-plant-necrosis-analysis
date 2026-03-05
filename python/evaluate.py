"""
GreenR — Cross-Validation Accuracy Evaluation

Python equivalent of the accuracy evaluation button (pushbutton5) in DetectDisease_GUI.m.

Original MATLAB:
    500 iterations of hold-out cross-validation (70% train / 30% test)
    using crossvalind('HoldOut', groups, 0.3).
    Reports maximum accuracy across all iterations.

Usage
-----
    python evaluate.py
    python evaluate.py --mat ../../Code/GreenR/Accuracy_Data.mat
    python evaluate.py --n-iter 100 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DEFAULT_MAT = Path(__file__).parent.parent / "Code" / "GreenR" / "Accuracy_Data.mat"


def evaluate(mat_path: Path, n_iter: int = 500, test_size: float = 0.3, seed: int | None = None) -> None:
    print(f"Loading: {mat_path}")
    mat = loadmat(str(mat_path))
    X = mat["Train_Feat"].astype(np.float64)
    y = mat["Train_Label"].flatten().astype(int)

    print(f"Dataset : {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"Running : {n_iter} hold-out iterations (test={test_size:.0%}) ...")

    splitter = StratifiedShuffleSplit(n_splits=n_iter, test_size=test_size, random_state=seed)
    accuracies = []

    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = SVC(kernel="linear", decision_function_shape="ovr")
        clf.fit(X_tr_s, y_tr)
        acc = clf.score(X_te_s, y_te) * 100.0
        accuracies.append(acc)

        if i % 50 == 0 or i == n_iter:
            print(f"  [{i:4d}/{n_iter}]  running max={max(accuracies):.2f}%  mean={np.mean(accuracies):.2f}%")

    accuracies = np.array(accuracies)
    print("\n" + "=" * 45)
    print(f"Max accuracy  : {accuracies.max():.2f}%")
    print(f"Mean accuracy : {accuracies.mean():.2f}%")
    print(f"Std deviation : {accuracies.std():.2f}%")
    print(f"Min accuracy  : {accuracies.min():.2f}%")
    print("=" * 45)


def main() -> None:
    parser = argparse.ArgumentParser(description="GreenR hold-out cross-validation evaluation")
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT,
                        help="Path to Accuracy_Data.mat (default: searches repo Code/ directory)")
    parser.add_argument("--n-iter", type=int, default=500,
                        help="Number of hold-out iterations (default: 500, matching MATLAB)")
    parser.add_argument("--test-size", type=float, default=0.3,
                        help="Fraction held out for testing (default: 0.3)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (MATLAB version was non-deterministic)")
    args = parser.parse_args()

    if not args.mat.exists():
        raise FileNotFoundError(
            f"Accuracy_Data.mat not found at: {args.mat}\n"
            "Provide the path with --mat, e.g.:\n"
            "  python evaluate.py --mat ../../Code/GreenR/Accuracy_Data.mat"
        )
    evaluate(args.mat, n_iter=args.n_iter, test_size=args.test_size, seed=args.seed)


if __name__ == "__main__":
    main()
