"""
Multi-class SVM classifier for plant disease detection.

Python equivalent of multisvm.m.

The original MATLAB code implements a manual One-vs-Rest loop using binary
svmtrain/svmclassify. Here we use scikit-learn's SVC which provides the
same One-vs-Rest strategy natively, plus StandardScaler (absent in the
original but recommended for SVM performance).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASS_NAMES = {
    0: "Alternaria Alternata",
    1: "Anthracnose",
    2: "Bacterial Blight",
    3: "Cercospora Leaf Spot",
    4: "Healthy Leaf",
}

_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "svm_greenr.pkl"
_DEFAULT_MAT_PATH = Path(__file__).parent.parent.parent / "Code" / "GreenR" / "Training_Data.mat"


class DiseaseClassifier:
    """
    One-vs-Rest linear SVM for leaf disease classification.

    Attributes
    ----------
    svm : sklearn.svm.SVC
    scaler : sklearn.preprocessing.StandardScaler
    """

    def __init__(self, kernel: str = "linear") -> None:
        self.svm = SVC(kernel=kernel, decision_function_shape="ovr", probability=True)
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DiseaseClassifier":
        """
        Train the SVM on feature matrix X (N × 13) with labels y (N,).

        Feature scaling is applied internally via StandardScaler — absent in
        the original MATLAB code but important for SVM with heterogeneous
        feature scales (features range from ~0.016 to ~10666).
        """
        X_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_scaled, y)
        self._fitted = True
        return self

    def predict(self, features: np.ndarray) -> tuple[int, str, np.ndarray]:
        """
        Classify a single sample or a batch.

        Parameters
        ----------
        features : np.ndarray  shape (13,) or (N, 13)

        Returns
        -------
        label     : int or np.ndarray   numeric class label(s)
        class_name: str or list[str]    human-readable disease name(s)
        proba     : np.ndarray          class probabilities (N × n_classes)
        """
        if not self._fitted:
            raise RuntimeError("Classifier not trained. Call fit() or load() first.")

        single = features.ndim == 1
        X = features.reshape(1, -1) if single else features
        X_scaled = self.scaler.transform(X)

        labels = self.svm.predict(X_scaled)
        proba = self.svm.predict_proba(X_scaled)

        if single:
            label = int(labels[0])
            return label, CLASS_NAMES[label], proba[0]
        return labels, [CLASS_NAMES[int(l)] for l in labels], proba

    def save(self, path: str | Path = _DEFAULT_MODEL_PATH) -> None:
        """Persist model + scaler to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"svm": self.svm, "scaler": self.scaler}, path)

    @classmethod
    def load(cls, path: str | Path = _DEFAULT_MODEL_PATH) -> "DiseaseClassifier":
        """Load a previously saved model."""
        obj = cls()
        data = joblib.load(path)
        obj.svm = data["svm"]
        obj.scaler = data["scaler"]
        obj._fitted = True
        return obj

    @classmethod
    def from_mat(cls, mat_path: str | Path = _DEFAULT_MAT_PATH) -> "DiseaseClassifier":
        """
        Train directly from the original Training_Data.mat file.

        Equivalent to loading training data in Detect.m / DetectDisease_GUI.m.
        """
        mat = loadmat(str(mat_path))
        X = mat["Train_Feat"].astype(np.float64)      # (125, 13)
        y = mat["Train_Label"].flatten().astype(int)  # (125,)
        return cls().fit(X, y)
