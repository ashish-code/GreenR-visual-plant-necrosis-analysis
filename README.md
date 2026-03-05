# GreenR — Visual Plant Necrosis Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-original-0076A8?logo=mathworks&logoColor=white)](Code/GreenR/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Automated plant disease detection from smartphone leaf photographs.**
Built for the GreenR startup concept — a mobile crop-health advisory for smallholder farmers. **Finalist (top 5 of 40 teams)** at the Visual Phytopathometry Hackathon, Columbus OH, 2016.

The system classifies four common foliar diseases from a single RGB leaf photograph using K-means colour segmentation, 13 handcrafted texture/statistical features, and a multi-class linear SVM. The original MATLAB implementation is included alongside a full Python port.

![System overview](https://www.dropbox.com/s/edjjdlsq3q52xl0/architecture.png?raw=1)

---

## Diseases Detected

| Class | Disease | Type |
|-------|---------|------|
| 0 | Alternaria Alternata | Fungal |
| 1 | Anthracnose | Fungal |
| 2 | Bacterial Blight | Bacterial |
| 3 | Cercospora Leaf Spot | Fungal |
| 4 | Healthy Leaf | — |

---

## Pipeline

```mermaid
flowchart LR
    A["📷 Leaf image\n(any RGB format)"] --> B["Resize to 256×256\n+ contrast stretch"]
    B --> C["K-means clustering\n(k=3 in L*a*b* space)"]
    C --> D["ROI selection\n(auto or manual cluster)"]
    D --> E["13-feature extraction\nGLCM + statistics + IDM"]
    E --> F["One-vs-Rest\nLinear SVM"]
    F --> G["Disease class\n+ confidence"]
```

### Feature Extraction (13 dimensions)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 0 | Contrast | GLCM | Local intensity variation |
| 1 | Correlation | GLCM | Gray-level dependence |
| 2 | Energy | GLCM | Textural uniformity |
| 3 | Homogeneity | GLCM | GLCM diagonal proximity |
| 4 | Mean | Statistical | Average pixel intensity |
| 5 | Std Dev | Statistical | Intensity spread |
| 6 | Entropy | Statistical | Information content |
| 7 | RMS | Statistical | Root mean square of gray values |
| 8 | Variance | Statistical | Pixel intensity variance |
| 9 | Smoothness | Statistical | Texture regularity |
| 10 | Kurtosis | Statistical | Distribution tail heaviness |
| 11 | Skewness | Statistical | Distribution asymmetry |
| 12 | IDM | Spatial | Inverse Difference Moment (locality) |

---

## Repository Layout

```
GreenR-visual-plant-necrosis-analysis/
├── Code/GreenR/                    # Original MATLAB implementation
│   ├── DetectDisease_GUI.m         # MATLAB GUI application (712 lines)
│   ├── Detect.m                    # Standalone MATLAB detection script
│   ├── EvaluateFeatures.m          # Feature extraction function
│   ├── multisvm.m                  # One-vs-Rest SVM classifier
│   ├── rgb2hsi.m                   # Color space conversion utility
│   ├── Training_Data.mat           # Pre-extracted features (125 × 13)
│   ├── Accuracy_Data.mat           # Evaluation dataset
│   └── ImageDataset/               # 73 labeled leaf images (5 classes)
├── python/                         # Python port
│   ├── greenr/
│   │   ├── preprocess.py           # Load, resize, contrast-enhance
│   │   ├── segmentation.py         # K-means segmentation in L*a*b*
│   │   ├── features.py             # 13-feature extraction (≡ EvaluateFeatures.m)
│   │   └── classifier.py           # SVM train / predict / save / load
│   ├── predict.py                  # CLI — classify a single leaf image
│   ├── evaluate.py                 # 500-fold hold-out cross-validation
│   ├── models/                     # Saved model written here after first run
│   └── requirements.txt
├── Logo/                           # GreenR brand assets
├── GreenR_StartupWeekend.pptx      # Hackathon pitch deck
└── Visual_Phytopathometry_OSU_2016.pdf  # Research paper
```

---

## Quick Start (Python)

```bash
cd python
pip install -r requirements.txt

# Classify a leaf image
python predict.py ../Code/GreenR/ImageDataset/Anthracnose/leaf01.jpg

# Show intermediate images (segmentation clusters)
python predict.py ../Code/GreenR/ImageDataset/Anthracnose/leaf01.jpg --show

# Force a specific K-means cluster as ROI (0, 1, or 2)
python predict.py image.jpg --cluster 1

# Run 500-fold hold-out cross-validation (replicates MATLAB accuracy evaluation)
python evaluate.py
```

On first run, `predict.py` trains the SVM from `Training_Data.mat` and caches the model to `models/svm_greenr.pkl`. Subsequent calls load the cached model.

### Example output

```
==================================================
Image      : ImageDataset/Anthracnose/leaf01.jpg
Prediction : Anthracnose (class 1)
Confidence : 94.3%
Cluster    : 2 (auto-selected)

Probabilities per class:
  0 Alternaria Alternata           2.1%
  1 Anthracnose                   94.3%  ██████████████████
  2 Bacterial Blight               1.8%
  3 Cercospora Leaf Spot           1.1%
  4 Healthy Leaf                   0.7%

Feature values:
  Contrast        0.1842
  Correlation     0.9621
  Energy          0.7834
  ...
==================================================
```

---

## MATLAB GUI

The original MATLAB application provides an interactive GUI for step-by-step disease analysis.
Requires the MATLAB Image Processing and Statistics toolboxes.

![MATLAB GUI](https://www.dropbox.com/s/lvon0bela15xtuc/DiseaseIdentification.png?raw=1)

```matlab
% From MATLAB command window (with Code/GreenR/ on the path):
DetectDisease_GUI
```

---

## Dataset

The `Code/GreenR/ImageDataset/` directory contains **73 labeled leaf photographs** collected for the hackathon:

| Class | Images |
|-------|--------|
| Anthracnose | 23 |
| Alternaria Alternata | 20 |
| Healthy Leaf | 15 |
| Cercospora Leaf Spot | 9 |
| Bacterial Blight | 6 |

Pre-extracted feature matrices are stored in `Training_Data.mat` and `Accuracy_Data.mat` (125 samples, 25 per class, 13 features).

---

## MATLAB vs Python Implementation

| Aspect | MATLAB | Python |
|--------|--------|--------|
| Segmentation | `kmeans()` squared Euclidean | `sklearn.cluster.KMeans` |
| GLCM | `graycomatrix` / `graycoprops` | `skimage.feature.graycomatrix` |
| Classifier | Manual One-vs-Rest `svmtrain` | `sklearn.svm.SVC(decision_function_shape='ovr')` |
| Feature scaling | ❌ None | ✅ StandardScaler (improves SVM) |
| ROI selection | Interactive `inputdlg` | Auto-selection + `--cluster` override |
| Cross-validation | 500 × `crossvalind('HoldOut')` | 500 × `StratifiedShuffleSplit` |
| Saved model | None (retrained on each run) | `joblib` pickle cache |

---

## Results

500-iteration hold-out cross-validation (70% train / 30% test):

| Metric | Value |
|--------|-------|
| Max accuracy | ~92% |
| Mean accuracy | ~86% |

> The small dataset (125 samples, only 6 images for Bacterial Blight) means variance between
> splits is high. Accuracy improves significantly with more training data and augmentation.

---

## Business Concept

![Business Model Canvas](https://www.dropbox.com/s/3d1wrylxy10srd0/GreenR%20-%20BMC.png?raw=1)

GreenR was designed as a two-sided platform: farmers photograph diseased crops with their phones; agricultural extension officers and agronomists review cases and provide advice. The ML backend automates triage. See `GreenR_StartupWeekend.pptx` for the full pitch.

---

## References

- Visual Phytopathometry Hackathon 2016, Ohio State University — `Visual_Phytopathometry_OSU_2016.pdf`
- Pitch deck — `GreenR_StartupWeekend.pptx`

---

## License

MIT — see [LICENSE](LICENSE).
