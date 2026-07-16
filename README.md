# Hyperspectral Anomaly Detection

An unsupervised machine learning project for detecting anomalous objects in hyperspectral imagery using dimensionality reduction, statistical modelling, and deep learning techniques.

The project investigates how unusual spectral signatures can be identified in high-dimensional remote-sensing data without requiring anomaly labels during model training. The current experiments use the **AVIRIS-1 San Diego hyperspectral dataset**, which contains aircraft anomalies embedded within a predominantly urban background.

## Project Overview

Hyperspectral images contain hundreds of spectral measurements for every image pixel. This rich spectral information can help distinguish materials and objects that may appear visually similar in conventional RGB imagery.

However, hyperspectral anomaly detection is challenging because:

* Hyperspectral data are highly dimensional.
* Anomalous pixels are extremely rare.
* Spectral bands are often strongly correlated.
* Background distributions may be complex.
* Small detection thresholds can produce many false positives.

This project develops and evaluates unsupervised approaches for assigning an anomaly score to every pixel in a hyperspectral scene.

The uploaded baseline experiment uses:

* Spectral standardization
* Principal Component Analysis
* Ledoit–Wolf covariance regularization
* Reed–Xiaoli anomaly detection
* Pixel-level ROC and precision–recall evaluation
* Percentile-based anomaly thresholding

## Dataset

The baseline experiment uses the **AVIRIS-1 San Diego dataset**.

### Dataset characteristics

| Property           |            Value |
| ------------------ | ---------------: |
| Spatial dimensions | 100 × 100 pixels |
| Spectral bands     |              189 |
| Total pixels       |           10,000 |
| Background pixels  |            9,936 |
| Anomaly pixels     |               64 |
| Anomaly proportion |            0.64% |

The MATLAB dataset file contains:

```text
data: Hyperspectral image cube with shape (100, 100, 189)
map: Binary ground-truth anomaly mask with shape (100, 100)
```

The anomaly mask uses:

```text
0 = Background
1 = Anomaly
```

The anomalous objects in this scene correspond to aircraft.

> The dataset is not included in this repository. Place `aviris_1.mat` inside the `data/` directory before running the notebooks.

## Repository Structure

```text
hyperspectral-anomaly-detection/
│
├── data/
│   └── aviris_1.mat
│
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_RX_Anomaly_Detection.ipynb
│   ├── 03_Autoencoder_Anomaly_Detection.ipynb
│   ├── 04_Variational_Autoencoder.ipynb
│   ├── 05_Spectral_VAE.ipynb
│   └── 06_Model_Comparison.ipynb
│
├── results/
│   ├── aviris_rx_results.png
│   ├── aviris_standard_scaler.joblib
│   ├── aviris_pca.joblib
│   └── aviris_rx_covariance.joblib
│
├── requirements.txt
├── README.md
└── .gitignore
```

Rename the notebook files in the structure above if your actual notebook names are different.

## Experimental Workflow

### 1. Data loading

The hyperspectral cube and ground-truth anomaly map are loaded from the MATLAB file using `scipy.io.loadmat`.

The hyperspectral data are converted to `float32`, while the ground-truth map is converted into a binary anomaly mask.

### 2. Data-quality inspection

The notebook examines:

* Minimum spectral value
* Maximum spectral value
* Mean spectral value
* Standard deviation
* Missing values
* Infinite values
* Number of background and anomaly pixels

The uploaded dataset contains no missing or infinite values.

### 3. Hyperspectral visualization

Several forms of visualization are generated:

* Individual spectral-band images
* A false-color representation of the scene
* The ground-truth aircraft mask
* Background and anomaly spectral signatures
* Mean background and anomaly spectra
* Principal-component images

A false-color image is constructed using selected bands and percentile-based contrast stretching.

### 4. Pixel-level feature preparation

The three-dimensional hyperspectral cube is reshaped into a two-dimensional feature matrix:

```text
Original cube: 100 × 100 × 189
Feature matrix: 10,000 × 189
```

Each row represents one image pixel, and each column represents one spectral band.

### 5. Spectral standardization

Each spectral band is standardized using `StandardScaler`.

Standardization prevents bands with large numerical values from dominating the PCA transformation and covariance calculation.

### 6. Principal Component Analysis

Principal Component Analysis is used to reduce the original 189 spectral bands to 30 principal components.

```text
Original dimensions: 189
Reduced dimensions: 30
Explained variance retained: 99.98%
```

PCA reduces spectral redundancy, computational cost, and covariance instability while retaining nearly all of the dataset’s variance.

### 7. Regularized RX anomaly detection

The project uses the Reed–Xiaoli detector, commonly known as the **RX detector**, as the statistical baseline.

For each pixel, the RX detector calculates its squared Mahalanobis distance from the estimated background distribution:

[
RX(x) = (x-\mu)^T\Sigma^{-1}(x-\mu)
]

where:

* (x) is the spectral feature vector of a pixel.
* (\mu) is the estimated background mean.
* (\Sigma) is the estimated covariance matrix.
* A higher RX score indicates a greater likelihood of being anomalous.

Instead of using an unregularized sample covariance matrix, the experiment uses the **Ledoit–Wolf covariance estimator**. Covariance regularization improves numerical stability when working with highly correlated spectral features.

### 8. Anomaly-score map

The RX score for every pixel is reshaped into a spatial anomaly map with dimensions:

```text
100 × 100
```

The score map is normalized to the range from 0 to 1 for visualization.

Pixels with larger values are considered more spectrally unusual relative to the estimated background.

### 9. Threshold-based classification

Although RX produces continuous anomaly scores, a binary prediction mask is also created for evaluation.

The baseline notebook classifies pixels at or above the **99th percentile** of RX scores as anomalies.

```text
Threshold percentile: 99
Predicted anomaly pixels: 100
```

This threshold is used only to illustrate binary detection behaviour. It is not treated as an optimized operating threshold.

## Baseline Results

The PCA and regularized RX experiment produced the following pixel-level results:

| Metric                     |    Score |
| -------------------------- | -------: |
| ROC-AUC                    |   0.9753 |
| PR-AUC / Average Precision |   0.1122 |
| Precision                  |   0.0100 |
| Recall                     |   0.0156 |
| F1-score                   |   0.0122 |
| Specificity                |   0.9900 |
| False-positive rate        | 0.009964 |

### Confusion matrix

|                   | Predicted Background | Predicted Anomaly |
| ----------------- | -------------------: | ----------------: |
| Actual Background |                9,837 |                99 |
| Actual Anomaly    |                   63 |                 1 |

## Interpretation of the Results

The high ROC-AUC of **0.9753** shows that the RX anomaly scores generally rank anomalous pixels above background pixels.

However, the lower PR-AUC and threshold-based recall demonstrate the difficulty of identifying rare anomalies in a highly imbalanced dataset.

Only 64 of the 10,000 pixels are anomalous. Therefore, even a small false-positive rate can produce more false detections than true detections.

The results also show an important distinction between:

* **Anomaly ranking performance**, measured using ROC-AUC and PR-AUC
* **Binary detection performance**, which depends heavily on the selected threshold

The 99th-percentile threshold provides high background specificity but detects only one of the 64 anomalous pixels. Future experiments should therefore investigate threshold-selection methods that achieve a better balance between detection probability and false-alarm rate.

## Generated Outputs

The notebook generates and saves:

* False-color hyperspectral visualization
* Ground-truth anomaly mask
* PCA component visualizations
* RX anomaly-score map
* Thresholded anomaly prediction
* ROC curve
* Precision–recall curve
* Combined experiment figure
* Trained preprocessing and covariance objects

The main experiment figure is saved as:

```text
results/aviris_rx_results.png
```

The fitted objects are saved as:

```text
results/aviris_standard_scaler.joblib
results/aviris_pca.joblib
results/aviris_rx_covariance.joblib
```

These files can be reused to transform data and reproduce the RX scoring pipeline.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/hyperspectral-anomaly-detection.git
cd hyperspectral-anomaly-detection
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Alternatively, install the main dependencies directly:

```bash
pip install numpy scipy pandas matplotlib scikit-learn spectral joblib jupyter
```

## Requirements

The project uses:

```text
numpy
scipy
pandas
matplotlib
scikit-learn
spectral
joblib
jupyter
```

Deep-learning notebooks may additionally require:

```text
torch
torchvision
```

## Running the Project

Place the dataset in the following location:

```text
hyperspectral-anomaly-detection/data/aviris_1.mat
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebooks and run them in numerical order.

For portability, define the dataset path relative to the repository:

```python
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "aviris_1.mat"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
```

Avoid using a machine-specific path such as:

```python
"/Users/username/Desktop/hyperspectral-anomaly-detection/data/aviris_1.mat"
```

## Methodology Summary

```text
Hyperspectral Cube
        ↓
Data Inspection and Visualization
        ↓
Pixel-Level Reshaping
        ↓
Spectral Standardization
        ↓
Principal Component Analysis
        ↓
Ledoit–Wolf Covariance Estimation
        ↓
RX Mahalanobis-Distance Scoring
        ↓
Spatial Anomaly-Score Map
        ↓
ROC-AUC and Precision–Recall Evaluation
        ↓
Percentile-Based Binary Prediction
```

## Current Limitations

The current RX baseline has several limitations:

* It models the background using a single global distribution.
* It does not explicitly use spatial neighbourhood information.
* The binary result is sensitive to the selected percentile threshold.
* Rare anomalies make accuracy and ROC-AUC insufficient by themselves.
* A global covariance model may not capture complex or multimodal backgrounds.
* PCA retains variance but does not specifically optimize anomaly separability.
* The experiment currently evaluates a single hyperspectral scene.

## Future Work

Future extensions of the project may include:

* Local RX anomaly detection
* Kernel RX detection
* Robust covariance estimation
* Autoencoder reconstruction-error detection
* Variational autoencoder anomaly detection
* Spectral variational autoencoders
* Spatial–spectral feature learning
* Convolutional autoencoders
* Patch-based anomaly detection
* Threshold optimization
* False-alarm-controlled evaluation
* Multiple-dataset validation
* Model comparison using consistent preprocessing
* Ablation studies for PCA dimensionality
* Statistical comparison of detection methods

## Evaluation Considerations

Because this dataset is highly imbalanced, the following metrics are emphasized:

* ROC-AUC
* Precision–recall AUC
* Precision
* Recall or probability of detection
* F1-score
* Specificity
* False-positive rate
* Confusion matrix

Overall accuracy should not be interpreted alone because a model can achieve very high accuracy by predicting nearly every pixel as background.

## Reproducibility

The project uses a fixed random seed:

```python
RANDOM_STATE = 42
```

The preprocessing models and covariance estimator are saved with `joblib`, allowing the fitted components of the RX pipeline to be reused.

To improve reproducibility across machines:

* Use relative file paths.
* Store dependencies in `requirements.txt`.
* Record package versions.
* Keep raw datasets separate from generated results.
* Do not commit large data or model files unless necessary.
* Run notebooks from a clean kernel before publishing them.

## Technologies Used

* Python
* NumPy
* SciPy
* Pandas
* Matplotlib
* Scikit-learn
* Spectral Python
* Joblib
* Jupyter Notebook

## Research Objective

The broader objective of this project is to compare statistical and representation-learning approaches for hyperspectral anomaly detection and examine their behaviour under severe class imbalance.

The experiments focus on:

* Spectral feature representation
* Unsupervised anomaly scoring
* Dimensionality reduction
* Covariance modelling
* Reconstruction-based detection
* Threshold sensitivity
* Rare-event evaluation
* Reproducible scientific computing

## Author

**Monica Chaganti**

Master of Science in Information Systems
Artificial Intelligence and Data Science

Research interests include machine learning, deep learning, scientific machine learning, computer vision, remote sensing, representation learning, and anomaly detection.

## License

This repository is intended for educational and research purposes.


## Acknowledgements

This project uses the AVIRIS-1 San Diego hyperspectral dataset for research and educational experimentation. Please cite the original dataset providers and relevant anomaly-detection literature when using this work in academic publications.
