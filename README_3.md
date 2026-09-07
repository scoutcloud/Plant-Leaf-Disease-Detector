# Plant Leaf Disease Detector

A computer vision project I built to detect diseases in tomato leaves — no deep learning, just classic image processing and machine learning. I wanted to see how far you could get with handcrafted features and simple classifiers before reaching for a CNN.

Given a tomato leaf image, the pipeline predicts one of three classes:

- Healthy
- Early Blight
- Late Blight

## Why traditional CV instead of deep learning?

Mostly curiosity. CNNs are the obvious go-to for this kind of task, but I wanted to actually understand what's happening under the hood — segmenting the leaf, isolating diseased regions, engineering features by hand, and reducing them with PCA — rather than treating the whole thing as a black box. It also meant I could train and run everything without needing a GPU.

## How it works

The pipeline goes through a few stages:

1. **Preprocessing** — resize the image, convert to grayscale, apply Gaussian filtering and histogram equalization to normalize lighting.
2. **Segmentation** — separate the leaf from its background using HSV color thresholds and morphological operations.
3. **Disease detection** — identify likely diseased regions based on color and intensity, and estimate how much of the leaf is affected (severity).
4. **Feature extraction** — pull out 12 handcrafted features describing color, texture, and the disease regions.
5. **PCA** — compress those 12 features down to 5 principal components.
6. **Classification** — feed the reduced features into a trained classifier to get a final prediction.

I trained two classifiers — KNN and Gaussian Naive Bayes — and compared them:

| Model | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| KNN | 88.78% | 89.07% |
| Naive Bayes | 78.78% | 79.35% |

KNN came out ahead, so that's the model `predict.py` uses by default.

## Dataset

Built on the [PlantVillage dataset](https://github.com/urayoanm/plantvillage-dataset), using only the three tomato classes:

- `Tomato___healthy`
- `Tomato___Early_blight`
- `Tomato___Late_blight`

Split into train/validation/test sets. The dataset itself isn't included in this repo (it's too large) — you'll need to download it separately and place it in the project directory.

## Project structure

```
plant-leaf-disease-detector/
├── main.py                      # trains the models end-to-end
├── predict.py                   # runs inference on a single leaf image
├── leaf.jpg                     # sample image for testing predict.py
├── models/                      # saved model files
│   ├── knn_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── pca.pkl
│   └── scaler.pkl
├── outputs/                     # prediction results & processed images land here
└── src/
    ├── dataset_loader.py
    ├── segmentation.py
    ├── disease_analysis.py
    ├── severity_analysis.py
    ├── feature_extraction.py
    ├── pca_analysis.py
    ├── knn_classifier.py
    ├── naive_bayes_classifier.py
    └── evaluation_visualization.py
```

## Getting it running

Clone the repo:

```bash
git clone https://github.com/scoutcloud/Plant-Leaf-Disease-Detector.git
cd Plant-Leaf-Disease-Detector
```

Set up the environment:

```bash
conda create -n plant_cv python=3.11
conda activate plant_cv
pip install opencv-python numpy matplotlib pandas scikit-learn pillow
```

Drop the PlantVillage dataset into the project directory, then train:

```bash
python main.py
```

This loads the dataset, extracts features, runs PCA, trains both classifiers, and prints out evaluation metrics.

## Predicting on a new leaf

```bash
python predict.py leaf.jpg
```

This runs the full pipeline on your image — segmentation, disease detection, severity estimation, feature extraction, PCA, and finally KNN classification — and saves the prediction plus the processed intermediate images to `outputs/`.

## Where it falls short

Being upfront about the limitations:

- Only handles three classes right now (healthy, early blight, late blight) — all tomato-specific.
- Disease-region detection relies on color/intensity thresholds, which aren't as robust as learned features. Lighting, background clutter, and leaf orientation can all throw it off.
- Severity and affected-area numbers are estimates, not ground-truth measurements — there's no labeled severity data to validate against.
- Handcrafted features cap out the accuracy compared to what a CNN could likely achieve.

## What I'd like to add next

- More plant species and disease types.
- Better segmentation, especially for messy backgrounds.
- Additional texture-based features.
- Data augmentation to make the classifiers more robust.
- A CNN baseline to compare directly against this traditional CV approach.
- Actual labeled severity data to validate the severity estimates.

## References

- [PlantVillage Dataset](https://github.com/urayoanm/plantvillage-dataset)
- [OpenCV](https://opencv.org/)
- [scikit-learn](https://scikit-learn.org/)
