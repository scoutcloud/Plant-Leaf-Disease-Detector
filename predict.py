import sys
import os
import cv2
import numpy as np
import joblib

from src.feature_extraction import extract_features
from src.segmentation import segment_leaf
from src.disease_analysis import detect_disease_regions
from src.severity_analysis import calculate_disease_area, determine_severity


def main():

    # ---------------------------------------
    # 1. Check image argument
    # ---------------------------------------

    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        return

    image_path = sys.argv[1]

    # ---------------------------------------
    # 2. Load saved models
    # ---------------------------------------

    knn_model = joblib.load("models/knn_model.pkl")
    pca = joblib.load("models/pca.pkl")
    scaler = joblib.load("models/scaler.pkl")

    print("\nSaved models loaded successfully!")

    # ---------------------------------------
    # 3. Load input image
    # ---------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        return

    print("Input image loaded successfully!")

    # Create output folder if needed
    os.makedirs("outputs", exist_ok=True)

    # ---------------------------------------
    # 4. Leaf segmentation
    # ---------------------------------------

    leaf_mask, segmented_leaf = segment_leaf(image)

    cv2.imwrite(
        "outputs/prediction_leaf_mask.jpg",
        leaf_mask
    )

    cv2.imwrite(
        "outputs/prediction_segmented_leaf.jpg",
        segmented_leaf
    )

    print("Leaf segmentation completed!")

    # ---------------------------------------
    # 5. Disease region detection
    # ---------------------------------------

    disease_mask, highlighted = detect_disease_regions(
        image,
        leaf_mask
    )

    cv2.imwrite(
        "outputs/prediction_disease_mask.jpg",
        disease_mask
    )

    cv2.imwrite(
        "outputs/prediction_disease_highlighted.jpg",
        highlighted
    )

    print("Disease region detection completed!")

    # ---------------------------------------
    # 6. Calculate affected area
    # ---------------------------------------

    disease_percentage = calculate_disease_area(
        leaf_mask,
        disease_mask
    )

    severity = determine_severity(
        disease_percentage
    )

    print(
        "\nEstimated affected area:",
        round(disease_percentage, 2),
        "%"
    )

    print("Estimated severity:", severity)

    # ---------------------------------------
    # 7. Feature extraction
    # ---------------------------------------

    features = extract_features(image)

    features = np.array(features).reshape(1, -1)

    print("\nFeatures extracted:", features.shape[1])

    # ---------------------------------------
    # 8. Apply saved scaler
    # ---------------------------------------

    features_scaled = scaler.transform(features)

    # ---------------------------------------
    # 9. Apply saved PCA
    # ---------------------------------------

    image_pca = pca.transform(features_scaled)

    print("PCA transformation successful!")
    print("PCA components:", image_pca.shape[1])

    # ---------------------------------------
    # 10. Predict using saved KNN
    # ---------------------------------------

    prediction = knn_model.predict(
        image_pca
    )[0]

    print("KNN prediction completed!")

    # ---------------------------------------
    # 11. Prediction confidence
    # ---------------------------------------

    probabilities = knn_model.predict_proba(
        image_pca
    )[0]

    confidence = np.max(probabilities) * 100

    # ---------------------------------------
    # 12. Display final result
    # ---------------------------------------

    print("\n======================================")
    print("          LEAF DISEASE RESULT")
    print("======================================")

    print("Predicted disease:", prediction)

    print(
        "Prediction confidence:",
        round(confidence, 2),
        "%"
    )

    print(
        "Estimated affected area:",
        round(disease_percentage, 2),
        "%"
    )

    print("Estimated severity:", severity)

    print("======================================")

    print("\nPrediction images saved in:")
    print("outputs/")


if __name__ == "__main__":
    main()