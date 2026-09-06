import cv2
import os
import numpy as np
from src.feature_extraction import extract_features


def load_dataset(dataset_path, split):
    X = []
    y = []

    class_names = [
        "Tomato___healthy",
        "Tomato___Early_blight",
        "Tomato___Late_blight"
    ]

    split_path = os.path.join(dataset_path, split)

    for class_name in class_names:

        class_path = os.path.join(split_path, class_name)

        if not os.path.exists(class_path):
            print("Folder not found:", class_path)
            continue

        print("Loading:", split, "-", class_name)

        for filename in os.listdir(class_path):

            image_path = os.path.join(class_path, filename)

            image = cv2.imread(image_path)

            if image is None:
                continue

            features = extract_features(image)

            X.append(features)
            y.append(class_name)

    X = np.array(X)
    y = np.array(y)

    return X, y