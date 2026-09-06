import cv2
import numpy as np


def extract_features(image):
    """
    Extract simple color, texture, and shape features
    from a leaf image.
    """

    # Resize image
    image = cv2.resize(image, (256, 256))

    # -----------------------------------------
    # Color features
    # -----------------------------------------

    # Convert BGR to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mean_rgb = np.mean(rgb, axis=(0, 1))
    std_rgb = np.std(rgb, axis=(0, 1))

    # -----------------------------------------
    # Grayscale features
    # -----------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    mean_gray = np.mean(gray)
    std_gray = np.std(gray)

    # -----------------------------------------
    # Edge features
    # -----------------------------------------

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = np.mean(edges > 0)

    # -----------------------------------------
    # HSV features
    # -----------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    mean_hsv = np.mean(
        hsv,
        axis=(0, 1)
    )

    # -----------------------------------------
    # Combine all features
    # -----------------------------------------

    features = np.concatenate([
        mean_rgb,
        std_rgb,
        [mean_gray],
        [std_gray],
        [edge_density],
        mean_hsv
    ])

    return features