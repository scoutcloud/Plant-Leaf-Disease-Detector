import cv2
import numpy as np


def detect_disease_regions(image, leaf_mask):
    """
    Detect small brown/dark lesion-like regions
    inside the segmented leaf.
    """

    # Convert BGR image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # -----------------------------------------
    # 1. Detect brown disease spots
    # -----------------------------------------

    brown_regions = (
        (hue >= 5) &
        (hue <= 25) &
        (saturation >= 70) &
        (value >= 45) &
        (value <= 190)
    )

    # -----------------------------------------
    # 2. Detect very dark lesions
    # -----------------------------------------

    dark_regions = (
        (value < 80) &
        (saturation > 50)
    )

    # -----------------------------------------
    # Combine candidates
    # -----------------------------------------

    disease_mask = (
        (brown_regions | dark_regions)
        & (leaf_mask > 0)
    )

    disease_mask = (
        disease_mask.astype(np.uint8) * 255
    )

    # -----------------------------------------
    # Morphological noise removal
    # -----------------------------------------

    kernel = np.ones((3, 3), np.uint8)

    disease_mask = cv2.morphologyEx(
        disease_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # -----------------------------------------
    # Connected component filtering
    # -----------------------------------------

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        disease_mask,
        connectivity=8
    )

    filtered_mask = np.zeros_like(disease_mask)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        # Keep lesion-sized regions
        if 10 <= area <= 1500:
            filtered_mask[labels == i] = 255

    disease_mask = filtered_mask

    # -----------------------------------------
    # Highlight detected regions
    # -----------------------------------------

    highlighted = image.copy()

    highlighted[disease_mask > 0] = [0, 0, 255]

    return disease_mask, highlighted