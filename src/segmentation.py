import cv2
import numpy as np


def segment_leaf(image):
    """
    Segment the green leaf from the background.

    Returns:
        mask: Binary mask of the detected leaf
        segmented: Image containing only the leaf
    """

    # Convert BGR image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define green color range
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])

    # Create binary mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # -------------------------------
    # Morphological Opening
    # -------------------------------

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # -------------------------------
    # Morphological Closing
    # -------------------------------

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # -------------------------------
    # Apply mask to original image
    # -------------------------------

    segmented = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    return mask, segmented