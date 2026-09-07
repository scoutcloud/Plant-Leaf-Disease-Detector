import cv2


def calculate_disease_area(leaf_mask, disease_mask):
    """
    Calculate the percentage of the leaf area
    identified as a potential disease region.
    """

    # Count leaf pixels
    leaf_pixels = cv2.countNonZero(leaf_mask)

    # Count disease pixels
    disease_pixels = cv2.countNonZero(disease_mask)

    # Avoid division by zero
    if leaf_pixels == 0:
        return 0.0

    # Calculate percentage
    disease_percentage = (
        disease_pixels / leaf_pixels
    ) * 100

    return disease_percentage


def determine_severity(disease_percentage):
    """
    Convert disease percentage into a project-defined
    severity category.
    """

    if disease_percentage <= 5:
        return "Very Low"

    elif disease_percentage <= 15:
        return "Mild"

    elif disease_percentage <= 30:
        return "Moderate"

    else:
        return "Severe"