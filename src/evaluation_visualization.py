import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import confusion_matrix, accuracy_score

from src.dataset_loader import load_dataset


def main():

    # ---------------------------------------
    # 1. Load dataset
    # ---------------------------------------

    dataset_path = "dataset"

    X_train, y_train = load_dataset(dataset_path, "train")
    X_val, y_val = load_dataset(dataset_path, "val")
    X_test, y_test = load_dataset(dataset_path, "test")

    print("Dataset loaded successfully!")

    # ---------------------------------------
    # 2. Load saved models
    # ---------------------------------------

    knn_model = joblib.load("models/knn_model.pkl")
    nb_model = joblib.load("models/naive_bayes_model.pkl")
    pca = joblib.load("models/pca.pkl")
    scaler = joblib.load("models/scaler.pkl")

    print("Saved models loaded successfully!")

    # ---------------------------------------
    # 3. Apply scaler
    # ---------------------------------------

    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------
    # 4. Apply PCA
    # ---------------------------------------

    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print("PCA transformation completed!")

    # ---------------------------------------
    # 5. Generate predictions
    # ---------------------------------------

    knn_val_pred = knn_model.predict(X_val_pca)
    nb_val_pred = nb_model.predict(X_val_pca)

    knn_test_pred = knn_model.predict(X_test_pca)
    nb_test_pred = nb_model.predict(X_test_pca)

    # ---------------------------------------
    # 6. Calculate accuracies
    # ---------------------------------------

    knn_val_accuracy = accuracy_score(y_val, knn_val_pred)
    nb_val_accuracy = accuracy_score(y_val, nb_val_pred)

    knn_test_accuracy = accuracy_score(y_test, knn_test_pred)
    nb_test_accuracy = accuracy_score(y_test, nb_test_pred)

    print("\n===== Accuracy Results =====")

    print(
        "KNN Validation Accuracy:",
        round(knn_val_accuracy * 100, 2),
        "%"
    )

    print(
        "Naive Bayes Validation Accuracy:",
        round(nb_val_accuracy * 100, 2),
        "%"
    )

    print(
        "KNN Test Accuracy:",
        round(knn_test_accuracy * 100, 2),
        "%"
    )

    print(
        "Naive Bayes Test Accuracy:",
        round(nb_test_accuracy * 100, 2),
        "%"
    )

    # ---------------------------------------
    # 7. Create evaluation folder
    # ---------------------------------------

    os.makedirs("outputs/evaluation", exist_ok=True)

    # ---------------------------------------
    # 8. PCA Explained Variance Graph
    # ---------------------------------------

    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 5))

    components = np.arange(
        1,
        len(explained_variance) + 1
    )

    plt.bar(
        components,
        explained_variance
    )

    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance")

    plt.xticks(components)

    plt.tight_layout()

    plt.savefig(
        "outputs/evaluation/pca_explained_variance.png"
    )

    plt.close()

    # ---------------------------------------
    # 9. Model Accuracy Comparison
    # ---------------------------------------

    models = ["KNN", "Naive Bayes"]

    validation_accuracy = [
        knn_val_accuracy * 100,
        nb_val_accuracy * 100
    ]

    test_accuracy = [
        knn_test_accuracy * 100,
        nb_test_accuracy * 100
    ]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(
        x - width / 2,
        validation_accuracy,
        width,
        label="Validation"
    )

    plt.bar(
        x + width / 2,
        test_accuracy,
        width,
        label="Test"
    )

    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.title("KNN vs Naive Bayes Accuracy")

    plt.xticks(x, models)

    plt.legend()

    plt.ylim(0, 100)

    plt.tight_layout()

    plt.savefig(
        "outputs/evaluation/model_accuracy_comparison.png"
    )

    plt.close()

    # ---------------------------------------
    # 10. KNN Confusion Matrix
    # ---------------------------------------

    knn_cm = confusion_matrix(
        y_test,
        knn_test_pred
    )

    class_names = [
        "Healthy",
        "Early Blight",
        "Late Blight"
    ]

    plt.figure(figsize=(7, 6))

    plt.imshow(knn_cm)

    plt.title("KNN Test Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks(
        range(3),
        class_names,
        rotation=30
    )

    plt.yticks(
        range(3),
        class_names
    )

    for i in range(3):
        for j in range(3):
            plt.text(
                j,
                i,
                knn_cm[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        "outputs/evaluation/knn_confusion_matrix.png"
    )

    plt.close()

    # ---------------------------------------
    # 11. Naive Bayes Confusion Matrix
    # ---------------------------------------

    nb_cm = confusion_matrix(
        y_test,
        nb_test_pred
    )

    plt.figure(figsize=(7, 6))

    plt.imshow(nb_cm)

    plt.title("Naive Bayes Test Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks(
        range(3),
        class_names,
        rotation=30
    )

    plt.yticks(
        range(3),
        class_names
    )

    for i in range(3):
        for j in range(3):
            plt.text(
                j,
                i,
                nb_cm[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        "outputs/evaluation/naive_bayes_confusion_matrix.png"
    )

    plt.close()

    # ---------------------------------------
    # 12. Finish
    # ---------------------------------------

    print("\n===== Evaluation Graphs Generated =====")

    print(
        "outputs/evaluation/pca_explained_variance.png"
    )

    print(
        "outputs/evaluation/model_accuracy_comparison.png"
    )

    print(
        "outputs/evaluation/knn_confusion_matrix.png"
    )

    print(
        "outputs/evaluation/naive_bayes_confusion_matrix.png"
    )


if __name__ == "__main__":
    main()