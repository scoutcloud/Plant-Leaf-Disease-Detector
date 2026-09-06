import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.dataset_loader import load_dataset
from src.pca_analysis import apply_pca
from src.knn_classifier import train_knn
from src.naive_bayes_classifier import train_naive_bayes


def main():

    # ---------------------------------------
    # 1. Load Dataset
    # ---------------------------------------

    dataset_path = "dataset"

    X_train, y_train = load_dataset(
        dataset_path,
        "train"
    )

    X_val, y_val = load_dataset(
        dataset_path,
        "val"
    )

    X_test, y_test = load_dataset(
        dataset_path,
        "test"
    )

    print("\nDataset loaded successfully!")
    print("Training images:", len(X_train))
    print("Validation images:", len(X_val))
    print("Test images:", len(X_test))
    print("Original features:", X_train.shape[1])


    # ---------------------------------------
    # 2. Apply PCA
    # ---------------------------------------

    X_train_pca, X_val_pca, X_test_pca, pca, scaler = apply_pca(
        X_train,
        X_val,
        X_test
    )

    print("\nPCA applied successfully!")
    print("Training shape after PCA:", X_train_pca.shape)
    print("Validation shape after PCA:", X_val_pca.shape)
    print("Test shape after PCA:", X_test_pca.shape)
    print("Number of PCA components:", X_train_pca.shape[1])


    # ---------------------------------------
    # 3. KNN Classifier
    # ---------------------------------------

    knn_model = train_knn(
        X_train_pca,
        y_train
    )

    print("\nKNN training successful!")


    # ---------------------------------------
    # 4. Save KNN, PCA and Scaler
    # ---------------------------------------

    joblib.dump(
        knn_model,
        "models/knn_model.pkl"
    )

    joblib.dump(
        pca,
        "models/pca.pkl"
    )

    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    print("KNN model saved successfully!")
    print("PCA model saved successfully!")
    print("Scaler saved successfully!")


    # ---------------------------------------
    # 5. KNN Validation Evaluation
    # ---------------------------------------

    knn_val_predictions = knn_model.predict(
        X_val_pca
    )

    knn_val_accuracy = accuracy_score(
        y_val,
        knn_val_predictions
    )

    print("\n===== KNN Validation Evaluation =====")

    print(
        "KNN Validation Accuracy:",
        knn_val_accuracy
    )

    print(
        "KNN Validation Accuracy (%):",
        knn_val_accuracy * 100
    )

    print("\nKNN Classification Report:")

    print(
        classification_report(
            y_val,
            knn_val_predictions,
            target_names=[
                "Healthy",
                "Early Blight",
                "Late Blight"
            ]
        )
    )

    print("KNN Confusion Matrix:")

    knn_val_cm = confusion_matrix(
        y_val,
        knn_val_predictions
    )

    print(knn_val_cm)


    # ---------------------------------------
    # 6. Naive Bayes Classifier
    # ---------------------------------------

    nb_model = train_naive_bayes(
        X_train_pca,
        y_train
    )

    print("\nNaive Bayes training successful!")


    # ---------------------------------------
    # 7. Save Naive Bayes Model
    # ---------------------------------------

    joblib.dump(
        nb_model,
        "models/naive_bayes_model.pkl"
    )

    print("Naive Bayes model saved successfully!")


    # ---------------------------------------
    # 8. Naive Bayes Validation Evaluation
    # ---------------------------------------

    nb_val_predictions = nb_model.predict(
        X_val_pca
    )

    nb_val_accuracy = accuracy_score(
        y_val,
        nb_val_predictions
    )

    print("\n===== Naive Bayes Validation Evaluation =====")

    print(
        "Naive Bayes Validation Accuracy:",
        nb_val_accuracy
    )

    print(
        "Naive Bayes Validation Accuracy (%):",
        nb_val_accuracy * 100
    )

    print("\nNaive Bayes Classification Report:")

    print(
        classification_report(
            y_val,
            nb_val_predictions,
            target_names=[
                "Healthy",
                "Early Blight",
                "Late Blight"
            ]
        )
    )

    print("Naive Bayes Confusion Matrix:")

    nb_val_cm = confusion_matrix(
        y_val,
        nb_val_predictions
    )

    print(nb_val_cm)


    # ---------------------------------------
    # 9. Model Comparison
    # ---------------------------------------

    print("\n===== Model Comparison =====")

    print(
        "KNN Validation Accuracy (%):",
        knn_val_accuracy * 100
    )

    print(
        "Naive Bayes Validation Accuracy (%):",
        nb_val_accuracy * 100
    )

    if knn_val_accuracy > nb_val_accuracy:

        print("Better validation model: KNN")

        best_model_name = "KNN"

    elif nb_val_accuracy > knn_val_accuracy:

        print("Better validation model: Naive Bayes")

        best_model_name = "Naive Bayes"

    else:

        print("Both models have the same validation accuracy.")

        best_model_name = "KNN"


    # ---------------------------------------
    # 10. KNN Final Test Evaluation
    # ---------------------------------------

    print("\n===== KNN Final Test Evaluation =====")

    knn_test_predictions = knn_model.predict(
        X_test_pca
    )

    knn_test_accuracy = accuracy_score(
        y_test,
        knn_test_predictions
    )

    print(
        "KNN Test Accuracy:",
        knn_test_accuracy
    )

    print(
        "KNN Test Accuracy (%):",
        knn_test_accuracy * 100
    )

    print("\nKNN Test Classification Report:")

    print(
        classification_report(
            y_test,
            knn_test_predictions,
            target_names=[
                "Healthy",
                "Early Blight",
                "Late Blight"
            ]
        )
    )

    print("KNN Test Confusion Matrix:")

    knn_test_cm = confusion_matrix(
        y_test,
        knn_test_predictions
    )

    print(knn_test_cm)


    # ---------------------------------------
    # 11. Naive Bayes Final Test Evaluation
    # ---------------------------------------

    print("\n===== Naive Bayes Final Test Evaluation =====")

    nb_test_predictions = nb_model.predict(
        X_test_pca
    )

    nb_test_accuracy = accuracy_score(
        y_test,
        nb_test_predictions
    )

    print(
        "Naive Bayes Test Accuracy:",
        nb_test_accuracy
    )

    print(
        "Naive Bayes Test Accuracy (%):",
        nb_test_accuracy * 100
    )

    print("\nNaive Bayes Test Classification Report:")

    print(
        classification_report(
            y_test,
            nb_test_predictions,
            target_names=[
                "Healthy",
                "Early Blight",
                "Late Blight"
            ]
        )
    )

    print("Naive Bayes Test Confusion Matrix:")

    nb_test_cm = confusion_matrix(
        y_test,
        nb_test_predictions
    )

    print(nb_test_cm)


    # ---------------------------------------
    # 12. Final Model Result
    # ---------------------------------------

    print("\n===== FINAL MODEL RESULT =====")

    print(
        "Selected model based on validation:",
        best_model_name
    )

    print(
        "KNN Test Accuracy (%):",
        knn_test_accuracy * 100
    )

    print(
        "Naive Bayes Test Accuracy (%):",
        nb_test_accuracy * 100
    )

    print("\nSaved models:")

    print("models/knn_model.pkl")
    print("models/naive_bayes_model.pkl")
    print("models/pca.pkl")
    print("models/scaler.pkl")


if __name__ == "__main__":
    main()