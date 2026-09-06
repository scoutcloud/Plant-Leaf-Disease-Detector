from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def apply_pca(X_train, X_val, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=0.95)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return (
        X_train_pca,
        X_val_pca,
        X_test_pca,
        pca,
        scaler
    )