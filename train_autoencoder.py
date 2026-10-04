import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

DATA_PATH = "data/landmarks/landmark_data.csv"
MODEL_PATH = "models/autoencoder_pca.pkl"
SCALER_PATH = "models/autoencoder_scaler.pkl"
THRESHOLD_PATH = "models/autoencoder_threshold.pkl"
os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop("label", axis=1).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# PCA acts as a linear autoencoder: compress to n_components, then reconstruct.
# Fewer components = more compression = stricter "normal pose" definition.
n_components = 16
pca = PCA(n_components=n_components, random_state=42)
pca.fit(X_train)

# Reconstruct test data and measure error, same idea as a neural autoencoder
X_test_compressed = pca.transform(X_test)
X_test_reconstructed = pca.inverse_transform(X_test_compressed)
errors = np.mean(np.square(X_test - X_test_reconstructed), axis=1)

threshold = np.percentile(errors, 95) * 1.2
print(f"Reconstruction error threshold set to: {threshold:.5f}")
print(f"Explained variance with {n_components} components: {pca.explained_variance_ratio_.sum():.4f}")

joblib.dump(pca, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(threshold, THRESHOLD_PATH)

print(f"PCA autoencoder saved to {MODEL_PATH}")
print(f"Scaler saved to {SCALER_PATH}")
print(f"Threshold saved to {THRESHOLD_PATH}")