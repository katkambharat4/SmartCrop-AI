import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATASET_PATH = os.path.join("data", "crop_recommendation.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.pkl")

# Actual features available in dataset and accepted from user
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "rainfall"]
TARGET_COLUMN = "label"

def train():
    print("==================================================")
    print("SmartCrop AI - Model Training Pipeline")
    print("==================================================")

    # 1. Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print("Error: Dataset not found. Please place crop_recommendation.csv inside the data folder.")
        sys.exit(1)

    # 2. Load dataset
    print(f"\n[1] Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    # 3. Display dataset shape and inspect columns
    print(f"\n[2] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns in CSV: {df.columns.tolist()}")

    # Check for Season column in dataset
    if "season" not in [col.lower() for col in df.columns]:
        print("\nNote: 'Season' column is NOT present in the dataset.")
        print("As per project specifications, 'Season' will NOT be used as an ML training feature.")
        print(f"Model will be trained strictly on actual features: {FEATURE_COLUMNS}")

    # 4. Check missing values
    print("\n[3] Checking missing values:")
    missing = df.isnull().sum()
    print(missing)
    if missing.sum() > 0:
        print("Handling missing values by dropping null rows...")
        df = df.dropna()

    # 5. Check feature columns availability
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            print(f"Error: Required feature column '{col}' not found in dataset.")
            sys.exit(1)
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in dataset.")
        sys.exit(1)

    # 6. Separate features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    unique_crops = sorted(y.unique())
    print(f"\n[4] Total crop classes ({len(unique_crops)}):")
    print(", ".join(unique_crops))

    # 7. Split data into training and testing sets
    print("\n[5] Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")

    # 8. Train Random Forest Classifier
    print("\n[6] Training Random Forest Classifier (n_estimators=100, random_state=42)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 9. Evaluate model
    print("\n[7] Model Evaluation:")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Actual Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Actual Testing Accuracy:  {test_accuracy * 100:.2f}%")

    print("\nDetailed Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    # 10. Save trained model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n[8] Trained model successfully saved to: {MODEL_PATH}")

    # Save metadata dictionary for convenience and verification
    metadata = {
        "features": FEATURE_COLUMNS,
        "classes": unique_crops,
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "n_samples": int(df.shape[0]),
    }
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.pkl")
    joblib.dump(metadata, metadata_path)
    print(f"Model metadata successfully saved to: {metadata_path}")
    print("\n==================================================")
    print("Training Completed Successfully!")
    print("==================================================")

if __name__ == "__main__":
    train()
