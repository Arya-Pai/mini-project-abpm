import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from sklearn.model_selection import StratifiedKFold


data, meta = arff.loadarff("ABPM-dataset.arff")
df = pd.DataFrame(data)


for col in df.select_dtypes([object]):
    df[col] = df[col].str.decode("utf-8")

print("✅ Data loaded. Shape:", df.shape)
print(df.head())


if "Hypertension" not in df.columns:
    df["Hypertension"] = (
        ((df["BPS-Day24"] >= 135) | (df["BPD-Day24"] >= 85)) |     # Daytime criteria
        ((df["BPS-Night24"] >= 120) | (df["BPD-Night24"] >= 70))   # Nighttime criteria
    ).astype(int)


features = ["HRecord", "Perc", "Interrupt", "Age", "Weight", "Height", 
           "BPS-Day24","BPD-Day24","BPS-Night24","BPD-Night24", "Sexe", "BP-Load","Max-Sys","Min-Sys","Max-Dia","Min-Dia"]

target = "Hypertension"

X = df[features]
y = df[target]

# Encode categorical
if df["Interrupt"].dtype == "object":
    le = LabelEncoder()
    df["Interrupt"] = le.fit_transform(df["Interrupt"])
    X.loc[:, "Interrupt"] = df["Interrupt"]

if df["Sexe"].dtype == "object":
    le = LabelEncoder()
    df["Sexe"] = le.fit_transform(df["Sexe"])
    X.loc[:, "Sexe"] = df["Sexe"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n✅ Model Evaluation Metrics (Train/Test Split):")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No HTN", "HTN"], yticklabels=["No HTN", "HTN"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Train/Test Split")
plt.show()


fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"ROC curve (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


importances = rf_model.feature_importances_
feat_importances = pd.Series(importances, index=features)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importance in Hypertension Prediction")
plt.show()



def add_realistic_noise(X, noise_level=0.05):
    """Adds Gaussian noise to numeric features to simulate real-world data."""
    X_noisy = X.copy()
    numeric_cols = X_noisy.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        std = X_noisy[col].std()
        noise = np.random.normal(0, noise_level * std, size=X_noisy[col].shape)
        X_noisy[col] += noise
    
    return X_noisy


def cross_validate_model(model, X, y, folds=50, noise_level=0.05):
    # Add Gaussian noise to simulate real-world imperfections
    X_noisy = add_realistic_noise(X, noise_level=noise_level)

    # 50-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}

    for train_idx, test_idx in skf.split(X_noisy, y):
        X_train, X_test = X_noisy.iloc[train_idx], X_noisy.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

       
        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision"].append(precision_score(y_test, y_pred))
        metrics["recall"].append(recall_score(y_test, y_pred))
        metrics["f1"].append(f1_score(y_test, y_pred))
        try:
            metrics["roc_auc"].append(roc_auc_score(y_test, y_pred))
        except:
            pass

    print(f"\n📊 Cross-Validation Results ({folds} folds):")
    for metric, values in metrics.items():
        mean, std = np.mean(values), np.std(values)
        print(f"{metric.capitalize():<10}: {mean:.4f} ± {std:.4f}")

    
    final_model = model.fit(X_noisy, y)
    y_pred_final = final_model.predict(X_noisy)

    cm = confusion_matrix(y, y_pred_final)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No HTN", "HTN"], yticklabels=["No HTN", "HTN"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Cross Validation (Aggregated)")
    plt.show()


cross_validate_model(rf_model, X, y, folds=50, noise_level=0.05)

import joblib

def save_model_and_summary(model, X, label_encoders=None, filename="rf_model.pkl"):
    """Save trained model, encoders and print feature importance summary."""
    
    joblib.dump(model, filename)
    print("\nMODEL TRAINING COMPLETED")
    print("-" * 40)
    print(f"Model saved: {filename}")

    
    if label_encoders:
        joblib.dump(label_encoders, "label_encoders.pkl")
        print("Encoders saved: label_encoders.pkl")

    
    print("\nFEATURE IMPORTANCE")
    print("-" * 40)
    feature_importance = model.feature_importances_
    importance_df = (
        pd.DataFrame({
            'feature': X.columns,
            'importance': feature_importance
        })
        .sort_values('importance', ascending=False)
    )

    for i, (idx, row) in enumerate(importance_df.iterrows(), 1):
        print(f"{i:2d}. {row['feature']:15s} ({row['importance']:.3f})")

    return model, label_encoders


label_encoders = {
    "Interrupt": le if "Interrupt" in df else None,
    "Sexe": le if "Sexe" in df else None
}
save_model_and_summary(rf_model, X, label_encoders, filename="rf_model.pkl")
