import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import joblib
import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("Telecom_Network_Data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Shape:", df.shape)


# ==========================================
# 2. PREPROCESSING
# ==========================================

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Convert weather into numerical values
# Storm is also included
df["weather"] = df["weather"].map({
    "Clear": 0,
    "Cloudy": 1,
    "Rain": 2,
    "Snow": 3,
    "Storm": 4
})

# Remove rows with missing values
df = df.dropna()

print("\nMissing values after preprocessing:")
print(df.isnull().sum())


# ==========================================
# 3. INPUT FEATURES AND TARGET
# ==========================================

features = [
    "users_connected",
    "download_speed",
    "upload_speed",
    "latency",
    "weather"
]

X = df[features]

y = df["congestion"]

print("\nInput Features:")
print(X.head())

print("\nTarget:")
print(y.head())


# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 5. ITERATION 1 - LOGISTIC REGRESSION
# ==========================================

print("\n====================================")
print("ITERATION 1 - LOGISTIC REGRESSION")
print("====================================")

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train, y_train)

lr_pred = logistic_model.predict(X_test)
lr_prob = logistic_model.predict_proba(X_test)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred, zero_division=0)
lr_recall = recall_score(y_test, lr_pred, zero_division=0)
lr_f1 = f1_score(y_test, lr_pred, zero_division=0)
lr_auc = roc_auc_score(y_test, lr_prob)

print("\nLogistic Regression Results:")
print("Accuracy :", round(lr_accuracy * 100, 2), "%")
print("Precision:", round(lr_precision * 100, 2), "%")
print("Recall   :", round(lr_recall * 100, 2), "%")
print("F1-Score :", round(lr_f1 * 100, 2), "%")
print("AUC-ROC  :", round(lr_auc, 4))

print("\nClassification Report:")
print(classification_report(y_test, lr_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))


# ==========================================
# 6. ITERATION 2 - RANDOM FOREST
# ==========================================

print("\n====================================")
print("ITERATION 2 - RANDOM FOREST")
print("====================================")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, zero_division=0)
rf_recall = recall_score(y_test, rf_pred, zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
rf_auc = roc_auc_score(y_test, rf_prob)

print("\nRandom Forest Results:")
print("Accuracy :", round(rf_accuracy * 100, 2), "%")
print("Precision:", round(rf_precision * 100, 2), "%")
print("Recall   :", round(rf_recall * 100, 2), "%")
print("F1-Score :", round(rf_f1 * 100, 2), "%")
print("AUC-ROC  :", round(rf_auc, 4))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


# ==========================================
# 7. SAVE RANDOM FOREST MODEL
# ==========================================

joblib.dump(rf_model, "network_model.pkl")

print("\nRandom Forest Model Saved Successfully!")


# ==========================================
# 8. FEATURE IMPORTANCE
# ==========================================

importance = rf_model.feature_importances_

plt.figure(figsize=(8, 5))
plt.bar(features, importance)

plt.title("Feature Importance in Network Congestion Prediction")
plt.xlabel("Network Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ==========================================
# 9. INTERACTIVE PREDICTION
# ==========================================

print("\n====================================")
print("NETWORK FAILURE PREDICTION")
print("====================================")

users = float(input("Enter number of users connected: "))
download = float(input("Enter download speed: "))
upload = float(input("Enter upload speed: "))
latency = float(input("Enter latency: "))

weather = int(
    input(
        "Enter weather value "
        "(0=Clear, 1=Cloudy, 2=Rain, 3=Snow, 4=Storm): "
    )
)

user_input = pd.DataFrame(
    [[users, download, upload, latency, weather]],
    columns=features
)

result = rf_model.predict(user_input)

probability = rf_model.predict_proba(user_input)[0][1]

if result[0] == 1:
    print("\nPrediction: Network Failure / Congestion Detected")
else:
    print("\nPrediction: Network is Normal")

print("Failure/Congestion Probability:",
      round(probability * 100, 2), "%")


# ==========================================
# 10. FINAL SUMMARY FOR REPORT
# ==========================================

print("\n====================================")
print("FINAL RESULTS FOR PBL REPORT")
print("====================================")

print("\nIteration 1 - Logistic Regression")
print("Accuracy :", round(lr_accuracy * 100, 2), "%")
print("Precision:", round(lr_precision * 100, 2), "%")
print("Recall   :", round(lr_recall * 100, 2), "%")
print("F1-Score :", round(lr_f1 * 100, 2), "%")
print("AUC-ROC  :", round(lr_auc, 4))

print("\nIteration 2 - Random Forest")
print("Accuracy :", round(rf_accuracy * 100, 2), "%")
print("Precision:", round(rf_precision * 100, 2), "%")
print("Recall   :", round(rf_recall * 100, 2), "%")
print("F1-Score :", round(rf_f1 * 100, 2), "%")
print("AUC-ROC  :", round(rf_auc, 4))