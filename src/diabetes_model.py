import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# --------------------------------
# 1. Load dataset
# --------------------------------

df = pd.read_csv("data/diabetes.csv")

print("Original dataset shape:")
print(df.shape)


# --------------------------------
# 2. Replace invalid zero values
# --------------------------------

invalid_zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[invalid_zero_columns] = df[invalid_zero_columns].replace(0, np.nan)


# --------------------------------
# 3. Separate features and target
# --------------------------------

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# --------------------------------
# 4. Train-test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("Testing data shape:")
print(X_test.shape)


# --------------------------------
# 5. Create ML Pipeline
# --------------------------------

model_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )
])


# --------------------------------
# 6. Train the complete pipeline
# --------------------------------

model_pipeline.fit(X_train, y_train)

print("\nRandom Forest pipeline trained successfully!")


# --------------------------------
# 7. Predictions
# --------------------------------

y_pred = model_pipeline.predict(X_test)

y_probability = model_pipeline.predict_proba(X_test)[:, 1]


# --------------------------------
# 8. Evaluation
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)


print("\nFINAL RANDOM FOREST PERFORMANCE")
print("-------------------------------")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# --------------------------------
# 9. Confusion Matrix
# --------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------
# 10. Feature Importance
# --------------------------------

rf_model = model_pipeline.named_steps["model"]

feature_importance = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print("-------------------------")
print(importance_df)


# --------------------------------
# 11. Save the complete pipeline
# --------------------------------

joblib.dump(
    model_pipeline,
    "models/diabetes_model.pkl"
)

print("\nModel saved successfully!")
print("Location: models/diabetes_model.pkl")