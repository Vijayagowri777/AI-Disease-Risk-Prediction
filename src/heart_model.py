import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# --------------------------------------------------
# 1. Column names
# --------------------------------------------------

columns = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBloodPressure",
    "Cholesterol",
    "FastingBloodSugar",
    "RestingECG",
    "MaxHeartRate",
    "ExerciseInducedAngina",
    "STDepression",
    "STSlope",
    "MajorVessels",
    "Thalassemia",
    "Target"
]


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(
    "data/processed.cleveland.data",
    header=None,
    names=columns,
    na_values="?"
)


# --------------------------------------------------
# 3. Convert target to binary
# --------------------------------------------------
# 0 = No heart disease
# 1,2,3,4 = Heart disease
#
# Final:
# 0 = No heart disease
# 1 = Heart disease

df["Target"] = (df["Target"] > 0).astype(int)


# --------------------------------------------------
# 4. Select ONLY the 6 user-friendly features
# --------------------------------------------------

selected_features = [
    "Sex",
    "ChestPainType",
    "RestingBloodPressure",
    "Cholesterol",
    "MaxHeartRate",
    "ExerciseInducedAngina"
]

X = df[selected_features]
y = df["Target"]


# --------------------------------------------------
# 5. Define numerical features
# --------------------------------------------------

numerical_features = [
    "RestingBloodPressure",
    "Cholesterol",
    "MaxHeartRate"
]


# --------------------------------------------------
# 6. Define categorical features
# --------------------------------------------------

categorical_features = [
    "Sex",
    "ChestPainType",
    "ExerciseInducedAngina"
]


# --------------------------------------------------
# 7. Numerical preprocessing
# --------------------------------------------------

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# --------------------------------------------------
# 8. Categorical preprocessing
# --------------------------------------------------

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


# --------------------------------------------------
# 9. Combine preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# --------------------------------------------------
# 10. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 11. Create model pipeline
# --------------------------------------------------

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])


# --------------------------------------------------
# 12. Train model
# --------------------------------------------------

model_pipeline.fit(X_train, y_train)


# --------------------------------------------------
# 13. Predictions
# --------------------------------------------------

y_pred = model_pipeline.predict(X_test)

y_probability = model_pipeline.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 14. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)


print("\nHEART DISEASE MODEL - 6 FEATURES")
print("--------------------------------")

print("Selected features:")
for feature in selected_features:
    print(f"- {feature}")

print("\nMODEL PERFORMANCE")
print("-----------------")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# --------------------------------------------------
# 15. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# 16. Feature importance
# --------------------------------------------------

model = model_pipeline.named_steps["model"]
preprocessor_fitted = model_pipeline.named_steps["preprocessor"]

feature_names = preprocessor_fitted.get_feature_names_out()
coefficients = model.coef_[0]

importance_df = pd.DataFrame({
    "ProcessedFeature": feature_names,
    "Coefficient": coefficients,
    "AbsoluteImportance": abs(coefficients)
})


# --------------------------------------------------
# 17. Map processed features to original features
# --------------------------------------------------

def get_original_feature(processed_feature):

    for feature in numerical_features:
        if f"numerical__{feature}" in processed_feature:
            return feature

    for feature in categorical_features:
        if f"categorical__{feature}_" in processed_feature:
            return feature

    return processed_feature


importance_df["OriginalFeature"] = (
    importance_df["ProcessedFeature"]
    .apply(get_original_feature)
)


# --------------------------------------------------
# 18. Group importance by original feature
# --------------------------------------------------

original_importance = (
    importance_df
    .groupby("OriginalFeature")["AbsoluteImportance"]
    .sum()
    .sort_values(ascending=False)
)


print("\nFEATURE IMPORTANCE")
print("------------------")

print(original_importance)


# --------------------------------------------------
# 19. Save the 6-feature model
# --------------------------------------------------

joblib.dump(
    model_pipeline,
    "models/heart_model.pkl"
)

print("\nModel saved successfully:")
print("models/heart_model.pkl")