import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#Load Dataset
cancer = datasets.load_breast_cancer()

# Mapping
diagnosis_map = {
    0: "Malignant",
    1: "Benign"
}

# Create DataFrame
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["diagnosis"] = cancer.target
df["diagnosis_name"] = df["diagnosis"].map(diagnosis_map)

#Basic EDA
print("\nDataset Info:")
df.info()

print("\nDataset Shape:", df.shape)

print("\nFirst 5 Rows:\n", df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["diagnosis"].value_counts())

#Correlation Heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", linewidths=0.5)
plt.title("Breast Cancer Correlation Matrix")
plt.show()

#Features & Target
X = df.drop(["diagnosis", "diagnosis_name"], axis=1)
y = df["diagnosis"]

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)

#Pipeline
pipe = Pipeline([("scaler", StandardScaler()),("model", LogisticRegression(max_iter=1000))])

#GridSearchCV
param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(pipe,param_grid,cv=5,scoring="accuracy")
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

# Best model
best_model = grid.best_estimator_

#Predictions
y_pred = best_model.predict(X_test)

#Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.2%}")

# Diagnosis Mapping Print
print("\nDiagnosis Mapping:")
for k, v in diagnosis_map.items():
    print(k, "=", v)

# Actual vs Predicted
comparison = pd.DataFrame({
    "Actual": pd.Series(y_test).values,
    "Predicted": y_pred
})

comparison["Actual"] = comparison["Actual"].map(diagnosis_map)
comparison["Predicted"] = comparison["Predicted"].map(diagnosis_map)

print("\nActual vs Predicted (First 10 Records):")
print(comparison.head(10))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=cancer.target_names)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

#Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Coefficients
coef_df = pd.DataFrame(best_model.named_steps["model"].coef_.T, index=X.columns, columns=["Coefficient"])

#Sort by importance
coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values(ascending=False).index)
coef_df = coef_df.round(4)
print("\nTop Feature Coefficients:")
print(coef_df.head(10))

print("\nIntercept:")
print(best_model.named_steps["model"].intercept_)