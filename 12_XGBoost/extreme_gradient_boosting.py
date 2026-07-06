import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset Shape:", X.shape)
print("\nTarget Classes:")
print(data.target_names)

# Exploratory Data Analysis
print("\nFirst 5 Rows")
print(X.head())

print("\nDataset Info")
X.info()

print("\nMissing Values")
print(X.isnull().sum())

print("\nStatistical Summary")
print(X.describe())

# Target Distribution
plt.figure(figsize=(5,4))
sns.countplot(x=y)
plt.xticks([0, 1], data.target_names)
plt.title("Target Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(X.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Pairplot (First 5 Features)
df_pair = X.iloc[:, :5].copy()
df_pair["Target"] = y.map({0: "Malignant", 1: "Benign"})

sns.pairplot(df_pair, hue="Target")
plt.show()

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Pipeline
pipeline = Pipeline([
    ("xgb", XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ))
])

# Hyperparameter Tuning
param_grid = {
    "xgb__n_estimators": [50, 100, 150],
    "xgb__learning_rate": [0.01, 0.1, 0.2],
    "xgb__max_depth": [3, 5],
    "xgb__subsample": [0.8, 1.0]
}

grid = GridSearchCV(pipeline,param_grid,cv=5,scoring="accuracy",n_jobs=-1)

grid.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid.best_params_)

print("\nBest Cross Validation Accuracy:")
print(grid.best_score_)

# Best Model
best_model = grid.best_estimator_

# Prediction
y_pred = best_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", accuracy)

# Cross Validation
cv_scores = cross_val_score(best_model,X,y,cv=5,scoring="accuracy")

print("\nCross Validation Scores:")
print(cv_scores)

print("Average CV Accuracy:", cv_scores.mean())

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=data.target_names,yticklabels=data.target_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report\n")
print(classification_report(y_test,y_pred,target_names=data.target_names))

# Sample Predictions
sample_predictions = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nSample Predictions")
print(sample_predictions.head(10))