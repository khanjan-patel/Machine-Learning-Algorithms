import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Display Dataset Information
print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# Remove Outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

print("\nDataset Shape After Removing Outliers:")
print(df.shape)

# Features and Target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

# Create Pipeline
# StandardScaler + LassoCV
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(
        alphas=np.logspace(-4, 1, 100),
        cv=5,
        random_state=42,
        max_iter=10000
    ))
])

# Train Model
pipeline.fit(X_train, y_train)

# Best Alpha
best_alpha = pipeline.named_steps["lasso"].alpha_
print("\nBest Alpha Selected:", best_alpha)

# Predict Test Data
y_pred = pipeline.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print(f"Mean Absolute Error      : {mae:.4f}")
print(f"Mean Squared Error       : {mse:.4f}")
print(f"Root Mean Squared Error  : {rmse:.4f}")
print(f"R² Score                 : {r2:.4f}")

# Cross Validation Score
cv_scores = cross_val_score(pipeline,X,y,cv=5,scoring="r2")

print("\nCross Validation R² Scores:")
print(np.round(cv_scores, 4))

print(f"Average CV R² Score : {cv_scores.mean():.4f}")

# Actual vs Predicted
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": np.round(y_pred, 4)
})

print("\nActual vs Predicted (First 10 Records)")
print(comparison.head(10))

# Scatter Plot
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()

# Residual Plot
residuals = y_test - y_pred

plt.figure(figsize=(7,5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

sns.histplot(residuals, kde=True)

plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Residual Distribution")

plt.show()

# Feature Coefficients
coefficients = pipeline.named_steps["lasso"].coef_

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": coefficients
})

coef_df["Absolute Coefficient"] = coef_df["Coefficient"].abs()

coef_df = coef_df.sort_values(by="Absolute Coefficient",ascending=False)

print("\nFeature Coefficients")
print(coef_df[["Feature", "Coefficient"]])

# Feature Coefficient Bar Chart
plt.figure(figsize=(9,5))
plt.bar(coef_df["Feature"], coef_df["Coefficient"])
plt.xlabel("Features")
plt.ylabel("Coefficient")
plt.title("Feature Importance (Lasso Coefficients)")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.show()