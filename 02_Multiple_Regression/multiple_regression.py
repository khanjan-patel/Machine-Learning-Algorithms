import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#LOAD DATASET

df = pd.read_csv("car_data.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

#DATA UNDERSTANDING

print("\nData Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe().round())

print("\nMissing Values:")
print(df.isnull().sum())

#DATA CLEANING

print("\nNumber of Duplicate Rows =", df.duplicated().sum())

df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print("After Removing Duplicate Rows, Shape =", df.shape)

# OUTLIER REMOVAL USING IQR

outlier_cols = ["Selling_Price", "Present_Price", "Kms_Driven"]

print("\nShape Before Outlier Removal:", df.shape)

for col in outlier_cols:
    rows_before = len(df)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    rows_after = len(df)
    print(f"{col}: Removed {rows_before - rows_after} outlier rows")

df.reset_index(drop=True, inplace=True)

print("\nShape After Outlier Removal:", df.shape)

#CATEGORICAL DATA ANALYSIS

print("\nFuel Type Categories:")
print(df["Fuel_Type"].unique())

print("\nSeller Type Categories:")
print(df["Seller_Type"].unique())

print("\nTransmission Categories:")
print(df["Transmission"].unique())

print("\nFuel Type Counts:")
print(df["Fuel_Type"].value_counts())

print("\nSeller Type Counts:")
print(df["Seller_Type"].value_counts())

print("\nTransmission Counts:")
print(df["Transmission"].value_counts())

#FEATURE ENGINEERING

# Remove Car Name
df.drop("Car_Name", axis=1, inplace=True)

# Create Car Age
current_year = datetime.now().year
df["Car_Age"] = current_year - df["Year"]

# Remove Year column

df.drop("Year", axis=1, inplace=True)

#ONE-HOT ENCODING

df = pd.get_dummies(df,columns=["Fuel_Type", "Seller_Type", "Transmission"],drop_first=True)

#CORRELATION ANALYSIS

corr = df.corr(numeric_only=True)

print("\nCorrelation with Selling Price:")
print(corr["Selling_Price"].sort_values(ascending=False))

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#Convert Boolean Columns to Integer
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

print("\nData Types After Encoding:")
print(df.dtypes)

#FEATURES AND TARGET

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# VIF ANALYSIS

vif_df = pd.DataFrame()
vif_df["Feature"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nVariance Inflation Factor (VIF)")
print(vif_df.sort_values(by="VIF", ascending=False))

#TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=99)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

#MODEL TRAINING

model = LinearRegression()
model.fit(X_train, y_train)

# CROSS VALIDATION

kf = KFold(n_splits=5,shuffle=True,random_state=99)

cv_scores = cross_val_score(model,X,y,cv=kf,scoring="r2")

print("CV scores :",np.round(cv_scores,4))
print("Average:", round(cv_scores.mean(),4))

#PREDICTION

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])

#MODEL EVALUATION

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

adjusted_r2 = 1 - (
    (1 - r2) * (len(y_test) - 1)
    / (len(y_test) - X_test.shape[1] - 1)
)

print("\nModel Evaluation:")
print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R² Score (%):", round(r2 * 100, 2))
print("Adjusted R²:", round(adjusted_r2, 4))

# ACTUAL VS PREDICTED COMPARISON

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

comparison["Absolute_Error"] = abs(comparison["Actual"] - comparison["Predicted"])

print("\nActual vs Predicted (First 10 Rows)")
print(comparison.head(10))

print("\nTop 10 Highest Prediction Errors")
print(comparison.sort_values(by="Absolute_Error",ascending=False).head(10))

#COEFFICIENT ANALYSIS

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coef_df = coef_df.sort_values(by="Coefficient",ascending=False)

print("\nRegression Coefficients:")
print(coef_df)

print("\nIntercept:")
print(model.intercept_)

#ACTUAL VS PREDICTED SCATTER PLOT

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")

plt.show()

# RESIDUAL PLOT

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.show()

sns.histplot(residuals, kde=True)

plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Residual Distribution")

plt.show()