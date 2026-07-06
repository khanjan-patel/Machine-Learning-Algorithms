import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("placement.csv", dtype=str)

df["cgpa"] = pd.to_numeric(df["cgpa"], errors="coerce")
df["salary"] = pd.to_numeric(df["salary"], errors="coerce")

print("\nDataset Shape:", df.shape)

print("\nData Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nNumber of Duplicate Rows =", df.duplicated().sum())
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print("After Removing Duplicate Rows, Shape =", df.shape)

df["cgpa"] = df["cgpa"].fillna(df["cgpa"].mean())
df["salary"] = df["salary"].fillna(df["salary"].mean())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

correlation = df["cgpa"].corr(df["salary"])
print("\nCGPA-Salary Correlation:", round(correlation, 4))

X = df[["cgpa"]]
y = df["salary"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict Test Data
y_pred = model.predict(X_test)

print("\nSlope:", round(model.coef_[0], 4))
print("Intercept:", round(model.intercept_, 4))

# Predict Salary for New CGPA
new_data = pd.DataFrame({"cgpa": [6.99]})
predicted_salary = model.predict(new_data)

print("Predicted Salary =", round(predicted_salary[0], 2))

# Plot Regression Line
plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Actual Data")
X_sorted = X.sort_values(by="cgpa")
plt.plot(X_sorted, model.predict(X_sorted), color="black", label="Regression Line")

plt.xlabel("CGPA")
plt.ylabel("Salary")
plt.title("CGPA vs Salary")
plt.legend()
plt.show()

# Model Evaluation
print("\nModel Evaluation")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 4))
print("R² Score:", round(r2_score(y_test, y_pred), 4))