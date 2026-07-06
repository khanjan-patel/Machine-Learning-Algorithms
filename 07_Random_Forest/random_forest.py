import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
iris = datasets.load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

# Display Dataset
print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# Features and Target
X = df.drop("species", axis=1)
y = df["species"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Random Forest Model
model = RandomForestClassifier(n_estimators=100,criterion="gini",max_depth=5,random_state=42,oob_score=True)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy, 4))

# OOB Score
print("Out-of-Bag Score:", round(model.oob_score_, 4))

# Cross Validation
cv_scores = cross_val_score(model, X, y, cv=5)

print("\nCross Validation Scores:")
print(np.round(cv_scores, 4))

print("Average CV Accuracy:", round(cv_scores.mean(), 4))

# Display Species Mapping
print("\nSpecies Mapping:")
for i, species in enumerate(iris.target_names):
    print(f"{i} = {species}")

# Actual vs Predicted Comparison
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

species_map = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

comparison["Actual"] = comparison["Actual"].map(species_map)
comparison["Predicted"] = comparison["Predicted"].map(species_map)

print("\nActual vs Predicted (First 10 Records):")
print(comparison.head(10))

# Pair Plot
df_plot = df.copy()
df_plot["species"] = df_plot["species"].map(species_map)

sns.pairplot(df_plot, hue="species")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.drop("species", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - Iris Dataset")
plt.show()

# User Input
print("\nEnter flower measurements:")

try:
    sepal_length = float(input("Sepal Length (cm): "))
    sepal_width = float(input("Sepal Width (cm): "))
    petal_length = float(input("Petal Length (cm): "))
    petal_width = float(input("Petal Width (cm): "))
except ValueError:
    print("\nInvalid input! Please enter numeric values only.")
    exit()

# Create DataFrame with user input
user_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)

# Predict Species
prediction = model.predict(user_data)

# Predict Probabilities
probabilities = model.predict_proba(user_data)

species_name = iris.target_names[prediction[0]]

print("\nPrediction Probabilities:")
for i, species in enumerate(iris.target_names):
    print(f"{species.capitalize():<10}: {probabilities[0][i]:.2%}")

print(f"\nPredicted Class   : {prediction[0]}")
print(f"Predicted Species : {species_name}")

confidence = max(probabilities[0]) * 100
print(f"Model Confidence  : {confidence:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=iris.target_names)
disp.plot()

plt.title("Random Forest Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test,y_pred,target_names=iris.target_names))

# Feature Importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(by="Importance",ascending=False)

print("\nFeature Importance:")
print(importance_df)

# Feature Importance Graph
plt.figure(figsize=(7,4))
plt.bar(importance_df["Feature"],importance_df["Importance"])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.show()