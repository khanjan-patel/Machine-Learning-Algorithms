import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
plt.xticks([0,1], data.target_names)
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

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA()

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Explained Variance Ratio
print("\nExplained Variance Ratio")
print(pca.explained_variance_ratio_)

# Cumulative Explained Variance
cumulative_variance = pca.explained_variance_ratio_.cumsum()

print("\nCumulative Explained Variance")
print(cumulative_variance)

# Scree Plot
plt.figure(figsize=(8,5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),pca.explained_variance_ratio_,marker="o")
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)
plt.tight_layout()
plt.show()

# Cumulative Variance Plot
plt.figure(figsize=(8,5))
plt.plot(range(1, len(cumulative_variance) + 1),cumulative_variance,marker="o")
plt.title("Cumulative Explained Variance")
plt.xlabel("Principal Component")
plt.ylabel("Cumulative Variance")
plt.grid(True)
plt.tight_layout()
plt.show()

# PCA with 2 Components for Visualization
pca_2d = PCA(n_components=2)

X_pca = pca_2d.fit_transform(StandardScaler().fit_transform(X))

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Target"] = y.map({0: "Malignant", 1: "Benign"})

# PCA Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df,x="PC1",y="PC2",hue="Target")
plt.title("PCA - Breast Cancer Dataset")
plt.tight_layout()
plt.show()

# First 10 Transformed Samples
print("\nFirst 10 PCA Transformed Samples")
print(pca_df.head(10))