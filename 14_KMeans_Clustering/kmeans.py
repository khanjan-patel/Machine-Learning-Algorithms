import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load Dataset
data = load_iris()

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
plt.xticks([0, 1, 2], data.target_names)
plt.title("Target Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Pairplot
df_pair = X.copy()
df_pair["Target"] = y.map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

sns.pairplot(df_pair, hue="Target")
plt.show()

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.show()

# K-Means Model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

clusters = kmeans.fit_predict(X_scaled)

print("\nCluster Centers")
print(kmeans.cluster_centers_)

# Add Cluster Labels
df_cluster = X.copy()
df_cluster["Cluster"] = clusters

print("\nFirst 10 Cluster Assignments")
print(df_cluster.head(10))

# Sample Predictions
sample_predictions = pd.DataFrame({
    "Actual Class": y.map({
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }),
    "Predicted Cluster": clusters
})

print("\nSample Predictions")
print(sample_predictions.head(10))