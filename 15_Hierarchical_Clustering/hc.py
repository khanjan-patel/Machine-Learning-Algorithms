import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

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

# Dendrogram
plt.figure(figsize=(10,6))
linkage_matrix = linkage(X_scaled, method="ward",optimal_ordering=True)

dendrogram(linkage_matrix)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.show()

# Find Best Model Using Silhouette Score
best_score = -1
best_k = None
best_linkage = None

for linkage_method in ["ward", "complete", "average", "single"]:
    for k in range(2, 11):
        if linkage_method == "ward":
            hc = AgglomerativeClustering(n_clusters=k,metric="euclidean",linkage="ward")

        else:
            hc = AgglomerativeClustering(n_clusters=k,metric="euclidean",linkage=linkage_method)

        labels = hc.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print()
        print(
            f"Linkage: {linkage_method:9} | "
            f"Clusters: {k} | "
            f"Silhouette Score: {score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_k = k
            best_linkage = linkage_method

print("\nBest Linkage:", best_linkage)
print("Best Number of Clusters:", best_k)
print("Best Silhouette Score:", best_score)

# Final Hierarchical Clustering Model
hc = AgglomerativeClustering(n_clusters=best_k,metric="euclidean",linkage=best_linkage)
clusters = hc.fit_predict(X_scaled)

# Add Cluster Labels
df_cluster = X.copy()
df_cluster["Cluster"] = clusters

print("\nFirst 10 Cluster Assignments")
print(df_cluster.head(10))

# Cluster Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_cluster,x="sepal length (cm)",y="petal length (cm)",hue="Cluster",palette="Set1",s=100)
plt.title("Agglomerative Hierarchical Clustering")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

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