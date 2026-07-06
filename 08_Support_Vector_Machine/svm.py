import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load Dataset
iris = load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)

df["Species"] = iris.target
df["Species"] = df["Species"].map(
    {
        0: iris.target_names[0],
        1: iris.target_names[1],
        2: iris.target_names[2]
    }
)

print("\nFirst Five Records\n")
print(df.head())

# Basic Information
print("\nDataset Information\n")
df.info()

print("\nMissing Values\n")
print(df.isnull().sum())

print("\nStatistical Summary\n")
print(df.describe())

# Exploratory Data Analysis
print("\nClass Distribution\n")
print(df["Species"].value_counts())

# Histogram
df.hist(figsize=(10,8))
plt.suptitle("Feature Distributions")
plt.show()

# Pair Plot
sns.pairplot(df, hue="Species")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.drop("Species", axis=1).corr(),annot=True,cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()

# Prepare Data
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Pipeline
pipeline = Pipeline([('scaler', StandardScaler()),('svm', SVC(probability=True))])

# Grid Search
param_grid = {
    'svm__C':[0.1,1,10,100],
    'svm__gamma':[1,0.1,0.01,0.001],
    'svm__kernel':['linear','rbf','poly']
}

grid = GridSearchCV(pipeline,param_grid,cv=5,scoring='accuracy',n_jobs=-1)
grid.fit(X_train,y_train)

print("\nBest Parameters")
print(grid.best_params_)

print("\nBest Cross Validation Score : {:.2f}%".format(grid.best_score_ * 100))

best_model = grid.best_estimator_

print("\nBest Model")
print(best_model)

# Cross Validation
scores = cross_val_score(best_model,X,y,cv=5)

print("\nCross Validation Scores")
print(scores)

print("Average Cross Validation Accuracy : {:.2f}%".format(scores.mean() * 100))
    
# Prediction
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

print("\nTest Accuracy : {:.2f}%".format(accuracy * 100))

# Confusion Matrix
cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=iris.target_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report\n")
print(classification_report(y_test,y_pred,target_names=iris.target_names))

# User Prediction
print("\nEnter Flower Details")

sl = float(input("Sepal Length : "))
sw = float(input("Sepal Width  : "))
pl = float(input("Petal Length : "))
pw = float(input("Petal Width  : "))

sample = [[sl,sw,pl,pw]]

prediction = best_model.predict(sample)
probability = best_model.predict_proba(sample)

print("\nPredicted Class :",iris.target_names[prediction[0]])
print("\nPrediction Probabilities")

for i,name in enumerate(iris.target_names):
    print(f"{name} : {probability[0][i]*100:.2f}%")

confidence = np.max(probability)*100
print("\nModel Confidence : {:.2f}%".format(confidence))