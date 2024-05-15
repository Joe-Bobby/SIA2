import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("zoo.csv")

# Define your features and target variable
features = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'catsize', 'class_type']
target = 'domestic'

# Fill X with features and y with target variable
X = df[features].values
y = df[target].values

plt.figure(figsize=(10, 10))
plt.scatter(X[:,6], X[:,15], c=y, marker='.', s=100, edgecolors='black')
plt.xlabel('Predator')
plt.ylabel('Domestic')
plt.title('Plot of predator vs domestic')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train[:, [6, 15]], y_train)  # Using only 'predator' and 'domestic' features for training
y_pred_5 = knn5.predict(X_test[:, [6, 15]])  # Using only 'predator' and 'domestic' features for prediction

# Calculating metrics
accuracy = accuracy_score(y_test, y_pred_5)
precision = precision_score(y_test, y_pred_5)
recall = recall_score(y_test, y_pred_5)
specificity = recall_score(y_test, y_pred_5, pos_label=0)
f1 = f1_score(y_test, y_pred_5)

print("Accuracy with k=5:", accuracy * 100)
print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("Specificity:", specificity)
print("F1 Score:", f1)

# Create meshgrid using 'predator' and 'domestic' features
x_min, x_max = X[:, 6].min() - 1, X[:, 6].max() + 1
y_min, y_max = X[:, 15].min() - 1, X[:, 15].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

plt.figure(figsize=(15, 5))
Z = knn5.predict(np.c_[xx.ravel(), yy.ravel()])  # Predict using meshgrid points
Z = Z.reshape(xx.shape)
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_test[:, 6], X_test[:, 15], c=y_pred_5, marker='.', s=100, edgecolors='black')
plt.title("Number of neighbors k=5", fontsize=20)
plt.xlabel('Predator')
plt.ylabel('Class_type')
plt.show()
