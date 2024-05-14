import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Cargar datos desde el archivo CSV
data = pd.read_csv('zoo.csv')

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop('animal_name', axis=1)  # Ajusta 'etiqueta' al nombre real de la columna de tus etiquetas
y = data['animal_name']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el clasificador KNN
k = 5  # Número de vecinos
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Entrenar el clasificador KNN
knn_classifier.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = knn_classifier.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo KNN:", accuracy)

# Visualización de los resultados
plt.figure(figsize=(10, 6))

# Graficar las características del conjunto de prueba
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap='viridis', marker='o', label='Predicción')

# Marcar los puntos de prueba reales
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap='viridis', marker='x', label='Real')

plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Resultados de la predicción vs. Realidad')
plt.legend()
plt.colorbar(label='Clase')
plt.show()
