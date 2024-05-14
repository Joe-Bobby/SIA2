import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Paso 2: Cargar los datos desde el archivo CSV
data = pd.read_csv('zoo.csv')

# Paso 3: Preprocesar los datos si es necesario
# Por ejemplo, puedes convertir variables categóricas en variables dummy

# Paso 4: Dividir los datos en conjuntos de entrenamiento y prueba
X = data.drop('class_type', axis=1) # features
y = data['class_type'] # labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 5: Entrenar el modelo Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Paso 6: Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Paso 7: Visualizar los resultados
# Por ejemplo, puedes crear una matriz de confusión
confusion_matrix = metrics.plot_confusion_matrix(model, X_test, y_test)
plt.title('Matriz de Confusión')
plt.show()
