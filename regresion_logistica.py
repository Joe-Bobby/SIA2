import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class RegresionLogistica:
    def __init__(self, archivo, X):
        self.data = pd.read_csv(archivo)
        self.X = self.data.iloc[:, X].values
        self.y = self.data.iloc[:, 7].values

    def entrenar_modelo(self, test_size=0.2, random_state=42, degree=3):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        self.poly = PolynomialFeatures(degree=degree)
        X_train_poly = self.poly.fit_transform(X_train)
        self.model = LogisticRegression()
        self.model.fit(X_train_poly, y_train)
        accuracy, precision, recall, specificity, f1 = self.evaluar_modelo(X_test, y_test)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Sensitivity:", recall)
        print("Specificity:", specificity)
        print("F1 Score:", f1)

        
    def dibujar_clasificacion(self):
        fig, ax = plt.subplots()
        ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='bwr', edgecolors='k', s=20)
        ax.set_xlabel('Tienen dientes')
        ax.set_ylabel('Numero de piernas')
        x_min, x_max = self.X[:, 0].min() - 4, self.X[:, 0].max() + 4
        y_min, y_max = self.X[:, 1].min() - 4, self.X[:, 1].max() + 4
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        Z = self.model.predict(self.poly.transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
        return fig
    ...
    def evaluar_modelo(self, X_test, y_test):
        # Transforma las características de prueba con el mismo polinomio
        X_test_poly = self.poly.transform(X_test)
        
        # Realiza la predicción en el conjunto de prueba
        y_pred = self.model.predict(X_test_poly)
        
        # Calcula las métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calcula la matriz de confusión para obtener specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        
        # Devuelve las métricas calculadas
        return accuracy, precision, recall, specificity, f1
