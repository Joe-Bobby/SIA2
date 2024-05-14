import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

class RegresionLogistica:
    def __init__(self, archivo, X):
        self.data = pd.read_csv(archivo)
        self.X = self.data.iloc[:, X].values
        self.y = self.data.iloc[:, 11].values

    def entrenar_modelo(self, test_size=0.2, random_state=42, degree=3):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        self.poly = PolynomialFeatures(degree=degree)
        X_train_poly = self.poly.fit_transform(X_train)
        self.model = LogisticRegression()
        self.model.fit(X_train_poly, y_train)
        
    def dibujar_clasificacion(self):
        fig, ax = plt.subplots()
        ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='bwr', edgecolors='k', s=20)
        ax.set_xlabel('Característica 1')
        ax.set_ylabel('Característica 2')
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        Z = self.model.predict(self.poly.transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
        return fig
    




# class RegresionLogistica:
#     def __init__(self, archivo, X):
#         self.data = pd.read_csv(archivo)
#         self.X = self.data.iloc[:, X].values
#         self.y = self.data.iloc[:, -1].values

#     def entrenar_modelo(self, test_size=0.2, random_state=42, degree=3):
#         X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
#         self.poly = PolynomialFeatures(degree=degree)
#         X_train_poly = self.poly.fit_transform(X_train)
#         self.model = LogisticRegression()
#         self.model.fit(X_train_poly, y_train)
        
#     def dibujar_clasificacion(self):
#         fig, ax = plt.subplots()
#         ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='bwr', edgecolors='k', s=20)
#         ax.set_xlabel('Característica 1')
#         ax.set_ylabel('Característica 2')
#         x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
#         y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
#         xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                              np.arange(y_min, y_max, 0.1))
#         Z = self.model.predict(self.poly.transform(np.c_[xx.ravel(), yy.ravel()]))
#         Z = Z.reshape(xx.shape)
#         ax.contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
#         return fig