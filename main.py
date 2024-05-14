import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from regresion_logistica import RegresionLogistica

    
def regresion_logistica():
    # Crear instancias de la clase RegresionLogistica
    visualizar1 = RegresionLogistica("zoo.csv", [8,13])
    visualizar1.entrenar_modelo()

    # Establecer un tamaño fijo para las figuras
    figura1 = visualizar1.dibujar_clasificacion()

    figura1.set_size_inches(5, 5)

    # Convertir las figuras en objetos de lienzo de Tkinter
    canvas1 = FigureCanvasTkAgg(figura1, master=root)


    # Mostrar los lienzos
    canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)



# Crear la ventana de Tkinter
root = tk.Tk()
root.title("Visualización de Regresión Logística")

# Crear y colocar el botón para mostrar los gráficos
btn_mostrar = tk.Button(root, text="Regresion Logistica", command=regresion_logistica)
btn_mostrar.pack()



# Ejecutar el bucle principal de Tkinter
root.mainloop()