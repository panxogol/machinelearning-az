# -*- coding: utf-8 -*-

# Plantilla de Pre Procesado
# Cómo importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Tratamiento de NAs
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Codificar datos categóricos
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

columntransformer = ColumnTransformer(
    [("one_hot_encoder", OneHotEncoder(categories="auto"), [0])],
    remainder="passthrough"
    )

x = np.array(columntransformer.fit_transform(x), dtype=np.float_)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Dividir dataset en entrenamiento y testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=0)

# Escalado de variables
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

# En caso de no querer escalar los dummys, se pueden seleccionar 
# solo las columnas que se desean estandarizar.
x_train[:, 3:-1] = sc_x.fit_transform(x_train[:, 3:-1])
x_test[:, 3:-1] = sc_x.transform(x_test[:, 3:-1])
"""


