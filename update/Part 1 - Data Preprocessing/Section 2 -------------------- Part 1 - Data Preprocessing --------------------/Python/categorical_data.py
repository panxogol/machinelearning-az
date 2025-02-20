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

