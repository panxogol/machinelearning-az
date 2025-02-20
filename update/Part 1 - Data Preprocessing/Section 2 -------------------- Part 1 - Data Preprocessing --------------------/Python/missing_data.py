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