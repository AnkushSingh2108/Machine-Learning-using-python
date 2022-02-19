import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Wine.csv")
print(df)

x = df.iloc[:,-1].values
y  = df.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
