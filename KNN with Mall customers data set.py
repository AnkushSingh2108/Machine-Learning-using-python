import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Mall_Customers.csv")
print(df)

# extracting x&y
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

# splitting the x &ytrainning & test data
from sklearn.model_selection import train_test_split
x_tr,y_tr,y_te,x_te = train_test_split(x,y,test_size =0.2,random_state = 0)

# creating & traning the KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_tr,y_tr)

# predicting the output & printing it
y_pred = classifier.predict(x_te)
z = np.append(arr=y_pred.reshape(80,1),values=y_te.reshape(80,1))
print(z)