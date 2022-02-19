import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Social_Network_Ads.csv")
print(df)
print("----------------------------------------")

# EDA
print(df.isna().sum())
print("----------------------------------------")
print(df["Purchased"].value_counts())
print("----------------------------------------")

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

# splitting x & y into training & test set 
from sklearn.model_selection import train_test_split
x_tr,y_tr,x_te,y_te = train_test_split(x,y,test_size = 0.3,random_state = 0)

# creating & training the model
from sklearn.svm import SVC
classifier = SVC(kernel="linear",random_state=0)
classifier.fit(x_tr,y_tr)

# predicting the output & then comparing it 
y_pred = classifier.predict(x_te)

z = np.append(arr=y_pred.reshape(80,1),values=y_te.reshape(80,1))
print(z)
