import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('50_Startups.csv')
print('Dataframe',df)
print('Dataframe shape',df.shape)
print('df info',df.info)
x = df.iloc[:,:-1]
y = df.iloc[:-1]
print('x',x)
print('--------------------------------------')
print('y',y)

print('--------------------------------------')

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
print('x values',x)
print('--------------------------------------')
print('y values',y)

print('--------------------------------------')

# encoding the categorical column

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder = 'passthrough')
x = ct.fit_transform(x)
print(x)

from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size = 0.2, random_state = 0)
# creating & traininng the regression model

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_tr,y_tr)
y_pred = reg.predict(x_te)


from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_te,y_pred)
mse = mean_squared_error(y_te,y_pred)
rmse = np.sqrt(mse)

# VISUALIZATION
# since we have multiple column in the column index while plotting
plt.scatter(x_te[:,3],y_te,c ='red',label = 'original y')
plt.scatter(x_te[:,3],y_pred,c ='blue', label = 'calculated y')
plt.xlabel('Rnd')
plt.ylabel('profit')
plt.title('rnd v/s profit')
plt.legend()
plt.show()