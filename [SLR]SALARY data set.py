import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('SALARY DATASET.csv')
print("Our dataframe is",df)
print('---------------------------')
print("Dataset shape",df.shape)
print('---------------------------')
print("Dataframe information",df.info)
print('---------------------------')
x = df.iloc[:,:-1].values
y = df.iloc[:,1].values
print("x value after operating it with iloc",x)
print('---------------------------')
print("y value after operating it with iloc",y)
print('---------------------------')
# Since there is no Categorical data, missing values & only 1 column in x we are going to skip rest of the Data Preprocessinng
# steps except the splitting of training & test set bcoz it helps in supervised learning
# Splitting the x & y into the training & test set

from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size=0.25,random_state = 100)

# Importing & Training the Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
# we use the fit() to train the models
lr.fit(x_tr,y_tr)

# to test the model we use the x_test to get the input to get the calculated output & then compare it with the original output which is y_test to see how well the model is working

y_pred  = lr.predict(x_te)
print("Predicted y is",y_pred)
print('---------------------------')

from sklearn.metrics import mean_absolute_error,mean_squared_error
mae = mean_absolute_error(y_te,y_pred)
mse = mean_squared_error(y_te,y_pred)
rmse = np.sqrt(mse)
print(mae)
print(mse)
print(rmse)

# visualising the output
# we use the plot() to polt the points and then we plot the function
# to plot the regression for the plotting we use the matplolib library

plt.scatter(x_te,y_te,c = 'black')
plt.plot(x_te,y_pred,c = 'magenta')
plt.title("Yrs v/s Salary plot")
plt.xlabel('Years of EXP')
plt.ylabel("Salary")
plt.show()