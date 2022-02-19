import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv("POSITION_LEVEL_SALARY DATASET.csv")
print(df)

x = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values
print(x)
print(y)

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state=0)
reg.fit(x,y)

# Since we donot have any test set,what do we predict or how do we know that our algorithm is working or not 
# to solve this problem we are going to create pur own data set to get the answer 
# when we create our own data always keep in mind that it should the same sequence as x 
# meaning if x has one column then your data should also have one column if x has 2 ,your data should also have 2 
# the no. of rows does not matter bcoz  the test set is always smalller than the training set in case of the no. of rows 
# & your data set should always be nD array not 1D array

k = np.array([[7.5],[6.4],[11.0]])
print(k)



y_pred = reg.predict(k)
print(y_pred)
# our algorithm is not giving a good enough answer which is creating  a very huge problem
# the soln is either we increase the size of the data set or change the algorithm which is 
# better for the smaller dataset compared to this dataset


# for visualising the data

plt.scatter(x,y,c='black')
plt.plot(x,reg.predict(x),c='red')
plt.show()