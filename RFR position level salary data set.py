import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv("POSITION_LEVEL_SALARY DATASET.csv")
print(df)

x = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values
print(x)
print(y)

# since the dataset is very small we are not breaking it into training & testing data
# and as we saw that the decision tree algorithm was not able to work with this dataset 
# we are using the random forest algorithm

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=  65,random_state=41)
reg.fit(x,y)

k = np.array([[7.5],[6.4],[5.0]])
print(k)

y_pred = reg.predict(k)
print(y_pred)


pred = []

k = np.array([[7],[6],[5]])
for i in range(1,55):
    reg = RandomForestRegressor(n_estimators= i ,random_state=0)
    reg.fit(x,y)
    y_pred =reg.predict(k)
    pred.append(list(y_pred))
print(pred)
print(y_pred)

# visualising the dataset

plt.scatter(x,y,c = 'brown')
plt.plot(x, reg.predict(x),c = 'magenta')
plt.show()