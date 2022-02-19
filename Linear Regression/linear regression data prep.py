import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Section 1. Load and Visualize the data

X = pd.read_csv("./Linear Regression/Training Data/Linear_X_Train.csv")
Y = pd.read_csv("./Linear Regression/Training Data/Linear_Y_Train.csv")

print(type(X) , type(Y))

# Convert X,Y to numpy arrays
x = X.values
y = Y.values

print(type(x) , type(y))

# Normalisation

u = x.mean()
std = x.std()

x = (x-u)/std
print(x)


# Visualisation 

plt.scatter(X,y)
plt.show()

#theta = [theta0,theta1]
def hypothesis(x,theta):
    y_ = theta[0] + theta[1]*x
    return y_


def gradient(X,Y,theta):
    m = X.shape[0]
    grad = np.zeros((2,))
    for i in range(m):
        y = Y[i]
        x = X[i]
        y_ = hypothesis(x,theta)
        grad[0] += (y_ - y)  # gradient wrt theta0
        grad[1] += (y_ - y)*x # gradient wrt theta1    
    return grad/m


def error(X,Y,theta):
    m = x.shape[0]
    total_error= 0.0 
    for i in range(m):
        y_ = hypothesis(X[i],theta)
        total_error += (y_ - Y[i])**2
    return total_error/m


def gradientDescent(x,y,max_steps = 100,learning_rate = 0.1):
    theta = np.zeros((2,))
    error_list = []
    
    for i in range(max_steps):
        # Compute Grad
        grad = gradient(X,Y,theta)
        e = error(x.y,theta)
        error_list.append(e)
        
        # Update Theta
        theta[0] = theta[0] - learning_rate*grad[0]
        theta[1] = theta[1] - learning_rate*grad[1]
        
    return theta,error_list


theta,errorlist = gradientDescent(X,y)

print(theta, errorlist)