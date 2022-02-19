# We import these libraries in every ML algorithm no matter for what we are using the algorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the Dataset
# we import the dataset as a  dataframe and for this we use the pandas librarfk
df = pd.read_csv('Data.csv')
print(df)
print('---------------------------------------------------------')
# where we have to identify or classify(we are not claculating anything) comes into the classification problem stmnt of the supervised learning

#EDA
print(df.shape)
print('---------------------------------------------------------')

print(df.info())
print('---------------------------------------------------------')

# to directly know the number of missing value in the dataframe we can use the below command 
#df.isna.sum()

# to find the outliers we can compare the mean and median of the numeric columns
print('Mean of Age',df['Age'].mean())
print('---------------------------------------------------------')

print('Median of Age',df['Age'].median())
print('---------------------------------------------------------')


print('Mean of Salary',df['Salary'].mean())
print('---------------------------------------------------------')

print('Median of Salary',df['Salary'].median())
print('---------------------------------------------------------')


# identifying outliers in the categorical columns
print('Value counts of Purchased',df['Purchased'].value_counts())
print('---------------------------------------------------------')

print('Value counts of Country',df['Country'].value_counts())
print('---------------------------------------------------------')

# DATA PREPROCESSING
# if we have a supervised learning problem stmnt we always start the data preprocessing by extracting x & y from our 
# dataset and store them as the numpy arrays
# the iloc function helps us to acces the dataframe columns & rowsthough there index number 
# we write iloc[row index range, column index range] 
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
print('x values after iloc applied with values',x)
print('---------------------------------------------------------')

print('y values after iloc applied with values',y)
print('---------------------------------------------------------')

# once we are done with the extracting we take care of the missing values 
# we have 2 methods to take care of the missing values
# 1. Remove the complete row containing the missing values ........ (this method is not prefered bcoz of the loss of info)
# 2. Impute(replace) the missing value with the mean ,median or mode values of their respective columns.
# for this we use the SimpleImputer class of the impute module from the sklearn library

from sklearn.impute import SimpleImputer
# after importing the class we  need to create an object of this class with the arguments we need

impt = SimpleImputer(missing_values = np.nan, strategy ='mean')
x[:,1:] = impt.fit_transform(x[:,1:])
print('By mean',x)
print('---------------------------------------------------------')


impt = SimpleImputer(missing_values = np.nan, strategy ='median')
x[:,1:] = impt.fit_transform(x[:,1:])
print('By median',x)
print('---------------------------------------------------------')


impt = SimpleImputer(missing_values = np.nan, strategy ='most_frequent')
x[:,1:] = impt.fit_transform(x[:,1:])
print('By most_frequent',x)
print('---------------------------------------------------------')


impt = SimpleImputer(missing_values = np.nan, strategy ='constant')
x[:,1:] = impt.fit_transform(x[:,1:])
print('By constant',x)
print('---------------------------------------------------------')

# since our dataset is having "Y" column means it is a supervised learning example & if it is supervised learning example 
# we have to do traning testing and splitting


# ENCODING THE CATEGORICAL DATA
# The ML algos can't work with the string/categorical values. 
# To solve this problem we have to encode these categorical value into the numeric values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])],remainder = 'passthrough')
# just for precaution we use np.array to convert the output of the column transformerin a numpy array
x = np.array(ct.fit_transform(x))
print(x)
print('---------------------------------------------------------')

# since in ths dataset the y column is also having the categorical data we also have to encode the y column 
# as y can't have more than one column theerefore we never use the above method on the y column

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
print('---------------------------------------------------------')

# FEATURE SCALING
# In Some datasets the difference between the values of the columns(LEAVING DUMMY VARIABLES) of x might be very huge 
# for example in this particaular dataset there is a huge scale difference between the age column and the Salary column
# here the Age column has 2 digit values where as the salary column values are 5 digit values now to stop the salary column 
# from dominating we have to bring themm one the same scale . This process is known as FEATURE SCALING.

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
# the output of the standard scaler class can be in the scientific notation
print(x)
print('---------------------------------------------------------')

# TRAINING TESTING & SPLITTING
# (splitting the x & y into the training and test set (only happens in the case of SUPERVISED LEARNING)
# for the splitting we use the train_test_split function which splits the x & y into the defined ratio by shuffling the 'ROWS' (COMPLETE ROWS ARE BEING SHUFFLED) the default ratio is 75:25 as train:test the train test split function shuffles iin the random manner .this shuffling will be different for all the PCs. to prevent the randome shuffling of the train_test_split function by provinding any positive integer number to the random state argument we can fix the shuffling the shuffling will be for every system(PC)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size = 0.2, random_state = 0)
# this train_test_split function gives 4 outputs ie x_train,y_train,x_test,y_test 
# x_train & y_train will contain the ROWS of the TRAINING set x_test and y_test will contain the ROWS of the test set
print(x_train)
print(x_test)
