#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes Clasifier - Mushroom Dataset
# - Goal is to predict the class of mushrooms, given some fearures of mushrooms. We will use Naive Bayes Model for this classification
# # Load the dataset

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("./mushrooms.csv")
df.head()


# In[3]:


df.columns, df.shape


# # Encode the categroical data into Numerical Data

# In[4]:


le = LabelEncoder()

# Applies transformation on each column
ds = df.apply(le.fit_transform)


# In[5]:


ds.head()


# In[6]:


type(ds)


# In[7]:


data = ds.values


# In[8]:


type(data)


# In[9]:


print(data.shape)
print(data[:5,:])


# # break the data into train and test data

# In[10]:


data_y = data[:,0]
data_x = data[:,1:]


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size= 0.2)


# In[12]:


print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[13]:


np.unique(y_train)


# # Building our CLasifier

# In[14]:


def prior_prob(y_train,label):
    total_examples = y_train.shape[0]
    class_examples = np.sum(y_train == label)
    
    return (class_examples)/float(total_examples)


# In[15]:


y = np.array([1,1,1,1,5,5,4,5,5,5])
print(len(y))
prior_prob(y,1)


# In[16]:


def cond_prob(x_train,y_train,feature_col,feature_val,label):
    x_filtered = x_train[y_train == label]
    numerator = np.sum(x_filtered[:,feature_col] == feature_val)
    denominator = np.sum(y_train == label)
    
    return numerator/float(denominator)


# # Next Step: Compute Posterior Probability for each test example and make predictions

# In[17]:


def predict(x_train,y_train, xtest):
    """xtest is a single testing point with n nuber of features"""
    classes = np.unique(y_train)
    n_features = x_train.shape[1]
    posterior_probs = [] # list of probabilities for all classes and given a singlee testing point
    
    # Compute posterior probabilities for each class
    
    for label in classes:
        # posterior_prob = likelihood*prior
        likelihood = 1.0
        for f in range(n_features):
            cond = cond_prob(x_train,y_train,f,xtest[f], label)
            likelihood *= cond
            
        prior = prior_prob(y_train,label)
        post = likelihood*prior
        
        posterior_probs.append(post)
        
    pred = np.argmax(posterior_probs)
    
    return pred


# In[18]:


# np.unique(y_train)


# In[19]:


output = predict(x_train,y_train, x_test[2])
print(output)
print(y_test[2])


# In[20]:


def score(x_train,y_train,x_test,y_test):
    pred = []
    
    for i in range(x_test.shape[0]):
        pred_label = predict(x_train, y_train , x_test[i])
        pred.append(pred_label)
        
        pred = np.array(pred)
        
        accuracy = np.sum(pred == y_test)/y_test.shape[0]  # this is our score
        
        return accuracy


# In[21]:


print(score(x_train,y_train,x_test,y_test))


# In[ ]:




