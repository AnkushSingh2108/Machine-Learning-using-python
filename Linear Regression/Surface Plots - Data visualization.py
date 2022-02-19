#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# a = np.array([1,2,3])
# b = np.array([4,5,6,7])

a = np.arange(-1,1,0.02)
b = a


a,b = np.meshgrid(a,b)
# print(a)
# print(b)


# In[8]:


fig = plt.figure()
axes = fig.gca(projection = '3d')
axes.plot_surface(a,b,a**2+b**2,cmap = 'rainbow')   # the parameters are : x_value,y_value,z_eqn
plt.title("Surface plot")
plt.show()


# In[9]:


fig = plt.figure()
axes = fig.gca(projection = '3d')
axes.contour(a,b,a**2+b**2,cmap = 'rainbow')   # the parameters are : x_value,y_value,z_eqn
plt.title("Contour Plot")
plt.show()


# In[ ]:





# In[ ]:




