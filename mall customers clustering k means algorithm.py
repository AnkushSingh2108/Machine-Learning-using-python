import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Mall_Customers.csv")

print(df)
print(df['Genre'].value_counts())

x= df.iloc[:,3:].values
print(x)

# when we start with the Kmeans we always do the elbow method first to show the correct number of clusters we can form

from sklearn.cluster import KMeans
wcss=[] # creating an empty list to store the wcss values for different k
for i in range(1,6):
    elbow = KMeans(n_clusters=i, init='k-means++',random_state=12)
    elbow.fit(x)
    wcss.append(elbow.inertia_)  # inertia attribute of the kmeans calculates the wcss values
x1 = range(1,6)
plt.scatter(x1,wcss)
plt.plot(x1,wcss)
plt.title('Elbow Plot')
plt.xlabel('K cluster')
plt.ylabel('WCSS')
plt.show()


# training the model & predicting the o/p
cluster = KMeans(n_clusters=i, init='k-means++',random_state=42)
cluster.fit(x)
y_pred = cluster.predict(x)
print('ypred',y_pred)

output = np.append(arr = x, values = y_pred.reshape(200,1), axis = 1)
print(output)

# visualization of the clusters
plt.scatter(x[y_pred==0,0],x[y_pred==0,1],c ='red', s=80,label = 'cluster 1')
plt.scatter(x[y_pred==1,0],x[y_pred==1,1],c ='green', s=80,label = 'cluster 2')
plt.scatter(x[y_pred==2,0],x[y_pred==2,1],c ='yellow', s=80,label = 'cluster 3')
plt.scatter(x[y_pred==3,0],x[y_pred==3,1],c ='black', s=80,label = 'cluster 4')
plt.scatter(x[y_pred==4,0],x[y_pred==4,1],c ='violet', s=80,label = 'cluster 5')
plt.scatter(cluster.cluster_centers_[:,0],cluster.cluster_centers_[:,1],c='violet',s=500,label='Centroid')
plt.xlabel("Annual Income")
plt.ylabel("Spending Scores")
plt.legend()
plt.show()