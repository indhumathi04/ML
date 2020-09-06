#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[ ]:


c=pd.read_csv('credit_card_clients.csv')


# In[ ]:


x=c.iloc[1:30001,12:24].values


# In[4]:


wcss=[]#within cluster sum of squares
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='random',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
print(wcss)


# In[5]:


plt.plot(range(1,11),wcss)
plt.title('elbow method')
plt.xlabel('num of clusters')
plt.ylabel('wcss')
plt.show()


# In[ ]:


kmeans=KMeans(n_clusters=4,init='random',max_iter=200,n_init=10,random_state=0)
pred_y=kmeans.fit_predict(x)


# In[ ]:


plt.scatter(x[:,0],x[:,1])
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='red')


# In[ ]:




