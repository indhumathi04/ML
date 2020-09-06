#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[3]:


w=pd.read_csv('winequality.csv')
w


# In[4]:


w.describe()


# In[6]:


w.isnull()


# In[7]:


w=w.fillna(method='ffill')


# In[9]:


y=w['quality'].values
x=w[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','density','pH','sulphates','alcohol']].values


# In[10]:


plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(w['quality'])


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[12]:


model=LinearRegression()


# In[13]:


model.fit(x_train,y_train)


# In[18]:


x1=w[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','density','pH','sulphates','alcohol']]
coef_df=pd.DataFrame(model.coef_,x1.columns,columns=['coefficient'])


# In[19]:


coef_df


# In[20]:


y_pred=model.predict(x_test)


# In[21]:


df=pd.DataFrame({'actual':y_test,'pred':y_pred})


# In[22]:


df


# In[23]:


print(metrics.mean_absolute_error(y_test,y_pred))


# In[24]:


print(metrics.mean_squared_error(y_test,y_pred))


# In[25]:


print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[ ]:




