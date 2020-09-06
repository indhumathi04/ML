#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_digits
dataset=load_digits()


# In[3]:


print('Image data shape',dataset.data.shape)
print('Target data shape',dataset.target.shape)


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
for index,(image,label) in enumerate(zip(dataset.data[0:5],dataset.target[0:5])):
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title('instance: %i\n'%label,fontsize=20)


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(dataset.data,dataset.target,test_size=0.2,random_state=0)


# In[7]:


from sklearn.linear_model import LogisticRegression


# In[8]:


model=LogisticRegression(max_iter=200)


# In[9]:


model.fit(x_train,y_train)


# In[10]:


model.predict(x_test[0].reshape(1,-1))


# In[11]:


print(model.predict(x_test[0:10]))


# In[15]:


acc=model.score(x_test,y_test)
print(acc)
pred=model.predict(x_test)


# In[16]:


import seaborn as sns
from sklearn import metrics
cm=metrics.confusion_matrix(y_test,pred)


# In[17]:


cm


# In[30]:


plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True,fmt="0.2f",linewidth=0.1,cmap='Blues_r')
plt.xlabel('Actual label')
plt.ylabel('Predicted label')
all_samples_title='Accuracy score:{0}'.format(acc)
plt.title(all_samples_title,size=15)


# In[ ]:




