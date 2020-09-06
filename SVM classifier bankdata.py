#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


bankdata = pd.read_csv("bill_authentication.csv")
bankdata.shape
bankdata.head()


# In[ ]:


X = bankdata.drop('Class', axis=1)
y = bankdata['Class']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[ ]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)


# In[ ]:


y_pred = svclassifier.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

