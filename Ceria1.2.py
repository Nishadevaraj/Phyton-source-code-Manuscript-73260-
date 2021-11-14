#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LinearRegression #interface of LR
import pandas as pd #importfile


# In[2]:


fl = pd.read_csv("CM1.2.csv")


# In[3]:


fl


# In[4]:


train=fl.head(3)
test= fl.iloc[3:4]
train2=fl.iloc[4:7]
test2=fl.tail(1)


# In[5]:


train


# In[6]:


train2


# In[7]:


train.columns


# In[8]:


train_x, train_y = train[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],train[['logqe']]


# In[9]:


test_x,test_y = test[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],test[['logqe']]


# In[10]:


model = LinearRegression(fit_intercept = True)


# In[11]:


model.fit(train_x,train_y)


# In[12]:


pred = model.predict(test_x)


# In[13]:


pred


# In[14]:


E=test_y-pred


# In[15]:


AE=E.abs()


# In[16]:


import math
qe = math.exp(pred)


# In[17]:


pe=AE/test_y*100


# In[18]:


pe


# In[19]:


AE


# In[20]:


train2_x, train2_y = train2[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],train2[['logqe']]


# In[21]:


test2_x,test2_y = test2[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],test2[['logqe']]


# In[22]:


model.fit(train2_x,train2_y)


# In[23]:


pred2 = model.predict(test2_x)


# In[24]:


pred2


# In[25]:


E2=test2_y-pred2


# In[26]:


AE2=E2.abs()


# In[27]:


pe2=AE2/test2_y*100


# In[28]:


pe2


# In[29]:


AE2


# In[30]:


pred,pred2


# In[31]:


E/0.737419


# In[32]:


AE


# In[33]:


E


# In[ ]:





# In[ ]:




