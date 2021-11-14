#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LinearRegression #interface of LR
import pandas as pd #importfile


# In[2]:


fl = pd.read_csv("CeriaF1.1.csv")


# In[3]:


fl


# In[4]:


train = fl.head(3)
test = fl.tail(1)


# In[5]:


train.columns


# In[6]:


train_x, train_y = train[['logce', '1/n', 'kf','SS','time']], train[['logqe']]


# In[7]:


train_x


# In[8]:


test_x,test_y = test[['logce','1/n','kf','SS','time']],test[['logqe']]


# In[9]:


test_x


# In[10]:


model = LinearRegression(fit_intercept = True)


# In[11]:


model.fit(train_x,train_y)


# In[12]:


pred = model.predict(test_x)


# In[13]:


pred


# In[14]:


pred-test_y


# In[15]:


E = pred-test_y


# In[16]:


AE = E.abs()


# In[17]:


AE


# In[18]:


import math
qe = math.exp(pred)


# In[19]:


qe


# In[20]:


Oqe = math.exp(fl.iloc[3]['logqe'])


# In[21]:


Oqe


# In[22]:


pe=((Oqe-qe)/Oqe)


# In[23]:


pe


# In[24]:


pe*100


# In[ ]:




