#!/usr/bin/env python
# coding: utf-8

# from sklearn.linear_model import LinearRegression #interface of LR
# import pandas as pd #importfile

# In[2]:


fl = pd.read_csv("HA1_6.csv")


# In[3]:


fl


# In[4]:


train=fl.head(3)
test= fl.iloc[3:4]
train2=fl.iloc[4:7]
test2=fl.iloc[7:8]
train3=fl.iloc[8:11]
test3= fl.iloc[11:12]
train4=fl.iloc[12:14]
test4= fl.iloc[14:15]
train5=fl.iloc[15:18]
test5= fl.iloc[18:19]
train6=fl.iloc[19:22]
test6= fl.tail(1)


# In[6]:


train.columns


# In[7]:


train_x, train_y = train[['lnce', 'int', 'm(B)', 'Kt', 'ss', 'time']],train[['qe']]
train2_x, train2_y = train2[['lnce', 'int', 'm(B)', 'Kt', 'ss', 'time']],train2[['qe']]
train3_x, train3_y = train3[[ 'lnce', 'int', 'm(B)', 'Kt', 'ss', 'time']],train3[['qe']]
train4_x, train4_y = train4[[ 'lnce', 'int', 'm(B)', 'Kt', 'ss', 'time']],train4[['qe']]
train5_x, train5_y = train5[[ 'lnce', 'int', 'm(B)', 'Kt', 'ss', 'time']],train5[['qe']]
train6_x, train6_y = train6[[ 'lnce', 'int', 'm(B)', 'Kt', 'ss', 'time']],train6[['qe']]

test_x,test_y = test[[ 'lnce', 'int', 'm(B)', 'Kt', 'ss', 'time']],test[['qe']]
test2_x,test2_y = test2[[ 'lnce', 'int', 'm(B)', 'Kt', 'ss', 'time']],test2[['qe']]
test3_x,test3_y = test3[[ 'lnce', 'int', 'm(B)', 'Kt', 'ss', 'time']],test3[['qe']]
test4_x,test4_y = test4[['lnce', 'int', 'm(B)', 'Kt', 'ss', 'time']],test4[['qe']]
test5_x,test5_y = test5[[ 'lnce', 'int', 'm(B)', 'Kt', 'ss', 'time']],test5[['qe']]
test6_x,test6_y = test6[[ 'lnce', 'int', 'm(B)', 'Kt', 'ss', 'time']],test6[['qe']]


# In[8]:


model = LinearRegression(fit_intercept = True)


# In[9]:


model.fit(train_x,train_y)
pred = model.predict(test_x)
E=(test_y-pred)
AE=E.abs()
pe=AE/test_y*100
pred,E,AE,pe


# In[10]:


model.fit(train2_x,train2_y)
pred2 = model.predict(test2_x)
E2=(test2_y-pred2)
AE2=E2.abs()
pe2=AE2/test2_y*100
pred2,E2,AE2,pe2


# In[11]:


model.fit(train3_x,train3_y)
pred3 = model.predict(test3_x)
E3=(test3_y-pred3)
AE3=E3.abs()
pe3=AE3/test3_y*100
pred3,E3,AE3,pe3


# In[12]:


model.fit(train4_x,train4_y)
pred4 = model.predict(test4_x)
E4=(test4_y-pred4)
AE4=E4.abs()
pe4=AE4/test4_y*100
pred4,E4,AE4,pe4


# In[13]:


model.fit(train5_x,train5_y)
pred5 = model.predict(test5_x)
E5=(test5_y-pred5)
AE5=E5.abs()
pe5=AE5/test5_y*100
pred5,E5,AE5,pe5


# In[14]:


model.fit(train6_x,train6_y)
pred6 = model.predict(test6_x)
E6=(test6_y-pred6)
AE6=E6.abs()
pe6=AE6/test6_y*100
pred6,E6,AE6,pe6


# In[ ]:




