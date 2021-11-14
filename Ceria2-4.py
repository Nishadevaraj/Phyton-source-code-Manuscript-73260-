#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LinearRegression #interface of LR
import pandas as pd #importfile


# In[2]:


fl = pd.read_csv("CM234.csv")


# In[3]:


fl


# In[4]:


train=fl.head(3)
test= fl.iloc[3:4]
train2=fl.iloc[4:7]
test2=fl.iloc[7:8]
train3=fl.iloc[8:11]
test3= fl.iloc[11:12]
train4=fl.iloc[12:15]
test4= fl.iloc[15:16]
train5=fl.iloc[16:19]
test5= fl.iloc[19:20]
train6=fl.iloc[20:23]
test6= fl.iloc[23:24]
train7=fl.iloc[24:27]
test7= fl.iloc[27:28]
train8=fl.iloc[28:31]
test8= fl.iloc[31:32]
train9=fl.iloc[32:35]
test9= fl.tail(1)


# In[5]:


train.columns


# In[6]:


train_x, train_y = train[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],train[['logqe']]
train2_x, train2_y = train2[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],train2[['logqe']]
train3_x, train3_y = train3[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],train3[['logqe']]
train4_x, train4_y = train4[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],train4[['logqe']]
train5_x, train5_y = train5[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],train5[['logqe']]
train6_x, train6_y = train6[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],train6[['logqe']]
train7_x, train7_y = train7[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],train7[['logqe']]
train8_x, train8_y = train8[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],train8[['logqe']]
train9_x, train9_y = train9[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],train9[['logqe']]


# In[7]:


test_x,test_y = test[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],test[['logqe']]
test2_x,test2_y = test2[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],test2[['logqe']]
test3_x,test3_y = test3[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],test3[['logqe']]
test4_x,test4_y = test4[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],test4[['logqe']]
test5_x,test5_y = test5[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],test5[['logqe']]
test6_x,test6_y = test6[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],test6[['logqe']]
test7_x,test7_y = test7[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],test7[['logqe']]
test8_x,test8_y = test8[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],test8[['logqe']]
test9_x,test9_y = test9[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],test9[['logqe']]


# In[8]:


model = LinearRegression(fit_intercept = True)


# In[9]:


model.fit(train_x,train_y)
model.fit(train2_x,train2_y)
model.fit(train3_x,train3_y)
model.fit(train4_x,train4_y)
model.fit(train5_x,train5_y)
model.fit(train6_x,train6_y)
model.fit(train7_x,train7_y)
model.fit(train8_x,train8_y)
model.fit(train9_x,train9_y)


# In[10]:


pred = model.predict(test_x)
pred2 = model.predict(test2_x)
pred3 = model.predict(test3_x)
pred4 = model.predict(test4_x)
pred5 = model.predict(test5_x)
pred6 = model.predict(test6_x)
pred7 = model.predict(test7_x)
pred8 = model.predict(test8_x)
pred9 = model.predict(test9_x)


# In[11]:


E=(test_y-pred)
E2=(test2_y-pred2)
E3=(test3_y-pred3)
E4=(test4_y-pred4)
E5=(test5_y-pred5)
E6=(test6_y-pred6)
E7=(test7_y-pred7)
E8=(test8_y-pred8)
E9=(test9_y-pred9)


# In[12]:


AE=E.abs()
AE2=E2.abs()
AE3=E3.abs()
AE4=E4.abs()
AE5=E5.abs()
AE6=E6.abs()
AE7=E7.abs()
AE8=E8.abs()
AE9=E9.abs()


# In[13]:


pe=AE/test_y*100
pe2=AE2/test2_y*100
pe3=AE3/test3_y*100
pe4=AE4/test4_y*100
pe5=AE5/test5_y*100
pe6=AE6/test6_y*100
pe7=AE7/test7_y*100
pe8=AE8/test8_y*100
pe9=AE9/test9_y*100


# In[14]:


pred,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9


# In[15]:


train_x, train_y = train[[ 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],train[['logqe']]


# In[16]:


test_x,test_y = test[[ 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],test[['logqe']]


# In[17]:


model.fit(train_x,train_y)


# In[18]:


pred = model.predict(test_x)


# In[19]:


pred


# In[20]:


pred2


# In[21]:


E=(test_y-pred)


# In[22]:


AE=E.abs()


# In[23]:


pe=AE/test_y*100


# In[24]:


pred,E,AE,pe


# In[25]:


train2_x, train2_y = train2[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],train2[['logqe']]


# In[26]:


test2_x,test2_y = test2[['logqe', 'logce', '1/n', 'kf', 'ss', 'mass', 'temp', 'time']],test2[['logqe']]


# In[27]:


model.fit(train2_x,train2_y)


# In[28]:


pred2 = model.predict(test2_x)


# In[29]:


E2=(test2_y-pred2)


# In[30]:


AE2=E2.abs()


# In[31]:


pe2=AE2/test2_y*100


# In[32]:


pred2,E2,AE2,pe2


# In[33]:


model.fit(train3_x,train3_y)


# In[34]:


pred3 = model.predict(test3_x)


# In[35]:


E3=(test3_y-pred3)


# In[36]:


AE3=E3.abs()


# In[37]:


pe3=AE3/test3_y*100


# In[38]:


pred3,E3,AE3,pe3


# In[39]:


model.fit(train4_x,train4_y)


# In[40]:


pred4 = model.predict(test4_x)
E4=(test4_y-pred4)
AE4=E4.abs()
pe4=AE4/test4_y*100


# In[41]:


pred4,E4,AE4,pe4


# In[42]:


model.fit(train5_x,train5_y)


# In[43]:


pred5 = model.predict(test5_x)
E5=(test5_y-pred5)
AE5=E5.abs()
pe5=AE5/test5_y*100
pred5,E5,AE5,pe5


# In[44]:


model.fit(train6_x,train6_y)


# In[45]:


pred6 = model.predict(test6_x)
E6=(test6_y-pred6)
AE6=E6.abs()
pe6=AE6/test6_y*100
pred6,E6,AE6,pe6


# In[46]:


model.fit(train7_x,train7_y)


# In[47]:


pred7 = model.predict(test7_x)
E7=(test7_y-pred7)
AE7=E7.abs()
pe7=AE7/test7_y*100
pred7,E7,AE7,pe7


# In[48]:


model.fit(train8_x,train8_y)


# In[49]:


pred8 = model.predict(test8_x)
E8=(test8_y-pred8)
AE8=E8.abs()
pe8=AE8/test8_y*100
pred8,E8,AE8,pe8


# In[50]:


model.fit(train9_x,train9_y)


# In[51]:


pred9 = model.predict(test9_x)
E9=(test9_y-pred9)
AE9=E9.abs()
pe9=AE9/test9_y*100
pred9,E9,AE9,pe9


# In[52]:


model.fit(train_x,train_y)
model.fit(train2_x,train2_y)
model.fit(train3_x,train3_y)
model.fit(train4_x,train4_y)
model.fit(train5_x,train5_y)
model.fit(train6_x,train6_y)
model.fit(train7_x,train7_y)
model.fit(train8_x,train8_y)
model.fit(train9_x,train9_y)


# In[53]:


pred = model.predict(test_x)
pred2 = model.predict(test2_x)
pred3 = model.predict(test3_x)
pred4 = model.predict(test4_x)
pred5 = model.predict(test5_x)
pred6 = model.predict(test6_x)
pred7 = model.predict(test7_x)
pred8 = model.predict(test8_x)
pred9 = model.predict(test9_x)


# In[ ]:




