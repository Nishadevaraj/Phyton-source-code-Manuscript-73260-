#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LinearRegression #interface of LR
import pandas as pd #importfile


# In[2]:


fl = pd.read_csv("M1234.csv")


# In[3]:


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
test9= fl.iloc[35:36]
traina=fl.iloc[36:39]
testa= fl.iloc[39:40]
trainb=fl.iloc[40:43]
testb= fl.iloc[43:44]
trainc=fl.iloc[44:47]
testc= fl.iloc[47:48]


# In[4]:


fl


# In[5]:


train3.columns


# In[6]:


train_x, train_y = train[['logce', '1/n', 'kf', 'ss', 'temp', 'time']],train[['logqe']]
train2_x, train2_y = train2[['logce', '1/n', 'kf', 'ss',  'temp', 'time']],train2[['logqe']]
train3_x, train3_y = train3[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],train3[['logqe']]
train4_x, train4_y = train4[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],train4[['logqe']]
train5_x, train5_y = train5[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],train5[['logqe']]
train6_x, train6_y = train6[[ 'logce', '1/n', 'kf', 'ss', 'temp', 'time']],train6[['logqe']]
train7_x, train7_y = train7[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],train7[['logqe']]
train8_x, train8_y = train8[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],train8[['logqe']]
train9_x, train9_y = train9[[ 'logce', '1/n', 'kf', 'ss', 'temp', 'time']],train9[['logqe']]
traina_x, traina_y = traina[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],traina[['logqe']]
trainb_x, trainb_y = trainb[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],trainb[['logqe']]
trainc_x, trainc_y = trainc[[ 'logce', '1/n', 'kf', 'ss', 'temp', 'time']],trainc[['logqe']]


# In[7]:


test_x,test_y = test[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],test[['logqe']]
test2_x,test2_y = test2[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],test2[['logqe']]
test3_x,test3_y = test3[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],test3[['logqe']]
test4_x,test4_y = test4[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],test4[['logqe']]
test5_x,test5_y = test5[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],test5[['logqe']]
test6_x,test6_y = test6[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],test6[['logqe']]
test7_x,test7_y = test7[[ 'logce', '1/n', 'kf', 'ss', 'temp', 'time']],test7[['logqe']]
test8_x,test8_y = test8[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],test8[['logqe']]
test9_x,test9_y = test9[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],test9[['logqe']]
testa_x,testa_y = testa[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],testa[['logqe']]
testb_x,testb_y = testb[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],testb[['logqe']]
testc_x,testc_y = testc[[ 'logce', '1/n', 'kf', 'ss',  'temp', 'time']],testc[['logqe']]


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


# In[15]:


model.fit(train7_x,train7_y)
pred7 = model.predict(test7_x)
E7=(test7_y-pred7)
AE7=E7.abs()
pe7=AE7/test7_y*100
pred7,E7,AE7,pe7


# In[16]:


model.fit(train8_x,train8_y)
pred8 = model.predict(test8_x)
E8=(test8_y-pred8)
AE8=E8.abs()
pe8=AE8/test8_y*100
pred8,E8,AE8,pe8


# In[17]:


model.fit(train9_x,train9_y)
pred9 = model.predict(test9_x)
E9=(test9_y-pred9)
AE9=E9.abs()
pe9=AE9/test9_y*100
pred9,E9,AE9,pe9


# In[18]:


model.fit(traina_x,traina_y)
preda = model.predict(testa_x)
Ea=(testa_y-preda)
AEa=Ea.abs()
pea=AEa/testa_y*100
preda,Ea,AEa,pea


# In[19]:


model.fit(trainb_x,trainb_y)
predb = model.predict(testb_x)
Eb=(testb_y-predb)
AEb=Eb.abs()
peb=AEb/testb_y*100
predb,Eb,AEb,peb


# In[20]:


model.fit(trainc_x,trainc_y)
predc= model.predict(testc_x)
Ec=(testc_y-predc)
AEc=Ec.abs()
pec=AEc/testc_y*100
predc,Ec,AEc,pec


# In[ ]:





# In[ ]:




