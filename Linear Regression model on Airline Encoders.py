
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Reading data

airline = pd.read_csv(r'C:\Users\francis\Downloads\encoders.csv')


# In[3]:


airline.head(5)


# In[4]:


never_failure = airline[airline.err < 12]


# In[5]:


routine_failure = airline[airline.err > 12]


# In[35]:


routine_failure


# In[6]:


airline = never_failure.append(routine_failure)


# In[7]:


airline.isnull().sum()


# In[8]:


airline = airline.drop(['created_at','updated_at'],axis=1)


# In[9]:


airline.corr().plot()


# In[39]:


airline.dtypes


# In[48]:


plt.figure(figsize=(10,7))
sns.heatmap(airline.corr(),annot=True,linewidths=2)


# In[10]:


dummies = pd.get_dummies(airline['scanner'])

airline = pd.concat([airline, dummies],axis=1)


# In[34]:


#drop the scanner column

final_data = airline.drop(['date','scanner','minf','maxf','errf'],axis = 1)


# In[36]:


final_data.head(5)


# In[38]:


y = final_data.iloc[:,4]


# In[37]:


X = final_data.drop(['err'], axis = 1)


# In[51]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[52]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train) #fit is used to put the data in equation of Stasndard scaler
standardized_X = scaler.transform(X_train)


# In[48]:


standardized_X


# In[53]:


#Model fit and training

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import linear_model


# In[54]:


lm = LinearRegression()


# In[55]:


#Fit the model on to the instantiated object itself

lm.fit(X_train,y_train)


# In[56]:


train_pred = lm.predict(X_train)


# In[57]:


metrics.mean_squared_error(y_train,train_pred)


# In[58]:


test_pred = lm.predict(X_test)


# In[59]:


metrics.mean_squared_error(y_test,test_pred)


# In[60]:


np.sqrt(metrics.mean_squared_error(y_train,train_pred))


# In[61]:


print("R-squared value of this fit:",round(metrics.r2_score(y_train,train_pred),3))


# In[62]:


#Prediction

regr = linear_model.LinearRegression().fit(X_train,y_train)


# In[63]:


regr.coef_

regr.intercept_


# In[64]:



#Prediction

predt = regr.predict(X_test)


# In[65]:


#Evaluation based on metric

regr.score(X_test,y_test)

from sklearn.metrics import mean_squared_error, r2_score

mean_squared_error(y_test,predt)

r2_score(y_test,predt)


# In[66]:


#Evaluation based on visualization

import matplotlib.pyplot as plt

plt.hist(y_test)
plt.hist(predt)


# In[67]:


plt.show()

plt.scatter(regr.predict(X_train),regr.predict( X_train)- y_train, c= 'b', s=40, alpha = 0.5)
plt.scatter(regr.predict(X_test), regr.predict(X_test)-y_test, c='g', s=40)
plt.hlines(y= 0, xmin = 0, xmax= 50)
plt.title('Residual Plot')


# In[68]:


#build the model for Errf

X = final_data1.drop(['errf'], axis = 1)


# In[18]:


y = final_data.iloc[:,4]


# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[20]:


#Model fit and training

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import linear_model


# In[21]:


#Prediction

regr = linear_model.LinearRegression().fit(X_train,y_train)


# In[22]:


regr.coef_

regr.intercept_


# In[23]:



#Prediction

predt = regr.predict(X_test)


# In[24]:


#Evaluation based on metric

regr.score(X_test,y_test)

from sklearn.metrics import mean_squared_error, r2_score

mean_squared_error(y_test,predt)

r2_score(y_test,predt)


# In[25]:


#Evaluation based on visualization

import matplotlib.pyplot as plt

plt.hist(y_test)
plt.hist(predt)


# In[26]:


plt.show()

plt.scatter(regr.predict(X_train),regr.predict( X_train)- y_train, c= 'b', s=40, alpha = 0.5)
plt.scatter(regr.predict(X_test), regr.predict(X_test)-y_test, c='g', s=40)
plt.hlines(y= 0, xmin = 0, xmax= 50)
plt.title('Residual Plot')

