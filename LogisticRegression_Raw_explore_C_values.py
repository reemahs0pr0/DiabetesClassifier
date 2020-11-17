#!/usr/bin/env python
# coding: utf-8

# # Importing and Reading Data

# In[1]:


### General libraries ###
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

### ML Models ###
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

##################################
### Metrics ###
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[2]:


#reading the data
df = pd.read_csv("diabetes.csv")

#get data info
df.info()


# In[3]:


#get shape of the data
df.shape


# # Visualise the Data

# In[4]:


sb.pairplot (df, hue='Outcome')
plt.show()


# # Train our Model

# In[4]:


#choose the columns for independent variables
X=df.iloc[:,0:8]
X.head() #validating X


# In[5]:


#choose column for dependent variable
y=df["Outcome"]
y.head()


# In[6]:


#split data to train and test in ratio 75% 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train.head()


# In[7]:


y_train.head()


# In[8]:


logReg = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter=1000,random_state = 42)


# In[9]:


import time

train_start_time = time.time()
# Train the classifier.
logReg.fit(X_train,y_train)

print("Duration of training: %s seconds" % (time.time() - train_start_time))


# # Evaluate our Model

# In[10]:


#Make Predictions
y_pred = logReg.predict(X_test)
print(y_pred) 
print(y_test)


# In[11]:


print(accuracy_score(y_test, y_pred))


# # Predict some Value

# In[12]:


pred_start_time = time.time()
print(logReg.predict([(2, 100, 70, 20, 100, 25, 0.5, 27)]))
print("Duration of prediction: %s seconds" % (time.time() - pred_start_time))


# The Outcome from model predicting data with Pregnancies = 2, Glucose = 100, BloodPressure = 70, SkinThickness = 20, Insulin = 100, BMI = 25, DiabetesPedigreeFunction = 0.5, Age = 27 is 0

# # Validation with Confusion Matrix

# In[13]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, cmap="Blues", fmt="d")


# In[14]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
#recall - #of true positivies rate
#precision # 


# # Exploring Different C Values

# In[27]:


C_array = np.arange(0.1, 1.1, 0.1)

C_array


# In[28]:


acc = []
for C in C_array:
    logReg_ex = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter=1000, random_state = 42, C = C)
    logReg_ex.fit(X_train,y_train)
    ac = accuracy_score(y_test, logReg_ex.predict(X_test))
    acc.append(ac)
    print("C = {0}".format(C))
    print("Accuracy = {0}".format(ac))


# In[30]:


fig = plt.figure()
ax = fig.add_subplot()
ax.plot(C_array, acc, color='red', linewidth=2, marker='o')
ax.set_xlim(0,1.1)
ax.set(title='Accuracy with increasing C values', ylabel='Accuracy', xlabel='c')
plt.show()

