#!/usr/bin/env python
# coding: utf-8

# # Importing and Reading Data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[2]:


df = pd.read_csv("diabetes.csv")
df


# # Visualise the Data

# In[3]:


sb.pairplot (df, hue='Outcome')
plt.show()


# # Train our Model

# In[4]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

knn = KNeighborsClassifier(n_neighbors = 5) 


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:8], df['Outcome'], random_state = 42)
X_train.head()


# In[6]:


y_train.head()


# In[7]:


import time

train_start_time = time.time()
knn.fit(X_train, y_train)
print("Duration of training: %s seconds" % (time.time() - train_start_time))


# # Evaluate our Model

# In[8]:


y_pred = knn.predict(X_test)
print(y_pred) 
print(y_test)


# In[9]:


from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))


# # Predict some Value

# In[10]:


pred_start_time = time.time()
print(knn.predict([(2, 100, 70, 20, 100, 25, 0.5, 27)]))
print("Duration of prediction: %s seconds" % (time.time() - pred_start_time))


# The Outcome from model predicting data with Pregnancies = 2, Glucose = 100, BloodPressure = 70, SkinThickness = 20, Insulin = 100, BMI = 25, DiabetesPedigreeFunction = 0.5, Age = 27 is 0

# 
# # Validation with Confusion Matrix

# In[11]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, cmap="Blues", fmt="d")


# In[12]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred, digits=3))


# # Exploring Different k Values

# In[13]:


k_array = np.arange(1, 30, 2)

k_array


# In[14]:


acc = []
for k in k_array:
    knn_ex = KNeighborsClassifier(n_neighbors = k)
    knn_ex.fit(X_train, y_train)
    ac = accuracy_score(y_test, knn_ex.predict(X_test))
    acc.append(ac)
    print("k = {0}".format(k))
    print("Accuracy = {0}".format(ac))


# In[15]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k_array, acc, color='red', linewidth=2, marker='o')
ax.set_xlim(0,30)
ax.set(title='Accuracy with increasing k values', ylabel='Accuracy', xlabel='k')
ax.xaxis.set(ticks=range(1,30,2))
plt.show()


# # k = 13 (Highest Accuracy)

# In[16]:


knn_13 = KNeighborsClassifier(n_neighbors = 13)

train_start_time = time.time()
knn_13.fit(X_train, y_train)
print("Duration of training: %s seconds" % (time.time() - train_start_time))

y_pred13 = knn_13.predict(X_test)
print("Accuracy: %s" % (accuracy_score(y_test, y_pred13)))

pred_start_time = time.time()
print("Prediction: %s" % (knn.predict([(2, 100, 70, 20, 100, 25, 0.5, 27)])))
print("Duration of prediction: %s seconds" % (time.time() - pred_start_time))

cm_13 = confusion_matrix(y_test, y_pred13)
sb.heatmap(cm_13, annot=True, cmap="Blues", fmt="d")


# In[17]:


print(classification_report(y_test,y_pred13, digits=3))

