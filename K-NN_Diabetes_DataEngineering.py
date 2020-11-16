#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[2]:


df = pd.read_csv("diabetes.csv")
df


# # Data Cleaning

# In[3]:


df['Glucose'].replace(0,np.NaN, inplace=True)
df['BloodPressure'].replace(0,np.NaN, inplace=True)
df['SkinThickness'].replace(0,np.NaN, inplace=True)
df['Insulin'].replace(0,np.NaN, inplace=True)
df['BMI'].replace(0,np.NaN, inplace=True)
df = df.dropna()
df


# In[4]:


outcome_0 = len(df[df['Outcome'] == 0])
outcome_1 = len(df[df['Outcome'] == 1])

sample_size = pd.DataFrame(
{
    'Outcome': range(2),
    'Number of Samples': [outcome_0, outcome_1]
})
sample_size


# In[5]:


df_0 = df[df['Outcome'] == 0]
df_1 = df[df['Outcome'] == 1]
df_1.reset_index(drop=True, inplace=True)
df_1


# # Data Balancing - Oversampling

# In[6]:


df_db1 = pd.DataFrame(columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
j = 0

for i in range(263):
    if j == 129:
        j = 0
    df_db1.loc[i] = df_1.loc[j]
    j += 1

df_db1


# In[7]:


df_db = pd.concat([df_0, df_db1])
df_db.reset_index(drop=True, inplace=True)
df_db['Outcome'] = df_db['Outcome'].astype(int)
df_db


# # Visualise the Data

# In[109]:


sb.pairplot (df_db, hue='Outcome')
plt.show()


# # Train our Model

# In[8]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

knn = KNeighborsClassifier(n_neighbors = 5) 


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(df_db.iloc[:,0:8], df_db['Outcome'], random_state = 42)
X_train.head()


# In[10]:


y_train.head()


# In[11]:


import time

train_start_time = time.time()
knn.fit(X_train, y_train)
print("Duration of training: %s seconds" % (time.time() - train_start_time))


# # Evaluate our Model

# In[12]:


y_pred = knn.predict(X_test)
print(y_pred) 
print(y_test)


# In[13]:


from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))


# # Predict some Value

# In[14]:


pred_start_time = time.time()
print(knn.predict([(2, 100, 70, 20, 100, 25, 0.5, 27)]))
print("Duration of prediction: %s seconds" % (time.time() - pred_start_time))


# The Outcome from model predicting data with Pregnancies = 2, Glucose = 100, BloodPressure = 70, SkinThickness = 20, Insulin = 100, BMI = 25, DiabetesPedigreeFunction = 0.5, Age = 27 is 0

# # Exploring Different k Values

# In[15]:


k_array = np.arange(1, 30, 2)

k_array


# In[16]:


acc = []
for k in k_array:
    knn_ex = KNeighborsClassifier(n_neighbors = k)
    knn_ex.fit(X_train, y_train)
    ac = accuracy_score(y_test, knn_ex.predict(X_test))
    acc.append(ac)
    print("k = {0}".format(k))
    print("Accuracy = {0}".format(ac))


# In[17]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k_array, acc, color='red', linewidth=2, marker='o')
ax.set_xlim(0,30)
ax.set(title='Accuracy with increasing k values', ylabel='Accuracy', xlabel='k')
ax.xaxis.set(ticks=range(1,30,2))
plt.show()


# # Validation with Confusion Matrix

# In[21]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, cmap="Blues", fmt="d")


# In[22]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred, digits=3))


# # k = 19 (Highest Accuracy)

# In[26]:


knn_19 = KNeighborsClassifier(n_neighbors = 19)

train_start_time = time.time()
knn_19.fit(X_train, y_train)
print("Duration of training: %s seconds" % (time.time() - train_start_time))

y_pred19 = knn_19.predict(X_test)
print("Accuracy: %s" % (accuracy_score(y_test, y_pred19)))

pred_start_time = time.time()
print("Prediction: %s" % (knn.predict([(2, 100, 70, 20, 100, 25, 0.5, 27)])))
print("Duration of prediction: %s seconds" % (time.time() - pred_start_time))

cm_19 = confusion_matrix(y_test, y_pred19)
sb.heatmap(cm_19, annot=True, cmap="Blues", fmt="d")


# In[25]:


print(classification_report(y_test,y_pred19, digits=3))

