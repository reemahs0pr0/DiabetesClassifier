#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[25]:


df = pd.read_csv("diabetes.csv")
df


# # Data Cleaning

# In[29]:


df['Glucose'].replace(0,np.NaN, inplace=True)
df['BloodPressure'].replace(0,np.NaN, inplace=True)
df['SkinThickness'].replace(0,np.NaN, inplace=True)
df['Insulin'].replace(0,np.NaN, inplace=True)
df['BMI'].replace(0,np.NaN, inplace=True)
df = df.dropna()
df


# In[30]:


outcome_0 = len(df[df['Outcome'] == 0])
outcome_1 = len(df[df['Outcome'] == 1])

sample_size = pd.DataFrame(
{
    'Outcome': range(2),
    'Number of Samples': [outcome_0, outcome_1]
})
sample_size


# In[31]:


df_0 = df[df['Outcome'] == 0]
df_1 = df[df['Outcome'] == 1]
df_1.reset_index(drop=True, inplace=True)
df_1


# # Data Balancing - Oversampling

# In[32]:


df_db1 = pd.DataFrame(columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
j = 0

for i in range(263):
    if j == 129:
        j = 0
    df_db1.loc[i] = df_1.loc[j]
    j += 1

df_db1


# In[33]:


df_db = pd.concat([df_0, df_db1])
df_db.reset_index(drop=True, inplace=True)
df_db['Outcome'] = df_db['Outcome'].astype(int)
df_db


# # Visualise the Data

# In[160]:


sb.pairplot (df_db, hue='Outcome')
plt.show()


# # Feature Selection

# In[35]:


sb.heatmap(df_db.corr(), annot=True, cmap='BuPu')
plt.show()


# - 'Insulin' is highly correlated to 'Glucose' and 'Insulin' has lower correlation with 'Outcome'
# - 'BMI' is highly correlated to 'SkinThickness' and 'SkinThickness' has lower correlation with 'Outcome'
# 
# Hence, we will discard 'Insulin' and 'SkinThickness'.
# 
# Correlation with 'Outcome' which is < 0.2 will be used to indicate weak correlation.
# 
# Hence, we will discard 'BloodPressure'.

# # Train our Model

# In[137]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

knn = KNeighborsClassifier(n_neighbors = 5) 


# In[138]:


X_train, X_test, y_train, y_test = train_test_split(df_db.iloc[:,[0,1,5,6,7]], df_db['Outcome'], random_state = 42)
X_train.head()


# In[139]:


y_train.head()


# In[140]:


import time

train_start_time = time.time()
knn.fit(X_train, y_train)
print("Duration of training: %s seconds" % (time.time() - train_start_time))


# # Evaluate our Model

# In[141]:


y_pred = knn.predict(X_test)
print(y_pred) 
print(y_test)


# In[142]:


from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))


# # Predict some Value

# In[153]:


pred_start_time = time.time()
print(knn.predict([(6, 200, 30, 0.1, 57)]))
print("Duration of prediction: %s seconds" % (time.time() - pred_start_time))


# The Outcome from model predicting data with Pregnancies = 6, Glucose = 200, BMI = 30, DiabetesPedigreeFunction = 0.1, Age = 57 is 1

# # Exploring Different k Values

# In[154]:


k_array = np.arange(1, 30, 2)

k_array


# In[155]:


acc = []
for k in k_array:
    knn_ex = KNeighborsClassifier(n_neighbors = k)
    knn_ex.fit(X_train, y_train)
    ac = accuracy_score(y_test, knn_ex.predict(X_test))
    acc.append(ac)
    print("k = {0}".format(k))
    print("Accuracy = {0}".format(ac))


# In[156]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k_array, acc, color='red', linewidth=2, marker='o')
ax.set_xlim(0,30)
ax.set(title='Accuracy with increasing k values', ylabel='Accuracy', xlabel='k')
ax.xaxis.set(ticks=range(1,30,2))
plt.show()


# # Validation with Confusion Matrix

# We can use Confusion Matrix to see how the prediction goes.
# The matrix has the format:
# 
# |                    |actual outcome A | actual outcome B|
# |--------------------|-----------------|-----------------|
# |predicted outcome A |                 |                 |
# |predicted outcome B |                 |                 |

# In[157]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, cmap="Blues", fmt="d")


# In[158]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred, digits=3))


# In[159]:


#k = 9 ==> highest accuracy
knn_9 = KNeighborsClassifier(n_neighbors = 9)
knn_9.fit(X_train, y_train)
y_pred9 = knn_9.predict(X_test)
print(accuracy_score(y_test, y_pred9))
cm_9 = confusion_matrix(y_test, y_pred9)
sb.heatmap(cm_9, annot=True, cmap="Blues", fmt="d")


# In[152]:


print(classification_report(y_test,y_pred9, digits=3))

