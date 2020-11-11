#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[18]:


df = pd.read_csv("diabetes.csv")
df


# In[19]:


df.info()


# In[4]:


sb.pairplot (df, hue='Outcome')
plt.show()


# # Train the Model

# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:8], df['Outcome'],test_size=0.25,random_state = 42)
X_train.head()


# # Evaluate the Model

# In[29]:


import time

dt = DecisionTreeClassifier(random_state = 42) 
train_start_time = time.time()
dt.fit(X_train, y_train)
print("Duration of training: %s seconds" % (time.time() - train_start_time))


# In[30]:


y_pred = dt.predict(X_test)
print(accuracy_score(y_test, y_pred))


# In[31]:


pred_start_time = time.time()
print(dt.predict([(2, 100, 70, 20, 100, 25, 0.5, 27)]))
print("Duration of prediction: %s seconds" % (time.time() - pred_start_time))


# The Outcome from model predicting data with Pregnancies = 2, Glucose = 100, BloodPressure = 70, SkinThickness = 20, Insulin = 100, BMI = 25, DiabetesPedigreeFunction = 0.5, Age = 27 is 0

# In[32]:


cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, cmap="Blues", fmt="d")


# In[33]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[78]:


depth_array = range(1, 50)
leaf_array = range(1, 50)
split_array = range(2, 50)

acc_depth = []
acc_leaf = []
acc_split = []

print("MAX_DEPTH\n---------")
for depth in depth_array:
    dt = DecisionTreeClassifier(max_depth = depth, random_state = 42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc_depth.append(accuracy_score(y_test, y_pred))
    print(depth)
    print(accuracy_score(y_test, y_pred))

print("\nMIN_SAMPLES_LEAF\n----------------")
for leaf in leaf_array:
    dt = DecisionTreeClassifier(min_samples_leaf = leaf, random_state = 42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc_leaf.append(accuracy_score(y_test, y_pred))
    print(leaf)
    print(accuracy_score(y_test, y_pred))

print("\nMIN_SAMPLES_SPLIT\n-----------------")
for split in split_array:
    dt = DecisionTreeClassifier(min_samples_split = split, random_state = 42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc_split.append(accuracy_score(y_test, y_pred))
    print(split)
    print(accuracy_score(y_test, y_pred))


# In[79]:


fig = plt.figure(figsize=(30,10))
ax = fig.add_subplot(111)
ax.plot(depth_array, acc_depth, color='red', linewidth=1, marker='o', label='max_depth')
ax.plot(leaf_array, acc_leaf, color='green', linewidth=1, marker='o', label='min_samples_leaf')
ax.plot(split_array, acc_split, color='blue', linewidth=1, marker='o', label='min_samples_split')
plt.legend(loc='lower right')
ax.set_xlim(0,50)
ax.set(title='Accuracy with increasing max_depth/min_samples_leaf/min_samples_split', ylabel='Accuracy')
ax.xaxis.set(ticks=range(1,50))
plt.show()


# In[81]:


dt = DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 40, min_samples_split = 45, random_state = 42) 

train_start_time = time.time()
dt.fit(X_train, y_train)
print("Duration of training: %s seconds" % (time.time() - train_start_time))

y_pred = dt.predict(X_test)
print("Accuracy : %s" % (accuracy_score(y_test, y_pred)))

pred_start_time = time.time()
print("Prediction: %s" % (dt.predict([(2, 100, 70, 20, 100, 25, 0.5, 27)])))
print("Duration of prediction: %s seconds" % (time.time() - pred_start_time))

cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, cmap="Blues", fmt="d")


# In[82]:


print(classification_report(y_test,y_pred))


# In[86]:


from sklearn import tree
import graphviz
from graphviz import Source

Source(tree.export_graphviz(dt, out_file=None, class_names=['Outcome 0', 'Outcome 1'], feature_names= X_train.columns))

