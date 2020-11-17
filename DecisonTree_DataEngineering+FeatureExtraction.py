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

# In[25]:


sb.pairplot (df_db, hue='Outcome')
plt.show()


# # Feature Extraction

# In[8]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

y = df_db.loc[:,'Outcome'].values
x = StandardScaler().fit_transform(df_db.iloc[:,0:-1])


# In[9]:


for i in range(1,9):
    pca = PCA(n_components=i)
    pc = pca.fit_transform(x)
    print('Explained Variance for ' + str(pca.n_components) + ' principal components: ', pca.explained_variance_ratio_.sum())


# In[10]:


pca = PCA(n_components=8)
pc = pca.fit_transform(x)
print(pc)

print('Explained Variance ratio: ', pca.explained_variance_ratio_)
print('Explained Variance for ' + str(pca.n_components) + ' principal components: ', pca.explained_variance_ratio_.sum())


# # Train our Model

# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[12]:


x = pd.DataFrame(pc, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])
x


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.25,random_state = 42)


# In[14]:


import time

dt = DecisionTreeClassifier(random_state = 42) 
train_start_time = time.time()
dt.fit(X_train, y_train)
print("Duration of training: %s seconds" % (time.time() - train_start_time))


# # Evaluate our Model

# In[15]:


y_pred = dt.predict(X_test)
print(accuracy_score(y_test, y_pred))


# # Predict some Value

# In[16]:


pred_start_time = time.time()
print(dt.predict([(0, 0, 0, 0, 0, 0, 0, 0)]))
print("Duration of prediction: %s seconds" % (time.time() - pred_start_time))


# The Outcome from model predicting data with each Principal Component = 0 is 0

# # Validation with Confusion Matrix

# In[17]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, cmap="Blues", fmt="d")


# In[18]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# # Exploring Different max_depth, min_samples_split, min_samples_leaf

# In[19]:


from sklearn.tree import DecisionTreeClassifier

depth_array = range(1, 15)
leaf_array = range(1, 15)
split_array = range(2, 15)

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


# In[20]:


fig = plt.figure(figsize=(30,10))
ax = fig.add_subplot(111)
ax.plot(depth_array, acc_depth, color='red', linewidth=1, marker='o', label='max_depth')
ax.plot(leaf_array, acc_leaf, color='green', linewidth=1, marker='o', label='min_samples_leaf')
ax.plot(split_array, acc_split, color='blue', linewidth=1, marker='o', label='min_samples_split')
plt.legend(loc='lower right')
ax.set_xlim(0,15)
ax.set(title='Accuracy with increasing max_depth/min_samples_leaf/min_samples_split', ylabel='Accuracy')
ax.xaxis.set(ticks=range(1,15))
plt.show()


# # max_depth = 11, min_samples_leaf = 1, min_samples_split = 2 (Highest Accuracy)

# In[21]:


dt = DecisionTreeClassifier(max_depth = 11, random_state = 42) 

train_start_time = time.time()
dt.fit(X_train, y_train)
print("Duration of training: %s seconds" % (time.time() - train_start_time))

y_pred = dt.predict(X_test)
print("Accuracy : %s" % (accuracy_score(y_test, y_pred)))

pred_start_time = time.time()
print(dt.predict([(0, 0, 0, 0, 0, 0, 0, 0)]))
print("Duration of prediction: %s seconds" % (time.time() - pred_start_time))

cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, cmap="Blues", fmt="d")


# In[22]:


print(classification_report(y_test,y_pred))


# In[24]:


from sklearn import tree
import graphviz
from graphviz import Source

Source(tree.export_graphviz(dt, out_file=None, class_names=True, feature_names= X_train.columns))

