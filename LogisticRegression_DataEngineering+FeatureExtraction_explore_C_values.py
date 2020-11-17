#!/usr/bin/env python
# coding: utf-8

# # Importing and Reading Data

# In[23]:


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
from sklearn.metrics import accuracy_score,confusion_matrix


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[24]:


#reading the data
df = pd.read_csv("diabetes.csv")


# # Data Cleaning

# In[25]:


#cleaning the data
df['Glucose'].replace(0,np.NaN, inplace=True)
df['BloodPressure'].replace(0,np.NaN, inplace=True)
df['SkinThickness'].replace(0,np.NaN, inplace=True)
df['Insulin'].replace(0,np.NaN, inplace=True)
df['BMI'].replace(0,np.NaN, inplace=True)
df = df.dropna()
df


# In[26]:


outcome_0 = len(df[df['Outcome'] == 0])
outcome_1 = len(df[df['Outcome'] == 1])

sample_size = pd.DataFrame(
{
    'Outcome': range(2),
    'Number of Samples': [outcome_0, outcome_1]
})
sample_size

#number below shows the outcome 1 data is a lot less than outcome 0 data


# In[27]:


df_0 = df[df['Outcome'] == 0]
df_1 = df[df['Outcome'] == 1]
#resetting index of outcome 1 to 0
df_1.reset_index(drop=True, inplace=True)
df_1


# # Data Balancing - Oversampling

# In[28]:


#to make sure all data are 500 sets each for Outcome 1
df_db1 = pd.DataFrame(columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
j = 0

for i in range(263):
    if j == 129:
        j = 0
    df_db1.loc[i] = df_1.loc[j]
    j += 1

df_db1


# In[29]:


#Final dataset with oversampling data for outcome 1
df_db = pd.concat([df_0, df_db1])
df_db.reset_index(drop=True, inplace=True)
df_db['Outcome'] = df_db['Outcome'].astype(int)
df_db


# # Visualise the Data

# In[8]:


sb.pairplot (df_db, hue='Outcome')
plt.show()


# # Feature Extraction

# In[30]:


y = df_db.loc[:,'Outcome'].values
X = StandardScaler().fit_transform(df_db.iloc[:,0:-1])


# In[31]:


for i in range(1,9):
    pca = PCA(n_components=i)
    pc = pca.fit_transform(X)
    print('Explained Variance for ' + str(pca.n_components) + ' principal components: ', pca.explained_variance_ratio_.sum())


# In[32]:


pca = PCA(n_components=7)
pc = pca.fit_transform(X)
print(pc)

print('Explained Variance ratio: ', pca.explained_variance_ratio_)
print('Explained Variance for ' + str(pca.n_components) + ' principal components: ', pca.explained_variance_ratio_.sum())


# # Train our Model

# In[33]:


#split data to train and test in ratio 75% 25%
X_train, X_test, y_train, y_test = train_test_split(pc, y, test_size=0.25, random_state=42)


# In[34]:


#initialise logistic regression
logReg = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial',  max_iter=1000,random_state = 42)


# In[35]:


import time
train_start_time = time.time()
# Train the classifier.
logReg.fit(X_train,y_train)

print("Duration of training: %s seconds" % (time.time() - train_start_time))


# # Evaluate our Model

# In[36]:


#Make Predictions
y_pred = logReg.predict(X_test)


# In[37]:


print(accuracy_score(y_test, y_pred)) 
#the accuracy has risen  to 84%


# # Predict some Value

# In[38]:


pred_start_time = time.time()
print(logReg.predict([(0, 0, 0, 0, 0, 0, 0)]))
print("Duration of prediction: %s seconds" % (time.time() - pred_start_time))


# The Outcome from model predicting data with each Principal Component = 0 is 1

# # Validation with Confusion Matrix

# In[39]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, cmap="Blues", fmt="d")


# In[40]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred, digits=3))


# # Exploring Different C Values

# In[41]:


C_array = np.arange(0.1, 1.1, 0.1)

C_array


# In[42]:


acc = []
for C in C_array:
    logReg_ex = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter=1000, random_state = 42, C = C)
    logReg_ex.fit(X_train,y_train)
    ac = accuracy_score(y_test, logReg_ex.predict(X_test))
    acc.append(ac)
    print("C = {0}".format(C))
    print("Accuracy = {0}".format(ac))


# In[43]:


fig = plt.figure()
ax = fig.add_subplot()
ax.plot(C_array, acc, color='red', linewidth=2, marker='o')
ax.set_xlim(0,1.1)
ax.set(title='Accuracy with increasing C values', ylabel='Accuracy', xlabel='c')
plt.show()

