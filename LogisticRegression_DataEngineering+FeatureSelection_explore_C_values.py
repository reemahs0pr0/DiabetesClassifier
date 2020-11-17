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


# # Feature Selection

# In[30]:


sb.heatmap(df_db.corr(), annot=True, cmap='BuPu')
plt.show()


# In[31]:


#4 filter all features with correlation of  x<-0.2 or x>0.2 of Outcome
corr_mat=df_db.corr()
outcome_col=corr_mat['Outcome']
outcome_col.drop(['Outcome'],inplace=True)
filtered_outcome=outcome_col[(outcome_col<-0.2)|(outcome_col>0.2)]
print('candidates w.r.t Outcome=\n',filtered_outcome,'\n',sep='')


# In[32]:


#for each feature above, check if it's NOT highly correlated with other features
#select two or more features if their correlation is >0.3, if they are highly correlated, select one with higher correlation

to_drop=list(set(corr_mat.index)-set(filtered_outcome.index))
workset=corr_mat.drop(index=to_drop,columns=to_drop)

skip=[]
accept=[]

for colname in workset.columns:
    if not colname in skip and not colname in accept:
        series=workset[colname]
        series=series[(series>0.4)]
        resp_outcome=filtered_outcome[series.index]
        print("\noutcome:\n",resp_outcome)
        top=resp_outcome.abs().idxmax()
        
        accept=accept+[top]
        print("accept:",accept)
        
        skip += set(resp_outcome.index) - set([top])
       
        print("skip:",skip)


# # Train our Model

# In[33]:


#choose column for dependent variable
y=df_db["Outcome"]
y.head()


# In[34]:


#choose colum for independent variable
X=df_db.iloc[:,[1,5,6,7]]
X.head()


# In[35]:


#split data to train and test in ratio 75% 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[36]:


#initialise logistic regression
logReg = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial',  max_iter=1000,random_state = 42)


# In[37]:


import time
train_start_time = time.time()
# Train the classifier.
logReg.fit(X_train,y_train)

print("Duration of training: %s seconds" % (time.time() - train_start_time))


# # Evaluate our Model

# In[38]:


#Make Predictions
y_pred = logReg.predict(X_test)


# In[39]:


print(accuracy_score(y_test, y_pred)) 
#the accuracy has risen from 81 to 82%


# # Predict some Value

# In[40]:


pred_start_time = time.time()
print(logReg.predict([(200, 30, 0.1, 57)]))
print("Duration of prediction: %s seconds" % (time.time() - pred_start_time))


# The Outcome from model predicting data with Glucose = 200, BMI = 30, DiabetesPedigreeFunction = 0.1, Age = 57 is 1

# # Validation with Confusion Matrix

# In[41]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sb.heatmap(cm, annot=True, cmap="Blues", fmt="d")


# In[42]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred, digits=3))


# # Exploring Different C Values

# In[43]:


C_array = np.arange(0.1, 1.1, 0.1)

C_array


# In[44]:


acc = []
for C in C_array:
    logReg_ex = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter=1000, random_state = 42, C = C)
    logReg_ex.fit(X_train,y_train)
    ac = accuracy_score(y_test, logReg_ex.predict(X_test))
    acc.append(ac)
    print("C = {0}".format(C))
    print("Accuracy = {0}".format(ac))


# In[45]:


fig = plt.figure()
ax = fig.add_subplot()
ax.plot(C_array, acc, color='red', linewidth=2, marker='o')
ax.set_xlim(0,1.1)
ax.set(title='Accuracy with increasing C values', ylabel='Accuracy', xlabel='c')
plt.show()

