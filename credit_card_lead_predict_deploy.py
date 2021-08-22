#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas.api.types as ptypes
import pickle
from xgboost.sklearn import XGBClassifier

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


cc_data = pd.read_csv(r'C:\Users\RONALD\Desktop\IMS-Classroom\Python Code\Resume Project - ML Algo\Credit Card Lead Prediction\Deployment\Train_Data.csv')


# In[3]:


cc_data.columns


# In[4]:


cc_data = cc_data.drop(['ID'], axis=1)


# In[5]:


cc_data.columns


# In[6]:


cc_data.isnull().sum()


# #### Treating missing values

# In[7]:


def treat_columns(col_list):
    for i in col_list:
        col_name = i
        
        if (cc_data[col_name].isnull().any().any() == True):
            
            if ptypes.is_numeric_dtype(cc_data[col_name]):
                cc_data[col_name].fillna(cc_data[col_name].median(),inplace=True)
                print(col_name, 'Done')
            else:
                cc_data[col_name].fillna(cc_data[col_name].mode()[0], inplace=True)
                print(col_name, 'Done')


# In[8]:


cc_data.columns


# In[9]:


cols = ['Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
       'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active',
       'Is_Lead']


# In[10]:


treat_columns(cols)


# In[11]:


cc_data.isnull().sum()


# In[12]:


num_cols = cc_data[cc_data.select_dtypes(include = [np.number]).columns.tolist()]
cat_cols = cc_data[cc_data.select_dtypes(exclude = [np.number]).columns.tolist()]


# In[13]:


num_cols.columns


# In[14]:


cat_cols.columns


# In[15]:


num_cols.head()


# In[16]:


num_cols['Avg_Account_Balance'] = np.log(num_cols['Avg_Account_Balance'])


# In[17]:


train_num = num_cols.drop(['Is_Lead'], axis=1)
train_num.columns


# In[18]:


cols_train = []
cols_train=train_num.columns
cols_train


# In[19]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_num = scaler.fit_transform(train_num.values)
train_num = pd.DataFrame(train_num, columns = cols_train)


# In[20]:


train_num.head()


# #### Categorical columns

# In[21]:


from sklearn.preprocessing import LabelEncoder

train_cat = cat_cols.apply(LabelEncoder().fit_transform)


# #### Combine Numerical & Categorical columns

# In[24]:


train_target = num_cols['Is_Lead']
train_target


# In[25]:


final_cc_data = pd.concat([train_num, train_cat, train_target], axis=1)
final_cc_data.head()


# #### Data Partition

# In[26]:


X = final_cc_data.drop(['Is_Lead'], axis=1)
Y = final_cc_data[['Is_Lead']]


# In[27]:


X.columns


# In[28]:


Y.columns


# In[29]:


df_X = X.values


# In[30]:


xgbclassifier = XGBClassifier()
xgbclassifier.fit(df_X, Y)


# In[31]:


pickle.dump(xgbclassifier, open('model.pkl', 'wb'))


# In[32]:


model = pickle.load(open('model.pkl', 'rb'))


# In[33]:


cols_when_model_builds = model.get_booster().feature_names

