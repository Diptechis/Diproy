#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd


# In[11]:


#1) Generate an array of zeroes using inbuilt numpy function.

arr= np.zeros((3,3))
print(arr)


# In[12]:


#2) Run the given code and find the dot product of both arrays.
a = np.random.randint(2,10, (3,4))
b = np.random.randint(2,10, (4,3))
print(a)
print(b)


# In[13]:


a.dot(b)


# In[16]:


print (np.dot(a,b))


# In[ ]:





# In[ ]:


#3)Read the csv file provided using pandas and display the first 5 entries


# In[57]:


data = pd.read_csv("data_ass.csv")
data.head()


# In[58]:


#4)Describe the data with all features.
data.describe()


# In[59]:


#5)Find the total count of missing values for each feature.
data.isnull().sum()


# In[60]:


#6)Display all the unique values from the 'DESCRIPTION’ column.
data['DESCRIPTION'].unique()


# In[61]:


#7)Create a grouped table usin 'DESCRIPTION' as the grouping columns with the means of all the other columns.
data.groupby('DESCRIPTION').mean()


# In[62]:


#8)Generate a random sample of 10 rows from the data.
data.sample(n=10)


# In[90]:


#9)Add a Feature called 'New_feature' to the new Dataset and Add Random float Values in between 0 and 1 using Numpy.

data['New_feature']= np.random.rand(25,1) 
data


# In[88]:


#10) Replace zero value of column 'OBJECTID' with mean value permanently.
mean_o= data.OBJECTID.mean()
mean_o
data.OBJECTID.replace(0,mean_o,inplace = True)


# In[72]:


data

