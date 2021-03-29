#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report


# In[32]:


df = pd.read_csv('E:\\Fake_News\\news.csv')


# In[33]:


df.shape
df.head()


# In[34]:


sns.countplot(df.label)
plt.xlabel('Label')
plt.title('Fake vs Real News')


# In[35]:


# Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], df['label'], test_size=0.2, random_state=7)


# In[36]:


# Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[37]:


# Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

# Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[38]:


# Build confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# In[41]:


print(classification_report(y_test,y_pred))


# In[43]:


f1 = 2*(0.94*0.92/(0.94+0.92))
print(f1)


# In[ ]:




