#!/usr/bin/env python
# coding: utf-8

# In[92]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[93]:


data = np.genfromtxt("hw02_data_set_images.csv", delimiter = ",")
labels = np.genfromtxt("hw02_data_set_labels.csv", dtype=str) 


# In[94]:


#create empty matrixes with the dim of train and test data sets
train_data = np.zeros((25*5,320))
test_data = np.zeros((14*5,320))
train_labels = np.zeros(25*5)
test_labels = np.zeros(14*5)
print(train_data.shape)
print(test_data.shape)
print(train_labels.shape)
print(test_labels.shape)


# In[95]:


for i in range(0,5):
    train_data[25*i:(25+25*i),:] = data[(39*i):(25+39*i),:]  # take the first 25 values of each 39 data as training data
    test_data[14*i:(14+14*i),:] = data[(25+i*39):39*(i+1),:] # take the last 14 values of each 39 data as test data
    train_labels[25*i:(25+25*i)] = labels[(39*i):(25+39*i)]  # take the first 25 values of each 39 labels as training labels
    test_labels[14*i:(14+14*i)] = labels[(25+i*39):39*(i+1)] # take the last 14 values of each 39 labels as test labels

print(train_data)
print(test_data)
print(train_labels)
print(test_labels)


# In[96]:


def gen_y_truth(train_labels):
    val = np.zeros(train_labels.shape[0])
    for i in range(train_labels.shape[0]):
        if train_labels[i] == 1:
            val[i] = 1
        elif train_labels[i] == 2:
            val[i] = 2
        elif train_labels[i] == 3:
            val[i] = 3
        elif train_labels[i] == 4:
            val[i] = 4
        elif train_labels[i] == 5:
            val[i] = 5
    return val


# In[97]:


gen_y_truth(train_labels).shape


# In[98]:


y_train_truth = gen_y_truth(train_labels)  # 125,1
y_test_truth = gen_y_truth(test_labels)    #70,1
print(y_train_truth.shape)
print(y_test_truth.shape)


# In[99]:


sample_means = np.array([np.sum(train_data[y_train_truth == (c + 1)], axis=0) for c in range(5)]) / 25
print(sample_means)
sample_means.shape


# In[100]:


class_priors = [np.mean(y_train_truth == (c + 1)) for c in range(5)]
print(class_priors)


# In[102]:


def safe_log(x):  #in order to avoid errors
    return(np.log(x + 1e-100))

def calc_score(x):  #calculate score
    scores = np.zeros(5)
    for i in range(5):
        scores[i] = scores[i] + safe_log(class_priors[i])
        scores[i] = scores[i] + np.sum( x*safe_log(sample_means[i]) + ( (np.ones(sample_means.shape[1]) - x)*safe_log(np.ones(sample_means.shape[1]) - sample_means[i])) )
    return scores


# In[105]:


train_predictions = np.zeros(125)
for i in range(125):
    train_predictions[i] = np.argmax(calc_score(train_data[i])) + 1

confusion_matrix = pd.crosstab(train_predictions,y_train_truth,rownames = ['y_predicted'],colnames = ['y_test'])
print("training performance")
print(confusion_matrix)


# In[106]:


test_predictions = np.zeros(70)
for i in range(70):
    test_predictions[i] = np.argmax(calc_score(test_data[i])) + 1


test_confusion_matrix = pd.crosstab(test_predictions,y_test_truth,rownames = ['y_predicted'], colnames = ['y_test'])
print("test performance")
print(test_confusion_matrix)


# In[ ]:




