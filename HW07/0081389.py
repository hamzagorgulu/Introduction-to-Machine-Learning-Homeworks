#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance
from scipy import stats


# In[5]:


X = np.genfromtxt(r"C:\Users\Hamza\Desktop\Course Contents\ENGR-421 Introduction to Machine Learning\Homeworks\HW07\hw07_data_set_images.csv",delimiter=",")
Y = np.genfromtxt(r"C:\Users\Hamza\Desktop\Course Contents\ENGR-421 Introduction to Machine Learning\Homeworks\HW07\hw07_data_set_labels.csv",delimiter=",")


# In[9]:


N_train=X.shape[0]
N_test=Y.shape[0]
X_train = np.zeros(N_train)
Y_train = np.zeros(N_train)
X_test = np.zeros(N_test)
Y_test = np.zeros(N_test)

X_train = X[0:2000,:]
X_test = X[0:2000,:]
Y_train = Y[0:2000]
Y_test = Y[2000:4001]
#Y_train=Y_train[:, None]
#Y_test=Y_test[:, None]

print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


# In[11]:


N = len(Y_train)
D = X_train.shape[1]
K = int(max(Y_train))
print(N,D,K)


# In[12]:


class_means = []

for i in range(K):
    class_means.append( np.mean(X_train[Y_train == i + 1,] , axis=0))
    
class_means = np.array(class_means)

X_train_minus_mean = []

for i in range(N):
    X_train_minus_mean.append( X_train[i, :] - class_means[np.int(Y_train[i]) - 1, :] )
    
X_train_minus_mean= np.array(X_train_minus_mean)

t_mean = np.mean(class_means, axis = 0)


# In[14]:


def within_class_scatter():
    ret = np.zeros((D,D))
    class_covariances = [(np.dot(np.transpose(X_train[Y_train == (c + 1)] - class_means[c]), (X_train[Y_train == (c + 1)] - class_means[c]))) for c in range(K)]
    ret = class_covariances[0] + class_covariances[1] + class_covariances[2]
    return ret
        
def between_class_scatter():
    ret = np.zeros((D,D))
    for i in range(K):
        X_c = X_train[Y_train == i+1]
        mean_c = np.mean(X_c, axis = 0)
        n_c = X_c.shape[0]
        mean_d = (mean_c - t_mean).reshape(D,1)
        ret += n_c * np.dot(mean_d, np.transpose(mean_d))
    return ret
      


# In[15]:


within_class_scatter_mat = within_class_scatter()
between_class_scatter_mat = between_class_scatter()

for d in range(D):
    within_class_scatter_mat[d,d] = within_class_scatter_mat[d,d] + 1e-10   


# In[20]:


#eigen values and eigen vectors
within_scatter_inversed = np.linalg.inv(within_class_scatter_mat)
values, vectors = la.eigh(np.dot(within_scatter_inversed, between_class_scatter_mat))

two_vectors = vectors[:, 0:2]
Z_train = np.dot(X_train, two_vectors)
Z_test = np.dot(X_test, two_vectors)


# In[52]:


point_colors = ["#fc051a", "#004cff", "#00d150","#FFFF00","#acbf63","#bf6363","#63bf91","#bf637a","#a77b86","#c2a3ab"]
labels=["t-shirt top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker",
"bag","ankle boot"]
plt.figure()
plt.title("training points")
for i in range(N):
    plt.scatter(Z_train[i,0], -Z_train[i,1], c=point_colors[np.int(Y_train[i])-1], s=5)
plt.legend(labels,bbox_to_anchor=(1.05,1))
plt.show()


# In[53]:


plt.figure()
plt.title("tests points")
for i in range(len(Y_test)):
    plt.scatter(Z_test[i,0],  -Z_test[i,1], color=point_colors[np.int(Y_test[i])-1], s=5)
plt.legend(labels,bbox_to_anchor=(1.05,1))
plt.show()


# In[54]:


train_predictions = []

for i in range(len(Z_train[:,1])):
    v = Z_train[i, :]
    initial_distances = np.zeros(Z_train.shape[0])
    for j in range(len(Z_train[:,1])):
        initial_distances[j] = distance.euclidean(v, Z_train[j, :])
    smallest_dists_indices = np.argsort(initial_distances)[:11]  # k=11
    temp_labels = []
    for x in smallest_dists_indices:
        temp_labels.append(Y_train[x])
    prediction= stats.mode(temp_labels)[0]
    train_predictions.append(prediction)
    
print(np.transpose(np.array(confusion_matrix(train_predictions, Y_train))))


# In[55]:


test_predictions = []

for i in range(len(Z_test[:,1])):
    v = Z_test[i, :]
    initial_distances = np.zeros(Z_train.shape[0])
    for j in range(len(Z_train[:,1])):
        initial_distances[j] = distance.euclidean(v, Z_train[j, :])
    smallest_dists_indices = np.argsort(initial_distances)[:11]  #k is 11
    temp_labels = []
    for x in smallest_dists_indices:
        temp_labels.append(Y_train[x])
    prediction= stats.mode(temp_labels)[0]
    test_predictions.append(prediction)
    
print(np.transpose(np.array(confusion_matrix(test_predictions, Y_test))))


# In[ ]:




