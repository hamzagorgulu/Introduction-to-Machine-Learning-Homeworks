#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(421)


# In[95]:


# Class Means
m1 = np.array([+0.0, +4.5])
m2 = np.array([-4.5, -1.0])
m3 = np.array([+4.5, -1.0])
m4 = np.array([0.0, -4.0])
means = np.array([m1, m2, m3, m4])
print(means)


# In[96]:


# Class Covariances
c1 = np.array([[+3.2, +0.0], [+0.0, +1.2]])
c2 = np.array([[+1.2, +0.8], [+0.8, +1.2]])
c3 = np.array([[+1.2, -0.8], [-0.8, +1.2]])
c4 = np.array([[+1.2, +0.0], [+0.0, +3.2]])
class_covariances = np.array([c1, c2, c3, c4])
print(class_covariances)


# In[32]:


# Class Sizes
n1 = 105
n2 = 145
n3 = 135
n4 = 115


# In[97]:


# Generating random samples from a multivariate normal distribution
points1 = np.random.multivariate_normal(m1, c1, n1)
points2 = np.random.multivariate_normal(m2, c2, n2)
points3 = np.random.multivariate_normal(m3, c3, n3)
points4 = np.random.multivariate_normal(m4, c4, n4)

points = np.concatenate((points1, points2, points3,points4))

# Generate corresponding labes
y = np.concatenate((np.repeat(1, n1), np.repeat(2, n2), np.repeat(3, n3), np.repeat(4, n4)))

plt.figure(figsize = (10, 10))
plt.plot(points1.T[0], points1.T[1], 'ro')
plt.plot(points2.T[0], points2.T[1], 'go')
plt.plot(points3.T[0], points3.T[1], 'bo')
plt.plot(points4.T[0], points4.T[1], 'yo')

plt.show()


# In[34]:


#Number of classes and points
K = np.max(y)
N = points.shape[0]


# In[35]:


K


# In[36]:


N


# In[98]:


# sample_means
sample_means = np.array([np.mean(points[y == (c + 1)], axis=0) for c in range(K)])
print(sample_means)


# In[105]:


# sample_covariances
sample_covariances = [np.cov(points[y == (c + 1)].T) for c in range(K)]

print(sample_covariances[0])
print(sample_covariances[1])
print(sample_covariances[2])
print(sample_covariances[3])


# In[106]:


# prior probabilities
class_priors = [np.mean(y == (c + 1)) for c in range(K)]
print(class_priors)


# In[107]:


# Wc, wc, wc0
W = [(-0.5 * np.linalg.inv(sample_covariances[c])) for c in range(K)]
w = [(np.linalg.inv(sample_covariances[c]) @ sample_means[c]) for c in range(K)]
w0 = [(-0.5 * (sample_means[c].T @ sample_covariances[c] @ sample_means[c]) - 0.5 * np.log(np.linalg.det(sample_covariances[c])) + np.log(class_priors[c])) for c in range(K)]

# Needed to convert to arrays because cannot take the transpose of a list
W = np.array(W)
w = np.array(w)


# In[108]:


# for gc(x) function
def g(points):
    predictions = []
    for i in range(N):
        results = [points[i].T @ W[c] @ points[i] + w[c].T @ points[i] + w0[c] for c in range(K)]
        predictions.append(results)
    return predictions


# In[109]:


y_predictions = g(points)
y_pred = np.argmax(y_predictions, axis = 1) + 1

# Confusion matrix
confusion_matrix = pd.crosstab(y_pred, y, rownames = ['y_predicted'], colnames = ['y_truth'])
print(confusion_matrix)


# In[127]:


# Visualization

x1_interval = np.linspace(-8, +8, N)
x2_interval = np.linspace(-8, +8, N)

x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))
discriminant_values_f = np.zeros((len(x1_interval), len(x2_interval), K))

for c in range(K):
    discriminant_values[:,:,c] = W[c, 0, 0] * pow(x1_grid, 2) + W[c, 0, 1] * x1_grid * x2_grid + W[c, 1, 0] * x1_grid * x2_grid + w[c, 0] * x1_grid + W[c, 1, 1] * pow(x2_grid, 2) + w[c, 1] * x2_grid + w0[c]


# For contour
A = discriminant_values[:,:,0]
B = discriminant_values[:,:,1]
C = discriminant_values[:,:,2]
D = discriminant_values[:,:,3]


A[(A < B) & (A < C) & (A < D)] = np.nan
B[(B < A) & (B < C) & (B < D)] = np.nan
C[(C < A) & (C < B) & (C < D)] = np.nan
D[(D < A) & (D < B) & (D < C)] = np.nan


plt.figure(figsize = (10, 10))
plt.plot(points[y == 1, 0], points[y == 1, 1], "r.", markersize = 10)
plt.plot(points[y == 2, 0], points[y == 2, 1], "g.", markersize = 10)
plt.plot(points[y == 3, 0], points[y == 3, 1], "b.", markersize = 10)
plt.plot(points[y == 4, 0], points[y == 4, 1], "y.", markersize = 10)
plt.plot(points[y_pred != y, 0], points[y_pred != y, 1], "ko", markersize = 12, fillstyle = "none")


plt.contour(x1_grid, x2_grid, A-B, levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, A-C, levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, B-D, levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, C-D, levels = 0, colors = "k")

plt.contourf(x1_grid, x2_grid, A-B-C , levels = 0, colors = "r", alpha=0.35)
plt.contourf(x1_grid, x2_grid, B-A-D, levels = 0, colors = "g", alpha=0.35)
plt.contourf(x1_grid, x2_grid, C-A-D, levels = 0, colors = "b", alpha=0.35)
plt.contourf(x1_grid, x2_grid, D-B-C , levels = 0, colors = "y", alpha=0.35)

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# In[ ]:




