#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa


# In[2]:


X = np.genfromtxt("hw09_data_set.csv", delimiter=',')
X.shape


# In[3]:


N = len(X)
print(N)


# In[4]:


plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], s=40, color="k")

plt.xlabel("$x_1$", fontsize=12)
plt.ylabel("$x_2$", fontsize=12)

plt.show()


# ## Eucledian Distance and Visualization

# In[5]:


def euclidean_distance(x1, x2):
    return np.sqrt((x2[0] - x1[0])**2 + (x2[1] - x1[1])**2)


# In[6]:


euc = np.array([[euclidean_distance(X[i], X[k]) for k in range(N)] for i in range(N)])  #eucledian distances

threshold = 2

B = np.array([[1 if euc[i, k] < threshold and i != k else 0 for k in range(N)] for i in range(N)])  #B matrice
B.shape


# In[7]:


print(f"Euclidean Distances:\n{euc}")


# In[8]:


print(f"B matrix:\n{B}")


# In[9]:


plt.figure(figsize=(8, 8))

for i in range(N):
    for k in range(N):
        if B[i, k] == 1 and i != k:
            plt.plot(np.array([X[i, 0], X[k, 0]]), np.array([X[i, 1], X[k, 1]]), linewidth=0.5, c="grey")

plt.scatter(X[:, 0], X[:, 1], s=50, color="k", zorder=3)

plt.xlabel("$x_1$", fontsize=12)
plt.ylabel("$x_2$", fontsize=12)

plt.show()


# # D and L Matrices

# In[15]:


D = np.zeros((N,N), dtype=int)

for i in range(N):
    D[i, i] = B[i].sum()
    
print(f"Degree Matrix:\n{D}")


# In[16]:


L = D - B
print(f"Laplacian Matrix:\n{L}")


# In[17]:


D_new = np.zeros((N, N))

for i in range(N):
    D_new[i, i] = D[i, i]**(-1/2)


# In[18]:


L_symmetric = np.eye(N) - (D_new @ B @ D_new)
print(f"Normalized Laplacian Matrix:\n{L_symmetric}")


# ## Eigenvectors and Z matrix

# In[19]:


R = 5
eig_vals, eig_vecs = np.linalg.eig(L_symmetric)
Z = eig_vecs[:,np.argsort(eig_vals)[1:R+1]]
print(f"Z Matrix:\n{Z}")
print(Z.shape)


# ## Centroids

# In[20]:


centroids = np.vstack([Z[242], Z[528], Z[570], Z[590], Z[648],Z[667],Z[774],Z[891],Z[955]])   # K is 9
print(f"Initial Centroids:\n{centroids}")


# ## Run K-means Algorithm

# In[21]:


def update_memberships(centroids, X):
    D = spa.distance_matrix(centroids, X)
    memberships = np.argmin(D, axis = 0)
    return(memberships)

def update_centroids(memberships, X):
    centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(K)])
    return(centroids)


# In[22]:


memberships = update_memberships(centroids, Z)
iteration = 1
K = 9

while True:
    print(f"Iteration #{iteration}")
    print(f"Centroids:\n{centroids}")
    print(f"Memberships:\n{memberships}\n")
    old_centroids = centroids
    centroids = update_centroids(memberships, Z)
    if np.alltrue(centroids == old_centroids):
        break
    old_memberships = memberships
    memberships = update_memberships(centroids, Z)
    iteration += 1
    
centroids = update_centroids(memberships, X)


# ## Clustering Results

# In[25]:


colors = ["#00FFFF", "#8B8378", "#000000", "#CD3333", "#8B7355","#D2691E","#EE3B3B","#CAFF70","#C85C44"]

plt.figure(figsize=(8, 8))

for c in range(K):
    plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 15, color=colors[c])

for c in range(K):
    plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize = 12, 
             markerfacecolor = colors[c], markeredgecolor = "black")
    
plt.xlabel("$x_1$", fontsize=12)
plt.ylabel("$x_2$", fontsize=12)
    
plt.show()


# In[ ]:




