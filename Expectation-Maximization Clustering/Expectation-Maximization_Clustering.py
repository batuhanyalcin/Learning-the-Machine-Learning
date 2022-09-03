#!/usr/bin/env python
# coding: utf-8

# # Homework 08: Linear Discriminant Analysisn
# ## Batuhan Yalçın
# ### May 19, 2022

# In[38]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
from scipy.stats import multivariate_normal


# ## Step 1, Step 2 and Step 3

# ## Importing Data

# In[57]:


X = np.genfromtxt("hw08_data_set.csv", delimiter = ",")
# Initializations
initial_centroids = np.genfromtxt("hw08_initial_centroids.csv", delimiter = ",")

N = np.shape(X)[0] 
K = np.shape(initial_centroids)[0] 
D = np.shape(X)[1]

initial_means = np.array([[+5.0, +5.0],
                          [-5.0, +5.0],
                          [-5.0, -5.0],
                          [+5.0, -5.0],
                          [+5.0, +0.0],
                          [+0.0, +5.0],
                          [-5.0, +0.0],
                          [+0.0, -5.0],
                          [+0.0, +0.0]])

initial_covariances = np.array([[[+0.8, -0.6], [-0.6, +0.8]],
                                [[+0.8, +0.6], [+0.6, +0.8]],
                                [[+0.8, -0.6], [-0.6, +0.8]],
                                [[+0.8, +0.6], [+0.6, +0.8]],
                                [[+0.2, +0.0], [+0.0, +1.2]],
                                [[+1.2, +0.0], [+0.0, +0.2]],
                                [[+0.2, +0.0], [+0.0, +1.2]],
                                [[+1.2, +0.0], [+0.0, +0.2]],
                                [[+1.6, +0.0], [+0.0, +1.6]]])


# ## Algorithm Steps

# In[58]:


def update_centroids(memberships, X):
    if memberships is None:
        # initialize centroids
        centroids = initial_centroids # I just changed the initialization step. 
    else:
        # update centroids
        centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)



def iteration_h(X, centroid, cov, prior):
    posterior = []
    for i in range(K):
        #print(centroids[i], class_cov[i])
        mix_density = multivariate_normal(centroids[i], class_cov[i]).pdf(X)*class_prior[i]  
        posterior.append(mix_density)
    H = np.vstack([posterior[k]/np.sum(posterior, axis = 0) for k in range(K)])
    return H


def iteration_update_centorids(H, X):
    mean = np.vstack([np.matmul(H[i], X)/np.sum(H[i], axis = 0) for i in range(K)])
    return mean

def iteration_update_cov(X,H,mean):
    cov = []   
    for i in range(K):
        temp = np.zeros((2,2))
        for j in range(N):
            c = np.matmul((X[j] - mean[i])[:, None], (X[j] - mean[i])[None, :])*H[i,j]
            temp += c
        
        cov.append(temp / np.sum(H[k], axis = 0))
    return(cov)
    
def iteration_update_priors(H):
    prior = np.vstack([np.sum(H[i], axis = 0)/N for i in range(K)])
    return prior

def iteration_H(X, centroid, cov, prior):
    posterior = []
    for k in range(K):
        #print(centroids[k], class_cov[k]) 
        mix_density = multivariate_normal(centroids[k], class_cov[k]).pdf(X)*class_prior[k]
        posterior.append(mix_density)
    H = np.vstack([posterior[k]/np.sum(posterior, axis = 0) for k in range(K)])
    return H

def iteration_update_centorids(H, X):
    mean = np.vstack([np.matmul(H[k], X)/np.sum(H[k], axis = 0) for k in range(K)])
    return mean

def iteration_update_cov(X,H,mean):
    cov = []  
    for k in range(K):
        temp = np.zeros((2,2))
        for i in range(N):
            c = np.matmul((X[i] - mean[k])[:, None], (X[i] - mean[k])[None, :])*H[k,i]
            temp += c
        
        cov.append(temp / np.sum(H[k], axis = 0))
    return(cov)
    
def iteration_update_priors(H):
    prior = np.vstack([np.sum(H[k], axis = 0)/N for k in range(K)])
    return prior


# ## Visualization

# In[52]:


def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = "black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize = 12, 
                 markerfacecolor = cluster_colors[c], markeredgecolor = "black")
    plt.xlabel("x1")
    plt.ylabel("x2")


# ## Iterations

# In[59]:


centroids = None
memberships = None
centroids = update_centroids(memberships, X)
memberships = update_memberships(centroids, X)
class_means = centroids
class_sizes = [np.sum(memberships == c) for c in range(K)]
class_prior = [np.mean(memberships == c) for c in range(K)]
class_cov = []
for k in range(K):
    temp = np.zeros((2,2))
    for i in range(class_sizes[k]):
        cov = np.matmul(((X[memberships == k])[i,:] - centroids[k,:])[:, None], ((X[memberships == k])[i,:] - centroids[k,:][None, :]))
        temp += cov
    class_cov.append(temp / class_sizes[k])

for iteration in range(1,101):
    H = iteration_H(X,centroids,class_cov, class_prior)

    # M-step in EM algorithm:
    centroids = iteration_update_centorids(H, X)
    class_cov = iteration_update_cov(X, H, centroids)
    class_prior = (iteration_update_priors(H)).reshape(K,) # the function gives (5,1) vector but it has to be (5,) for EM_stepE function. I couldn't understand why this is the case. Anyway, reshape solved the problem.
    memberships=np.argmax(H, axis = 0) # Membership values are the index which maximize the h(ik)  

print("print(means)")
print(centroids)




# In[88]:


plt.figure(figsize=(8, 8))
cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99"])


x1, x2 = np.mgrid[-6:+6:.05, -6:+6:.05] # grid for plt.contour
pos = np.dstack((x1, x2))

for i in range(K):
    predicted_classes = multivariate_normal(centroids[i], class_cov[i] * 2).pdf(pos)
    test_classes = multivariate_normal(initial_means[i], initial_covariances[i] *2).pdf(pos)
    plt.contour(x1, x2, test_classes, levels=1, linestyles="dashed", colors="k")
    plt.contour(x1, x2, predicted_classes, levels=1, colors=cluster_colors[i])
    plt.plot(X[memberships == i, 0], X[memberships == i, 1], ".", markersize=10,
             color=cluster_colors[i])

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()



# In[ ]:




