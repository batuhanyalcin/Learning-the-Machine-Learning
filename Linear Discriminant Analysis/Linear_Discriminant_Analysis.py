#!/usr/bin/env python
# coding: utf-8

# # Homework 07: Linear Discriminant Analysisn
# ## Batuhan Yalçın
# ### May 6, 2022

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
from scipy import stats
from scipy.spatial import distance
import pandas as pd
#from sklearn.metrics import confusion_matrix


# ## Importing Data

# ## Step 1

# In[2]:


# read data into memory
images_data = np.genfromtxt("hw07_data_set_images.csv", delimiter = ",")
labels_data = np.genfromtxt("hw07_data_set_labels.csv", delimiter = ",")


# ## Step 2

# In[3]:


train_images = images_data[:2000,:]
test_images = images_data[2000:,:]
train_label = labels_data[:2000]
test_label = labels_data[2000:]

# get x and y values
x_train = train_images
y_train = train_label.astype(int)

x_test = test_images
y_test = test_label.astype(int)

# get number of classes, number of samples, and number of features
K = int(np.max(train_label))
#N = x_train.shape[0]
D = x_train.shape[1]

# get numbers of train and test samples
N_train = len(y_train)
N_test = len(y_test)


# ## STEP 3

# ## Calculate Sw and SB

# In[4]:


sample_means = [np.mean(x_train[y_train==(c + 1)], axis = 0) for c in range(K)]
point_mean = np.mean(sample_means,axis=0)
np.shape(sample_means)
mean_overall = np.mean(x_train, axis=0)

SW = np.zeros((784,784))
SB = np.zeros((784, 784))
class_labels = np.unique(y_train)

for c in class_labels:
    X_c = x_train[y_train == c]
    mean_c = np.mean(X_c, axis=0)
    
    SW += (X_c-mean_c).T.dot((X_c-mean_c))
    
    n_c = X_c.shape[0]
    mean_diff = (mean_c - mean_overall).reshape(784, 1)
    SB += n_c * (mean_diff).dot(mean_diff.T)
    
print("print(SW[0:5, 0:5])")
print("print(SB[0:5, 0:5])")    
print(SW[0:5, 0:5])
print(SB[0:5, 0:5])


# ## STEP 4

# ## Calculate the largest nine eigenvalues

# ## Principal Component Analysis

# In[5]:


SWSB_Matrix = np.linalg.inv(SW).dot(SB)
# Get eigenvalues and eigenvectors of SW^-1 * SB
eigenvalues, eigenvectors = np.linalg.eig(SWSB_Matrix)
eigenvectors = eigenvectors.T


# sort eigenvalues largest to smallest
idxs = np.argsort(abs(eigenvalues))[::-1]
eigenvectors = eigenvectors[idxs]
vectors = np.real(eigenvectors)

#Get largest eigenvalues
eigenvalues = eigenvalues[idxs]
values = np.real(eigenvalues)


print("\nprint(values[0:9])")
print(values[0:9])
## Principal Component Analysis


# ## STEP 5

# ## Plotting

# In[6]:


# get two eigenvectors that correspond to the largest two eigenvalues
linear_discriminants = eigenvectors[0 : 2]
# calculate two-dimensional projections for train
X_projected = np.dot(x_train, linear_discriminants.T)


# In[7]:


# plot two-dimensional projections using lab taslak
Z = np.real(X_projected)

# plot two-dimensional projections
plt.subplots(figsize=(16,6))
plt.subplot(1, 2, 1)
point_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])
for c in range(K):
    plt.plot(Z[y_train == c + 1, 0], Z[y_train == c + 1, 1], marker = "o", markersize = 4, linestyle = "none", color = point_colors[c])
plt.legend(["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", 
"bag", "ankle boot"],
           loc = "upper left", markerscale = 2)
plt.xlabel("Component#1")
plt.ylabel("Component#2")
plt.xlim(-6, 6)
plt.ylim(-6, 6)

# calculate two-dimensional projections for test
X_projected_test = np.dot(x_test, linear_discriminants.T)
Z_test = np.real(X_projected_test)

plt.subplot(1, 2, 2)
for c in range(K):
    plt.plot(Z_test[y_test == c + 1, 0], Z_test[y_test == c + 1, 1], marker = "o", markersize = 4, linestyle = "none", color = point_colors[c])
plt.legend(["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", 
"bag", "ankle boot"],
           loc = "upper left", markerscale = 2)
plt.xlabel("Component#1")
plt.ylabel("Component#2")
plt.xlim(-6, 6)
plt.ylim(-6, 6)

plt.show()


# ## STEP 6

# 

# In[8]:


## Project of the training and test data points in a 9-dimensional subspace
linear_discriminants_9 = eigenvectors[0 : 9]
X_projected_9 = np.dot(x_train, linear_discriminants_9.T)
Z_9 = np.real(X_projected_9)
X_projected_test_9 = np.dot(x_test, linear_discriminants_9.T)
Z_test_9 = np.real(X_projected_test_9)## STEP 5## STEP 6


# ## Slower solver

# In[9]:


#Slover solver using own implemntation of KNN
def KNN_algorithm(X1_set,X2_set,N_train,K):
    Y_hat = []
    for i in range(N_train):
        test_set = X1_set[i,:]
        initial_distances = np.zeros(X2_set.shape[0])
        for j in range(N_train):
            initial_distances[j] = distance.euclidean(test_set, X2_set[j, :])
        smallest_dists_indices = np.argsort(initial_distances)[:K]
        
        temp_classes = []
        for k in smallest_dists_indices:
            temp_classes.append(y_train[k])
        prediction= stats.mode(temp_classes)[0]
        Y_hat.append(prediction)
        
    return Y_hat


# ## Training Performance

# In[10]:


print("print(confusion_matrix_test)")
y_hat_train_predict = KNN_algorithm(Z_9,Z_9,N_train,11)
confusion_matrix_train = pd.crosstab(np.reshape(y_hat_train_predict, N_train), y_train,
                               rownames = ["y_hat"], colnames = ["y_train"])
print(confusion_matrix_train)

y_hat_test = KNN_algorithm(Z_test_9,Z_9,N_train,11)
confusion_matrix_test = pd.crosstab(np.reshape(y_hat_test, N_train), y_test,
                               rownames = ["y_hat"], colnames = ["y_test"])
print("print(confusion_matrix_test)")
print(confusion_matrix_test)


# In[ ]:




