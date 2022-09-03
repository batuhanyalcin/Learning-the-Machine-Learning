#!/usr/bin/env python
# coding: utf-8

# # HW01: Multivariate Parametric Classification
# ## Batuhan Yalçın
# ### March 12, 2022

# In[418]:


import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd


# ## Parameters

# In[458]:


np.random.seed(521)

# mean parameters
class_means = np.array([[0.0, 4.5], [-4.5, -1.0], [4.5, -1.0], [0.0, -4.0]])

# standard deviation parameters
class_deviations = np.array([[[3.2, 0.0,], [0.0, 1.2]],
                            [[1.2, 0.8], [0.8, 1.2]],
                            [[1.2, -0.8], [-0.8, 1.2]],
                            [[1.2, 0.0], [0.0, 3.2]]]
                             )

# sample sizes
class_sizes = np.array([105, 145, 135, 115])


# ## Data Generation

# In[420]:


# generate random samples
# note that it should not be normal it should be multivariate_normal
points1 = np.random.multivariate_normal(class_means[0,:], class_deviations[0,:,:], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1,:], class_deviations[1,:,:], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2,:], class_deviations[2,:,:], class_sizes[2])
points4 = np.random.multivariate_normal(class_means[3,:], class_deviations[3,:,:], class_sizes[3])

#Generate Points arrays in sequence vertically
points = np.concatenate((points1, points2, points3,points4))

# generate corresponding labels
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2]), np.repeat(4, class_sizes[3])))


# ## Step 2 Plotting the Data

# In[421]:


print("\nStep 2 solution\n")
# given plot size
plt.figure(figsize = (8, 8))
plt.plot(points1[:,0], points1[:,1], "r.", markersize = 10)
plt.plot(points2[:,0], points2[:,1], "g.", markersize = 10)
plt.plot(points3[:,0], points3[:,1], "b.", markersize = 10)
plt.plot(points4[:,0], points4[:,1], "m.", markersize = 10)
plt.xlabel("x1"),plt.ylabel("x2"),plt.show()


# ## Exporting Data

# In[422]:


# write data to a file
np.savetxt("64274_hw01_data_set.csv", np.hstack((points, y[:, None])), fmt = "%f,%f,%d")


# ## Importing Data

# In[423]:


# read data into memory
data_set = np.genfromtxt("64274_hw01_data_set.csv", delimiter = ",")

# get x and y values
x = data_set[:,[0, 1]]
y = data_set[:,2].astype(int)

# get number of classes and number of samples
K = np.max(y)
N = data_set.shape[0]


# ## Parameter Estimation

# $\hat{\mu}_{c} = \dfrac{\sum\limits_{i = 1}^{N} x_{i} \mathbb{1}(y_{i} = c)}{\sum\limits_{i = 1}^{N} \mathbb{1}(y_{i} = c)}$

# In[450]:


# calculate sample means
sample_means =np.stack([np.mean(x[y == (c + 1)], axis=0) for c in range(K)])


# $\hat{\Sigma} = \dfrac{\sum\limits_{i = 1}^{N} (x_{i} - \widehat{\mu_{c}}) (x_{i} - \widehat{\mu_{c}})^{T}}{N}$
# 

# In[451]:


# calculate sample covariances
sample_covariances = [
    (np.matmul(np.transpose(x[y == (c + 1)] - sample_means[c]), 
               (x[y == (c + 1)] - sample_means[c])) / class_sizes[c]) for c in range(K)]


# $\hat{P}(y_{i} = c) = \dfrac{\sum\limits_{i = 1}^{N} \mathbb{1}(y_{i} = c)}{N}$

# In[452]:


# calculate prior probabilities
class_priors = [np.mean(y == (c + 1)) for c in range(K)]## Parametric Classification


# ## Step 3 Prints

# In[455]:


# Print of the sample means
print("\nStep 3 solution\n")
print("\n   Print of the sample means\n")
print(sample_means,"\n")
#Print of sample covariances 
print("\n   Print of sample covariance\n")
for sample_covariance in sample_covariances:
    print(sample_covariance,"\n")
#Print of prior probabilities 
print("\n   Print of prior probabilities \n")
print(class_priors,"\n")


# ## Parametric Classification

# In[428]:


data_interval = np.linspace(-8, +8, 1401)
#From this point I don't know packages for the confusion matrix but I learned from ethernet so I will go backfarward from imlplentation
#confusion_matrix = pd.crosstab(predicted_values, y_truth, rownames=['y_pred'], colnames=['y_truth'])

# The y_truth is our y predicted_values should calculate
#For predicted_values gscore should calculate for that score should calculate
#To calculate score values we need calculate Wc wc and wco
#Since I couldn't add the formula as you added I couldn't comment it

Wc = np.array([np.linalg.inv(sample_covariances[c]) / -2 for c in range(K)])
wc = np.array([np.matmul(np.linalg.inv(sample_covariances[c]), sample_means[c]) for c in range(K)])
wc0 = np.array([-(np.matmul(np.matmul(np.transpose(sample_means[c]), np.linalg.inv(sample_covariances[c])),
                             sample_means[c])) / 2 - np.log(np.linalg.det(sample_covariances[c])) / 2
                 + np.log(class_priors[c]) for c in range(K)])


# ## Score Functions

# In[429]:


#Function to get maxiumum g scores 
def get_g_maxes(xx):
    # Calculate g_scores for each point
    g = np.stack(
        # Calculate scores for each class
        np.matmul(np.matmul(np.transpose(xx), Wc[c]), xx) + np.matmul(np.transpose(wc[c]), xx) + wc0[c]
        for c in range(K))
    # Get maximum values
    g_max = max(g[0], g[1], g[2], g[3])
    # Checking g_max in loop for each g_scores then update predicted
    if g_max == g[0]:
        return 1
    elif g_max == g[1]:
        return 2
    elif g_max == g[2]:
        return 3
    else:
        return 4

predicted_values = []

#Below loop calculates g scores than adds to maximum ones to the predicted values
predicted_values = np.array([get_g_maxes(x[i])for i in range(len(x))])


# ## Step 4 Print

# In[430]:


print("\nStep 3 solution\n")
confusion_matrix = pd.crosstab(predicted_values, y, rownames=['y_pred'], colnames=['y_truth'])
print(confusion_matrix)


# ## Posteriors

# In[457]:


print("\nStep 5 solution\n")
# Define Interval
x1_interval = np.linspace(-8, +8, 333)
x2_interval = np.linspace(-8, +8, 333)
# open the grid
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)

# Intilize descirimant values matrix for more efficient calculatin
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))
# Calculating discriminant values for each class 
for c in range(K):
    discriminant_values[:, :, c] = (Wc[c, 0, 0] * x1_grid ** 2) + (Wc[c, 0, 1] * x1_grid * x2_grid) + (
               wc[1, 0] * x2_grid * x1_grid) + (wc[1, 1] * x2_grid ** 2) + (wc[c, 0] * x1_grid) + (
                                             wc[c, 1] * x2_grid) + wc0[c]

#Since I coulnd't succesfully implement counterf I used this function to pair implement
#basicly while zipping the x1_grid and x2_gird pairs them with the predicted values
data = np.array([get_g_maxes([x1, x2]) for x1, x2 in zip(np.ravel(x1_grid), np.ravel(x2_grid))])
Data = data.reshape(x1_grid.shape)

A = discriminant_values[:, :, 0]
B = discriminant_values[:, :, 1]
C = discriminant_values[:, :, 2]
D = discriminant_values[:, :, 3]

A[(A < B) & (A < C) & (A < D)] = np.nan
B[(B < A) & (B < C) & (B < D)] = np.nan
C[(C < A) & (C < B) & (C < D)] = np.nan
D[(D < A) & (D < B) & (D < C)] = np.nan


discriminant_values[:, :, 0] = A
discriminant_values[:, :, 1] = B
discriminant_values[:, :, 2] = C
discriminant_values[:, :, 3] = D

# From know on things increase since 4 class define 5 boundiers so I use y_truth instead of y
y_truth=y
plt.figure(figsize=(8, 8))
plt.plot(x[y_truth == 1, 0], x[y_truth == 1, 1], "r.", markersize=10)
plt.plot(x[y_truth == 2, 0], x[y_truth == 2, 1], "g.", markersize=10)
plt.plot(x[y_truth == 3, 0], x[y_truth == 3, 1], "b.", markersize=10)
plt.plot(x[y_truth == 4, 0], x[y_truth == 4, 1], "m.", markersize=10)


#Counter plot
plt.contour(x1_grid, x2_grid, Data, 0, colors="m")
plt.xlim([-8, 8])
plt.ylim([-8, 8])
plt.imshow(Data, origin='lower', extent=[-8, 8, -8, 8],cmap="gist_rainbow", alpha=0.4)
plt.imshow

# Inccorrect predictions
plt.plot(x[predicted_values != y, 0], x[predicted_values != y, 1], "ko", markersize=12, fillstyle="none")

#3!-1 boundries there should be 5 boundries
#boundriy between class 1 and 2
#plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 1], levels=0, colors="k")
#plt.contourf(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 1], levels=0, colors=["g", "r"], alpha=0.3)
#boundriy between class 1 and 3
#plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 2], levels=0, colors="k")
#plt.contourf(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 2], levels=0, colors=["b", "r"], alpha=0.3)
#boundriy between class 1 and 4
#plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 3], levels=0, colors="k")
#plt.contourf(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 3], levels=0, colors=["m", "r"], alpha=0.3)
#boundriy between class 2 and 3
#plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 1] - discriminant_values[:, :, 2], levels=0, colors="k")
#plt.contourf(x1_grid, x2_grid, discriminant_values[:, :, 1] - discriminant_values[:, :, 2], levels=0, colors=["b", "g"], alpha=0.3)
#boundriy between class 2 and 3
#plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 1] - discriminant_values[:, :, 3], levels=0, colors="k")
#plt.contourf(x1_grid, x2_grid, discriminant_values[:, :, 1] - discriminant_values[:, :, 3], levels=0, colors=["m", "g"], alpha=0.3)
#boundriy between class 3 and 4
#plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 2] - discriminant_values[:, :, 1], levels=0, colors="k")                                                                                 
#plt.contourf(x1_grid, x2_grid, discriminant_values[:, :, 2] - discriminant_values[:, :, 3], levels=0, colors=["m", "b"], alpha=0.3)

plt.xlabel("x1"), plt.ylabel("x2"), plt.show()


# In[459]:


print("Summary: it was very simliar to the lab01 however, plotting was not there and it was very hard then lab3 puted it helped lot")
print("Emotinal summary: it was very enjoyable until last plot, last plot gets little annoying but at the and final satisfying fix it")
print("Thank you lot for homework!")


# In[ ]:




