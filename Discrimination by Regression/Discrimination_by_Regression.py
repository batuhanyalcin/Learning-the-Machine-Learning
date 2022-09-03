#!/usr/bin/env python
# coding: utf-8

# # Homework 03: Discrimination by Regression
# ## BATUHAN YALÇIN 64274
# ### March 25, 2022

# In[378]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import math
import warnings
from sklearn.metrics import confusion_matrix
from collections import Counter
def safelog(x):
    return(np.log(x + 1e-100))


# ## STEP 2

# ## Importing Data

# In[379]:


#This part same as homework 2
# read data into memory
images_set = np.genfromtxt("hw03_data_set_images.csv", delimiter = ",")
labels_set = np.genfromtxt("hw03_data_set_labels.csv", delimiter = ",")

#For me test prints
#print(images_set)
#print(labels_set)
#ys, xs = np.where(images_set.astype(bool))
#plt.imshow(images_set)
#plt.scatter(xs[::2], ys[::2])
#print(labels_set)


# ## STEP 3

# ## Dividing the data set into two parts

# In[380]:


#This part same as homework 2
# First 25 data to the training rest of 25 to the test
# Seperate for each 5 class
warnings.filterwarnings("ignore", category=DeprecationWarning) 
images_training_set=[]
label_training_set=[]
images_test_set=[]
label_test_set=[]
labels_set= labels_set.astype(np.int)
labels_set-=1
for i in range(5):
    # First 25 to the trainning
    for j in range(25):
        images_training_set.append(images_set[j+ 39*i])
        label_training_set.append(labels_set[j+ 39*i])
    # Rest of 14 to the test
    for k in range(14):
        images_test_set.append(images_set[25 +k+ 39*i])
        label_test_set.append(labels_set[25 +k+ 39*i])
images_training_set=np.array(images_training_set)
label_training_set=np.array(label_training_set)
images_test_set=np.array(images_test_set)
label_test_set=np.array(label_test_set)

# Test prints to get datas succesfully
#print(images_training_set)
#print(np.shape(label_training_set))
#print(label_test_set)


# ## Sigmoid Function

# In[381]:


def sigmoid(X, W, w0):
    return 1 / (1 + np.exp(-(np.matmul(X,W) + w0)))


# ## STEP 4

# ## Algorithm Parameters

# In[382]:


# set learning parameters
eta = 0.001
epsilon = 0.001


# ## Gradient Functions

# In[383]:


# define the gradient functions
def gradient_W(X, y_truth, y_predicted):
    return np.array([-np.matmul(
        (y_truth[:, c] - y_predicted[:, c]) * y_predicted[:, c] * (1 - y_predicted[:, c]), X)
                     for c in range(K)]).transpose()

def gradient_w0(y_truth, y_predicted):
    return -np.sum((y_truth - y_predicted) * y_predicted * (1 - y_predicted), axis=0)


# ## Parameter Initialization

# In[384]:


# get number of classes
K = (len(np.unique(label_test_set)))
# get number of samples
N = images_training_set.shape[0]
values = Counter(label_training_set).values()
class_sizes= values


# In[385]:


# randomly initalize W and w0
np.random.seed(521)
W = np.random.uniform(low = -0.01, high = 0.01, size = (images_training_set.shape[1], K))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, K))
#Z = sigmoid(np.matmul(np.hstack((np.ones((N,  1)), X)), W))
#y_predicted = sigmoid(np.matmul(np.hstack((np.ones((N,  1)), Z)), v))
#objective_values = -np.sum(y_truth * safelog(y_predicted) + (1 - y_truth) * safelog(1 - y_predicted))


# ## Iterative Algorithm

# \begin{align*}
# \Delta v_{h} &= \eta (y_{i} - \hat{y}_{i}) z_{ih} \\
# \Delta w_{hd} &= \eta (y_{i} - \hat{y}_{i}) v_{h} z_{ih} (1 - z_{ih}) x_{id}
# \end{align*}

# In[386]:


# learn W and w0 using gradient descent
iteration = 1
objective_values = []
Y_truth = np.zeros((len(images_training_set), K)).astype(int)
for i in range(len(images_training_set)):
    Y_truth[i][label_training_set[i]]=1
while 1:
    Y_predicted = sigmoid(images_training_set, W, w0)

    objective_values = np.append(objective_values, 0.5*np.sum((Y_truth-Y_predicted)**2))
    
    W_old = W
    w0_old = w0

    W = W - eta * gradient_W(images_training_set, Y_truth, Y_predicted)
    w0 = w0 - eta * gradient_w0(Y_truth, Y_predicted)

    if np.sqrt(np.sum((w0 - w0_old)**2) + np.sum((W - W_old)**2)) < epsilon:
        break
        
    iteration = iteration + 1

#print(np.shape(objective_values))
#print(np.shape(images_training_set))
#print(iteration)
#print(np.shape(Y_truth))
#print(np.shape(Y_predicted))
print("\nprint(W)")
print("print(w0)")
print(W)
print(w0)


# ## STEP 5

# ## Convergence

# In[387]:


# plot objective function during iterations
plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# ## STEP 6

# ## Training Performance

# In[388]:


# calculate confusion matrix
Y_predicted = np.argmax(Y_predicted, axis = 1) + 1
confusion_matrix = pd.crosstab(Y_predicted, label_training_set, rownames = ["y_pred"], colnames = ["y_truth"])
print("\nConfusion matrix for the data points in trainning set:\n")
print("print(confusion_matrix)")
print(confusion_matrix)


# ## STEP 7

# ## Test Confusion Matrix

# In[389]:


# calculate confusion matrix of test perfornmans
Y_test_prediction = sigmoid(images_test_set,W,w0)
Y_test_prediction = np.argmax(Y_test_prediction, axis = 1) + 1
confusion_matrix = pd.crosstab(Y_test_prediction, label_test_set+1, rownames = ["y_pred"], colnames = ["y_truth"])
print("\nConfusion matrix for the data points in test set:\n")
print("print(confusion_matrix)")
print(confusion_matrix)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




