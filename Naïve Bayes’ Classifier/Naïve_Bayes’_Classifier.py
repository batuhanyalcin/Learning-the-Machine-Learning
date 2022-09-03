#!/usr/bin/env python
# coding: utf-8

# # HW2 Naïve Bayes’ Classifier
# ## Batuhan Yalçın 64274
# ### March 19, 2022

# In[61]:


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
def safedivide(x):
    return((x + 1e-7))


# ## STEP 2

# ## Importing Data

# In[63]:


from PIL import Image
# read data into memory
images_set = np.genfromtxt("hw02_data_set_images.csv", delimiter = ",")
labels_set = np.genfromtxt("hw02_data_set_labels.csv", delimiter = ",")

#For me test prints
#print(images_set)
#print(labels_set)
#ys, xs = np.where(images_set.astype(bool))

#plt.imshow(images_set)
#plt.scatter(xs[::2], ys[::2])


# ## STEP 3

# ## Dividing the data set into two parts

# In[64]:


# First 25 data to the training rest of 25 to the test
# Seperate for each 5 class
images_training_set=[]
label_training_set=[]
images_test_set=[]
label_test_set=[]
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
#print(label_training_set)
#print(label_test_set)


# ## STEP 4

# ## Parameter Estimation

# In[65]:


# get number of classes
K = (len(np.unique(label_test_set)))
# get number of samples
N = images_training_set.shape[0]

values = Counter(label_training_set).values()
class_sizes= np.zeros(5)
for i in range(N):
    class_sizes[int(label_training_set[i]-1)]+=1

#Test Prints
#print(values)
#print(class_sizes)
#print(N)


# $\hat{\mu}_{c} = \dfrac{\sum\limits_{i = 1}^{N} x_{i} \mathbb{1}(y_{i} = c)}{\sum\limits_{i = 1}^{N} \mathbb{1}(y_{i} = c)}$

# In[66]:


# calculate sample means
sample_means =np.stack([np.mean(images_training_set[label_training_set == (c + 1)], axis=0) for c in range(K)])
pcd=sample_means


# $\hat{P}(y_{i} = c) = \dfrac{\sum\limits_{i = 1}^{N} \mathbb{1}(y_{i} = c)}{N}$

# In[67]:


# calculate prior probabilities
class_priors = [np.mean(label_training_set == (c + 1)) for c in range(K)]## Parametric Classification


# ## Answer of STEP 4

# In[68]:


print("\nprint(pcd)")
print(pcd)
print("\nprint(class_priors)")
print(class_priors)


# ## Parametric Classification

# In[69]:


score_values = np.array([0, 0, 0, 0, 0])
# evaluate score functions
def score_def(x,pcd,class_priors):
    score_values = np.array([0, 0, 0, 0, 0])
    #Uses given score Function
    for i in range(K):
        score_values[i] = np.sum(np.dot(np.transpose(x),safelog(pcd[i]))+np.dot((1-np.transpose(x)), safelog(1-pcd[i])))+np.log(class_priors[i])
    return score_values

#Gets maximum g_score values
def get_g_maxes(predicted_values,g_scores):
    for i in range(len(g_scores)):
        max_g=np.max(g_scores[i])
        if g_scores[i][0]==max_g:
            predicted_values.append(1)
        elif g_scores[i][1]==max_g:
            predicted_values.append(2)
        elif g_scores[i][2]==max_g:
            predicted_values.append(3)
        elif g_scores[i][3]==max_g:
            predicted_values.append(4)
        else :
            predicted_values.append(5)     
    predicted_values=np.array(predicted_values)
    return predicted_values

#Calls score_def to get gscores of the trainning set
trainnig_gscore = [score_def(images_training_set[i],pcd,class_priors) for i in range(np.shape(images_training_set)[0])]
#Calls score_def to get gscores of the test set
test_gscore = [score_def(images_test_set[i],pcd,class_priors) for i in range(np.shape(images_test_set)[0])]

#Calls get_g_maxes(which I used similiar function in HW1) to get maximum gscores for trainning set
trainning_prediction = get_g_maxes([],trainnig_gscore)
#Calls get_g_maxes(which I used similiar function in HW1) to get maximum gscores for test set
test_prediction = get_g_maxes([],test_gscore)


# 
# ## Training Performance

# In[70]:


# calculate confusion matrix of training perfornmance
confusion_matrix = pd.crosstab(trainning_prediction, label_training_set, rownames = ["y_pred"], colnames = ["y_truth"])
print("\nConfusion matrix for the data points in trainning set:\n")
print("print(confusion_matrix)")
print(confusion_matrix)


# ## Test Confusion Matrix

# In[71]:


# calculate confusion matrix of test perfornmans
confusion_matrix = pd.crosstab(test_prediction, label_test_set, rownames = ["y_pred"], colnames = ["y_truth"])
print("\nConfusion matrix for the data points in test set:\n")
print("print(confusion_matrix)")
print(confusion_matrix)


# In[ ]:




