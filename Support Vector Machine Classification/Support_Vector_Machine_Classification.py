#!/usr/bin/env python
# coding: utf-8

# # Homework 06: Support Vector Machine Classification
# ## Batuhan Yalçın
# ### April 29, 2022

# In[1]:


import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt


# ## Importing Data

# ## Step 1

# In[2]:


# read data into memory
images_data = np.genfromtxt("hw06_data_set_images.csv", delimiter = ",")
labels_data = np.genfromtxt("hw06_data_set_labels.csv", delimiter = ",")


# ## Step 2

# In[ ]:


train_images = images_data[:1000,:]
test_images = images_data[1000:,:]
train_label = labels_data[:1000]
test_label = labels_data[1000:]

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

# ## CALCULATE  Histogram Intersection Kernel

# In[31]:


bins = 64
min_value = 0;
max_value = 256
bin_width = (max_value- min_value)/bins 

left_borders = np.arange(min_value,max_value, bin_width)
right_borders = np.arange(min_value + bin_width, max_value + bin_width ,bin_width)

H_train = np.zeros((N_train, len(left_borders)))
H_test = np.zeros((N_train, len(left_borders)))

for i in range(N_train):
    
    phat_test = np.asarray([np.sum((left_borders[b] <= x_test[i,:]) & (x_test[i,:] < right_borders[b]))
                            for b in range(len(left_borders))]) / 784
    phat_train = np.asarray([np.sum((left_borders[b] <= x_train[i,:]) & (x_train[i,:] < right_borders[b]))
                             for b in range(len(left_borders))]) / 784
    
    H_test[i,:] = phat_test[:]
    H_train[i,:] = phat_train[:]
    
print("print(H_train[0:5,0:5])")
print("print(H_test[0:5,0:5])")
print(H_train[0:5,0:5])
print(H_test[0:5,0:5])


# ## STEP 4

# ## CALCULATE HISTOGRAMS

# In[33]:


def histogram_intersection(h1, h2):
    result = []
    for j in range(N_train):
        sums = []
        for k in range(N_train):
                sums.append(np.sum(np.minimum(h1[j], h2[k])))
        result.append(sums)
    result=np.array(result)
    return(result)

K_train = histogram_intersection(H_train, H_train)
K_test = histogram_intersection(H_test,H_train)

print("print(K_train[0:5,0:5])")
print("print(K_test[0:5,0:5])")
print(K_train[0:5,0:5])
print(K_test[0:5,0:5])


# ## Learning Algorithm

# #### Primal Problem
# $\begin{equation}
# 	\begin{split}
# 		\mbox{minimize}\;\;& \dfrac{1}{2} ||\boldsymbol{w}||_{2}^{2} + C \sum\limits_{i = 1}^{N} \xi_{i} \\
# 		\mbox{with respect to}\;\;& \boldsymbol{w} \in \mathbb{R}^{D},\;\; \boldsymbol{\xi} \in \mathbb{R}^{N},\;\; w_{0} \in \mathbb{R} \\
# 		\mbox{subject to}\;\;& y_{i} (\boldsymbol{w}^{\top} \boldsymbol{x}_{i} + w_{0}) \geq 1 - \xi_{i} \;\;\;\; i = 1,2,\dots,N \\
# 		& \xi_{i} \geq 0\;\;\;\; i = 1,2,\dots,N \\
# 		\mbox{where}\;\;& C \in \mathbb{R}_{+}
# 	\end{split}
# \end{equation}$
# 
# #### Dual Problem
# $\begin{equation}
# 	\begin{split}
# 		\mbox{maximize}\;\;& \sum\limits_{i = 1}^{N} \alpha_{i} - \dfrac{1}{2} \sum\limits_{i = 1}^{N} \sum\limits_{j = 1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j} k(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}) \\
# 		\mbox{with respect to}\;\;& \boldsymbol{\alpha} \in \mathbb{R}^{N} \\
# 		\mbox{subject to}\;\;& \sum\limits_{i = 1}^{N} \alpha_{i} y_{i} = 0 \\
# 		& 0 \leq \alpha_{i} \leq C\;\;\;\; i = 1,2,\dots,N \\
# 		\mbox{where}\;\;& C \in \mathbb{R}_{+}
# 	\end{split}
# \end{equation}$
# 
# #### Dual Problem in Matrix-Vector Form
# $\begin{equation}
# 	\begin{split}
# 		\mbox{minimize}\;\;&-\boldsymbol{1}^{\top} \boldsymbol{\alpha} + \dfrac{1}{2} \boldsymbol{\alpha}^{\top} ((y y^{\top}) \odot \mathbf{K}) \boldsymbol{\alpha} \\
# 		\mbox{with respect to}\;\; & \boldsymbol{\alpha} \in \mathbb{R}^{N} \\
# 		\mbox{subject to}\;\;& \boldsymbol{y}^{\top} \boldsymbol{\alpha} = 0 \\
# 		& \boldsymbol{0} \leq \boldsymbol{\alpha} \leq C \boldsymbol{1} \\
# 		\mbox{where}\;\;& C \in \mathbb{R}_{+}
# 	\end{split}
# \end{equation}$

# In[34]:


def f_learning(c):
    yyK = np.matmul(y_train[:,None], y_train[None,:]) * K_train

    # set learning parameters
    C = c
    epsilon = 0.001

    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N_train, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
    h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
    A = cvx.matrix(1.0 * y_train[None,:])
    b = cvx.matrix(0.0)

    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
    alpha = np.reshape(result["x"], N_train)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C

    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(y_train[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))

    # calculate predictions on training samples
    f_predicted_test = np.matmul(K_test, y_train[:,None] * alpha[:,None]) + w0
    y_predict_test = 2 * (f_predicted_test > 0.0) - 1
    
    f_predicted_train = np.matmul(K_train, y_train[:,None] * alpha[:,None]) + w0
    y_predict_train = 2 * (f_predicted_train > 0.0) - 1
    return y_predict_train, y_predict_test 


# ## Training Performance

# In[36]:


# calculate confusion matrix
y_predicted_train, y_predicted_test = f_learning(10)

confusion_matrix = pd.crosstab(np.reshape(y_predicted_train, N_train), y_train,
                               rownames = ["y_predicted"], colnames = ["y_train"])
print(confusion_matrix)

print()
confusion_matrix = pd.crosstab(np.reshape(y_predicted_test, N_train), y_test,
                               rownames = ["y_predicted"], colnames = ["y_test"])
print(confusion_matrix)


# ## STEP 6

# ## Train vector machines

# In[37]:


accuracies_of_train = []
accuracies_of_test = []
C_values = [pow(10, -1),pow(10, -0.5), pow(10, 0), pow(10, 0.5), pow(10, 1), pow(10, 1.5), pow(10, 2), pow(10, 2.5), pow(10, 3)]

for c in C_values:

    #finding predictions for training
    y_predicted_trains, y_predicted_tests = f_learning(c)
    #finding predictions for test
    
    # Accuracy counters
    cntr_train = 0
    cntr_test = 0
    for i in range(N_train):
        if y_train[i] == y_predicted_trains[i]:
            cntr_train +=1
        if y_test[i] == y_predicted_tests[i]:
            cntr_test +=1
            
    # Accuracy Calculator
    accuracy_train = (cntr_train / N_train)
    accuracy_test = (cntr_test / N_test)
    
    # Accuracy Storage 
    accuracies_of_train.append(accuracy_train)
    accuracies_of_test.append(accuracy_test)  
    
    


# ## Visualization

# In[39]:


plt.figure(figsize = (10, 4))
plt.plot(C_values, accuracies_of_train, marker = ".", markersize = 10, linestyle = "-", color = "b",label='train')
plt.plot(C_values, accuracies_of_test, marker = ".", markersize = 10, linestyle = "-", color = "r",label='test')
plt.xscale('log')
plt.legend(['training', 'test'])
plt.legend(loc='upper left')
plt.xlabel("Regularization parameter (C)")
plt.ylabel("Accuracy")
plt.show()


# In[ ]:




