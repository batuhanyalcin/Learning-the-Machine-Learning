#!/usr/bin/env python
# coding: utf-8

# # Homework 05: Decision Tree Regression
# ## Batuhan Yalçın
# ### April 14, 2022

# In[10]:


import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
def safelog(x):
    return(np.log(x + 1e-100))
def safelog2(x):
    if x == 0:
        return(0)
    else:
        return(np.log2(x))


# ## PART 1)

# ## Importing Data

# In[11]:


# read data into memory
data_set_train = np.genfromtxt("hw05_data_set_train.csv", delimiter = ",")
data_set_test = np.genfromtxt("hw05_data_set_test.csv", delimiter = ",")

# get x and y values
x_train = data_set_train[:,0]
y_train = data_set_train[:,1]

x_test = data_set_test[:,0]
y_test = data_set_test[:,1]

# get number of classes, number of samples, and number of features
K = np.max(y_train)
N = x_train.shape[0]
#D = x_train.shape[1]

# get numbers of train and test samples
N_train = len(y_train)
N_test = len(y_test)


# ## PART 2

# ## Initialization

# In[12]:


# create necessary data structures
node_indices = {}
is_terminal = {}
need_split = {}

node_features = {}
node_means = {}
node_splits = {}
node_frequencies = {}

# put all training instances into the root node
node_indices[1] = np.array(range(N_train))
is_terminal[1] = False
need_split[1] = True


# ## Tree Inference

# In[13]:


def learnTreeAlgorithm(x_train,y_train,P):
    while True:
        # find nodes that need splitting
        split_nodes = [key for key, value in need_split.items()
                       if value == True]
        # check whether we reach all terminal nodes
        if len(split_nodes) == 0:
            break
        # find best split positions for all nodes
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            node_mean = np.mean(y_train[data_indices])

            #Pruning
            if x_train[data_indices].size <=P:
                is_terminal[split_node] = True
                node_means[split_node] = node_mean

            else:
                is_terminal[split_node] = False
                x_sorted = np.sort(np.unique(x_train[data_indices]))
                split_positions = (x_sorted[1:len(x_sorted)] +x_sorted[0:(len(x_sorted)-1)])/2
                split_scores = np.repeat(0.0,len(split_positions))

                for s in range(len(split_positions)):
                    left_indices = data_indices[x_train[data_indices] < split_positions[s]]
                    right_indices = data_indices[x_train[data_indices] >= split_positions[s]]
                    total_err = 0
                    if len(left_indices)>0:
                        total_err += np.sum((y_train[left_indices] - np.mean(y_train[left_indices])) ** 2)
                    if len(right_indices)>0:
                        total_err += np.sum((y_train[right_indices] - np.mean(y_train[right_indices])) ** 2)
                    split_scores[s] = total_err/(len(left_indices)+len(right_indices))
                # If only one item then the mean is equal that one and it's terminal
                if len(x_sorted) == 1 : 
                    is_terminal[split_node] = True
                    node_means[split_node] = node_mean
                    continue
                # decide where to split on which feature
                best_split = split_positions[np.argmin(split_scores)]
                node_splits[split_node] = best_split

                # create left node using the selected split
                left_indices = data_indices[(x_train[data_indices] < best_split)]
                node_indices[2 * split_node] =left_indices
                is_terminal[2 * split_node]  = False
                need_split[2 * split_node] = True

                # create right node using the selected split
                right_indices = data_indices[(x_train[data_indices] >= best_split)]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1]  =True
    return is_terminal,node_splits,node_means

#Learning parameter
P=30
is_terminal,node_splits,node_means = learnTreeAlgorithm(x_train,y_train,P)


# ## PART 3

# ## LEARNING

# In[14]:



def Learning(is_terminal, node_splits, node_means,x_data, root):
    index = root
    while True:
        #Terminal
        if is_terminal[index] == True:
            return node_means[index]
        #Right child
        if x_data > node_splits[index]:
            index = index*2 + 1
        #Left chiild
        else:
            index = index*2  



learned = []
for x_data in x_test:
    learned.append(Learning(is_terminal,node_splits,node_means,x_data,1))
learned= np.array(learned)

learned_train = []
for x_data in x_train:
    learned_train.append(Learning(is_terminal,node_splits,node_means,x_data,1))
learned_train= np.array(learned_train)


# ## PLOT RESULT

# In[15]:


minimum_value = 0
maximum_value = max(x_train)
data_interval = np.arange(minimum_value,maximum_value,0.001)
fig = plt.figure(figsize=(15,5))
plt.plot(x_train, y_train, "b.", label = 'training', markersize = 10)
regressogram=[]
[regressogram.append(Learning(is_terminal,node_splits,node_means,data_interval[i],1)) for i in range(len(data_interval))]
plt.plot(data_interval,regressogram,color="black")
      
plt.ylabel("Signal (milivolt)")
plt.xlabel("Time (sec)")
plt.legend(loc='upper right')
plt.show()

fig = plt.figure(figsize=(15,5))
plt.plot(x_test, y_test, "r.", label = 'test', markersize = 10)
regressogram=[]
[regressogram.append(Learning(is_terminal,node_splits,node_means,data_interval[i],1)) for i in range(len(data_interval))]
plt.plot(data_interval,regressogram,color="black")
      
plt.ylabel("Signal (milivolt)")
plt.xlabel("Time (sec)")
plt.legend(loc='upper right')
plt.show()


# ## PART 4

# In[16]:


def rmse(truth,prediction):
    rmse=np.sqrt(np.mean((truth - prediction) ** 2))
    return rmse

rmse_test = rmse(y_test,learned)
rmse_train = rmse(y_train,learned_train)

print("RMSE on training set is",rmse_train,"when P is",  P)
print("RMSE on test set is",rmse_test,"when P is",  P)


# ## PART 5
# 

# ## Learn by setting several pre-pruning parameter

# In[17]:


rmse_train_values= []
rmse_test_values= []
for P in range(10,55,5):
    
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_features = {}
    node_means = {}
    node_splits = {}
    node_frequencies = {}

    # put all training instances into the root node
    node_indices[1] = np.array(range(len(x_train)))
    is_terminal[1] = False
    need_split[1] = True
    
    is_terminal,node_splits,node_means = learnTreeAlgorithm(x_train,y_train,P)
    
    learned = []
    for x_data in x_test:
        learned.append(Learning(is_terminal,node_splits,node_means,x_data,1))
    learned= np.array(learned)
    #print(learned)

    learned_train = []
    for x_data in x_train:
        learned_train.append(Learning(is_terminal,node_splits,node_means,x_data,1))
    learned_train= np.array(learned_train)
    
    rmse_test_values.append(rmse(y_test,learned))
    rmse_train_values.append(rmse(y_train,learned_train))
    
rmse_test_values = np.array(rmse_test_values)
rmse_train_values = np.array(rmse_train_values)


# ## PART 5 RESULT PLOT

# In[18]:


fig = plt.figure(figsize=(15,5))
X_Range = range(10,55,5)

plt.plot(X_Range,rmse_test_values, color= "red", marker = ".", markersize=10)
plt.scatter(X_Range,rmse_test_values, color="red", label="Test") 

plt.plot(X_Range,rmse_train_values, color= "blue", marker = ".", markersize=10)
plt.scatter(X_Range,rmse_train_values, color="blue", label="Training") 

plt.xlabel("Pre-pruning size(P)")
plt.ylabel("RMSE")
plt.legend(["training", "test"], loc='upper left')
plt.show()


# In[ ]:




