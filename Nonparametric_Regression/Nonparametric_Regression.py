#!/usr/bin/env python
# coding: utf-8

# # Homework 04: Nonparametric Regression
# ## Batuhan Yalçın
# ### April 8, 2022

# In[12]:


import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
def safelog(x):
    return(np.log(x + 1e-100))


# ## Part 2

# ## Importing Data

# In[13]:


# read data into memory
data_set_train = np.genfromtxt("hw04_data_set_train.csv", delimiter = ",")
data_set_test = np.genfromtxt("hw04_data_set_test.csv", delimiter = ",")

# get x and y values
x_train = data_set_train[:,0]
y_train = data_set_train[:,1]

x_test = data_set_test[:,0]
y_test = data_set_test[:,1]
# get number of classes and number of samples
K = np.max(y_train)
N = data_set_train.shape[0]


# ## PART 3 Learn a regressogram

# In[14]:


# Assign given parameters:
bin_width = 0.1
origin = 0


# In[15]:


point_colors = np.array(["red", "blue"])
minimum_value = 0
maximum_value = max(x_train)
data_interval = np.linspace(minimum_value, maximum_value,1601)


# In[16]:


left_borders = np.arange(minimum_value, maximum_value -bin_width, bin_width)
right_borders =np.arange(minimum_value + bin_width, maximum_value, bin_width)

regressogram = np.zeros(len(left_borders))

for b in range(len(left_borders)):
    conditionalpart=((left_borders[b] < x_train) & (x_train <= right_borders[b]))
    regressogram[b] = np.sum(conditionalpart*y_train) / np.sum(conditionalpart)
    

plt.figure(figsize = (10, 6))
plt.plot(x_train,y_train, "b.",  markersize = 10, label="Training")


for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [regressogram[b], regressogram[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [regressogram[b], regressogram[b + 1]], "k-")  
      
plt.xlabel("Signal (milivolt)")
plt.ylabel("Time (sec)")
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize = (10, 6))

plt.plot(x_test,y_test, "r.",  markersize = 10, label="Test")

for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [regressogram[b], regressogram[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [regressogram[b], regressogram[b + 1]], "k-")  
      
plt.xlabel("Signal (milivolt)")
plt.ylabel("Time (sec)")
plt.legend(loc='upper right')
plt.show()


# ## PART 4 Calculate RMSE

# In[17]:


rmse1 = 0
for i in range(len(left_borders)):
    for j in range(len(y_test)):        
        if ((left_borders[i] < x_test[j]) & (x_test[j] <= right_borders[i])):
            rmse1 = rmse1 + (y_test[j] - regressogram[int((x_test[j] - origin)/bin_width)])**2
            
rmse1 = np.sqrt(rmse1/len(x_test))
#print("Regressogram => RMSE is "+str(rmse1)+" when h is "+str(bin_width))

#RMSE corrected
rmse0 = 0
for i in range(y_test.shape[0]):
    for j in range(left_borders.shape[0]):
        if(left_borders[j] < x_test[i] and x_test[i] <= right_borders[j]):
            err = (y_test[i] - regressogram[j])**2
            rmse0 += err
rmse0=np.sqrt(rmse0 / len(x_test))
print("Regressogram => RMSE is "+str(rmse0)+" when h is "+str(bin_width))


# ## PART 5 Learn a running mean smoother

# In[18]:


mean_smooth = np.asarray([ np.sum( (np.abs( (x - x_train)/bin_width ) <=  0.5 )*y_train  ) / (np.sum( (np.abs( (x - x_train)/bin_width ) <=0.5))) for x in data_interval])


plt.figure(figsize = (10, 6))
plt.plot(x_train,y_train, "b.",  markersize = 10, label="Training")
plt.plot(data_interval, mean_smooth, "k-")
plt.xlabel("Signal (milivolt)")
plt.ylabel("Time (sec)")
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize = (10, 6))
plt.plot(x_test,y_test, "r.",  markersize = 10, label="Test")
plt.plot(data_interval, mean_smooth, "k-")
plt.xlabel("Signal (milivolt)")
plt.ylabel("Time (sec)")
plt.legend(loc='upper right')
plt.show()


# ## PART 6 Calculated RMSE of mean smoother for test data

# In[19]:


#Optimum solution
m_s_rmse = np.array([np.sum((abs((x - x_train) / bin_width)<= 0.5) * y_train)
                     / np.sum((abs((x - x_train) / bin_width)<= 0.5)) for x in x_test])

original_rmse2 = np.sqrt(np.sum((y_test - m_s_rmse) ** 2) / len(y_test))

#Data which adapted data interval
rmse2 = 0
for j in range(len(y_test)):

          rmse2 = rmse2 + (y_test[j] - mean_smooth[ int((x_test[j] - origin) / ((maximum_value - origin)/(1601)))] )**2
        
rmse2 = np.sqrt(rmse2/len(x_test))
#print("Original")
print("Regressogram => RMSE is "+str(original_rmse2)+" when h is "+str(bin_width))
#print("Adapted")
#print("Regressogram => RMSE is "+str(rmse2)+" when h is "+str(bin_width))


# ## PART 7 Learn a kernel smoother

# In[20]:


bin_width = 0.02
# defining K(u) function for simplicity.
def K_u_(u):
    return (1/np.sqrt(2*np.pi)) * np.exp(-0.5*u**2)


kernel_smooth = np.zeros(len(data_interval))

for i in range(len(data_interval)):
    
    kernel_smooth[i] = np.sum( K_u_((data_interval[i] - x_train)/bin_width ) * y_train) / np.sum(K_u_((data_interval[i] - x_train)/bin_width ))


plt.figure(figsize = (10, 6))
plt.plot(x_train,y_train, "b.",  markersize = 10, label="Training")
plt.plot(data_interval, kernel_smooth, "k-")
plt.xlabel("Signal (milivolt)")
plt.ylabel("Time (sec)")
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize = (10, 6))
plt.plot(x_test,y_test, "r.",  markersize = 10, label="Test")
plt.plot(data_interval, kernel_smooth, "k-")
plt.xlabel("Signal (milivolt)")
plt.ylabel("Time (sec)")
plt.legend(loc='upper right')
plt.show()


# ## PART 8 Calculated RMSE of kernel smoother for test data points

# In[21]:


rmse3 = 0

k_smoother_rmse = np.array([np.sum(K_u_((x - x_train) / bin_width) * y_train)
                                 / np.sum(K_u_((x - x_train) / bin_width)) for x in x_test])


rmse3 = np.sqrt(np.sum((y_test - k_smoother_rmse) ** 2) / len(y_test))

print("Regressogram => RMSE is "+str(rmse3)+" when h is "+str(bin_width))


# In[ ]:





# In[ ]:




