#!/usr/bin/env python
# coding: utf-8

# # Assignment 2
# ### Instructor: Vagelis Papalexakis
# ### Credit for  Assignment 2: 20/35 points of the final grade
# 
# In this assignment we will implement two different supervised learning models: 1) linear regression (using gradient descent), and 2) k-nearest neighbor classification. As we did in Assignment 1, here we will also use the Iris dataset. Below are some useful imports and some data bookkeeping:

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import random as rand
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
data_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
data = pd.read_csv('iris.data', names = data_names)


# ## Question 1: Linear Regression [50%]
# The first model we will implement is Linear Regression using Gradient Descent. 
# 
# ###  Getting data
# In order to properly test linear regression, we first need to find a set of correlated variables, so that we use one to predict the other. Consider the following scatterplots:

# In[73]:


sb.pairplot(data[['petal_length','sepal_length','label']], hue = 'label')


# We observe that sepal length and petal width for Iris-versicolor and Iris-virginica are reasonably correlated, so we are going to take those two variables for those two classes and use one to regress on the other.

# In[137]:


sub_data = data.loc[data['label'] != 'Iris-setosa', :]
y = sub_data['petal_length'].values
x = sub_data['sepal_length'].values
x = x.reshape(-1, 1)


# # Question 1a: Gradient descent for linear regression [40%]
# As we saw in class, here we will implement the gradient descent version of linear regression.
# In particular, the function implemented should follow the following format:
# ```python
# def linear_regression_gd(x,y,learning_rate = 0.00001,max_iter=10000,tol=pow(10,-5)):
# ```
# Where 'x' is the training data feature(s), 'y' is the variable to be predicted, 'learning_rate' is the learning rate used, 'max_iter' defines the maximum number of iterations that gradient descent is allowed to run, and 'tol' is defining the tolerance for convergence (which we'll discuss next).
# 
# The return values for the above function should be (at the least) 1) 'theta' which are the regression parameters, 2) 'all_cost' which is an array where each position contains the value of the objective function $J(\theta)$ for a given iteration, 3) 'iters' which counts how many iterations did the algorithm need in order to converge to a solution.
# 
# Gradient descent is an iterative algorithm; it keeps updating the variables until a convergence criterion is met. In our case, our convergence criterion is whichever of the following two criteria happens first:
# 
# - The maximum number of iterations is met
# - The relative improvement in the cost is not greater than the tolerance we have specified. For this criterion, you may use the following snippet into your code:
# ```python
# np.absolute(all_cost[it] - all_cost[it-1])/all_cost[it-1] <= tol
# ```

# In[138]:


#your code here
def linear_regression_gd(x, y, learning_rate, max_iter, tol):
    all_cost = []
    it = 0
    m = 1
    b = 0
    
    for it in range(max_iter):
        totalError = 0
        b_gradient = 0
        m_gradient = 0
        N = float(len(x))
        for i in range(len(x)):
            x_i = x[i][0]
            y_i = y[i]
            b_gradient += (y_i - ((m * x_i) + b))
            m_gradient += x_i * (y_i - ((m * x_i) + b))
            totalError += (y_i - (m * x_i + b)) ** 2
        all_cost.append(totalError / float(len(x)))
        m = m + (learning_rate * m_gradient)
        b = b + (learning_rate * b_gradient)
        
        if(it > 0):
            if(np.absolute(all_cost[it] - all_cost[it-1])/all_cost[it-1] <= tol):
                break
    theta = [m, b]
    
    return theta, all_cost, it

linear_regression_gd(x,y,learning_rate = 0.00001,max_iter=10000,tol=pow(10,-5))


# ### Question 1b: Convergence plots [10%]
# After implementing gradient descent for linear regression, we would like to test that indeed our algorithm converges to a solution. In order see this, we are going to look at the value of the objective/loss function $J(\theta)$ as a function of the number of iterations, and ideally, what we would like to see is $J(\theta)$ drops as we run more iterations, and eventually it stabilizes. 
# 
# As we discussed in class, the learning rate plays a big role in how fast our algorithm converges: a larger learning rate means that the algorithm is making faster strides to the solution, whereas a smaller learning rate implies slower steps. In this question we are going to test two different values for the learning rate:
# - 0.00001
# - 0.000001
# 
# while keeping the default values for the max number of iterations and the tolerance.
# 
# 
# - Plot the two convergence plots (cost vs. iterations) [5%]
# 
# - What do you observe? [5%]
# 
# <b>Important</b>: Remember that as we discussed in class, in reality, when we are running gradient descent, we should be checking convergence based on the <i>validation</i> error (i.e., we would have to split our training set into a e.g., 70/30 training'/validation subsets, use the new training' set to calculate the gradient descent updates and evaluate the error both on the training' set and the validation set, and as soon as the validation loss stops improving, we stop training. <b>In order to keep things simple, in this assignment we are only looking at the training loss</b>, but as long as you have a function 
# ```python
# def compute_cost(x,theta,y):
# ```
# that calculates the loss for a given x, y, and set of parameters you have, you can always compute it on the validation portion of x and y (that are <b>not</b> used for the updates).  

# In[139]:


#your code here
#This may also be useful: augment x with a new column for the bias term
#x_new = np.concatenate((np.ones(len(x)).reshape(-1, 1),x),axis=1)
# Plot line against data

plt.subplot(2, 1, 1)
theta, all_cost, it = linear_regression_gd(x,y,learning_rate = 0.00001,max_iter=10000,tol=pow(10,-5))
iterations = np.arange(it + 1)
plt.plot(iterations, all_cost)
plt.xlabel('iterations')
plt.ylabel('cost')

plt.subplot(2, 1, 2)
theta, all_cost, it = linear_regression_gd(x,y,learning_rate = 0.000001,max_iter=10000,tol=pow(10,-5))
iterations = np.arange(it + 1)
plt.plot(iterations, all_cost)
plt.xlabel('iterations')
plt.ylabel('cost')
plt.show()

# Don't know if I needed to make this function but made it just in case
def compute_cost(x,theta,y):
    cost = 0
    for i in range(len(x)):
        cost += (y[i] - (theta[0] * x[i][0] + theta[1])) ** 2
    return cost


# The smaller learning rate will require more iterations but both runs of gradient descent converge to a similar total cost

# ## Question 2: K-Nearest Neighbors Classifier [50%]
# The K-Nearest Neighbors Classifier is one of the most popular instance-based (and in general) classification models. In this question, we will implement our own version and test in different scenarios.
# 
# ### Question 2a: Implement the K-NN Classifier [30%]
# For the implementation, your function should have the format:
# ```python
# def knnclassify(test_data,training_data, training_labels, K=1):
# ```
# where 'test_data' contains test data points, 'training_data' contains training data points, 'training_labels' holds the training labels, and 'K' is the number of neighbors. 
# 
# The output of this function should be 'pred_labels' which contains the predicted label for each test data point (it should, therefore, have the same number of rows as 'test_data').

# The piece of code below prepares the Iris dataset by converting the labels from strings to integers (which is quite easier to move around and do calculations with):

# In[4]:


all_vals = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
all_labels = data['label'].values
unique_labels = np.unique(all_labels)
#change string labels to numbers
new_labels = np.zeros(len(all_labels))
for i in range(0,len(unique_labels)):
    new_labels[all_labels == unique_labels[i]] = i
all_labels = new_labels


# In[6]:


#your code here
from sklearn.model_selection import train_test_split
def knnclassify(test_data, training_data, training_labels, K):
    pred_labels = np.zeros(len(test_data))
    
    for i in range(len(test_data)):
        temp_labels = training_labels
        dist_arr = np.zeros(len(training_data))
        for j in range(len(training_data)):
            # find distance between test point and training point
            dist_arr[j] = sqrt((test_data[i][0] - training_data[j][0])**2 + (test_data[i][1] - training_data[j][1])**2)
        
        # find majority label
        label_majority = []
        for k in range(K):
            index = np.argmin(dist_arr)
            label_majority.append(temp_labels[index])
            dist_arr = np.delete(dist_arr, index)
            temp_labels = np.delete(temp_labels, index)
        counts = np.bincount(label_majority)
        label = np.argmax(counts)
        
        pred_labels[i] = label
    return pred_labels

(training_data, test_data, training_labels, test_labels) = train_test_split(all_vals, all_labels, test_size=0.3)
knnclassify(test_data, training_data, training_labels, K=3)


# ### Question 2b: Measuring performance [10%]
# 
# In this question you will have to evaluate the average performance of your classifier for different values of $K$. In particular, $K$ will range in $\{1,\cdots,8\}$. We are going to measure the performance using classification accuracy. For computing the accuracy, you may use
# ```python
# accuracy = sum(test_labels == pred_labels)/len(test_labels)
# ```
# where 'test_labels' are the actual class labels and 'pred_labels' are the predicted labels
# 
# 
# In order to get a proper estimate for the accuracy for every K, we need to run multiple iterations where for each iteration we get a different randomized split of our data into train and test. In this question, we are going to run 100 iterations for every K, and for every random splitting, you may use:
# 
# ```python
#     (training_data, test_data, training_labels, test_labels) = train_test_split(all_vals, all_labels, test_size=0.3)
# ```
# where the train/test ratio is 70/30. 
# 
# After computing the accuracy for every $K$ for every iteration, you will have 100 accuracies per $K$. The best way to store those accuracies is in a matrix that has as many rows as values for $K$ and 100 columns, each one for each iteration.
# 
# Compute average accuracy as a function of $K$. Because we have a randomized process, we also need to compute how certain/uncertain our estimation for the accuracy per $K$ is. For that reason, we also need to compute the standard deviation of the accuracy for every $K$. Having computed both average accuracy and standard deviation, make a figure that shows the average accuracy as a function of $K$ with each point of the figure being surrounded by an error-bar encoding the standard deviation. You may find 
# ```python
# plt.errorbar()
# ```
# useful for this plot.

# In[204]:


#your code here
from sklearn.model_selection import train_test_split
from numpy import zeros
Matrix = zeros([8,100])
avg_arr = []
std_arr = []
for k in range(1, 9):
    for i in range(100):
        (training_data, test_data, training_labels, test_labels) = train_test_split(all_vals, all_labels, test_size=0.3)
        pred_labels = knnclassify(test_data, training_data, training_labels, K=k)
        accuracy = sum(test_labels == pred_labels)/len(test_labels)
        Matrix[k-1][i] = accuracy
    avg_arr.append(np.mean(Matrix[k-1][:]))
    std_arr.append(np.std(Matrix[k-1][:]))

x = np.arange(8)
plt.figure()
plt.errorbar(x, avg_arr, xerr = None, yerr = std_arr)
plt.xlabel("K value")
plt.ylabel("Accuracy")


# ### Question 2c: Feature selection [10%]
# 
# We have extensively discussed in class the fact that a good or bad set of features can make or break our model! Here we will see what happens when we operate on a subset of the features, and in particular in
# - a subset that has good separability of classes
# - a subset that has poor separability of classes
# 
# Recall from Assignment 1 where we did the scatterplots of the Iris dataset that a pair of features with high visual separability is (petal length, sepal width), whereas a set that confuses at least two classes is (sepal length, sepal width). 

# In[9]:


sb.pairplot(data[['petal_length','sepal_width','sepal_length','label']], hue = 'label')


# Apply K-NN classification with K = 1 on two datasets (using the same train/test split for both datasets, and the same method you used to split as above) and measure the classification accuracy for:
# - Only (petal length, sepal width) [2.5%]
# - Only (sepal length, sepal width) [2.5%]
# 
# What do you observe regarding the classification accuracy? [5%]

# In[206]:


#your code here
all_vals = data[['sepal_width', 'petal_length']].values
all_labels = data['label'].values
unique_labels = np.unique(all_labels)
#change string labels to numbers
new_labels = np.zeros(len(all_labels))
for i in range(0,len(unique_labels)):
    new_labels[all_labels == unique_labels[i]] = i
all_labels = new_labels

avg_accuracy = [];
for i in range(100):
    (training_data, test_data, training_labels, test_labels) = train_test_split(all_vals, all_labels, test_size=0.3)
    pred_labels = knnclassify(test_data, training_data, training_labels, K=k)
    accuracy = sum(test_labels == pred_labels)/len(test_labels)
    avg_accuracy.append(accuracy)
print(np.mean(avg_accuracy))

all_vals = data[['sepal_width', 'sepal_length']].values
all_labels = data['label'].values
unique_labels = np.unique(all_labels)
#change string labels to numbers
new_labels = np.zeros(len(all_labels))
for i in range(0,len(unique_labels)):
    new_labels[all_labels == unique_labels[i]] = i
all_labels = new_labels

avg_accuracy = [];
for i in range(100):
    (training_data, test_data, training_labels, test_labels) = train_test_split(all_vals, all_labels, test_size=0.3)
    pred_labels = knnclassify(test_data, training_data, training_labels, K=k)
    accuracy = sum(test_labels == pred_labels)/len(test_labels)
    avg_accuracy.append(accuracy)
print(np.mean(avg_accuracy))


# Petal length and sepal width are much easier to classify than sepal length and sepal width. This could be because the points for sepal width vs petal length are more spread out and therefore easier to define. Sepal width and sepal length are more clustered together and therefore make it less accurate
