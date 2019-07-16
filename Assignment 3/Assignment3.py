#!/usr/bin/env python
# coding: utf-8

# # Assignment 3
# ### Instructor: Vagelis Papalexakis
# ### Credit for  Assignment 3: 10/35 points of the final grade
# 
# In this assignment we will implement the K-means clustering algorithm. We are going to use the same dataset as in the previous two assignments (<b>Note</b>: make sure you copy the dataset from Assignment 1 to the folder of this assignment!).

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import random as rand
from sklearn.model_selection import train_test_split


data_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
data = pd.read_csv('iris.data', 
                   names = data_names)


# ## Question 1: Implementing and testing K-means clustering [100%]
# ### Question 1a: Implementing K-Means clustering [50%]
# In this question you should implement a function that performs k-means clustering, using the Euclidean distance (you may use Numpy libraries for the distance computation). For calculation of the centroid you should use the 'mean' function.
# 
# For uniformity, you should implement a function with the following specifications:
# ```python
# def kmeans_clustering(all_vals,K,max_iter = 100, tol = pow(10,-3) ):
# ```
# where 1) 'all_vals' is the $N \times M$ matrix that contains all data points ($N$ is the number of data points and $M$ is the number of features, each row of the matrix is a data point), 2) 'K' is the number of clusters, 3) 'max_iter' is the maxium number of iterations, and 4) 'tol' is the tolerance for the change of the sum of squares of errors that determines convergence.
# 
# Your function should return the following variables: 1) 'assignments': this is a $N\times 1$ vector (where $N$ is the number of data points) where the $i$-th position of that vector contains the cluster number that the $i$-th data point is assigned to, 2) 'centroids': this is a $K\times M$ matrix, each row of which contains the centroid for every cluster, 3) 'all_sse': this is a vector that contains all the sum of squares of errors per iteration of the algorithm, and 4) 'iters': this is the number of iterations that the algorithm ran.
# 
# Here we are going to implement the simplest version of K-means, where the initial centroids are chosen entirely at random among all the data points.
# 
# As we saw in class, the K-means algorithm iterates over the following steps:
# - Given a set of centroids, assign all data points to the cluster represented by its nearest centroid (according to Euclidean distance)
# - Given a set of assignments of points to clusters, compute the new centroids for every cluster, by taking the mean of all the points assigned to each cluster.
# 
# Your algorithm should converge if 1) the maximum number of iteratiosn is reached, or 2) if the SSE between two consecutive iterations does not change a lot (as in the gradient descent for linear regression we saw in Assignment 2). In order to check for the latter condition, you may use the following piece of code:
# ```python
# if np.absolute(all_sse[it] - all_sse[it-1])/all_sse[it-1] <= tol
# ```
# 
# In order to calculate the SSE (sum of squares of error) first you need to define what an 'error' is. In k-means, error per data point refers to the Euclidean distance of that particular point from its assigned centroid. SSE sums up all those squared Euclidean distances for all data points and comes up with a number that reflects the total error of approximating every data points by its assigned centroid.
# 
# 
# 

# In[144]:


#k-means clustering
from numpy import zeros
import random
def kmeans_clustering(all_vals,K,max_iter = 100, tol = pow(10,-3)):
    
    shape = all_vals.shape
    N = shape[0]
    M = shape[1] - 1 # subtract 1 to exclude label
    centroids = zeros([K,M])
    iters = 0
    assignments = zeros([N])
    all_sse = []
    
    
    # Gets random points from sample 
    randIndex = random.sample(range(N), K)
    randPoints = [0] * K
    for i in range(K):
        randPoint = all_vals.loc[randIndex[i], "sepal_length":"petal_width"]
        randPoints[i] = randPoint
        for j in range(len(randPoint)):
            centroids[i][j] = randPoint[j]
    
    for it in range(max_iter):
        error = 0
        
        # Find smallest distance from cluster point to each data point
        for n in range(N):

            # Data row (point)
            point = all_vals.loc[n, "sepal_length":"petal_width"]
            
            # Distance is an array of distances from point to a cluster point
            Distance = [0] * K
            for i in range(len(randPoints)):
                Distance[i] = np.linalg.norm(point - centroids[i])
            
            assignments[n] = Distance.index(min(Distance))
            error += (min(Distance) ** 2)

        all_sse.append(error / float(N))

        if(it > 0):
            if(np.absolute(all_sse[it] - all_sse[it-1])/all_sse[it-1] <= tol):
                break
                
        # Compute the new centroids for every cluster, by taking the mean of all the points assigned to each cluster
        totalSum = [0] * K
        count = [0] * K
        for n in range(N):
            cluster = assignments[n]
            totalSum[int(cluster)] += all_vals.loc[n, "sepal_length":"petal_width"]
            count[int(cluster)] += 1
        mean = totalSum
        for i in range(K):
            if(count[i] == 0):
                centroids[i] = 0
            else:
                mean[i] = mean[i] / count[i]
                centroids[i] = mean[i]
    
    return assignments, centroids, all_sse, it


kmeans_clustering(all_vals = data, K = 3, max_iter = 100, tol = pow(10,-3))


# ### Question 1b: Visualizing K-means [10%]
# In this question we wll visualize the result of the K-means algorithm. For ease of visualization, we will focus on a scatterplot of two of the four features of the Iris dataset. In particular: run your K-means code with K=3 and default values for the rest of the inputs. Subsequently, make a single scatterplot that contains all data points of the dataset for features 'sepal_length' and 'petal_length' and color every data point according to its cluster assignment.

# In[118]:


assignments, centroids, all_sse, it = kmeans_clustering(all_vals = data, K = 3, max_iter = 100, tol = pow(10,-3))
print(assignments)
x = data["sepal_length"]
y = data["petal_length"]
for i in range(len(x)):
    cluster = int(assignments[i])
    if(cluster == 0):
        plt.scatter(x[i], y[i], c = "red")
    if(cluster == 1):
        plt.scatter(x[i], y[i], c = "blue")
    if(cluster == 2):
        plt.scatter(x[i], y[i], c = "green")


# ### Question 1c: Testing K-means [40%]
# Selecting the right number of clusters $K$ is a very challenging problem, especially when we don't have some side-information or domain expertise that can help us narrow down a few reasonable values for that parameter. 
# 
# In the absence of any other information, a very useful exercise is to create the plot of SSE (sum of squares of errors) as a function of $K$. Ideally, for a very small $K$, the error will be high (since we are trying to approximate a whole lot of points with a very small number of centroids) and as $K$ increases, the error decreases. However, after a certain value (or a couple of values) for $K$, we will notice diminishing returns, i.e., the error will be decreasing, but not to a great degree. Typically, the value(s) for $K$ where this behavior is observed (the threshold point after which we observe diminishing returns) is usually a good guess for the number of clusters. 
# 
# In this question, we will have to create the SSE vs. K plot for $K = 1\cdots10$. Furthermore, because K-means uses randomized initialization, we need to do a number of iterations per value of $K$ in order to get a good estimate of the actual SSE (which may not be caused by randomness in the initialization). For this question, you will have to run the entire K-means algorithm to completion, and repeat it 50 different times per $K$, and collect all SSEs. In the figure, you should report the mean SSE per $K$, surrounded by error-bars which will encode the standard deviation.

# In[148]:


# WARNING!!!
# THIS WILL TAKE A LONG TIME TO RUN
K = 10
meanError = [0] * K
devError = [0] * K

for k in range(1, K + 1):
    totalError = []
    for i in range(50):
        assignments, centroids, all_sse, it = kmeans_clustering(all_vals = data, K = k, max_iter = 100, tol = pow(10,-3))
        totalError.append(all_sse[len(all_sse) - 1])
    meanError[k - 1] = (np.mean(totalError))
    devError[k - 1] = (np.std(totalError))
    print("K: ", k, " ", meanError[k-1], " ", devError[k-1])

x = np.arange(1, K + 1)
plt.errorbar(x, y = meanError, yerr=devError)

