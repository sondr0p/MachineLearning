#!/usr/bin/env python
# coding: utf-8

# # CS171 - Spring 2019 - Assignment 1
# ### Instructor: Vagelis Papalexakis
# 
# In this first assignment you will explore a dataset, visualizing the dataset in various ways, and doing a preliminary analysis on the data. 
# 
# For this assignment we are going to use the functionality of Pandas (the library, *not* the unbearably cute animal): https://pandas.pydata.org/ in order to manipulate datasets.
# In addition to Pandas, we are going to use Matplotlib (https://matplotlib.org/) and Numpy (http://www.numpy.org/) and you may also find Seaborn (https://seaborn.pydata.org/) useful for some data visualization.
# 
# Unless you are explicitly asked to *implement* a particular functionality, you may assume that you may use an existing implementation from the libraries above (or some other library that you may find, as long as you *document* it).
# 
# Before you start, make sure you have installed all those packages in your local Jupyter instance, as follows:
# 
# conda install numpy pandas matplotlib seaborn
# 
# ## Academic Integrity
# Each assignment should be done  individually. You may discuss general approaches with other students in the class, and ask questions to the TAs, but  you must only submit work that is yours . If you receive help by any external sources (other than the TA and the instructor), you must properly credit those sources, and if the help is significant, the appropriate grade reduction will be applied. If you fail to do so, the instructor and the TAs are obligated to take the appropriate actions outlined at http://conduct.ucr.edu/policies/academicintegrity.html . Please read carefully the UCR academic integrity policies included in the link.
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import seaborn as sb
import random as rand
import random
from numpy import linalg as LA


# ## Question 0: Getting real data [0%] 
# 
# In this assignment you are going to use data from the UCI Machine Learning repository ( https://archive.ics.uci.edu/ml/index.php ). In particular, you are going to use the famous Iris dataset: https://archive.ics.uci.edu/ml/datasets/Iris
# 

# In[2]:


data_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
data = pd.read_csv('iris.data', 
                   names = data_names)
data.head()


# ## Question 1: Data Visualization [20%]
# 

# ### Question 1a: Scatterplots [10%]
# 1. Plot the scatterplot of all pairs of features and color the points by class label [5%]
# 2. Which pair of features is (visually) the most correlated?  [2.5%]
# 3. Can you think of a reason why looking at this plot would be useful in a task where we would have to classify flowers by label? [2.5%]

# In[3]:


# Plots each feature of the data against each other

fig, axs = plt.subplots(4, 4, figsize=(10, 10))

# get data by label
setosa = data[data['label'].isin(['Iris-setosa'])]
versicolor = data[data['label'].isin(['Iris-versicolor'])]
virginica = data[data['label'].isin(['Iris-virginica'])]

# scatter plots (probably could have used for loop but small enough features to copy paste)
sep_l = setosa['sepal_length']
sep_w = setosa['sepal_width']
pet_l = setosa['petal_length']
pet_w = setosa['petal_width']

axs[0, 0].scatter(sep_l, sep_l, c = 'red')
axs[1, 0].scatter(sep_l, sep_w, c = 'red')
axs[2, 0].scatter(sep_l, pet_l, c = 'red')
axs[3, 0].scatter(sep_l, pet_w, c = 'red')

axs[0, 1].scatter(sep_w, sep_l, c = 'red')
axs[1, 1].scatter(sep_w, sep_w, c = 'red')
axs[2, 1].scatter(sep_w, pet_l, c = 'red')
axs[3, 1].scatter(sep_w, pet_w, c = 'red')

axs[0, 2].scatter(pet_l, sep_l, c = 'red')
axs[1, 2].scatter(pet_l, sep_w, c = 'red')
axs[2, 2].scatter(pet_l, pet_l, c = 'red')
axs[3, 2].scatter(pet_l, pet_w, c = 'red')

axs[0, 3].scatter(pet_w, sep_l, c = 'red')
axs[1, 3].scatter(pet_w, sep_w, c = 'red')
axs[2, 3].scatter(pet_w, pet_l, c = 'red')
axs[3, 3].scatter(pet_w, pet_w, c = 'red', label = 'Iris Setosa')

sep_l = versicolor['sepal_length']
sep_w = versicolor['sepal_width']
pet_l = versicolor['petal_length']
pet_w = versicolor['petal_width']

axs[0, 0].scatter(sep_l, sep_l, c = 'green')
axs[1, 0].scatter(sep_l, sep_w, c = 'green')
axs[2, 0].scatter(sep_l, pet_l, c = 'green')
axs[3, 0].scatter(sep_l, pet_w, c = 'green')

axs[0, 1].scatter(sep_w, sep_l, c = 'green')
axs[1, 1].scatter(sep_w, sep_w, c = 'green')
axs[2, 1].scatter(sep_w, pet_l, c = 'green')
axs[3, 1].scatter(sep_w, pet_w, c = 'green')

axs[0, 2].scatter(pet_l, sep_l, c = 'green')
axs[1, 2].scatter(pet_l, sep_w, c = 'green')
axs[2, 2].scatter(pet_l, pet_l, c = 'green')
axs[3, 2].scatter(pet_l, pet_w, c = 'green')

axs[0, 3].scatter(pet_w, sep_l, c = 'green')
axs[1, 3].scatter(pet_w, sep_w, c = 'green')
axs[2, 3].scatter(pet_w, pet_l, c = 'green')
axs[3, 3].scatter(pet_w, pet_w, c = 'green', label = 'Iris Versicolor')

sep_l = virginica['sepal_length']
sep_w = virginica['sepal_width']
pet_l = virginica['petal_length']
pet_w = virginica['petal_width']

axs[0, 0].scatter(sep_l, sep_l, c = 'blue')
axs[1, 0].scatter(sep_l, sep_w, c = 'blue')
axs[2, 0].scatter(sep_l, pet_l, c = 'blue')
axs[3, 0].scatter(sep_l, pet_w, c = 'blue')

axs[0, 1].scatter(sep_w, sep_l, c = 'blue')
axs[1, 1].scatter(sep_w, sep_w, c = 'blue')
axs[2, 1].scatter(sep_w, pet_l, c = 'blue')
axs[3, 1].scatter(sep_w, pet_w, c = 'blue')

axs[0, 2].scatter(pet_l, sep_l, c = 'blue')
axs[1, 2].scatter(pet_l, sep_w, c = 'blue')
axs[2, 2].scatter(pet_l, pet_l, c = 'blue')
axs[3, 2].scatter(pet_l, pet_w, c = 'blue')

axs[0, 3].scatter(pet_w, sep_l, c = 'blue')
axs[1, 3].scatter(pet_w, sep_w, c = 'blue')
axs[2, 3].scatter(pet_w, pet_l, c = 'blue')
axs[3, 3].scatter(pet_w, pet_w, c = 'blue', label = 'Iris Virginica')

# labels for color and features
axs[0, 0].annotate("sepal length", xy=(4.5,7))
axs[1, 1].annotate("sepal width", xy=(2,4))
axs[2, 2].annotate("petal length", xy=(2,6))
axs[3, 3].annotate("petal width", xy=(0,2))
axs[3,3].legend(loc='lower right')

plt.show()


# Your answer here:
# 2. petal length and petal width
# 3. To visualize and see the correlation between two characteristics between multiple flower labels

# ### Question 1b: Boxplot and Histogram [10%]
# 
# 1. Plot the boxplot for each feature of the dataset (you can put all boxplots on a single figure) [4%]
# 2. Plot the histogram only for petal length [4%]
# 3. Does the histogram for petal length give more information than the boxplot? If so, what information? [2%]

# In[4]:


# Produces a boxplot of each of the 4 features
# Also creates a histogram of the petal length

# boxplots
sep_len = data['sepal_length']
sep_wid = data['sepal_width']
pet_len = data['petal_length']
pet_wid = data['petal_width']

fig, axs = plt.subplots(1, 4, figsize=(10, 10))

axs[0].boxplot(sep_len)
axs[1].boxplot(sep_wid)
axs[2].boxplot(pet_len)
axs[3].boxplot(pet_wid)

axs[0].set_title('sepal length')
axs[1].set_title('sepal width')
axs[2].set_title('petal length')
axs[3].set_title('petal width')

plt.show()

# Histogram
fig, ax = plt.subplots()
ax.hist(pet_len)
ax.set_title('petal length')

plt.show()


# Your answer here:
# 
# 3. Histogram because it gives more information about where the data is distributed.

# ## Question 2: Distance computation [40%]
# 
# 

# ### Question 2a: Implement the Lp distance function [20%]
# 1. Write code that implements the Lp distance function between two data points as we saw it in class [15%]
# 2. Verify that it is correct by comparing it for p=2 against an existing implementation in Numpy for the two selected data points below. Note that the difference of the distances may not be exactly 0 due to numerical precision issues. [5%]

# In[5]:


# Computes Minkowski distance between two vectors
# Compares distance to numpy implementation

# my implementation
def lp(v, p):
    d = sum(pow((abs(v)), p))
    return pow(d, (1/p))

a = data.values[0][0:4]
b = data.values[1][0:4]
p = 2
d = lp(np.array(a) - np.array(b), p)

# numpy implementation 
d_np = np.linalg.norm(np.array(a) - np.array(b), p)

print("My dist: ", d)
print("Numpy dist: ", d_np)


# ### Question 2b: Compute the distance matrix between all data points [20%]
# 1. Compute an $N\times N$ distance matrix between all data points (where $N$ is the number of data points) [5%]
# 2. Plot the above matrix and include a colorbar. [5%]
# 3. What is the minimum number of distance computations that you can do in order to populate every value of this matrix? (note: it is OK if in the first two questions you do all the $N^2$ computations) [5%]
# 4. Note that the data points in your dataset are sorted by class. What do you observe in the distance matrix? [5%]

# In[6]:


fig, ax = plt.subplots()
cm = 'RdBu_r'

# Create NxN matrix
N = 150
DistanceMatrix = [[0 for i in range(N)] for j in range(N)]  
for i in range(N):
    for j in range(N):
        v = lp(data.values[i][0:4] - data.values[j][0:4], 2) 
        DistanceMatrix[i][j] = v

# Plot matrix
pcm = ax.pcolormesh(DistanceMatrix, cmap=cm)
fig.colorbar(pcm, ax=ax)
plt.show()


# Your answer here:
# 3. N^2 computations
# 4. There is a difference when comparing data points of two different classes. It creates boxes where there is a distinct transition when comparing classes. For example, note that comparing data point 0 to data points [0:49] is mostly blue showing there is little lp distance between them. When comparing the same data point to data points [50:99] and [100:149], the lp distance increases.

# ## Question 3: Data Sampling [40%]
# 
# Sometimes datasets are too big, or come in a streaming fashion, and it is impossible for us to process every single data point, so we have to resort to sampling methods. In this question, you will implement the popular "reservoir sampling" method, which is mostly used to obtain a uniform random sample of a data stream. Subsequently, you will experiment with sampling directly all the data and conducting stratified sampling (by class label) and observe the results in the data distribution.

# ### Question 3a: Reservoir Sampling [20%]
# 1. Implement reservoir sampling as we saw it in class. Create a 'reservoir_sampling' function because it will be useful for the next question. [15%]
# 2. Run reservoir sampling with reservoir size $M = 15$ and plot the histogram of the petal length feature for the sampled dataset [5%]

# In[7]:


def reservoir_sampling(stream,M):
    # Create reservoir and fill it with M samples
    i = 0;
    res = [0] * M
    for i in range(M):
        res[i] = stream.iloc[i]
        
    # Reservoir is full so begin rejecting or keeping data
    while(i < len(stream)):
        # Get random index j from 0 to i
        j = random.randrange(i+1); 
        
        # Kick or keep
        if(j < M): 
            res[j] = stream.iloc[i]; 
        i += 1; 
    return res

res = reservoir_sampling(data['petal_length'], 15)

# Plot Histogram
fig, ax = plt.subplots()
ax.hist(res)
ax.set_title('petal length: Reservoir')

plt.show()


# ### Question 3b: Stratified Sampling [20%]
# 1. Implement stratified sampling by class label, and within each stratum use the reservoir sampling function you implemented. [15%]
# 2. Run your stratified sampler with $M=5$ samples per class (so that we have 15 data points in total) and plot the histogram of the petal length feature for the sampled dataset [2.5%]
# 3. Do you observe any difference between the stratified and the non-stratified histograms? Which one resembles the original petal length distribution more closely? In order to answer this question you may want to run both sampling procedures a few times and observe which one gives a more accurate result on average. [2.5%]

# In[8]:


# Get data by label
setosa_data = data[data['label'].isin(['Iris-setosa'])]
versicolor_data = data[data['label'].isin(['Iris-versicolor'])]
virginica_data = data[data['label'].isin(['Iris-virginica'])]

setosa = setosa_data['petal_length']
versicolor = versicolor_data['petal_length']
virginica = virginica_data['petal_length']

M = 5

res_set = (reservoir_sampling(setosa, M))
res_ver = (reservoir_sampling(versicolor, M))
res_vir = (reservoir_sampling(virginica, M))

sample = res_set + res_ver + res_vir

# Plot Histogram
fig, ax = plt.subplots()
ax.hist(sample)
ax.set_title('petal length: Stratified')
plt.show()


# Your answer here:
# 3. The stratified histogram distributes the data more accurately. For example, the non-stratified sample could plot 0 data points between 0 and 2 (not often though) while the stratified always plots data from these points due to the grouping by label.
