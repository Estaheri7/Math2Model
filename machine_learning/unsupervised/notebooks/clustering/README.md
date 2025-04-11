# Clustering with K-Means

This notebook demonstrates the implementation of the K-Means clustering algorithm from scratch using Python. The K-Means algorithm is a popular unsupervised learning method used for clustering data into distinct groups.

## Algorithm Explanation

K-Means clustering aims to partition `n` observations into `k` clusters in which each observation belongs to the cluster with the nearest mean. The algorithm follows these steps:

1. **Initialization**: Randomly select `k` initial centroids from the data points.
2. **Assignment**: Assign each data point to the nearest centroid, forming `k` clusters.
3. **Update**: Calculate the new centroids as the mean of the data points assigned to each cluster.
4. **Repeat**: Repeat the assignment and update steps until the centroids do not change significantly or a maximum number of iterations is reached.
