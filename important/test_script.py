from sklearn.cluster import k_means
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# create a sample dataSet
dataSet, clusterAssgn = make_blobs(n_samples=100, centers=3,
                                   n_features=2, random_state=0)

# kmeans = k_means(x,n_clusters=3,random_state = 0)
kmeans = k_means(dataSet, init='k-means||', sampling_factor=3,
                 n_clusters=3, random_state=0)

x = dataSet[:, 0]
y = dataSet[:, 1]
Cluster = kmeans[1]

centers = kmeans[0]

print 'cluster:', Cluster

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x, y, c=Cluster, s=50)
# s parameter shows how big will be the plus symbol

centers = np.mat(centers)
for ele in centers:
    i = ele[0, 0]
    j = ele[0, 1]
    ax.scatter(i, j, s=50, c='red', marker='+')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.show()
