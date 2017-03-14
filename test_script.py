from sklearn.cluster import k_means
from sklearn.datasets.samples_generator import make_blobs

x, y = make_blobs(n_samples=100,centers=3,n_features=2,random_state=0)

#kmeans = k_means(x,n_clusters=3,random_state = 0)


kmeans =  k_means(x,init='k-means||',sampling_factor=3,n_clusters=3,random_state = 0)

print kmeans
