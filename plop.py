import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.cluster import KMeans
from sklearn import datasets

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = TfidfVectorizer()
# newsgroups = fetch_20newsgroups(subset='all')
# X = vectorizer.fit_transform(newsgroups.data)
# print X.shape

n_samples = 100000
X, y = datasets.make_blobs(n_samples=n_samples,
                           n_features=20,
                           centers=15,
                           cluster_std=1)
print "Xshape = {}, size = {}MB".format(X.shape, X.nbytes / float(1e6))

n_clusters = 50
n_init = 6
  
kmpp = KMeans(n_clusters=50, init='k-means++', n_init=n_init, n_jobs=-1)
t_start = time.time()
kmpp.fit(X)
kppt = time.time() - t_start
print "k-means++", kmpp.inertia_, kppt
 
 
NN = [n_clusters / 10, n_clusters / 2, n_clusters, n_clusters * 2,
       n_clusters * 5]
II = []
TT = []
for sampling_factor in NN:
    kmscala = KMeans(n_clusters=50, init='k-means||',
                     n_init=n_init, sampling_factor=sampling_factor, n_jobs=-1)
    t_start = time.time()
    kmscala.fit(X)
    tt = time.time() - t_start
    inertia = kmscala.inertia_
    II.append(inertia)
    TT.append(tt)
    print "k-means||", inertia, tt,
    print "l={}, r={}".format(sampling_factor, 5)
 
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(NN, II, linewidth=3, color="red")
ax1.axhline(kmpp.inertia_, linestyle='--')
ax1.text(200, kmpp.inertia_, r'k-means++', fontsize=15, color='blue')
ax1.set_title('Inertia')
ax1.set_ylabel("inertia")
ax1.set_xlabel(r"$l$")
ax2.plot(NN, TT, linewidth=3, color='red')
ax2.axhline(kppt, linestyle='--')
ax2.text(200, kmpp.inertia_, r'k-means++', fontsize=15, color='blue')
ax2.set_title('Runtime (sec)')
ax2.set_ylabel("time")
ax2.set_xlabel(r"$l$")
plt.suptitle("k-means++ VS k-means|| (r=5, n_clusters=50)", fontsize=18)
plt.show()

