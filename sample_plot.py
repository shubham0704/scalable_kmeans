import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import kMeans
import kmeanspp
'''
x = np.random.randn(10)
y = np.random.randn(10)
Cluster = np.array([0, 1, 1, 1, 3, 2, 2, 3, 0, 2])    # Labels of cluster 0 to 3
centers = np.random.randn(3, 2) 
'''
dataMat = np.mat(kMeans.loadDataSet('testSet.txt'))
x = dataMat[:,0]
y = dataMat[:,1]
centers,clusterAssgn = kMeans.kMeans(dataSet=dataMat,k=4,createCent = kmeanspp.createCent)
#centers,clusterAssgn = kMeans.kMeans(dataSet=dataMat,k=4)
#print '\nthe centers are\n'
#print centers
#print clusterAssgn
Cluster = np.array(clusterAssgn[:,0])
#print Cluster
print centers
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x,y,c=Cluster,s=50)
#s parameter shows how big will be the plus symbol

#cNorm = colors.Normalize(vmin = 0,vmax = 2)
centers = np.mat(centers) 
for ele in centers:
    i = ele[0,0]
    j = ele[0,1]
    ax.scatter(i,j,s=50,c='red',marker='+')
ax.set_xlabel('x')
ax.set_ylabel('y')
#plt.colorbar(scatter)

plt.show()

