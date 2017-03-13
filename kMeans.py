
from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA-vecB,2)))
'''
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    #n specifies the number of columns
    #shape(dataSet)[0] gives the number of rows
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ =min(dataSet[:,j])
        #for all values of each column subtract minJ from each of them
        rangeJ = float(max(dataSet[:,j]-minJ))
        # also you can find maxJ and do maxJ - minJ
        centroids[:,j] = minJ +rangeJ * random.rand(k,1)
        #minJ + anywhere within the range are the centroids
    return centroids
'''

def randCent(dataSet, K):
    n =shape(dataSet)[0]
    #initialize 2-d mat with first center selected arbitrarily
    elected_centers = [dataSet[0,:].tolist()[0]]
    
    for k in range(1,K):
        Dx = array([min([distEclud(c,x) for c in elected_centers]) for x in dataSet])
        '''
        for each point:
            for each center:
                find the distance of center to point
            find the nearest center for that point
        Dx is an array of minimum-distances of all points from the any of the elected_centers
        
        take a new center from the list of points in the dataSet which has the maximum probability
        
        suppose we have probs like: [0.1,0.4,0.2,0.2,0.1]
        then we have cumsums like: [0.1,0.5,0.7,0.9,1]
        so the intervals are like |.....|..|..|.| that means that a point which is far from both centers say a and b
        creates a bigger interval so any random point taken has to be in this.
        
        suppose all points are grouped together very close to two cluster centers except one and the all are placed in an ascending 
        order array ex.
        probs = [0.001,0.0011,0.2,.3,0.4088]
        cumsprobs = [.001,.0012,0.2012,.5012,1]
        
        Now a random point can take any values bw 0 and 1 the interval which accomodates a higher range of numbers bw 0 and 1 will be
        the most likely place for point to be in.
        So if the distance of any number is very high its probability is very high its probability is high which means its distance to
        both the centers is very large
        irrespective of where the point occurs in the array it will increase the interval and other intervals 
        will decrease automatically
        
        what I would think is find length of each range and take a point like min(x)+some number which puts it in that interval
        '''
        r = random.rand()
        probs = Dx/sum(Dx)
        cumsprobs = scipy.cumsum(probs)
        
        for j,p in enumerate(cumsprobs):
            if r < p:
                i = j
                break
        elected_centers.append(dataSet[i].tolist()[0])
         
    return matrix(elected_centers)    
    #C = n
    '''
     calculate the distance of the points from the centres and find the minimum of them
     and then 
      let D(x)
     denote the shortest distance from a data point to the closest center we have already chosen 
     '''  

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = mat(createCent(dataSet, k))
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf;minIndex = -1
            for j in range(k):
                 distJI = distMeas(centroids[j,:],dataSet[i,:])
                 if distJI < minDist:
                     minDist = distJI;minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        #print centroids
        for cent in range(k):
            #matrix.A is used to create a matrix into a numpy array
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment 


