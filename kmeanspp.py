
from math import log,ceil
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


def createCent(dataSet, K,sampling_factor=2):
    '''
        Parameters:
        -----------
        
        dataSet: numpy.matrix
        K: integer, number of clusters
        sampling_factor: integer
    
    '''
        
    n =shape(dataSet)[0]
    elected_centers = []
    candidate_centers = []
    dataSet_temp = dataSet.A.tolist()
    
    #initialize 2-d mat with first center selected arbitrarily
    candidate_centers.append(dataSet[0, :].tolist()[0])
    
    #find the nearest distance to the closest centers in candidate_centers
    Dx = array([min([distEclud(c, x) for c in candidate_centers]) for x in dataSet])
    psi = sum(Dx)
    l = int(ceil(log(psi)))

    for k in range(l):
        
        r_points = random.random_sample((sampling_factor, ))
        Dx = array([min([distEclud(c, x) for c in candidate_centers]) for x in dataSet])
        probs = (sampling_factor * Dx)/psi
        cumsumprobs = cumsum(probs)
        psi = sum(Dx)
        # parallel job start
        for r in r_points: 
            for j,p in enumerate(cumsumprobs):
                if r < p:
                    i = j
                    break       
            candidate_centers.append(dataSet[i, :].tolist()[0])        
        #parallel job stop
        
    w = [0 for _ in range(len(candidate_centers))]
    for i in range(len(dataSet_temp)):
        minDist = inf
        for j,c in enumerate(candidate_centers):
                dist = distEclud(c, array(dataSet_temp[i]))
                if dist < minDist:
                    minDist = dist
                    index = j
        w[index]+=1
        
    #select k-clusters according to kmeans++
    w = array(w)    
    probs = w/float(sum(w))
    cumsprobs = cumsum(probs)
    
    for k in range(K):
        r = random.rand()
        for j,p in enumerate(cumsprobs):
            if r < p and candidate_centers[j] not in elected_centers:
                index = j
                elected_centers.append(candidate_centers[index])
                break
                
               
    return elected_centers
        
         
                    

