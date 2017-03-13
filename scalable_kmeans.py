from kMeans import kMeans
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
    n =shape(dataSet)[0]
    elected_centers = []
    candidate_centers = []
    dataSet_temp = dataSet.A.tolist()
    #initialize 2-d mat with first center selected arbitrarily
    candidate_centers.append(dataSet[0,:].tolist())
    #dataSet1 = dataSet.tolist()
    #find the nearest distance to the closest centers in candidate_centers
    Dx = array([min([distEclud(c,x) for c in candidate_centers]) for x in dataSet])
    psi = sum(Dx)
    l = int(ceil(log(psi)))

    for k in range(l):
        
        #now I am expected to select sampling_factor times the number 
        #elements say two elements each time from dataSet
        '''
        take Dx array if sampling_factor is say 3 split it into three halfs and find the
        do the probability thing find candidate cluster points
        '''
        r_points = random.random_sample((sampling_factor, ))
        Dx = array([min([distEclud(c,x) for c in candidate_centers]) for x in dataSet])
        probs = (sampling_factor * Dx)/psi
        cumsumprobs = cumsum(probs)
        psi = sum(Dx)
        # parallel job start
        for r in r_points:
            
            #need some optimizations in probs and next step can be clubbed together
            #r = random.rand()
            
            #do i need to do the cumsum of probs as done in kmeans++ step
            
            for j,p in enumerate(cumsumprobs):
                if r < p:
                    i = j
                    break
            candidate_centers.append(dataSet[i,:].tolist())
            dataSet_temp.pop(i)
        #parallel job stop
    #return dataSet_temp,dataSet,candidate_centers
    w = {}
    #for each element in setC find the closest centroids among candidates
    #and update the weight for the closest center
    
    for ele in dataSet_temp:		
		ele = tuple(ele)
		minDist = inf
		if ele not in w:
			w[ele] = 0
		else:
			for i,c in enumerate(candidate_centers):
				dist = distEclud(c, ele)
				if dist < minDist:
					minDist = dist
					index = i
			w[candidate_centers[i]]+=1
    #select k-clusters according to kmeans++
    total = sum(w.values())
    probs = array(w.values())/total
    cumsprobs = cumsum(probs)
    print len(cumsprobs)
    for k in range(K):
        r = random.rand()
        for j,p in enumerate(cumsprobs):
            if r < p:
                i = j
                break
        elected_centers.append(candidate_centers[i])
        
        
    return elected_centers
        
         
                    
