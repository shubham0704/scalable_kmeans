
from math import log,ceil
import numpy as np

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))


def createCent(dataSet, K,sampling_factor=2):
    '''
        Parameters:
        -----------

        dataSet: numpy.matrix
        K: integer, number of clusters
        sampling_factor: integer

    '''
    elected_centers = []
    candidate_centers = []
    dataSet_temp = dataSet.tolist()
    dataSet = np.mat(dataSet)

    # initialize 2-d mat with first center selected arbitrarily
    candidate_centers.append(dataSet[0, :].tolist()[0])

    # Step-2 find the nearest distance to the closest centers
    # in candidate_centers
    Dx = np.array([min([distEclud(c, x) for c in candidate_centers])
                   for x in dataSet])
    psi = sum(Dx)
    l = int(ceil(log(psi)))

    # Step-3 Start of loop for log(psi) times
    for k in range(l):

        # Step-4 sample each point independently
        r_points = np.random.random_sample((sampling_factor, ))
        Dx = np.array([min([distEclud(c, x) for c in candidate_centers])
                       for x in dataSet])
        probs = (sampling_factor * Dx)/psi
        cumsumprobs = np.cumsum(probs)
        psi = sum(Dx)

        # parallel job start (in future after this works)
        for r in r_points:
            for j, p in enumerate(cumsumprobs):
                if r < p:
                    i = j
                    break
            # Step-5 add the point to the candidate center
            candidate_centers.append(dataSet[i, :].tolist()[0])
        # parallel job stop

    # Step-6 end for

    # Step-7 weight each candidate point according to how many points its
    # closest to
    w = [0 for _ in range(len(candidate_centers))]
    for i in range(len(dataSet_temp)):
        minDist = float("inf")
        for j, c in enumerate(candidate_centers):
                dist = distEclud(c, np.array(dataSet_temp[i]))
                if dist < minDist:
                    minDist = dist
                    index = j
        w[index] += 1

    # Step-8 Recluster-select k-clusters according to kmeans++
    w = np.array(w)
    probs = w/float(sum(w))
    cumsprobs = np.cumsum(probs)

    for k in range(K):
        r = np.random.rand()
        for j, p in enumerate(cumsprobs):
            if r < p and candidate_centers[j] not in elected_centers:
                index = j
                elected_centers.append(candidate_centers[index])
                break

    return np.array(elected_centers)


