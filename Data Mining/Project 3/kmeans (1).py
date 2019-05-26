import numpy as np
import math

def norm(x):
    """
    >>> Function you should not touch
    """
    max_val = np.max(x, axis=0)
    x = x/max_val
    return x

def rand_center(data,k):
    """
    >>> Function you need to write
    >>> Select "k" random points from "data" as the initial centroids.
    """
    centroids = []
    numbers = np.random.choice(data.shape[0],k,False)
    for i in range(k):
        centroids.append(data[numbers[i]])
    return centroids

def converged(centroids1, centroids2):
    """
    >>> Function you need to write
    >>> check whether centroids1==centroids
    >>> add proper code to handle infinite loop if it never converges
    """
    status = True
    for i in range(centroids1.shape[0]):
        difference = centroids2[i] - centroids1[i]
        if (difference[0]> .025 or difference[1]> .025 or difference[2]> .025 or difference[3]> .025) :
            status = False
    
    return status

def eucl_distance(point1,point2):
    w = (point1[0] - point2[0]) ** 2
    x = (point1[1] - point2[1]) ** 2
    y = (point1[2] - point2[2]) ** 2
    z = (point1[3] - point2[3]) ** 2
    return math.sqrt(w+x+y+z)

def update_centroids(data, centroids, k=3):
    """
    >>> Function you need to write
    >>> Assign each data point to its nearest centroid based on the Euclidean distance
    >>> Update the cluster centroid to the mean of all the points assigned to that cluster
    """
    label = np.empty((data.shape[0]))
    
    for i in range(data.shape[0]):
        distance = 10
        for j in range(k):
            if eucl_distance(data[i],centroids[j]) < distance:
                distance = eucl_distance(data[i],centroids[j])
                label[i] = j
                
    newcentroids = np.empty(centroids.shape)
    clusterOne, clusterTwo, clusterThree = 0,0,0
    
    for j in range(data.shape[0]):
        if label[j] == 0:
                clusterOne += 1
                newcentroids[0] += data[j]
        elif label[j] == 1:
                clusterTwo += 1
                newcentroids[1] += data[j]
        elif label[j] == 2:
                clusterThree += 1
                newcentroids[2] += data[j]
    newcentroids[0] = newcentroids[0]/clusterOne       
    newcentroids[1] = newcentroids[1]/clusterTwo
    newcentroids[2] = newcentroids[2]/clusterThree   
    return newcentroids, label


def kmeans(data,k=3):
    """
    >>> Function you should not touch
    """
    # step 1:
    centroids = rand_center(data,k)
    converge = False
    while not converge:
        old_centroids = np.copy(centroids)
        # step 2 & 3
        centroids, label = update_centroids(data, old_centroids)
        # step 4
        converge = converged(old_centroids, centroids)
    print(">>> final centroids")
    print(centroids)
    print SSE(centroids,label,data)
    return centroids, label

def evaluation(predict, ground_truth):
    gini(predict, ground_truth)
    pass

def gini(predict, ground_truth):
    """
    >>> use the ground truth to do majority vote to assign a flower type for each cluster
    >>> accordingly calculate the probability of missclassifiction and correct classification
    >>> finally, calculate gini using the calculated probabilities
    """
    a,b,c = 0.,0.,0.
    anot,bnot,cnot = 0.,0.,0.
    cluster1 = [0,0,0]
    cluster2 = [0,0,0]
    cluster3 = [0,0,0]
    
    for i in range(150):
        index = int(ground_truth[i])
        if predict[i] == 0.:
            cluster1[index] += 1. 
        elif predict[i] == 1.:
            cluster2[index] += 1.
        else:
            cluster3[index]+= 1.
    
    if cluster1[0]> cluster1[1] and cluster1[0]> cluster1[2]:
        cluster1index = 0
    elif cluster1[1] > cluster1[0] and cluster1[1]> cluster1[2]:
        cluster1index = 1
    else:
        cluster1index = 2
        
    if cluster2[0]> cluster2[1] and cluster2[0]> cluster2[2]:
        cluster2index = 0
    elif cluster2[1] > cluster2[0] and cluster2[1]> cluster2[2]:
        cluster2index = 1
    else:
        cluster2index = 2
        
    if cluster3[0]> cluster3[1] and cluster3[0]> cluster3[2]:
        cluster3index = 0
    elif cluster3[1] > cluster3[0] and cluster3[1]> cluster3[2]:
        cluster3index = 1
    else:
        cluster3index = 2
        
    aprob = float(cluster1[cluster1index]/(cluster1[0]+cluster1[1]+cluster1[2]))
    bprob = float(cluster2[cluster2index]/(cluster2[0]+cluster2[1]+cluster2[2]))
    cprob = float(cluster3[cluster3index]/(cluster3[0]+cluster3[1]+cluster3[2]))

    gini =  1- ((aprob)**2 +(1-aprob)**2 +(bprob)**2 +(1-bprob)**2 +(cprob)**2 +(1-cprob)**2)/3.
    print "Gini:", gini
    pass

def SSE(centroids, label,data):
    """
    >>> Calculate the sum of squared errors for each cluster
    """
    clusterOne, clusterTwo, clusterThree = 0,0,0
    for j in range(data.shape[0]):
        if label[j] == 0:
                clusterOne += eucl_distance(data[j], centroids[0]) ** 2
        elif label[j] == 1:
                clusterTwo += eucl_distance(data[j], centroids[1]) ** 2
        elif label[j] == 2:
                clusterThree += eucl_distance(data[j], centroids[2]) ** 2
    print "SSE: ", clusterOne, clusterTwo, clusterThree
    return
