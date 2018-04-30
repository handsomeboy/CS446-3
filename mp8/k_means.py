from copy import deepcopy
import numpy as np
import numpy.linalg as npla
import pandas as pd
import sys


'''
In this problem you write your own K-Means
Clustering code.

Your code should return a 2d array containing
the centers.

'''
# Import the dataset
data_row = pd.read_csv("data/iris.data", header = 0,parse_dates=True,sep=',')
data = np.array(data_row)[:,:4]
#print(data)
# Make 3  clusters
k = 3
# Initial Centroids
C = [[2.,  0.,  3.,  4.], [1.,  2.,  1.,  3.], [0., 2.,  1.,  0.]]
C = np.array(C)
print("Initial Centers")
print(C)
def k_means(C):
    # Write your code here!
    error = 1
    tol = 1E-3
    k = len(C)
    def cluster(C):
        d0 = data - C[0]
        d1 = data - C[1]
        d2 = data - C[2]
        D0 = np.sum(d0**2, axis=1)
        D1 = np.sum(d1**2, axis=1)
        D2 = np.sum(d2**2, axis=1)
        return [1*(d2 < d1 and d2 < d3)+ 2 * ( d3< d1 and d3 < d2) for d1, d2, d3 in zip(D0, D1, D2)]
        
        
    Cluster = cluster(C)
    print(Cluster)
    cnt = 0
    while (error > tol):
        Cluster = cluster(C)
        C_old = C.copy()
        for i in range(k):
            C[i] = np.mean(data[[j for j in range(len(data)) if Cluster[j]==i ]], axis=0)
        error = npla.norm(C-C_old)
        cnt += 1
        print(cnt)
        print(error)
        print(C)
    
    C_final = C
    return C_final

C_final = k_means(C)
print("Final center")
print(C_final)




