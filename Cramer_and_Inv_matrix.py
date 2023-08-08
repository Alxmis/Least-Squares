import numpy as np
from math import sqrt
import random
import time

lamb = 0.01

Xv = open("test.txt")
Yv = open("testY.txt")

def MainFunk(X, Y, b, K, lam):
    sum = 0
    for i in range(len(Y)):
        m = 0
        for j in range(len(Y)-1):
            m -= K[j]*X[i][j]
        sum += (Y[i] - b - m)**2
    for i in K:
        sum += i**2*lam
    return sum

def MakeXMatrix(file):
    res = []
    for line in file:
        arr = list(map(int, line.split()))
        res.append(arr)
    sef = np.array(res)
    result = sef.transpose()
    return result

def MakeYVec(file):
    res = []
    for line in file:
        res.append(int(line))
    return res

def CreateCoeffMatrix(X, Y, lam, N, M):
    result = []
    for i in range(N-1):
        a = []
        for j in range(N-1):
            sum = 0
            for k in range(M):
                sum += -X[k][i]*X[k][j]
                if i==j:
                    sum+=2*lam
            a.append(-2*sum)
        lsum = 0
        for k in range(M):
            lsum+=-X[k][i]
        a.append(-2*lsum)
        result.append(a)
    v = []
    for j in range(N-1):
        sum = 0
        for k in range(M):
            sum+=-X[k][j]
        v.append(-2*sum)
    v.append(2*N)
    result.append(v)
    return result

def CreateFreeCoefMat(X, Y, N, M):
    result = []
    for i in range(N-1):
        sum = 0
        for j in range(M):
            sum+=X[j][i]*Y[j]
        result.append(sum*2)
    suml = 0
    for i in range(M):
        suml+=Y[i]
    result.append(2*suml)
    return result

def copyArr(arr):
    res = []
    for i in range(len(arr)):
        res.append([])
        for j in range(len(arr[0])):
            res[i].append(arr[i][j])
    return res

def replaceMatColl(mat, replaceVec, index):
    res = copyArr(mat)
    for i in range(len(res)):
        res[i][index] = replaceVec[i]
    return res

def MatrixMult(a, b):
    res = []
    for i in range(len(b)):
        s = 0
        for j in range(len(b)):
            s += a[i][j]*b[j]
        res.append(s)
    return res

def CalcY(X, K):
    res = 0
    for i in range(len(X)):
        res += X[i]*K[i]
    res += K[len(K)-1]
    return res

timeK = []
timeI = []
tk = 0
ti = 0
X = np.load(f"linearRegressionData/x_scale_10_50.npy")
Y = np.load(f"linearRegressionData/y_est_10_50.npy")[0]
X = X.transpose()

result = []
N = len(X[0])+1
M = len(Y)
t1 = time.time()
mainMat = CreateCoeffMatrix(X, Y, lamb, N, M)
freeC = CreateFreeCoefMat(X, Y, N, M)
t2 = time.time()
mainDet = np.linalg.det(mainMat)
for i in range(N):
    result.append(np.linalg.det(replaceMatColl(mainMat, freeC, i))/mainDet)
t3 = time.time()
result2 = MatrixMult(np.linalg.inv(mainMat), freeC)
t4 = time.time()

err = []
for i in range(N):
    err.append(CalcY(X[i], result2)-Y[i])

print(err)
print(timeK)
print(timeI)
