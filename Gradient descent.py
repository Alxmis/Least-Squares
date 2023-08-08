import matplotlib.pyplot as plt
import random
import numpy as np


def gradient_norm():
    global n, points
    sum = 0
    s = 0
    for j in range(len(points)):
        s1 = points[j][-1]
        for k in range(n-1):
            s1 -= points[j][k] * coefficients[k]
        s1 -= coefficients[-1]
        s += s1
    for i in range(n):
        if i == n-1:
            derivative = -2 * s
        else:
            derivative = 0
            for j in range(len(points)):
                s2 = points[j][-1]
                for k in range(n-1):
                    s2 -= points[j][k] * coefficients[k]
                s2 -= coefficients[-1]
                derivative += -2 * points[j][i] * s2 + 2 * lamb * coefficients[i]
        sum += derivative ** 2
    return sum ** 0.5
    
res = []
for n in range(10, 101, 10):
    r_score = 0
    for N in range(1, 101):
        print(n, N)
        yy = []
        y_estimate = []
        coefficients = [1 for _ in range(n)]
        lamb = 0.01
        eps = 5000 + 5000 * (N / 10) * 50 * (n / 10)
        points = []
        nameX = f"x_scale_{n}_{N}.npy"
        nameY = f"y_est_{n}_{N}.npy"
        X = np.load(nameX)
        Y = np.load(nameY)
        for l in range(len(X[0])):
            point = []
            for k in range(n-1):
                point.append(X[k][l])
            point.append(Y[0][l])
            points.append(point)

        a = [0.0000000007 for _ in range(n-1)]
        a.append(0.0000007)
        
        while gradient_norm() > eps:
            
            #count derivative
            s = 0
            for j in range(len(points)):
                s1 = points[j][-1]
                for k in range(n-1):
                    s1 -= points[j][k] * coefficients[k]
                s1 -= coefficients[-1]
                s += s1
            for i in range(n):
                if i == n-1:
                    derivative = -2 * s
                else:
                    derivative = 0
                    for j in range(len(points)):
                        s2 = points[j][-1]
                        for k in range(n-1):
                            s2 -= points[j][k] * coefficients[k]
                        s2 -= coefficients[-1]
                        derivative += -2 * points[j][i] * s2 + 2 * lamb * coefficients[i]
                        
                #gradient descent
                coefficients[i] = coefficients[i] - a[i] * derivative

        print(coefficients)

        for i in range(len(points)):
            y_estimate.append(points[i][-1])

        for i in range(len(points)):
            s = 0
            for j in range(n-1):
                s += points[i][j] * coefficients[j]
            s += coefficients[-1]

    
