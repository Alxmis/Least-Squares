import random
import numpy as np

def makeRegression(coefficient, dataSize):
    X = np.empty((len(coefficient) - 1, dataSize))
    for i in range(len(X)):
        random.seed()
        step = random.randint(1, 20)
        X[i] = [x for x in range(0, dataSize * step, step)]
    Y = np.empty((dataSize, 1))

    for i in range(dataSize):
        Y[i] = coefficient[0]
        for j in range(0, N - 1):
            Y[i] += coefficient[j + 1] * X[j][i]

    return X, Y

for N in range(10, 100 + 10, 10):
    for n in range(1, 100 + 1):
        random.seed()
        dataSize = random.randint(150, 200)  # generated dataset size

        random.seed()
        coefTrue = np.empty((N, 1))
        for i in range(len(coefTrue)):
            random.seed()
            coefTrue[i] = random.uniform(7.0, 45.0) # weighting coefficients
        xScale, yEst = makeRegression(coefTrue, dataSize)

        # Adding outliers
        nO = random.randint(10, 20) # number of outliers
        iO = np.arange(nO, dataSize, int(dataSize/nO) - random.randint(1, 5)) # indexes of outliers
        for i in range(0, nO):
            yEst[iO[i]] += random.randint(-256, 256)

        Y = np.empty([1, dataSize])
        for i in range(0, dataSize):
            Y[0][i] = yEst[i]

        np.save("x_scale_{}_{}".format(N, n), xScale)
        np.save("y_est_{}_{}".format(N, n), Y)