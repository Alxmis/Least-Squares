import time
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score


file = open("Global_min.txt", "w+")
# fileR = open("R2_score.txt", "w+")

clock = [x for x in range(1, 10 + 1)]
N_array = [x for x in range(10, 100 + 10, 10)]

# Cramer's method, Inverse matrix method, and Gradient descent data respectively
clock0 = [0.009996652603149414, 0.04899477958679199, 0.11196327209472656, 0.21691465377807617, 0.3408794403076172, 0.4318702220916748, 0.5718152523040771, 0.7507567405700684, 1.141620397567749, 1.4105327129364014]
clock1 = [0.009996652603149414, 0.047972917556762695, 0.10695695877075195, 0.2049405574798584, 0.3158872127532959, 0.3908724784851074, 0.5098261833190918, 0.6558089256286621, 1.0096747875213623, 1.1786189079284668]
clock2 = [0.19848326444625855, 0.3813626670837402, 0.5791351819038391, 0.8616371631622315, 1.3344751811027527, 1.561657998561859, 1.9900141525268555, 3.0867759346961976, 3.0710984587669374, 3.505057575702667]


# R_array = [0 for x in range(1, 100 + 1)]

for N in range(10, 100 + 10, 10):
    t0 = time.time()
    for n in range(1, 100 + 1):
        weight = [0 for x in range(N)]

        x_scale_prob = np.load(f"x_scale_{N}_{n}.npy")
        y_estimate_prob = np.load(f"y_est_{N}_{n}.npy")

        x_scale = np.array(x_scale_prob, dtype=float)
        y_estimate0 = np.array(y_estimate_prob, dtype=float)
        y_estimate = [0 for i in range(len(y_estimate0[0]))]
        for i in range(len(y_estimate)):
            y_estimate[i] = y_estimate0[0][i]

        w0 = cvx.Variable(1)
        wi = cvx.Variable(N - 1)

        gamma = 1e-15
        obj = cvx.Minimize(cvx.norm(y_estimate - w0 - wi @ x_scale) + gamma * cvx.norm(wi))

        prob = cvx.Problem(obj)
        prob.solve()


        weight[0] = w0.value[0]
        for i in range(1, N):
            weight[i] = wi.value[i - 1]

        yy = [0 for x in range(len(y_estimate))]
        for i in range(len(yy)):
            yy[i] = w0.value[0]
            for d in range(0, N - 1):
                yy[i] += wi.value[d]*x_scale[d][i]

        # R_array[n - 1] = r2_score(y_estimate, yy)
        file.write(f"{N}_{n} {weight} {yy}\n")

    t1 = time.time() - t0
    clock[int(N/10 - 1)] = (t1 / 100)

    # fileR.write(f"{N} {sum(R_array) / 100}\n")


file.close()

plt.plot(N_array, clock, "-om", label="CVX")
plt.plot(N_array, clock0, "-xg", label="Cramer's method")
plt.plot(N_array, clock1, "-vb", label="Inverse matrix method")
plt.plot(N_array, clock2, "-2c", label="Gradient descent")
plt.ylabel("Error")
plt.xlabel("Dimension")
plt.legend()
plt.show()