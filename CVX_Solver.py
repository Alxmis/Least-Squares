import time
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt


file = open("Global_min.txt", "w+")

clock = [x for x in range(1, 10 + 1)]
N_array = [x for x in range(10, 100 + 10, 10)]

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

        file.write(f"{N}_{n} {weight} {yy}\n")

    t1 = time.time() - t0
    clock[int(N/10 - 1)] = (t1 / 100)


file.close()

plt.plot(N_array, clock, "-om")
plt.ylabel("Error")
plt.xlabel("Dimension")
plt.show()
