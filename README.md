# Least-Squares
Various algorithms find the optimal parameters of the linear function using the Least squares method with l2-regularization.

## Description
1. 100(=:N) test cases are generated for each value of dimension n=[10,100] of x vector of variables (points in multidimensional space).
2. Each test case is solved using the CVX solver. 
3. A graph of the dependence of the average solution time on the dimensionality n is plotted. The values of the global minimum x* and the optimal value of the target function f(x*) for each test case are saved.
4. The following algorithms are implemented: Gradient descent, Cramer's method, and Inverse matrix method.
5. A graph of the dependence of the average running time of the algorithms on the dimensionality n is plotted.
