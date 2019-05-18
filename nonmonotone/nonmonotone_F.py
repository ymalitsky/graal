# An experiment with some highly nonlinear/nonconvex/nonmonotone
# problem for adaptive Golden Ratio Algorithm.

__author__ = "Yura Malitsky"
__license__ = "MIT License"
__email__ = "y.malitsky@gmail.com"
__status__ = "Development"


import numpy as np
import scipy as sp
import scipy.linalg as LA
from graal import *

# choose dimension, number or experiments, and maximal number of
# iterations to run the algorithm
n = 5000
n_exp = 100
N = 10000


one = np.ones(n)
zero = np.zeros(n)
fbf_array = np.empty((n_exp, N))
graal_array = np.empty((n_exp,N))
z0 = np.ones(n)
success = 0
err = 1e-6
list_numb_it = []

for i in range(n_exp):
    print("Iteration", i+1)
    gen = i
    np.random.seed(gen)
    A = np.random.normal(0,1, (n,n))
    B = np.random.normal(0,1, (n,n))

    def F(x):
        t1 = A @ np.sin(x)
        t2 = B @ np.exp(x)
        res = t1 * (t1 @ x) + t2 * (t2 @ x)
        return res

    prox_G = lambda z, rho: z
    J = lambda z: LA.norm(F(z))

    values1, z1, n_it = adaptive_graal_terminate(J, F, prox_G, z0,
                                                 numb_iter=N, phi=1.5,
                                                 output=False,
                                                 tol=err)
    if n*LA.norm(z1) > 1e-4 and values1[-1] <= err:
        success += 1
        list_numb_it.append(n_it)
    print("dimension = {0}, success = {1}/{2}, average number of
    iterations ={3}".format(n, success, n_exp, np.mean(list_numb_it)))
