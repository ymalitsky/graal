import numpy as np
import scipy as sp
import scipy.linalg as LA
from graal import *


n = 1000
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
        t1 = A.dot((np.sin(x)))
        t2 = B.dot(np.exp(x))
        res = (np.outer(t1,t1) + np.outer(t2,t2)).dot(x)
        return res

    prox_G = lambda z, rho: z
    J = lambda z: LA.norm(F(z))

    values1, z1, n_it = explicit_graal_terminate(J, F, prox_G, z0, numb_iter=N, phi=1.5, output=False, tol=err)
    if n*LA.norm(z1) > 1e-4 and values1[-1] <= err:
        success += 1
        list_numb_it.append(n_it)
    print("dimension = {0}, success = {1}, average number of iterations = {2}".format(n, success, np.mean(list_numb_it)))
