import numpy as np
import scipy as sp
import scipy.linalg as LA
from graal import *

n = 1000
n_exp = 100
N = 10000

zero = np.zeros(n)
fbf_array = np.empty((n_exp, N))
graal_array = np.empty((n_exp,N))

success = 0
err = 1e-6
list_numb_it = []
z0 = np.ones(n)

for i in range(n_exp):
    print("Iteration", i+1)
    gen = i
    np.random.seed(gen)

    A = np.random.normal(0,1, (n,n))
    T_tilda = lambda z:  np.log(1.1 +(A.dot((z)))**2)

    def T(z):
        S = T_tilda(z)
        n_z = LA.norm(z)
        n_S = LA.norm(S)
        Tz = n_z * S / (np.abs(n_z-1) + n_S)
        return Tz

    F = lambda z: z - T(z)
    prox_G = lambda z, rho: z
    J = lambda z: LA.norm(F(z))


    values1, z1, n_it = adaptive_graal_terminate(J, F, prox_G, z0, numb_iter=N, phi=1.5, output=False, tol=err)
    print("norm of z1", LA.norm(z1))
    if n*LA.norm(z1) > 1e-2 and np.abs(1-LA.norm(z1)) <= 1e-4 and values1[-1] <= err:
        success += 1
        list_numb_it.append(n_it)
    print("dimension = {0}, success = {1}, average number of iterations = {2}".format(n, success, np.mean(list_numb_it)))
