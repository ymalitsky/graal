# Implementation of Golden Ratio Algorithms for Variational
# inequalities.

__author__ = "Yura Malitsky"
__license__ = "MIT License"
__email__ = "y.malitsky@gmail.com"
__status__ = "Development"


import numpy as np
import scipy.linalg as LA
from time import perf_counter


def golden_ratio_alg(J, F, prox_g, x0, la, numb_iter=100):
    """  Golden Ratio algorithm for variational inequalities.

    Input 
    -----
    J : function that computes residual in every iteration.
        Takes x as input.
    F : main operator.
        Takes x as input.
    prox_g: proximal operator.
        Takes two parameters x and a scalar as input.
    x0: Starting point.
        np.array, be consistent with J, F and prox_g.
    la : a positive number, the stepsize.
    numb_iter: number of iteration to run rhe algorithm.

    Return
    ------
    values: 1d array
          Collects all values that were computed in every iteration with J(x)
    x, x_ : last iterates
    """

    x = x0
    x_ = x0
    phi = (np.sqrt(5) + 1) / 2
    values = [J(x0)]
    for i in range(numb_iter):
        x = prox_g(x_ - la * F(x), la)
        x_ = ((phi - 1) * x + x_) / phi
        values.append(J(x))
    return np.array(values), x, x_



def adaptive_graal(J, F, prox_g, x0, numb_iter=100, phi=1.5, output=False):
    """ Adaptive Golden Ratio algorithm.

    Input 
    -----
    J : function that computes residual in every iteration.
        Takes x as input.
    F : main operator.
        Takes x as input.
    prox_g: proximal operator.
        Takes two parameters x and a scalar as input.
    x0: Starting point.
        np.array, be consistent with J, F and prox_g.
    numb_iter: number of iteration to run rhe algorithm.
    phi: a key parameter for the algorithm.
         Must be between 1 and the golden ratio, 1.618... Choice
         phi=1.5 seems to be one of the best.
    output: boolean.  
         If true, prints the length of a stepsize in every iteration.  
         Useful for monitoring.

    Return
    ------
    values: 1d array
          Collects all values that were computed in every iteration with J(x)
    x, x_ : last iterates
    time_list: list of time stamps in every iteration. 
          Useful for monitoring.
    """
    
    begin = perf_counter()

    x, x_ = x1.copy(), x1.copy()
    x0 = x + np.random.randn(x.shape[0]) * 1e-9
    Fx = F(x)
    la = phi / 2 * LA.norm(x - x0) / LA.norm(Fx - F(x0))
    rho = 1. / phi + 1. / phi**2
    values = [J(x)]
    time_list = [perf_counter() - begin]
    th = 1

    for i in range(numb_iter):
        x1 = prox_g(x_ - la * Fx, la)
        Fx1 = F(x1)

        n1 = LA.norm(x1 - x)**2
        n2 = LA.norm(Fx1 - Fx)**2
        n1_div_n2 = np.exp(np.log(n1) - np.log(n2))
        la1 = min(rho * la, 0.25 * phi * th / la * n1_div_n2, 1e6)
        x_ = ((phi - 1) * x1 + x_) / phi
        if output:
            print (i, la)
        th = phi * la1 / la
        x, la, Fx = x1, la1, Fx1
        values.append(J(x))
        time_list.append(perf_counter() - begin)

    end = perf_counter()

    print("Time execution of adaptive GRAAL:", end - begin)
    return np.array(values), x, x_, time_list


def adaptive_graal_terminate(J, F, prox_g, x1, numb_iter=100, phi=1.5, tol=1e-6, output=False):
    """ Adaptive Golden Ratio algorithm with termination criteria.

    Input 
    -----
    J : function that computes residual in every iteration.
        Takes x as input.
    F : main operator.
        Takes x as input.
    prox_g: proximal operator.
        Takes two parameters x and a scalar as input.
    x1: Strating point.
        np.array, be consistent with J, F and prox_g.
    numb_iter: number of iteration to run rhe algorithm.
    phi: a key parameter for the algorithm.
         Must be between 1 and the golden ratio, 1.618... Choice
         phi=1.5 seems to be one of the best.
    tol: a positive number.
        Required accuracy for termination of the algorithm.
    output: boolean.  
         If true, prints the length of a stepsize in every iteration. Useful
         for monitoring.
    Return
    ------
    values: 1d array
          Collects all values that were computed in every iteration with J(x).
    x : last iterate.
    i: positive integer, number of iteration to reach desired accuracy.
    time_list: list of time stamps in every iteration. 
          Useful for monitoring.
    """

    begin = perf_counter()

    x, x_ = x1.copy(), x1.copy()
    x0 = x + np.random.randn(x.shape[0]) 
    Fx = F(x)
    la = phi / 2 * LA.norm(x - x0) / LA.norm(Fx - F(x0))
    rho = 1. / phi + 1. / phi**2
    values = [J(x)]
    time_list = [perf_counter() - begin]
    th = 1

    i = 1
    while i <= numb_iter and values[-1] > tol:
        i += 1
        x1 = prox_g(x_ - la * Fx, la)
        Fx1 = F(x1)
        n1 = LA.norm(x1 - x)
        n2 = LA.norm(Fx1 - Fx)
        n1_div_n2 = np.exp(2 * (np.log(n1) - np.log(n2)))
        la1 = min(rho * la, 0.25 * phi * th / la * n1_div_n2, 1e6)
        x_ = ((phi - 1) * x1 + x_) / phi
        if output:
            print (i, la, LA.norm(x1), LA.norm(Fx), LA.norm(x_))

        th = phi * la1 / la
        x, la, Fx = x1, la1, Fx1
        values.append(J(x))
        time_list.append(perf_counter() - begin)

    end = perf_counter()

    print("Time execution of aGRAAL:", end - begin)
    return np.array(values), x, i, time_list

