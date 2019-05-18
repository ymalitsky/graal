# A collection of algorithms to solve monotone variational
# inclusions. The following algorithms are implemented: Tseng's
# forward-backward-forward algorithm, adaptive Golden Ratio algorithm.


__author__ = "Yura Malitsky"
__license__ = "MIT License"
__email__ = "y.malitsky@gmail.com"
__status__ = "Development"


import numpy as np
import scipy as sp
import numpy.linalg as LA
from time import perf_counter


def tseng_fbf_linesearch(J, F, prox_g, x0, delta=2, numb_iter=100):
    """
    Tseng's forward-backward-forward algorithm with linesearch for
    monotone inclusion $0 \in F + \partial g.

    Notice that FBF algorithm in this form works only for Nash
    problem, as every time we project x onto the nonnegative orthant
    to make it feasible.
    
    Input 
    -----
    J : function that computes residual in every iteration.
        Takes x as input.
    F : main operator.
        Takes x as input.
    prox_g: proximal operator.
        Takes two parameters x and a scalar as input.
    x0: Strating point.
        np.array, must be consistent with J, F and prox_g.
    delta: a positive number.
        Allows stepsize to increase from iteration to iteration. 
    numb_iter: number of iteration to run rhe algorithm.

    Return
    ------
    iterates: a list of 
        values (another list that collects all values J(x)), 
        x : last iterate.
        la : a positive number, last stepsize.
        n_F: number of F evaluated in total.
        n_prox: number of proximal maps evaluated in total.

    """
    begin = perf_counter()
    beta = 0.7
    theta = 0.99

    x1 = x0 + np.random.randn(x0.shape[0]) * 1e-9
    Fx = F(x0)

    la0 = LA.norm(F(x1)-Fx)/ LA.norm(x1-x0)
    iterates = [[J(x0)], x0, la0, 1, 0]

    def iter_T(values, x, la, n_F, n_prox):
        Fx = F(x)
        la *= delta
        for j in range(100):
            z = prox_g(x - la * Fx, la)
            Fz = F(z)
            if la * LA.norm(Fz - Fx) <= theta * LA.norm(z - x):
                break
            else:
                la *= beta
        #x1 = z - la * (Fz - Fx)
        x1 = np.fmax(z - la * (Fz - Fx), 0)
        # print j, la
        values.append(J(z))
        # n_f += j+1
        n_F += j + 2
        n_prox += j + 1
        ans = [values, x1, la,  n_F, n_prox]
        return ans

    for i in range(numb_iter):
        iterates = iter_T(*iterates)

    end = perf_counter()
    #print("---- FBF ----")
    #print("Number of iterations:", numb_iter)
    #print("Number of gradients, n_grad:", iterates[-2])
    #print("Number of prox_g:", iterates[-1])
    print("CPU time for FBF:", end - begin)
    return iterates

def adaptive_graal(J, F, prox_g, x1, numb_iter=100, phi=1.5, output=False):
    """ Adaptive Golden Ratio algorithm.

    Input 
    -----
    J : function that computes residual in every iteration.
        Takes x as input.
    F : main operator.
        Takes x as input.
    prox_g: proximal operator.
        Takes two parameters x and a scalar as input.
    x1: Starting point.
        np.array, must be consistent with J, F and prox_g.
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
          Collects all values that were computed in every iteration
          with J(x)
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
        n1_div_n2 = n1/n2 if n2 != 0 else la*10

        la1 = min(rho * la, 0.25 * phi * th / la * n1_div_n2)
        x_ = ((phi - 1) * x1 + x_) / phi
        if output:
            print (i, la)
        th = phi * la1 / la
        x, la, Fx = x1, la1, Fx1
        values.append(J(x))
        time_list.append(perf_counter() - begin)
    end = perf_counter()

    print("CPU time for aGRAAL:", end - begin)
    return values, x, x_, time_list


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
        np.array, must be consistent with J, F and prox_g.
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
    x0 = x + np.random.randn(x.shape[0]) * 1e-9
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

    print("CPU time for aGRAAL:", end - begin)
    return values, x, i
