# A collection of algorithms to solve a fixed point equation Tx =x for
# nonexpansive operator T.

# The following algorithms are implemented: Krasnoselskii-Mann and  adaptive Golden Ratio algorithm.


__author__ = "Yura Malitsky"
__license__ = "MIT License"
__email__ = "y.malitsky@gmail.com"
__status__ = "Development"


import numpy as np
import scipy.linalg as LA
from time import perf_counter

# measure the gap ||x-Tx||
J = lambda x, Tx: LA.norm(x - Tx)


# measure the same gap ||x-Tx|| = ||Fx|| but for VIP
JF = lambda x: LA.norm(x)

def krasn_mann(T, x0, a=0.5, numb_iter=100):
    """ Krasnoselskii-Mann algorithm to solve Tx = x.
    
    Input 
    -----
    T : main operator.
        Takes x as input.
    x0: Strating point.
        np.array, must be consistent with T
    a: a real number.
      It is used for averaging T with identity.
    numb_iter: number of iteration to run rhe algorithm.

    Return
    ------
    values: a list of residuals: ||x - Tx||.
    x: the last iterate.
    """

    begin = perf_counter()

    x = x0
    Tx = T(x0)

    values = [J(x,Tx)]

    for _ in range(numb_iter):
        x = a * x + (1 - a) * Tx
        Tx = T(x)
        res = J(x,Tx)
        values.append(res)
    end = perf_counter()
    print("Time execution of K-M:", end - begin)

    return values, x

def fixed_point_agraal(T, x1,  numb_iter=100, phi=1.5, output=False):
    """
    Adaptive Golden Ratio Algorithm for x = Tx

    Input 
    -----
    T: main operator.
        Takes x as input.
    x1: Starting point.
        np.array, must be consistent with T.
    numb_iter: number of iteration to run rhe algorithm.
    phi: a key parameter for the algorithm.
         Must be between 1 and the golden ratio, 1.618... Choice
         phi=1.5 seems to be one of the best.
    output: boolean.  
         If true, prints the length of a stepsize in every iteration.  
         Useful for monitoring.

    Return
    ------
    values: a list of residuals: ||x - Tx||.
    x,  : last iterate.
    step_list: list of all stepsizes.
    """

    begin = perf_counter()

    x, x_ = x1.copy(), x1.copy()
    tau = 1. / phi + 1. / phi**2

    F = lambda x: x - T(x)
    x0 = x + np.random.randn(x.shape[0]) * 1e-6
    Fx = F(x)
    la = 1
    print(la)
    step_list = [la]
    th = 1

    values = [JF(Fx)]

    for i in range(numb_iter):
        x1 = x_ - la * Fx
        Fx1 = F(x1)

        n1 = LA.norm(x1 - x)**2
        n2 = LA.norm(Fx1 - Fx)**2
        la1 = min(tau * la, 0.25 * phi * th / la * (n1 / n2))
        x_ = ((phi - 1) * x1 + x_) / phi
        th = phi * la1 / la
        x, la, Fx = x1, la1, Fx1

        if output:
            print(la)
        res = JF(Fx)
        values.append(res)
        step_list.append(la1)
    end = perf_counter()

    print("Time execution of aGRAAL:", end - begin)
    return values, x, step_list
