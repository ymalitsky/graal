# algorithms for x = Tx

import numpy as np
import scipy as sp
import scipy.linalg as LA
from time import perf_counter

# measure the gap ||x-Tx||
J = lambda x, Tx: LA.norm(x-Tx)


# measure the same gap ||x-Tx|| = ||Fx|| but for VIP
JF = lambda x: LA.norm(x)

def krasn_mann(T, x0, a=0.5, numb_iter=100):
    """
    Krasnoselskii-Mann scheme for x = Tx
    In every iteration we measure the residual ||x - Tx||
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

def fixed_point_egraal(T, x1,  numb_iter=100, phi=1.5, output=False):
    """
    Golden Ratio Algorithm for the problem x = Tx

    T is the operator
    x0 is the starting point

    """
    begin = perf_counter()

    x, x_ = x1.copy(), x1.copy()
    tau = 1. / phi + 1. / phi**2

    F = lambda x: x - T(x)
    x0 = x + np.random.randn(x.shape[0]) * 1e-6
    Fx = F(x)
    #la = phi / 2 * LA.norm(x - x0) / LA.norm(Fx - F(x0))
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
        # la1 = min(tau * la, 0.25 * phi * th / la * safe_division(n1, n2))
        la1 = min(tau * la, 0.25 * phi * th / la * (n1 / n2))
        x_ = ((phi - 1) * x1 + x_) / phi
        th = phi * la1 / la
        x, la, Fx = x1, la1, Fx1

        if output:
            print(la)
        res = JF(Fx)
        #res = LA.norm(Fx1)
        values.append(res)
        step_list.append(la1)
    end = perf_counter()

    print("Time execution of EGRAAL:", end - begin)
    return values, x, step_list
