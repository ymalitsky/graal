import numpy as np
import scipy as sp
import scipy.linalg as LA
from time import perf_counter

# Golden ratio algorithms


def golden_ratio_alg(J, F, prox_g, x0, la, numb_iter=100):
    x = x0
    x_ = x0
    phi = (np.sqrt(5) + 1) / 2
    values = [J(x0)]
    for i in range(numb_iter):
        x = prox_g(x_ - la * F(x), la)
        x_ = ((phi - 1) * x + x_) / phi
        values.append(J(x))
    return values, x, x_



def adaptive_graal(J, F, prox_g, x0, numb_iter=100, phi=1.5, output=False):
    """
    Explcit GRAAL
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

    print("Time execution:", end - begin)
    return values, x, x_, time_list


def adaptive_graal_terminate(J, F, prox_g, x1, numb_iter=100, phi=1.5, tol=1e-6, output=False):
    """
    Explcit GRAAL but with termination criteria
    """
    begin = perf_counter()

    x, x_ = x1.copy(), x1.copy()
    #x0 = x + np.random.randn(x.shape[0]) * 1e-9
    x0 = x + np.random.randn(x.shape[0]) 
    ##x0 = x + np.ones(x.shape[0]) * 1e-6
    Fx = F(x)
    la = phi / 2 * LA.norm(x - x0) / LA.norm(Fx - F(x0))
    #la = 1e-9
    rho = 1. / phi + 1. / phi**2
    values = [J(x)]
    time_list = [perf_counter() - begin]
    th = 1

    i = 1
    while i <= numb_iter and values[-1] > tol:
        i += 1
    #for i in range(numb_iter):
        x1 = prox_g(x_ - la * Fx, la)
        Fx1 = F(x1)
        #print(LA.norm(Fx1))
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
    return values, x, i

