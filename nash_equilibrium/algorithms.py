
import numpy as np
import scipy as sp
import numpy.linalg as LA
from itertools import count
from time import perf_counter


def tseng_fbf_linesearch(J, F, prox_g, x0, delta=2, numb_iter=100):
    """
    Solve monotone inclusion $0 \in F + \partial g by Tseng
    forward-backward-forward algorithm with linesearch. Notice FBF
    algorithm in this form works only for Nash problem, see the update
    for x below
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
    return [iterates[i] for i in [0, 1,  -2, -1]]

def explicit_graal(J, F, prox_g, x1, numb_iter=100, phi=1.5, output=False):
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

    print("CPU time for EGRAAL:", end - begin)
    return values, x, x_, time_list


def explicit_graal_terminate(J, F, prox_g, x1, numb_iter=100, phi=1.5, tol=1e-6, output=False):
    """
    Explcit GRAAL but with termination criteria
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

    print("CPU time for EGRAAL:", end - begin)
    return values, x, i
