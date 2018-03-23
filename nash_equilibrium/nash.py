import numpy as np
import scipy as sp
import scipy.linalg as LA
import matplotlib.pyplot as plt
from algorithms import *

import matplotlib as mpl

mpl.rc('lines', linewidth=2)
mpl.rcParams.update(
    {'font.size': 12, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
mpl.rcParams['xtick.major.pad'] = 2
mpl.rcParams['ytick.major.pad'] = 2

n = 1000
n_exp = 10
N = 50000
#gamma = 1.1
gamma = 1.5

zero = np.zeros(n)
fbf_array = np.empty((n_exp, N))
graal_array = np.empty((n_exp,N))

for i in range(n_exp):
    print("Iteration", i+1)
    # fix a random generator
    gen = i
    np.random.seed(gen)
    c = np.random.uniform(1,100,n)
    L = np.random.uniform(0.5,5,n)
    #beta = np.random.uniform(0.5, 2,n)
    beta = np.random.uniform(0.3, 4,n)
    q0 = np.ones(n)

    p = lambda Q: (5000**(1./gamma)) * (Q**(-1./gamma))

    def f(q):
        t = 1./beta
        res = c * q + 1./(1.+t)*(L**t * q**(1+t))
        return  res

    def df(q):
        t = 1./beta
        res = c + (L* q)**t
        return res

    dp = lambda Q: -1./gamma * (5000**(1./gamma)) * (Q**(-1./gamma -1))

    def F(q):
        Q = q.sum()
        res = df(q) - p(Q) - q*dp(Q)
        return res


    def prox_g(q, eps):
        return np.fmax(q,0)

    def proj(q):
        return np.fmax(q,0)

    J = lambda x: LA.norm(x-prox_g(x-F(x),1))

    ans1 = tseng_fbf_linesearch(J, F, prox_g, q0, delta=1.5, numb_iter=N-1)
    ans2 = explicit_graal(J, F, prox_g, q0, numb_iter=N-1, phi=1.5, output=False)

    fbf_array[i] = ans1[0]
    graal_array[i] = ans2[0]





fig, ax, = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
fbf = ax.plot(fbf_array.T,'b')
gr = ax.plot(graal_array.T,'#FFD700')
ax.set_xlabel(u'iterations, $k$')
ax.set_ylabel('residual')
ax.set_yscale('log')

ax.legend([fbf[0], gr[0]], ['FBF', 'EGRAAL'])
plt.savefig('figures/nash-2.pdf',bbox_inches='tight')
plt.show()
plt.clf()
