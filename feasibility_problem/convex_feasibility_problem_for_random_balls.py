import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
from fixed_points import *
import matplotlib as mpl


mpl.rc('lines', linewidth=2)
mpl.rcParams.update(
    {'font.size': 12, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
mpl.rcParams['xtick.major.pad'] = 2
mpl.rcParams['ytick.major.pad'] = 2


n, m = 2000, 1000
n_exp = 100
N = 1000

zero = np.zeros(n)
km_array = np.empty((n_exp, N))
graal_array = np.empty((n_exp,N))



for i in range(n_exp):
    print("Iteration", i+1)
    # fix a random generator
    gen = i
    np.random.seed(gen)

    # define balls
    #C = np.random.uniform(-10, 10, (m, n))
    C = np.random.normal(0, 100, (m, n))
    R = LA.norm(C,axis=1) + 0.5

    # starting point
    #x0 = np.random.uniform(-100,100,n)
    x0 = np.mean(C, axis=0)



    # Define operator
    def T(x):
        dist = LA.norm(x - C, axis=1)
        ind = np.where(dist > R)
        # define the number of projection that we have to compute
        n_ind = ind[0].shape[0]
        C_ind = C[ind]
        # compute projections only for those balls that are needed
        Y = (R[ind] / dist[ind] * (x - C_ind).T).T + C_ind
        return ((np.sum(Y, axis=0) + (m - n_ind)*x)) / m

    # run algorithms
    ans1 = krasn_mann(T, x0, 0, numb_iter=N-1)
    ans2 = fixed_point_egraal(T, x0, numb_iter=N-1, phi=1.5, output=False)

    km_array[i] = ans1[0]
    graal_array[i] = ans2[0]

# show and save results
fig, ax, = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
km = ax.plot(km_array.T,'b')
gr = ax.plot(graal_array.T,'#FFD700')
ax.set_xlabel(u'iterations, $k$')
ax.set_ylabel('residual')
ax.set_yscale('log')
ax.set_xlim(0,N)

ax.legend([km[0], gr[0]], ['KM', 'EGRAAL'])
plt.savefig('figures/balls-n=2000.pdf',bbox_inches='tight')
plt.show()
plt.clf()
