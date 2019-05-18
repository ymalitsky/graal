import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
from fixed_points import *
import seaborn as sns


# choose dimensions m, n
# choose how many experiments n_exp
# choose number of iterations

n, m = 1000, 2000
n_exp = 100
N = 1000

# intialization
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
    ans2 = fixed_point_agraal(T, x0, numb_iter=N-1, phi=1.5, output=False)

    km_array[i] = ans1[0]
    graal_array[i] = ans2[0]

# show and save results
sns.set() # comment this line if seaborn is not installed

fig, ax, = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
km = ax.plot(km_array.T,'b')
gr = ax.plot(graal_array.T,'#FFD700')
ax.set_xlabel(u'iterations, $k$')
ax.set_ylabel('residual')
ax.set_yscale('log')
ax.set_xlim(0,N)

ax.legend([km[0], gr[0]], ['KM', 'aGRAAL'])
plt.savefig('figures/balls-n={}-darkgrid.pdf'.format(n), bbox_inches='tight')
plt.show()
plt.clf()

np.save("saved_data/km_array-n={}.npy".format(n), km_array)
np.save("saved_data/gr_array-n={}.npy".format(n), graal_array)
