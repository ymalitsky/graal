import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns

# mpl.rc('lines', linewidth=2)
# mpl.rcParams.update(
#     {'font.size': 13, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
# mpl.rcParams['xtick.major.pad'] = 2
# mpl.rcParams['ytick.major.pad'] = 2

sns.set()

def plot_results(values, colors, labels, linestyles, filename):
    directory = 'figures'
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig_name = directory + '/' + filename[5:] + '.pdf'
    #v_min = min([min(v) for v in values])
    v_min = values.min()
    plt.figure(figsize=(6,4))
    for i,v in enumerate(values):
        plt.plot(v - v_min, color=colors[i], label=labels[i], linestyle=linestyles[i])
    plt.yscale('log')
    plt.xlabel(u'iterations')
    plt.ylabel('$J(x^k)-J_{_*}$')
    #plt.ylabel('$h(x^k)-h_{_*}$')
    
    plt.legend()
    plt.savefig(fig_name, bbox_inches='tight')
    plt.clf()

    
