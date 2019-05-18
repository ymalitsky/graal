import os
from matplotlib import pyplot as plt
import seaborn as sns

# comment the following line if seaborn is not installed
sns.set()

def plot_results(values, colors, labels, linestyles, filename):
    directory = 'figures'
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig_name = directory + '/' + filename[5:] + '.pdf'
    v_min = values.min()
    plt.figure(figsize=(6,4))
    
    for i,v in enumerate(values):
        plt.plot(v - v_min, color=colors[i], label=labels[i],
                 linestyle=linestyles[i])
    plt.yscale('log')
    plt.xlabel(u'iterations')
    plt.ylabel('$J(x^k)-J_{_*}$')    
    plt.legend()
    plt.savefig(fig_name, bbox_inches='tight')
    plt.clf()

    
