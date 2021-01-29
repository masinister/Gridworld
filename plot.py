import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot(legend_var):
    data = np.load('data.npy', allow_pickle = True)
    cmap = [np.real(row[0][legend_var]) for row in data]
    std_cmap = (cmap - np.mean(cmap)) / np.std(cmap)
    for i in range(len(data)):
        v = np.real(data[i][0][legend_var])
        plt.plot(data[i][-1], label = np.float("{:.3f}".format(v)), c=cm.jet(std_cmap[i]))

    plt.title(legend_var)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    # plt.legend(loc = 'lower right')
    m = plt.cm.ScalarMappable(cmap=cm.jet)
    m.set_array(cmap)
    plt.colorbar(m)
    plt.show()

if __name__ == "__main__":
    plot("connectivity")
