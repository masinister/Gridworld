import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot(legend_var):
    pallete = cm.coolwarm
    data = np.load('data.npy', allow_pickle = True)
    cmap = [np.real(row[0][legend_var]) for row in data]
    std_cmap = (cmap - np.mean(cmap)) / np.std(cmap)
    norm_cmap = (cmap - np.min(cmap)) / (np.max(cmap) - np.min(cmap))

    for i in range(len(data)):
        v = np.real(data[i][0][legend_var])
        plt.plot(data[i][-1], label = np.float("{:.3f}".format(v)), c=pallete(std_cmap[i]))

    plt.title(legend_var)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    # plt.legend(loc = 'lower right')
    m = plt.cm.ScalarMappable(cmap=pallete)
    m.set_array(cmap)
    plt.colorbar(m)
    plt.show()

if __name__ == "__main__":
    plot("avg_eccentricity")
