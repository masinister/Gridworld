import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot(legend_var):
    pallete = cm.coolwarm
    data = np.load('data.npy', allow_pickle = True)
    cmap = [np.real(row[0][legend_var]) for row in data]
    std_cmap = (cmap - np.mean(cmap)) / np.std(cmap)
    norm_cmap = (cmap - np.min(cmap)) / (np.max(cmap) - np.min(cmap))

    plt.figure(1)

    for i in range(len(data)):
        v = np.real(data[i][0][legend_var])
        plt.plot(data[i][1], label = np.float("{:.3f}".format(v)), c=pallete(std_cmap[i]))

    plt.title("Learning Curves vs. {}".format(legend_var))
    plt.xlabel("Time Step")
    plt.ylabel("Error")
    # plt.legend(loc = 'lower right')
    m = plt.cm.ScalarMappable(cmap=pallete)
    m.set_array(cmap)
    plt.colorbar(m)

    plt.figure(2)
    y = [len(row[1]) for row in data]
    plt.scatter(cmap, y)
    plt.title("Covergence Time vs. {}".format(legend_var))
    plt.xlabel(legend_var)
    plt.ylabel("Covergence Time")

    plt.figure(3)
    y = [row[2] for row in data]
    plt.scatter(cmap, y)
    plt.title("||Q| - |Q_opt|| vs. {}".format(legend_var))
    plt.xlabel(legend_var)
    plt.ylabel("||Q| - |Q_opt||")

    plt.figure(4)
    y = [row[3] for row in data]
    plt.scatter(cmap, y)
    plt.title("Q / Q_opt vs. {}".format(legend_var))
    plt.xlabel(legend_var)
    plt.ylabel("Q / Q_opt")


    plt.show()

if __name__ == "__main__":
    plot("num shortest paths")

"diameter"
"cover time"
"conductance"
"avg eccentricity"
"connectivity"
"efficiency"
"min eigenvalue",
"max eigenvalue"
"closeness vitality"
