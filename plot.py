import numpy as np
import matplotlib.pyplot as plt

def plot():
    data = np.load('data.npy', allow_pickle = True)

    for row in data:
        plt.plot(row[-1][0], label = [np.float("{:.3f}".format(np.real(x))) for x in row[:-1][0]])

    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.legend(loc = 'upper left')
    plt.show()

if __name__ == "__main__":
    plot()
