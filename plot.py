import numpy as np
import matplotlib.pyplot as plt

def plot():
    data = np.load('data.npy', allow_pickle = True)

    for row in data:
        plt.plot(row[-1][0], label = row[:-1][0])

    plt.legend(loc = 'upper left')
    plt.show()

if __name__ == "__main__":
    plot()
