import matplotlib.pyplot as plt
def plotData(X, y):
    plot = plt.plot(X, y)
    plt.setp(plot, marker='x', markersize=5, markeredgecolor='red', linestyle='')
    plt.show()
