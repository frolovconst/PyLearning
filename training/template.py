import numpy as np
from os.path import exists
from pathlib import  Path
FILE_NAME = 'stump.in'
LOCAL_FOLDER_NAME = 'stump'
LOCALDATA_ROOT_FOLDER_NAME = './data'


def read_input(sort):
    on_server = exists(FILE_NAME)
    base_path = Path('.') if on_server else Path(DATA_ROOT_FOLDER_NAME) / LOCAL_FOLDER_NAME
    with open(base_path + FILE_NAME, 'r') as f:
        n = int(f.readline())
        if n == 0:
            result = None
        else:
            result = np.empty((n, 2), dtype=int)
            for i in range(n):
                result[i, 0], result[i, 1] = map(int, f.readline().split(' '))
            if sort:
                result = result[np.argsort(result[:, 0])]
    return result, n


def optimize():
    points, n = read_input(sort=True)
    if n == 0:
        a = b = c = 0
    else:
        sse_a = DynamicSSE()
        sse_b = DynamicSSE()
        for point in points:
            sse_b.add(point[1])
        c = points[0, 0]
        a = sse_a.dyn_mean.mean
        b = sse_b.dyn_mean.mean

        for i, point in enumerate(points[:-1]):
            sse = sse_a.sse + sse_b.sse
            sse_a.add(points[i, 1])
            sse_b.remove(points[i, 1])
            if (point[0] == points[i+1, 0]):
                continue
            if sse_a.sse + sse_b.sse < sse:
                a = sse_a.dyn_mean.mean
                b = sse_b.dyn_mean.mean
                c = (points[i+1, 0] + points[i, 0]) / 2
    print(f'{a:.6f} {b:.6f} {c:.6f}')


class DynamicMean:
    def __init__(self, init_arr=None):
        if init_arr is None:
            self.mean = 0
            self.n = 0
        else:
            self.mean = np.mean(init_arr)
            self.n = len(init_arr)

    def add(self, value, weight=1):
        self.n += weight
        self.mean += weight * (value - self.mean) / self.n

    def remove(self, value, weight=1):
        self.add(value, -1 * weight)


class DynamicSSE:
    def __init__(self, init_arr=None):
        if init_arr is None:
            self.dyn_mean = DynamicMean()
            self.sse = 0
        else:
            self.dyn_mean = DynamicMean(init_arr)
            self.sse = np.sum(np.square(init_arr-self.dyn_mean.mean))

    def add(self, value):
        a1 = self.dyn_mean.mean
        n_mult = self.dyn_mean.n
        self.dyn_mean.add(value)
        a2 = self.dyn_mean.mean
        self.sse += (value-a2)*(value+a2) + n_mult*(a1-a2)*(a1+a2)

    def remove(self, value):
        a2 = self.dyn_mean.mean
        self.dyn_mean.remove(value)
        n_mult = self.dyn_mean.n
        a1 = self.dyn_mean.mean
        self.sse -= (value - a2) * (value + a2) + n_mult * (a1 - a2) * (a1 + a2)


def main():
    optimize()


if __name__ == '__main__':
    main()
