from skimage.transform import radon
import numpy as np
from .transform import Transform


class Radon(Transform):

    def __init__(self, theta, b, data, cellsize=25):
        self.theta = theta
        self.b = b
        self.data = data
        self.cellsize = cellsize

    @classmethod
    def transform(cls, x, y, Z,tmin=0,tmax=180):
        """
        returns a radon transform of the data

        the returned value is is Radon object with the following important attributes:
        .theta     rotations
        .b         offsets
        .data      intensities
        .cellsize  cellsize of the input data

        and the following methods:
        .plot(axes)       plot the radon transform on the supplied axis

        :param x: numpy array (1D) with x coordinates
        :param y: numpy array (1D) with y coordinates
        :param Z: numpy array (2D) with data values (no numpy MaskedArray)
        :return: Radon object
        """
        cellsize = np.abs(x[1] - x[0])
        assert cellsize == np.abs(y[1] - y[0]), 'cellsize not equal in both directions'

        theta = np.linspace(tmin, tmax, max(Z.shape), endpoint=False)
        result = radon(Z, theta=theta) * cellsize
        b = cellsize * np.arange(-result.shape[0]//2, result.shape[0] - result.shape[0]//2)

        return cls(theta, b, result, cellsize=cellsize)



if __name__ == '__main__':
    x, y = np.linspace(0, 2*np.pi, 200), np.linspace(0, 2*np.pi, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(8*Y)

    import utils
    from matplotlib import pyplot as plt
    Z = utils.prepare(Z, unmask=True, taper=True, avg=True)

    fig = plt.figure()
    ax = fig.gca()
    ax.pcolormesh(x, y, Z)

    f = Radon.transform(x, y, Z)
    fig = plt.figure()
    ax = fig.gca()
    c = f.plot(ax)
    fig.colorbar(c)

    plt.show()