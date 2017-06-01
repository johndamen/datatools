from __future__ import division, print_function
import numpy as np
from .transform import Transform
from .utils import prepare as prepare_dataset


__all__ = ['Fourier', 'filter']


class Fourier(Transform):

    def __init__(self, kx, ky, S, cellsize=25, area=None):
        self.kx = kx
        self.ky = ky
        self.S = S

        if isinstance(cellsize, int):
            cellsize = (cellsize, cellsize)
        self.cellsize = cellsize

        if area is None:
            # area = cellsize[0] * S.shape[0] * cellsize[1] * S.shape[1]
            area = S.shape[0] * S.shape[1]
        self.area = area

    @property
    def shape(self):
        """return the shape of the fourier transform spectrum"""
        return self.S.shape

    @property
    def amplitude(self):
        """return the amplitude as 2*abs(S/datasize)"""
        return 2*np.absolute(self.S/self.area)

    @property
    def steepness(self):
        """return the steepness as H*K or H/L with H = 2*amplitude"""
        return 2* self.amplitude * self.K

    @property
    def K(self):
        """return a matrix with the wave numbers"""
        Kx, Ky = np.meshgrid(self.kx, self.ky)
        return (Kx**2+Ky**2)**.5

    @classmethod
    def transform(cls, x, y, Z, taper=False, remove_avg=False):
        """
        returns a shifted fourier transform (fft2) of the data

        :param x: numpy array (1D) with x coordinates
        :param y: numpy array (1D) with y coordinates
        :param Z: numpy array (2D) with data values (no numpy MaskedArray)
        :return: Fourier object

        the returned value is is Fourier object with the following important attributes:
        .kx         wave number in horizontal direction
        .ky         wave number in vertical direction
        .S          spectrum
        .K          absolute wave number (kx**2+ky**2)**.5
        .amplitude  amplitude representation
        .steepness  amplitude*absolute wave number
        .shape      shape of the spectrum

        and the following methods:
        .reverse()        reverse the fourier to retrieve the data
        .plot(axes)       plot the fourier transform on the supplied axis
        .apply_mask(mask) multiplies the spectrum by the supplied mask to modify the spectrum (eg. filtering)
        """

        Z = cls.prepare(Z, taper=taper, avg=remove_avg)

        # difference first 2 values along x and y
        cellsize = (np.abs(x[1] - x[0]), np.abs(y[1] - y[0]))

        if (np.absolute(np.diff(x) - cellsize[0]) > cellsize[0]*.001).any():
            raise ValueError('x not equally spaced')
        if (np.absolute(np.diff(y) - cellsize[1]) > cellsize[1]*.001).any():
            raise ValueError('y not equally spaced')

        if isinstance(Z, np.ma.MaskedArray):
            raise TypeError('input for fourier transform may not be of type MaskedArray')

        # # length x * length y
        # area = cellsize[0] * Z.shape[0] * cellsize[1] * Z.shape[1]

        # create wave number vectors
        kx = 2*np.pi*np.linspace(-.5, .5, Z.shape[1])/cellsize[1]
        ky = 2*np.pi*np.linspace(-.5, .5, Z.shape[0])/cellsize[0]

        # spectrum
        S = np.fft.fftshift(np.fft.fft2(Z))

        # return a fourier object
        return cls(kx, ky, S, cellsize=cellsize)

    def reverse(self, shape=None, nanmask=None):
        """
        return a dataset from the fourier spectrum

        :param shape: shape of the output (should be equal to the original dataset)
        :param nanmask: numpy bool array of nan values where true means to mask
        :return: new dataset
        """

        # determine shape
        if shape is None:
            shape = self.kx.shape[0], self.ky.shape[0]

        # shift spectrum
        S = np.fft.ifftshift(self.S)

        # reverse fourier of spectrum
        data = np.real(np.fft.ifft2(S, shape))

        # apply mask for missing values
        if nanmask is not None:
            return np.ma.array(data, mask=nanmask)
        else:
            return data

    def apply_mask(self, m, shift=False):
        """

        :param m:
        :param shift:
        :return:
        """
        if shift:
            m = np.fft.fftshift(m)

        S = self.S*m

        return type(self)(self.kx, self.ky, S,
                          cellsize=self.cellsize,
                          area=self.area)

    def filter(self, kmin=0, kmax=np.inf, theta=None, theta_offset=180, dirmode='deg', spatial_frequencies=True):
        """
        Filter the spectrum based on wave number or angle
        :param kmin: mimumum wave number as spatial or angular (see spatial_frequencies) with default 0
        :param kmax: maximum wave number as spatial or angular (see spatial_frequencies) with default np.inf
        :param theta: mean angle of features to keep
        :param theta_offset: offset from theta to either side
        :param dirmode: specifies angle definition (deg|rad)
        :param spatial_frequencies: specifies if wave numbers are spatial (1/L) or angular (2pi/L)
        :return: new Fourier object with masked frequencies
        """
        if spatial_frequencies:
            kmin = 2 * np.pi * kmin
            kmax = 2 * np.pi * kmax

        mask = (self.K > kmin) & (self.K < kmax)

        if theta is not None:
            if dirmode == 'deg':
                theta = theta * np.pi / 180
                theta_offset = theta_offset * np.pi / 180
            elif dirmode == 'rad':
                pass
            else:
                raise ValueError('invalid direction mode')

            x = np.linspace(-1, 1, self.kx.size)
            y = np.linspace(-1, 1, self.ky.size)
            X, Y = np.meshgrid(x, y)
            angle_diff = np.absolute((theta - np.arctan2(Y, X) + .5 * np.pi) % np.pi - .5 * np.pi)

            mask = np.logical_and(mask, angle_diff <= theta_offset)

        return self.apply_mask(mask)

    def plot(self, ax, attr='steepness', labelsize=14, zero_lines=True, spatial_frequencies=True, vmin=0, cmap='inferno', **kw):
        if spatial_frequencies:
            x = self.kx / 2 * np.pi
            y = self.ky / 2 * np.pi
            xlabel = r'$\xi_x$'
            ylabel = r'$\xi_y$'
        else:
            x = self.kx
            y = self.ky
            xlabel = r'$k_x$'
            ylabel = r'$k_y$'

        c = ax.pcolormesh(x, y, getattr(self, attr), cmap=cmap, vmin=vmin, **kw)
        ax.set_xlabel(xlabel, size=labelsize)
        ax.set_ylabel(ylabel, size=labelsize)
        ax.set_aspect('equal')

        if zero_lines:
            ax.axhline(0, color=(.9, .9, .9), lw=.5)
            ax.axvline(0, color=(.9, .9, .9), lw=.5)
        return c

    @classmethod
    def prepare(cls, data, taper=True, avg=True):
        return prepare_dataset(data, unmask=True, taper=taper, avg=avg)


def filter(x, y, data, **kwargs):
    return Fourier.transform(x, y, data)\
        .filter(**kwargs)\
        .reverse(nanmask=data.mask)
