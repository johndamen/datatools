from datatools import transformations
from matplotlib import pyplot as plt
import numpy as np

x = np.arange(0, 2*np.pi, .01)
y = np.arange(0, 2*np.pi, .01)
X, Y = np.meshgrid(x, y)
noise = 2 * np.random.random(X.shape) - 1
big_wave = np.sin(2 * X)
small_wave = .3*np.sin(10*Y)
Z = big_wave + small_wave + noise
Z /= Z.max()  # normalize


# make a fourier transform
# remove the average
# apply gaussian tapering function
F = transformations.Fourier.transform(x, y, Z, remove_avg=True)

# remove all frequencies not in the specified band
# in this case a wavenumber
# also possible to apply directional filter
F_filt = F.filter(kmin=1, kmax=2)

# revert to normal data from filtered fourier spectrum
filtered_Z = F_filt.reverse()


fig, axes = plt.subplots(figsize=(12, 12), nrows=2, ncols=2)
axes[0, 0].pcolormesh(x, y, Z, vmin=-1, vmax=1)

axes[0, 1].pcolormesh(F.kx, F.ky, F.amplitude, cmap='inferno', vmin=0)
axes[0, 1].set_xlim(-10, 10)
axes[0, 1].set_ylim(-10, 10)

axes[1, 0].pcolormesh(F_filt.kx, F_filt.ky, F_filt.amplitude, cmap='inferno', vmin=0)
axes[1, 0].set_xlim(-10, 10)
axes[1, 0].set_ylim(-10, 10)

axes[1, 1].pcolormesh(x, y, filtered_Z, vmin=-1, vmax=1)


# make a fourier transform
# remove the average
# apply gaussian tapering function
F = transformations.Fourier.transform(x, y, Z, remove_avg=True)

# remove all frequencies not in the specified band
# apply a directional filter
F_filt = F.filter(theta=90, theta_offset=50)

# revert to normal data from filtered fourier spectrum
filtered_Z = F_filt.reverse()


fig, axes = plt.subplots(figsize=(12, 12), nrows=2, ncols=2)
axes[0, 0].pcolormesh(x, y, Z, vmin=-1, vmax=1)

axes[0, 1].pcolormesh(F.kx, F.ky, F.amplitude, cmap='inferno', vmin=0)
axes[0, 1].set_xlim(-12, 12)
axes[0, 1].set_ylim(-12, 12)

axes[1, 0].pcolormesh(F_filt.kx, F_filt.ky, F_filt.amplitude, cmap='inferno', vmin=0)
axes[1, 0].set_xlim(-12, 12)
axes[1, 0].set_ylim(-12, 12)

axes[1, 1].pcolormesh(x, y, filtered_Z, vmin=-1, vmax=1)



# make a fourier transform for an area of 100x100 metres
F = transformations.Fourier.transform(np.linspace(0, 100, x.size), np.linspace(0, 100, y.size), Z, remove_avg=True)

# only keep the small wave (wavelength 10 m)
F_filt = F.filter(kmin=1./11, kmax=1./9)
filtered_Z = F_filt.reverse()


fig, axes = plt.subplots(figsize=(12, 12), nrows=2, ncols=2)
axes[0, 0].pcolormesh(x, y, Z, vmin=-1, vmax=1)

axes[0, 1].pcolormesh(F.kx, F.ky, F.amplitude, cmap='inferno', vmin=0)
axes[0, 1].set_xlim(-.12, .12)
axes[0, 1].set_ylim(-.12, .12)

axes[1, 0].pcolormesh(F_filt.kx, F_filt.ky, F_filt.amplitude, cmap='inferno', vmin=0)
axes[1, 0].set_xlim(-.12, .12)
axes[1, 0].set_ylim(-.12, .12)

axes[1, 1].pcolormesh(x, y, filtered_Z, vmin=-1, vmax=1)


plt.show()

