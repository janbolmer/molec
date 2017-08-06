#! usr/bin/env python

import pymc, math, argparse, os, sys, time

import numpy as np
import pandas as pd

import matplotlib as plt

from scipy.special import wofz
from scipy.interpolate import interp1d

sys.path.append('bin/')

from spec_functions import * # spec_functions.py
from syn_spec import * # syn_spec.py
from sns_plots import * # sns_plots.py

from astropy.convolution import Gaussian1DKernel, convolve
# physical constants

m_e = 9.1095e-28 # g
e = 4.8032e-10 # cgs
c = 2.998e10 # cm/s

l0 = 1608.45 # AA
f = 0.05399224 # unitless 
gamma = 2.434e+08 # 1/s
N = 19.0

velos = np.arange(-400, 400, 0.5)
norm_spec = np.ones(len(velos))

v0 = -10
line = 1608.45
A = 33
sigma = 12.0
gamma = (gamma * line * 10e-13) / (2 * math.pi)

const = (math.pi * e**2) / (m_e * c)
a = (gamma * c) / (4*math.pi*v0*sigma*math.sqrt(2))
u = velos / (sigma*math.sqrt(2))


wav_aa, n_flux, n_flux_err, flux, flux_err, grb_name, \
	res, psf_fwhm = get_data("spectra/GRB120815A_OB1VIS.txt", 2.358, wl_range=False)

velocity, fluxv, fluxv_err = aa_to_velo(wav_aa, n_flux,
		n_flux_err, line, 2.385, 20)

transform = np.median(np.diff(velocity))

fig = figure(figsize=(10, 6))
ax = fig.add_axes([0.13, 0.13, 0.85, 0.80])

ff = []
ff2 = []
for vv in velos:
	V = -A * voigt(vv, sigma, gamma)
	gauss_kernel = Gaussian1DKernel(stddev=28.0/((2*np.sqrt(2*np.log(2)))*transform), x_size=1, mode="oversample")
	V2 = convolve(V, gauss_kernel)

	ff.append(V + 1)
	ff2.append(V2 + 1)

ax.errorbar(velocity, fluxv, color="gray", fmt="o")
ax.plot(velos, ff, label='Voigt', color="blue", linewidth=2)
ax.plot(velos, ff2, label='Voigt', color="red", linewidth=2)

show()





