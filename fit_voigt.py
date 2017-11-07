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


def read_data(file, z, line, w):
	data = open(file, "r")
	wav_aa, flux, flux_err = [], [], []
	grb_name = ""
	res = 0
	psf_fwhm = 0

	wl1 = line*(1+z) - w
	wl2 = line*(1+z) + w

	for line in data:
		if not line.startswith(("GRB", "Resolution", "PSFFWHM")):
			if wl1 <= float(line.split()[0]) <= wl2:
				wav_aa = np.append(wav_aa,float(line.split()[0]))
				flux = np.append(flux,float(line.split()[1]))
				flux_err = np.append(flux_err,float(line.split()[2]))
			if line.startswith('GRB'):
				grb_name = str(line.split()[0]).split("_")[0]
			if line.startswith('Res'):
				res = float(line.split()[1])
			if line.startswith('PSF'):
				psf_fwhm = float(line.split()[1])

	return wav_aa, flux, flux_err, grb_name, res, psf_fwhm

def plot_spec(wav_aa, flux, flux_err, grb_name):

	sns.set_style("white", {'legend.frameon': True})

	fig = figure(figsize=(10, 6))
	ax = fig.add_axes([0.13, 0.15, 0.83, 0.78])

	ax.errorbar(wav_aa,flux,yerr=flux_err,color='gray',marker='o',
		ls='None',label='Observed')
	ax.plot(wav_aa,flux,drawstyle='steps-mid',color='gray',alpha=0.66)

	plt.title(grb_name, fontsize=24)

	#ax.set_xlim([wav_aa[0], wav_aa[:-1]])
	#ax.set_ylim([min(flux)-0.1*min(flux), max(flux)+0.1*max(flux)])

	ax.set_xlabel(r"$\sf Observed\, Wavelength (\AA)$", fontsize=24)
	ax.set_ylabel(r"$\sf Flux$", fontsize=24)

	for axis in ['top','bottom','left','right']:
	  ax.spines[axis].set_linewidth(2)
	ax.tick_params(which='major', length=8, width=2)
	ax.tick_params(which='minor', length=4, width=1.5)
	
	for tick in ax.xaxis.get_major_ticks():
	    tick.label.set_fontsize(18)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(18)

	show()


wav_aa, flux, flux_err, grb_name, res, psf_fwhm = read_data("spectra/GRB161023A_OB1VIS.txt",
	2.71, 1608.4506, 8)

plot_spec(wav_aa, flux, flux_err, grb_name)



















