#! /usr/bin/python

import pymc, math, argparse, os, sys, time

import numpy as np
import pandas as pd

import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *

sys.path.append('bin/')

from spec_functions import get_data, aa_to_velo

#========================================================================
#========================================================================


def get_H2_lines():

	h2_name, h2_wav, h2_f, h2_gamma, h2_str = [], [], [], [], []

	h2_file = open("atoms/h2.dat", "r")

	for line in h2_file:
		ss = line.split()
		h2_name.append(str(ss[0]))
		h2_wav.append(float(ss[1]))
		h2_f.append(float(ss[2]))
		h2_gamma.append(float(ss[3]))
		h2_str.append(str(ss[7]) + str(ss[8]) + str(ss[9]))

	h2_file.close()

	return h2_name, h2_wav, h2_f, h2_gamma, h2_str


def velo_plot():

	h2_name, h2_wav, h2_f, h2_gamma, h2_str = get_H2_lines()
	
	lines, names = [], []
	
	for i in np.arange(0, len(h2_name), 1):
		if h2_str[i] in ["L0R(0)", "L1R(0)", "L1R(1)", "L4R(0)", \
			"L8R(0)"]:

			lines.append(h2_wav[i])
			names.append(h2_str[i])
	
	
	redshift = 2.358
	spec_file = "spectra/GRB120815Auvb.txt"
	
	wav_aa, n_flux, n_flux_err, flux, flux_err = get_data(spec_file, \
		redshift)
	
	
	fig = figure(figsize=(10, 12))
	
	ax1 = fig.add_axes([0.07, 0.09, 0.92, 0.13])
	ax2 = fig.add_axes([0.07, 0.28, 0.92, 0.13])
	ax3 = fig.add_axes([0.07, 0.47, 0.92, 0.13])
	ax4 = fig.add_axes([0.07, 0.66, 0.92, 0.13])
	ax5 = fig.add_axes([0.07, 0.85, 0.92, 0.13])
	
	axis = [ax1, ax2, ax3, ax4, ax5]
	
	
	for i in np.arange(0, len(axis), 1):
	
		velocity, fluxv, fluxv_err = aa_to_velo(wav_aa, n_flux, \
			n_flux_err, lines[i], redshift, wrange=20)
	
		axis[i].errorbar(velocity, fluxv, yerr=fluxv_err, linestyle='-', \
			color="black", linewidth=2, drawstyle='steps-mid')
	
		axis[i].set_ylim([-1, 2])
		axis[i].set_title(names[i])
		

		#axis[i].text(-5, 1.8, names[i])
	
	
	ax1.set_xlabel(r"$\sf Velocity\, (km/s)$", fontsize=22)
	ax3.set_ylabel(r"$\sf Normalized\, Flux$", fontsize=22)
	
	
	fig.savefig("test.pdf")


velo_plot()





















