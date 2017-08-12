#! /usr/bin/python

__author__ = "Jan Bolmer"
__copyright__ = "Copyright 2017"
__version__ = "0.1"
__maintainer__ = "Jan Bolmer"
__email__ = "jbolmer@eso.org"
__status__ = "Production"

import pymc, math, argparse, os, sys, time

import numpy as np
import pandas as pd

import seaborn as sns

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


def velo_plot(spec_file, redshift, velo_range, comps, lvl=0):

	if lvl == 0:
		h2_list = ["L0R(0)", "L0R(1)", "L0P(1)", "L0R(2)", \
				"L0P(2)", "L0R(3)", "L0P(3)", "L0R(4)", "L0P(4)", \
				"L0R(5)", "L0P(5)"]

	if lvl == 1:
		h2_list = ["L1R(0)", "L1R(1)", "L1P(1)", "L1R(2)", \
				"L1P(2)", "L1R(3)", "L1P(3)", "L1R(4)", "L1P(4)", \
				"L1R(5)", "L1P(5)"]

	sns.set_style("white", {'legend.frameon': True})

	h2_name, h2_wav, h2_f, h2_gamma, h2_str = get_H2_lines()
	
	lines, names = [], []
	
	for i in np.arange(0, len(h2_name), 1):
		if h2_str[i] in h2_list:
			lines.append(h2_wav[i])
			names.append(h2_str[i])
	
	wav_aa, n_flux, n_flux_err, flux, flux_err, \
	grb_name, res, psf_fwhm = get_data(spec_file, redshift)
	
	fig = figure(figsize=(7, 12))
	
	ax1 = fig.add_axes([0.1, 0.07,  0.88, 0.082])
	ax2 = fig.add_axes([0.1, 0.152, 0.88, 0.082])
	ax3 = fig.add_axes([0.1, 0.234, 0.88, 0.082])
	ax4 = fig.add_axes([0.1, 0.316, 0.88, 0.082])
	ax5 = fig.add_axes([0.1, 0.398, 0.88, 0.082])
	ax6 = fig.add_axes([0.1, 0.480, 0.88, 0.082])
	ax7 = fig.add_axes([0.1, 0.562, 0.88, 0.082])
	ax8 = fig.add_axes([0.1, 0.644, 0.88, 0.082])
	ax9 = fig.add_axes([0.1, 0.726, 0.88, 0.082])
	ax10= fig.add_axes([0.1, 0.808, 0.88, 0.082])
	ax11= fig.add_axes([0.1, 0.890, 0.88, 0.082])

	axis = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11]
	
	for i in np.arange(0, len(axis), 1):
	
		velocity, fluxv, fluxv_err = aa_to_velo(wav_aa, n_flux, \
			n_flux_err, lines[i], redshift, wrange=50)
	
		for comp in comps:
			axis[i].axvline(comp, linestyle="dashed", color="#d7301f")

		axis[i].errorbar(velocity, fluxv, linestyle='-', \
			color="#08306b", linewidth=2, drawstyle='steps-mid')
	
		# yerr=fluxv_err, 

		axis[i].set_ylim([-0.25, 1.8])
		axis[i].set_xlim([-velo_range, velo_range])

		for side in ['top','bottom','left','right']:
	  		axis[i].spines[side].set_linewidth(2)
		axis[i].tick_params(which='major', length=8, width=2)
		axis[i].tick_params(which='minor', length=4, width=1.5)
	
		for tick in axis[i].xaxis.get_major_ticks():
			tick.label.set_fontsize(18)
		for tick in axis[i].yaxis.get_major_ticks():
			tick.label.set_fontsize(18)
	
		if axis[i] in [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11]:
			axis[i].tick_params(labelbottom='off')

		axis[i].axhline(1, linestyle="dotted", linewidth=1, color="black")
		axis[i].text(velo_range - velo_range/2.7, 1.45,
			names[i] + " " + str(round(lines[i],1)), fontsize=10)


	ax1.set_xlabel(r"$\sf Velocity\, (km/s)$", fontsize=22)
	ax6.set_ylabel(r"$\sf Normalized\, Flux$", fontsize=22)
	ax11.set_title(grb_name + " L" + str(lvl), fontsize=20)

	show()
	fig.savefig(grb_name + "_H2_" + "L" + str(lvl) + "_velo.pdf")


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('-f','--file',dest="file",
		default="spectra/GRB120815A_OB1UVB.txt",type=str)
	parser.add_argument('-z','--z', dest="z",
		default=2.358,type=float)
	parser.add_argument('-vr','--velo_range',dest="velo_range",
		default=410.0,type=float)
	parser.add_argument('-c','--components',dest="components", nargs='+',
		default=[])
	args = parser.parse_args()

	spec_file = args.file
	redshift = args.z
	velo_range = args.velo_range
	comps = args.components

	print comps

	velo_plot(spec_file, redshift, velo_range, comps, lvl=0)
	velo_plot(spec_file, redshift, velo_range, comps, lvl=1)






















