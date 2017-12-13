#! /usr/bin/python

import math, argparse, os, sys, time

import numpy as np
import pandas as pd

import matplotlib as plt

sys.path.append('bin/')

from spec_functions import * # spec_functions.py

def get_spec(file):

	wav_aa, flux, flux_err, model = [], [], [], []
	grb_name = ""
	res = 0
	psf_fwhm = 0

	data = open(file, "r")

	for line in data:
		if not line.startswith(("GRB", "Resolution", "PSFFWHM")):
			wav_aa = np.append(wav_aa,float(line.split()[0]))
			flux = np.append(flux,float(line.split()[1]))
			flux_err = np.append(flux_err,float(line.split()[2]))
			model = np.append(model,float(line.split()[4]))
		if line.startswith('GRB'):
			grb_name = str(line.split()[0]).split("_")[0]
		if line.startswith('Res'):
			res = float(line.split()[1])
		if line.startswith('PSF'):
			psf_fwhm = float(line.split()[1])

	return wav_aa, flux, flux_err, model, grb_name

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-f','--file',dest="file",
		default="spectra/GRB120815Auvb.txt", type=str)
	args = parser.parse_args()
	
	spec_file = args.file

	wav_aa, flux, flux_err, model, grb_name = get_spec(spec_file)
	
	fig = figure(figsize=(16, 4))
	
	ax1 = fig.add_axes([0.07, 0.15, 0.92, 0.78])
	
	ax1.errorbar(wav_aa, flux, linestyle='-', color="black", linewidth=0.4, drawstyle='steps-mid', label=r"$\sf Data$", alpha=0.6)

	ax1.plot(wav_aa, model, color="#2171b5", linewidth=3.0, alpha=1.0)
	
	ylim([1e-19, 1e-15])
	yscale("log")
	
	plt.title(str(grb_name))

	ax1.set_xlabel("Observed Wavelength")	
	ax1.set_ylabel("Flux")

	show()
	fig.savefig(grb_name+"_inspec.pdf")
	

if __name__ == "__main__":
	
	main()	


