#! /usr/bin/python

import sys
import math
import scipy
import numpy

import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *

import numpy.ma as ma
from argparse import ArgumentParser

from astropy.io import fits


def get_spec_data(file='sci-lred0073.fits'):

	hdulist = fits.open(file)

	table_data = hdulist[0].data
	
	wav_aa, flux, flux_err = [], [], []
	#qual, cont, cont_err = [], [], []
	#norm_flux, norm_flux_err, jy_flux = [], [], []

	for i in table_data:
		wav_aa.append(float(i[0]))
		flux.append(float(i[1]))
		flux_err.append(float(i[2]))
		#qual.append(float(i[3]))
		#cont.append(float(i[4]))
		#cont_err.append(float(i[5]))
		#norm_flux.append(float(i[1])/float(i[4]))
		#norm_flux_err.append(((float(i[2])+float(i[1]))/float(i[4])) - (float(i[1])/float(i[4])))
		#jy_flux.append(ergJy(float(i[1]), float(i[0])))

	hdulist.close()

	return wav_aa, flux, flux_err

wav_aa, flux, flux_err = get_spec_data('sci-lred0073.fits')

print wav_aa[0:10]

print flux[0:10]

print flux_err[0:10]



#errorbar(wav_aa, flux, yerr=flux_err, fmt="+")
#yscale("log")
#show()







