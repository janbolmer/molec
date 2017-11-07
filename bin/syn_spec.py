#! /usr/bin/python

import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *
import numpy as np
from scipy.ndimage import gaussian_filter1d as gauss1d
from scipy.interpolate import splrep, splev
from scipy.special import wofz

from spec_functions import *

# get H2, CO and other lines

h2 = open ('atoms/h2.dat', 'r')
h2_lines = h2.readlines()
h2.close()

co = open ('atoms/co.dat', 'r')
co_lines = co.readlines()
co.close()

hd = open ('atoms/hd.dat', 'r')
hd_lines = hd.readlines()
hd.close()

af = open ('atoms/atom_excited.dat', 'r')
atom = af.readlines()
af.close()

#H2 vib model from Draine 2000, 2002

vibH2 = ["h2abs_list_n3_b5_260_1000.dat",
"h2abs_spec_n3_b5_1e3_1000.dat",
"h2abs_spec_n3_b5_1e4_1000.dat",
"h2abs_spec_n3_b5_260_1000.dat",
"h2abs_spec_n3_b5_350_1000.dat",
"h2abs_spec_n3_b5_3e3_1000.dat",
"h2abs_spec_n3_b5_5e4_1000.dat"]

vibH2data = np.genfromtxt("atoms/vibH2/" + vibH2[2])
h2swl, modspec, tauspec = [vibH2data[:, i] for i in [0, 1, 2]]
tauspec = np.array(tauspec[::-1])

#========================================================================
#========================================================================

class SynSpec(object):
	"""
	Class for adding absorption lines to a normalized spectrum,
	including absorption lines from ions and excited transitions as
	well as H2, H2* and CO.

	Arguments:
		wav_range:	(rest-frame) wavelength range to be covered
		redshift: 	redshift

	Methods:
		add_ion: 	adds absorption lines for given ion/element
		add_H2:		adds Lyman and Werner bands of H2
		add_vibH2:	adds vibrational excited levels of H2*
		add CO:		adds CO bandheads
		add HD:		adds HD lines
	"""

	def __init__(self, wav_range, redshift):

		self.wav_range = wav_range
		self.redshift = redshift

	def __len__(self):
		return len(wav_range)

	def add_ion(self, spectrum, atom_name, broad, Natom, A_REDSHIFT):
		'''
		broad:			Broadening parameter in km/s
		atom_name:		Same naming convention as in atom.dat
		Natom:			Total column densities
		A_REDSHIFT:		Offset in redshift in #/100000.0
		'''

		redshift = self.redshift
		wav_range = self.wav_range
		spec = spectrum
		broad = broad * 1E5
		nion = 10**Natom
		redshift = redshift + A_REDSHIFT

		for a in atom:
			a = a.split()
			if str(a[0]) == str(atom_name):
				if min(wav_range) < float(a[1])*(1+redshift) < max(wav_range):
					lamb = float(a[1])
					f = float(a[2])
					gamma = float(a[3])
					spec *= addAbs(wav_range, nion, lamb, f, gamma, broad, \
						redshift)
		return spec

	def add_H2(self, spectrum, broad, NTOTH2, TEMP, A_REDSHIFT, NROT):
		'''
		Adding H2
		broad:		Broadening parameter in km/s
		NTOTH2:		Total column density of H2
		TEMP:		Temperature of H2 (Kelvin)
		A_REDSHIFT:	Offset in redshift in #/100000.0
		NROT:		Rotational levels to consider, maximum 7
		'''

		redshift = self.redshift
		wav_range = self.wav_range
		spec = spectrum

		nJ, NH2 = [], {}
		for J in NROT:
			nJ.append(fNHII(TEMP, J))
		for nj in NROT:
			NH2['H2J%i' %nj] = 10**NTOTH2/sum(nJ)*nJ[nj]

		broad = broad * 1E5
		redshift = redshift + A_REDSHIFT

		for h2 in h2_lines:
			h2 = h2.split()
			if h2[0] in NH2.keys():
				lamb = float(h2[1])
				f = float(h2[2])
				gamma = float(h2[3])
				nion = NH2[h2[0]]
				spec *= addAbs(wav_range, nion, lamb, f, gamma, broad, \
					redshift)

		return spec
	
	def add_vibH2(self, spectrum, h2swl, modspec, tauspec, RES=0.15, \
				MH2S=0.03, A_REDSHIFT=0.0):
		'''
		Adding H*
		RES:		Resolution of spectrum in AA
		MH2S: 		Multiplier of Draine model with NH2* = 6.73E+1
		A_REDSHIFT:	Offset in redshift in #/100000.0
		'''

		redshift = self.redshift
		wav_range = self.wav_range
		spec = spectrum

		# do this only for wav_range to save computing time?
		h2swl = np.array([1/i*1E8*(1+redshift+A_REDSHIFT) for i in h2swl][::-1])
		modscale = np.exp(-1*(tauspec-1.9845E-01)*MH2S)
		modsmooth = gauss1d(modscale, 0.07/RES)
		tckp = splrep(h2swl, modsmooth, s=3, k=2)
		modint = splev(wav_range, tckp)

		spec *= modint

		return spec

	def addCO(self, spectrum, broad, NCO, TEMP, A_REDSHIFT):

		# WORK IN PROGRESS - How to add CO band heads?
		redshift = self.redshift
		wav_range = self.wav_range
		spec = spectrum
		broad = broad * 1E5
		redshift = redshift + A_REDSHIFT

		nJ, NCO = [], {}
		for J in NROT:
			nJ.append(fNCO(TEMP, J))
		for nj in NROT:
			NCO['COJ%i' %nj] = 10**NCO/sum(nJ)*nJ[nj]

		for co in co_lines:
			co = co.split()
			lamb = float(co[1])
			f = float(co[2])
			gamma = float(co[3])
			nion = NH2[h2[0]]
			spec *= addAbs(wav_range, nion, lamb, f, gamma, broad, \
				redshift)

		return spec

	def addHD(self, spectrum, broad, NTOTHD, TEMP, A_REDSHIFT):

		print "hallo world"

#========================================================================
#========================================================================





