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

af = open ('atoms/atom.dat', 'r')
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
tauspec   = np.array(tauspec[::-1])


#========================================================================
#========================================================================

class SynSpec:

	def add_ion(self, wav_range, synspec, redshift, atom_name, broad, \
		Natom, A_REDSHIFT):
	
		spec = synspec
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

	def add_single_ion(self, wav_range, synspec, redshift, lamb, f, gamma, \
		broad, N_ion, A_REDSHIFT_ION):

		spec = synspec
		broad = broad * 1E5
		N_ion = 10**N_ion
		redshift = redshift + A_REDSHIFT_ION
		spec *= addAbs(wav_range, N_ion, lamb, f, gamma, broad, redshift)

		return spec

	def add_exc_ion(self, wav_range, synspec, redshift, exc_ion_list):
	
		spec = synspec
		exc_ion_dic = get_exc_ion_dic()

		for exc_ion in exc_ion_list:
			lamb, f, gamma = exc_ion_dic[exc_ion]
			if b.has_key(exc_ion):
				broad = b[exc_ion] * 1E5
			else:
				broad = b["ALL"] * 1E5
			N_exc_ion = 10**N_exc[exc_ion]
			spec *= addAbs(wav_range, N_exc_ion, lamb, f, gamma, broad, \
				redshift)

		return spec

	def add_H2(self, wav_range, synspec, redshift, broad, NTOTH2, \
		TEMP, A_REDSHIFT, NROT):


		spec = synspec

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
	
	def add_vibH2(self, wav_range, synspec, redshift, h2swl, modspec, \
		tauspec, RES=6000, MH2S=0.03, A_REDSHIFT=0.0):

		spec = synspec

		h2swl 	  = np.array([1/i*1E8*(1+redshift + A_REDSHIFT) for i in h2swl][::-1])
		modscale  = np.exp(-1*(tauspec-1.9845E-01)*MH2S)
		modsmooth = gauss1d(modscale, 0.07/RES)
		tckp = splrep(h2swl, modsmooth, s = 3, k = 2)
		modint = splev(wav_range, tckp)

		spec *= modint

		return spec


	def addCO(self, wav_range, synspec, redshift, broad, NTOTCO, \
		TEMP, A_REDSHIFT):

		# WORK IN PROGRESS

		spec = synspec
		broad = broad * 1E5
		redshift = redshift + A_REDSHIFT


		nJ, NCO = [], {}
		for J in NROT:
			nJ.append(fNHII(TEMP, J)) # change this function 
		for nj in NROT:
			NH2['H2J%i' %nj] = 10**NTOTH2/sum(nJ)*nJ[nj]


		for co in co_lines:
			co = co.split()
			lamb = float(co[1])
			f = float(co[2])
			gamma = float(co[3])
			nion = NH2[h2[0]]
			spec *= addAbs(wav_range, nion, lamb, f, gamma, broad, \
				redshift)

		return spec


#========================================================================
#========================================================================



















