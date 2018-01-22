#! /usr/bin/python

"""
Colection of various functions
=========================================================================
=========================================================================
get_data:	reading data
=========================================================================
=========================================================================
"""


import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *
import numpy as np
from scipy.ndimage import gaussian_filter1d as gauss1d
from scipy.interpolate import splrep, splev
from scipy.special import wofz

from numba import jit

from astropy.convolution import Gaussian1DKernel

import importlib

import bokeh
from bokeh.plotting import figure as bokeh_fig
from bokeh.plotting import show as bokeh_show
from bokeh.plotting import output_file
from bokeh.io import output_file
from bokeh.layouts import widgetbox, row
from bokeh.models import *

import seaborn as sns

 
e = 4.8032e-10
c = 2.998e10
c_kms = 2.99792458e5 # km/s
m_e = 9.10938291e-28 # g
hbar = 1.054571726e-27 # erg * s
alpha = 1 / 137.035999139 # dimensionless
K = np.pi * alpha * hbar / m_e


#========================================================================
#========================================================================

def get_data(file, z, wl_range=False, wl1 = 3300, wl2 = 5000):
	'''
	To do: use a dictionary and pandas to read the data
	'''

	wav_aa, n_flux, n_flux_err, flux, flux_err = [], [], [], [], []
	grb_name = ""
	res = 0
	psf_fwhm = 0

	data = open(file, "r")
	if wl_range==False:
		for line in data:
			if not line.startswith(("GRB", "Resolution", "PSFFWHM")):
				wav_aa = np.append(wav_aa,float(line.split()[0]))
				flux = np.append(flux,float(line.split()[1]))
				flux_err = np.append(flux_err,float(line.split()[2]))
				n_flux = np.append(n_flux,float(line.split()[6]))
				n_flux_err = np.append(n_flux_err,float(line.split()[7]))
			if line.startswith('GRB'):
				grb_name = str(line.split()[0]).split("_")[0]
			if line.startswith('Res'):
				res = float(line.split()[1])
			if line.startswith('PSF'):
				psf_fwhm = float(line.split()[1])
	if wl_range==True: 
		for line in data:
			if not line.startswith(("GRB", "Resolution", "PSFFWHM")):
				if (wl1*(1+z)) <= float(line.split()[0]) <= (wl2*(1+z)):
					wav_aa = np.append(wav_aa,float(line.split()[0]))
					flux = np.append(flux,float(line.split()[1]))
					flux_err = np.append(flux_err,float(line.split()[2]))
					n_flux = np.append(n_flux,float(line.split()[6]))
					n_flux_err = np.append(n_flux_err,float(line.split()[7]))
			if line.startswith('GRB'):
				grb_name = str(line.split()[0]).split("_")[0]
			if line.startswith('Res'):
				res = float(line.split()[1])
			if line.startswith('PSF'):
				psf_fwhm = float(line.split()[1])
	data.close()

	return wav_aa, n_flux, n_flux_err, flux, flux_err, grb_name, res, psf_fwhm


def get_data_ign(file, z, ignore_lst, wl1 = 3300, wl2 = 5000):

	wav_aa, n_flux, n_flux_err= [], [], []
	data = open(file, "r")

	wl1 = wl1*(1+z)
	wl2 = wl2*(1+z)

	wl_low 	= []
	wl_up	= []

	for wav_rng in ignore_lst:
		wl_low.append(wav_rng[0]*(1+z))
		wl_up.append(wav_rng[1]*(1+z))

	for line in data:
		if not line.startswith(("GRB", "Resolution", "PSFFWHM")):
			if wl1 <= float(line.split()[0]) <= wl2:
				wav = float(line.split()[0])
				tester = 0.0
				for i in np.arange(0, len(wl_low), 1):
					if wl_low[i] < wav < wl_up[i]:
						wav_aa = np.append(wav_aa,float(line.split()[0]))
						n_flux = np.append(n_flux,1.0)
						n_flux_err = np.append(n_flux_err,1.0)
						tester += 1.0
				if tester == 0.0:
					wav_aa = np.append(wav_aa,float(line.split()[0]))
					n_flux = np.append(n_flux,float(line.split()[6]))
					n_flux_err = np.append(n_flux_err,float(line.split()[7]))

	data.close()
	return wav_aa, n_flux, n_flux_err

#========================================================================
#========================================================================

def aa_to_velo(wav_aa, flux, flux_err, line, redshift, wrange=15):
	'''
	Angstrom to veocity (km/s) for given line (AA) at given redshift
	around the wavelength range: line (AA) +/- wrange (AA)
	'''

	c = 299792.458
	rline = line * (1 + redshift)
	velocity, fluxv, fluxv_err = [], [], []
	for i in np.arange(0, len(wav_aa), 1):
		velo = abs(wav_aa[i]-rline)*c/rline
		if wav_aa[i] < rline and wav_aa[i] > rline-wrange:
			velocity.append(-velo)
			fluxv.append(flux[i])
			fluxv_err.append(flux_err[i])
		if wav_aa[i] > rline and wav_aa[i] < rline+wrange:
			velocity.append(velo)
			fluxv.append(flux[i])
			fluxv_err.append(flux_err[i])
	return velocity, fluxv, fluxv_err


def aa_to_velo_ign(wav_aa, flux, flux_err, line, redshift, ignore_lst, wrange=15):
	'''
	Angstrom to veocity (km/s) for given line (AA) at given redshift
	around the wavelength range: line (AA) +/- wrange (AA)
	'''

	c = 299792.458
	rline = line * (1 + redshift)
	velocity, fluxv, fluxv_err = [], [], []

	v_low 	= []
	v_up	= []

	for v_rng in ignore_lst:
		v_low.append(v_rng[0])
		v_up.append(v_rng[1])

	for i in np.arange(0, len(wav_aa), 1):
		velo = abs(wav_aa[i]-rline)*c/rline
		if wav_aa[i] < rline and wav_aa[i] > rline-wrange:
			velocity.append(-velo)
			fluxv.append(flux[i])
			fluxv_err.append(flux_err[i])
		if wav_aa[i] > rline and wav_aa[i] < rline+wrange:
			velocity.append(velo)
			fluxv.append(flux[i])
			fluxv_err.append(flux_err[i])

	velocity2, fluxv2, fluxv_err2 = [], [], []

	for i in np.arange(0, len(velocity), 1):
		tester = 0.0
		for j in np.arange(0, len(v_low), 1):
			if v_low[j] < velocity[i] < v_up[j]:
				#velocity2.append(velocity[i])
				#fluxv2.append(1.0)
				#fluxv_err2.append(100.0)
				tester += 1.0
		if tester == 0.0:
			velocity2.append(velocity[i])
			fluxv2.append(fluxv[i])
			fluxv_err2.append(fluxv_err[i])

	return velocity2, fluxv2, fluxv_err2


#========================================================================
#========================================================================

def get_lines(redshift):

	# make this better!

	a_name, a_wav = [], []
	atom_file = open("atoms/atom_excited.dat", "r")
	for line in atom_file:
		if not line.startswith(("#", "HD", "H2", "CO")):
			s = line.split()
			if not s[0][-1] in 'cdefghijklmnopqrstuvw':
			#if float(s[2]) > 0.008:
				a_name.append(str(s[0]))
				a_wav.append(float(s[1])*(1+redshift))
	atom_file.close()
	
	# MGII lines for intervening system
	ai_name, ai_wav = [], []
	atomi_file = open("atoms/atom.dat", "r")
	for line in atomi_file:
		if not line.startswith(("#", "HD", "H2", "CO")):
			if line.startswith("Mg"):
				s = line.split()
				ai_name.append(str(s[0]))
				ai_wav.append(float(s[1])*(1+1.539))
	atomi_file.close()
	
	aex_name, aex_wav = [], []
	atomex_file = open("atoms/atom.dat", "r")
	for line in atomex_file:
		if not line.startswith(("#", "HD", "H2", "CO")):
			s = line.split()
			if not s[0][-1] in 'cdefghijklmnopqrstuvw':

				if float(s[2]) > 0.001:
					aex_name.append(str(s[0]))
					aex_wav.append(float(s[1])*(1+redshift))
	atomex_file.close()
	
	h2_name, h2_wav = [], []
	h2_file = open("atoms/h2.dat", "r")
	for line in h2_file:
		ss = line.split()			
		if not str(ss[0]) == "H2J2":
			h2_name.append(str(ss[0]).strip("H2"))
		if str(ss[0]) == "H2J2":
			h2_name.append("J2")
		h2_wav.append(float(ss[1])*(1+redshift))
	h2_file.close()

	return a_name, a_wav, ai_name, ai_wav, aex_name, aex_wav, h2_name, h2_wav

#========================================================================
#========================================================================

def get_osc_gamma(abs_line):
	'''
	Returning Oscillator strength f and decay rate gamma of given line
	Has to be identical to the wavlength value given in atom_excited.dat
	(or rouded to 2 decimal places)
	'''

	f = 0
	gamma = 0
	data = open("atoms/atom_excited.dat", "r")
	for line in data:
		if not line.startswith("#"):
			s = line.split()
			if float(s[1]) == abs_line:
				f += float(s[2])
				gamma += float(s[3])
			if round(float(s[1]), 2) == abs_line:
				f += float(s[2])
				gamma += float(s[3])
	data.close()
	if f == 0:
		sys.exit("ERROR: Line not found; Could not \
		return Oscillator strength") 

	return f, gamma

#========================================================================
#========================================================================

def v_to_dz(v0, z0):
	'''
	Difference in velocity space into difference in redshift
	'''
	c = 299792.458
	dz = v0/c * (1+z0)
	nz = z0 + dz
	return nz

#========================================================================
#========================================================================

@jit
def voigt(x, sigma, gamma):
	'''
	1D voigt profile, e.g.:
	https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
	gamma: half-width at half-maximum (HWHM) of the Lorentzian profile
	sigma: the standard deviation of the Gaussian profile
	HWHM, alpha, of the Gaussian is: alpha = sigma * sqrt(2ln(2))
	'''
	
	z = (x + 1j*gamma) / (sigma * np.sqrt(2.0))
	V = wofz(z).real / (sigma * np.sqrt(2.0*np.pi))
	return V

def H(a,x):

	P = x**2 
	H0 = np.exp(-x**2)
	Q = 1.5/x**2
	return H0 - a / np.sqrt(np.pi) / P * ( H0 ** 2 * \
		(4. * P**2 + 7. * P + 4. + Q) - Q - 1.0)

#========================================================================
#======================================================================== 

@jit
def addAbs_unconv(wls, N_ion, lamb, f, gamma, broad, redshift):
	'''
	Adds an absorption line, which is not(!) convolved with the
	instrumental resolution

	wls: wavelenth in AA
	N_ion: Column denisty
	lamb:restframe wavelength of the transition in AA
	f: oscillator strength of transition
	gamma: Damping Constant
	broad: broadening parameter b
	redshift: redshift
	'''


	C_a = np.sqrt(np.pi) * e**2 * f * lamb * 1E-8 / m_e / c / broad
	a = lamb * 1E-8 * gamma / (4.*np.pi * broad)
	dl_D = broad/c * lamb
	x = (wls/(redshift+1.0) - lamb)/dl_D+0.01
	tau = C_a * N_ion * H(a, x)

	return np.exp(-tau)

@jit
def addAbs(wls, N_ion, lamb, f, gamma, broad, redshift, res):
	'''
	Adds an absorption line convolved with the instrumental 
	resolution R

	wls: wavelenth in AA
	N_ion: Column denisty
	lamb:restframe wavelength of the transition in AA
	f: oscillator strength of transition
	gamma: Damping Constant
	broad: broadening parameter b
	redshift: redshift
	res: spectral resolution R
	'''

	d_lamda = lamb/res  # delta_lambda = lambda/R (FWHM)
	sig_res = 2*np.sqrt(2*np.log(2)) # FWHM -> sigma_gauss
	tf = np.median(np.diff(wls)) # transform to pixels 

	#print rres, tf, rres/(2*np.sqrt(2*np.log(2))*tf)

	gauss_k = Gaussian1DKernel(stddev=d_lamda/(sig_res*tf),
		mode="oversample") # gaussian kernel for convolution

	C_a = np.sqrt(np.pi) * e**2 * f * lamb * 1E-8 / m_e / c / broad
	a = lamb * 1E-8 * gamma / (4.*np.pi * broad)
	dl_D = broad/c * lamb
	x = (wls/(redshift+1.0) - lamb)/dl_D+0.01
	tau = C_a * N_ion * H(a, x)

	return np.convolve(np.exp(-tau), gauss_k, mode='same')

#========================================================================
#========================================================================

def fNHII(T, J):
	'''
	returns the column density for rotatinal level J, given the
	given the Temperature T[K] the total H2 column density in the
	"add_H2" function of the SynSpec model
	'''

	# Para molecular hydrogen
	if J % 2 == 0:
		I = 0
	# Ortho molecular hydrogen
	else:
		I = 1
	# Statistical weights
	gj = (2*J + 1) * (2*I + 1)
	# Energy difference between the different states (J = X --> J = 0)
	# from Dabrowski, I. 1984, Can. J. Phys., 62, 1639
	dE0J = {0:0, 1:170.5, 2:509.9, 3:1015.2, 4:1681.7, \
	5:2503.9, 6:3474.4, 7:4586.4}
	nj = gj * np.exp(-dE0J[J]/T)
	return nj


def fNCO(T, J):

	dE0J = {0:0, 1:5.5, 2:16.6, 3:33.2,
		4:55.3, 5:82.9, 6:116.2}

	#SUM = 0
	#for i in range(0, J+1, 1):
	#	print 2*i +1
	#	SUM += 2*i +1

	PJ_T = ((2*J+1) * np.exp(-dE0J[J]/T))# / SUM

	#NJ_T = NCO * PJ_T

	#return NJ_T
	return PJ_T

#========================================================================
#========================================================================


def get_h2_dic():
	h2_dic = {}
	with open("atoms/h2.dat") as f:
		for line in f:
			if not line.startswith('#'):
				
				(line, lamb, f, gamma) = line.split()[0:4]
				h2_dic[line] = float(lamb), float(f), float(gamma)
	return h2_dic

def get_ion_dic():
	ion_dic = {}
	with open("atoms/atom.dat") as f:
		for line in f:
			if not line.startswith('#'):
				(line, lamb, f, gamma) = line.split()[0:4]
				ion_dic[line] = float(lamb), float(f), float(gamma)
	return ion_dic

def get_exc_ion_dic():
	exc_ion_dic = {}
	with open("atoms/atom_excited.dat") as f:
		for line in f:
			if not line.startswith('#'):
				(line, lamb, f, gamma) = line.split()[0:4]
				exc_ion_dic[line] = float(lamb), float(f), float(gamma)
	return exc_ion_dic


def get_co():
	co_dic = {}
	with open("atoms/co.dat") as f:
		for line in f:
			if not line.startswith('#'):
				(line, lamb, f, gamma) = line.split()[0:4]
				co_dic[line] = float(lamb), float(f), float(gamma)
	return co_dic


#========================================================================
#========================================================================

def tell_lines(intensity=-0.15, file = 'atoms/tel_lines_uves.dat', wav_aa=[]):

	file = open(file, 'r')
	lines = [line.strip().split() for line in file.readlines() \
			if not line.strip().startswith('#')]
	file.close()
	tell = []
	y_val = []
	for line in lines:
		if line != []:
			if float(line[-2]) < intensity:
				if min(wav_aa) < float(line[3]) < max(wav_aa):
					tell.append(float(line[3]))
					y_val.append(2.0)
	return tell, y_val

def skylines(intensity=10, file= "atoms/sky_lines.dat"):

	fin = open(file, 'r')
	lines = [line.strip().split() for line in fin.readlines() if not line.strip().startswith('#')]
	fin.close()  
	sky, wlprev, intprev = [], 0, 0
	yval = []
	for line in lines:
		if line != []:
			if float(line[-1]) > intensity:
				wlline = float(line[0])
				intens = float(line[1])
				if (wlline - wlprev) < 0.2: 
					sky.pop()
					sky.append((wlline*intens + wlprev*intprev)/(intens+intprev))
					yval.append(1.3)
				else:
					sky.append(wlline)
					yval.append(1.3)
			wlprev, intprev = float(line[0]), float(line[1])
	return sky, yval 

#========================================================================
#========================================================================

def get_paras(para_file):
	'''
	Reads the Parameters from the given .csv file
	Example:
	element,fixed,N_val,N_low,N_up,B_val,B_low,B_up,R_val,R_low,R_up
	HI, 0, 22.0, 21.80, 22.10, 20., 15.0, 40.0, 0.0, -100, 100
	FeII, 0, 18.0, 17.0, 20.0, 20., 0.0, 40.0, 0.0, -100, 100
	'''

	par_dic = {}
	par = pd.read_csv(para_file, delimiter=',')
	for i in np.arange(0, len(par), 1):
		par_dic[par['element'][i]] = par['fixed'][i], par['N_val'][i], \
		par['N_low'][i],par['N_up'][i],par['B_val'][i],par['B_low'][i],par['B_up'][i], \
		par['R_val'][i],par['R_low'][i],par['R_up'][i]

	return par_dic


#========================================================================
#========================================================================

def get_paras_velo(para_file):
	'''
	Reads the Parameters from the given .csv file
	Example:
	var,fixed,val,low,up
	vo1, 1, -20, -22, -18
	'''

	par_dic = {}
	par = pd.read_csv(para_file, delimiter=',')
	for i in np.arange(0, len(par), 1):
		par_dic[par['var'][i]] = par['fixed'][i], par['val'][i], \
		par['low'][i],par['up'][i]

	return par_dic


#========================================================================
#========================================================================


def plot_spec(wav_aa, n_flux, n_flux_err, y_min, y_max, y_min2, y_max2, y_fit,
	redshift, ignore_lst, a_name, a_wav, ai_name, ai_wav, aex_name,
	aex_wav, h2_name, h2_wav, target, fb, intv_lst):

	tell_x, tell_y = tell_lines(intensity=-0.2,file='atoms/tel_lines_uves.dat', wav_aa=wav_aa)

	sns.set_style("white", {'legend.frameon': True})

	wav_range = (max(wav_aa)-min(wav_aa))/5.0

	fig = figure(figsize=(10, 12))
	
	ax1 = fig.add_axes([0.08, 0.08, 0.90, 0.11])
	ax2 = fig.add_axes([0.08, 0.25, 0.90, 0.11])
	ax3 = fig.add_axes([0.08, 0.41, 0.90, 0.11])
	ax4 = fig.add_axes([0.08, 0.58, 0.90, 0.11])
	ax5 = fig.add_axes([0.08, 0.73, 0.90, 0.11])
	ax6 = fig.add_axes([0.08, 0.88, 0.90, 0.11])

	ax1.scatter(tell_x, tell_y, color="orange")

	for axis in [ax1, ax2, ax3, ax4, ax5, ax6]:

		axis.errorbar(wav_aa, n_flux, linestyle='-', color="black", linewidth=0.5, \
			drawstyle='steps-mid', label=r"$\sf Spec.$")

		axis.errorbar(wav_aa, n_flux_err, linestyle='dotted', color="black",
			linewidth=0.5, drawstyle='steps-mid', label=r"$\sf Error Spec.$", alpha=0.6)

		axis.plot(wav_aa[2:-2], y_fit[2:-2], label=r"$\sf Fit$", color="#2171b5", linewidth=1.8, alpha=0.9)

		# fill space between quantiles
		axis.fill_between(wav_aa[2:-2], y_min[2:-2], y_max[2:-2], color='#2171b5', alpha=0.2)
		axis.fill_between(wav_aa[2:-2], y_min2[2:-2], y_max2[2:-2], color='#2171b5', alpha=0.4)

		# plot 25% and 75% quantiles
		axis.plot(wav_aa[2:-2], y_max2[2:-2], color="#2171b5", linewidth=1, alpha=0.9, linestyle="dashed")
		axis.plot(wav_aa[2:-2], y_min2[2:-2], color="#2171b5", linewidth=1, alpha=0.9, linestyle="dashed")

		# plot 2.5& and 97.5% quantiles
		axis.plot(wav_aa[2:-2], y_max[2:-2], color="#2171b5", linewidth=1, alpha=0.9, linestyle=":")
		axis.plot(wav_aa[2:-2], y_min[2:-2], color="#2171b5", linewidth=1, alpha=0.9, linestyle=":")

		#axis.errorbar(x_tell, y_tell, color="gray", fmt='o', markersize=8)

		if "ele" in str(target):
			axis.set_ylim([-0.25, 2.25])
		else:
			axis.set_ylim([-0.85, 2.25])

		axis.axhline(0.0, linestyle="dashed", color="black", linewidth=2)

		y_fill = [1 for wav in wav_aa]

		#if not axis == ax1:
		#	axis.fill_between(wav_aa, y_fit, y_fill, color="#2171b5", alpha=0.3)


		for wav_rng in ignore_lst:
			axis.axvspan(wav_rng[0]*(1+redshift), wav_rng[1]*(1+redshift), \
				facecolor='black', alpha=0.25)

		for side in ['top','bottom','left','right']:
		  	axis.spines[side].set_linewidth(2)
		axis.tick_params(which='major', length=8, width=2)
		axis.tick_params(which='minor', length=6, width=1)
		for tick in axis.xaxis.get_major_ticks():
			tick.label.set_fontsize(18)
		for tick in axis.yaxis.get_major_ticks():
			tick.label.set_fontsize(18)

	for i in np.arange(0, len(a_name), 1):

		# Better Solution than this mess?
		if min(wav_aa) < a_wav[i] < (max(wav_aa)-wav_range*4):
			if i%2 == 0:
				ax6.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=5, color="#41ae76")
				ax6.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
			elif i%3 == 0:
				ax6.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=5, color="#41ae76")
				ax6.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
			else:
				ax6.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=5, color="#41ae76")
				ax6.axvline(a_wav[i],  ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)				

		if (max(wav_aa)-wav_range*4) < a_wav[i] < (max(wav_aa)-wav_range*3):
			if i%2 == 0:
				ax5.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=5, color="#41ae76")
				ax5.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
			elif i%3 == 0:
				ax5.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=5, color="#41ae76")
				ax5.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
			else:
				ax5.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=5, color="#41ae76")
				ax5.axvline(a_wav[i],  ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)	

		if (max(wav_aa)-wav_range*3) < a_wav[i] < (max(wav_aa)-wav_range*2):
			if i%2 == 0:
				ax4.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=5, color="#41ae76")
				ax4.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
			elif i%3 == 0:
				ax4.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=5, color="#41ae76")
				ax4.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
			else:
				ax4.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=5, color="#41ae76")
				ax4.axvline(a_wav[i],  ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)	

		if (max(wav_aa)-wav_range*2) < a_wav[i] < (max(wav_aa)-wav_range*1):
			if i%2 == 0:
				ax3.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=5, color="#41ae76")
				ax3.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
			elif i%3 == 0:
				ax3.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=5, color="#41ae76")
				ax3.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
			else:
				ax3.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=5, color="#41ae76")
				ax3.axvline(a_wav[i],  ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)	

		if (max(wav_aa)-wav_range*1) < a_wav[i] < max(wav_aa):
			if i%2 == 0:
				ax2.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=5, color="#41ae76")
				ax2.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
			elif i%3 == 0:
				ax2.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=5, color="#41ae76")
				ax2.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
			else:
				ax2.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=5, color="#41ae76")
				ax2.axvline(a_wav[i],  ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)			
	
	for i in np.arange(0, len(h2_name), 1):
		if min(wav_aa) < h2_wav[i] < (max(wav_aa)-wav_range*4):
			if i%2 == 0:
				ax6.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6,color="#a50f15")
				ax6.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
			else:
				ax6.text(h2_wav[i]+0.2, -0.8, h2_name[i], fontsize=6,color="#a50f15")
				ax6.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)

		if (max(wav_aa)-wav_range*4) < h2_wav[i] < (max(wav_aa)-wav_range*3):
			if i%2 == 0:
				ax5.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6,color="#a50f15")
				ax5.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
			else:
				ax5.text(h2_wav[i]+0.2, -0.8, h2_name[i], fontsize=6,color="#a50f15")
				ax5.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)

		if (max(wav_aa)-wav_range*3) < h2_wav[i] < (max(wav_aa)-wav_range*2):
			if i%2 == 0:
				ax4.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6,color="#a50f15")
				ax4.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
			else:
				ax4.text(h2_wav[i]+0.2, -0.8, h2_name[i], fontsize=6,color="#a50f15")
				ax4.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)

		if (max(wav_aa)-wav_range*2) < h2_wav[i] < (max(wav_aa)-wav_range*1):
			if i%2 == 0:
				ax3.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6,color="#a50f15")
				ax3.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
			else:
				ax3.text(h2_wav[i]+0.2, -0.8, h2_name[i], fontsize=6,color="#a50f15")
				ax3.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
		if (max(wav_aa)-wav_range*1) < h2_wav[i] < max(wav_aa):
			if i%2 == 0:
				ax2.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6,color="#a50f15")
				ax2.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
			else:
				ax2.text(h2_wav[i]+0.2, -0.8, h2_name[i], fontsize=6,color="#a50f15")
				ax2.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)

	lg = ax1.legend(numpoints=1, fontsize=10, loc=1, ncol=3)

	ax1.set_xlabel(r"$\sf Observed\, Wavelength (\AA)$", fontsize=24)
	ax3.set_ylabel(r"$\sf Normalized\, Flux$", fontsize=24)
	ax1.axhline(1, color="#2171b5", linewidth=2)

	ax6.set_xlim([min(wav_aa), max(wav_aa)-wav_range*4])
	ax5.set_xlim([max(wav_aa)-wav_range*4, max(wav_aa)-wav_range*3])
	ax4.set_xlim([max(wav_aa)-wav_range*3, max(wav_aa)-wav_range*2])
	ax3.set_xlim([max(wav_aa)-wav_range*2, max(wav_aa)-wav_range*1])
	ax2.set_xlim([max(wav_aa)-wav_range*1, max(wav_aa)])
	
	if fb == None:
		fig.savefig(target+"_H2_fit_spec.pdf")
	else:
		fig.savefig(target+"_b_"+str(fb)+"_H2_fit_spec.pdf")
	

#========================================================================
#========================================================================

def plot_H2_hist(res_file="save_results.dat", z = 2.358):
	'''
	Plotting histograms for Redshift, N_H2, B and T
	'''
	redshift, nh2, temp, broad = [], [], [], []

	file = open(res_file, "r")
	for line in file:
		if not "TEMP" in line:
			s = line.split(",")
			redshift.append(float(s[0])+float(z))
			nh2.append(float(s[1]))
			temp.append(float(s[2]))
			broad.append(float(s[3]))
	file.close()

	fig = figure(figsize=(10, 8))
	
	ax1 = fig.add_axes([0.12, 0.12, 0.35, 0.35])
	ax2 = fig.add_axes([0.12, 0.57, 0.35, 0.35])
	ax3 = fig.add_axes([0.55, 0.12, 0.35, 0.35])
	ax4 = fig.add_axes([0.55, 0.57, 0.35, 0.35])

	ax1.hist(redshift, bins=50)
	ax2.hist(nh2, bins=50)
	ax3.hist(temp, bins=50)
	ax4.hist(broad, bins=50)

	ax1.set_xlabel(r"$\sf Redshift$", fontsize=18)
	ax2.set_xlabel(r"$\sf N_{H2}$", fontsize=18)
	ax3.set_xlabel(r"$\sf Temperature$", fontsize=18)
	ax4.set_xlabel(r"$\sf b$", fontsize=18)

	fig.savefig("histograms.pdf")
	show()

#========================================================================
#======================================================================== 

def plot_H2_trace(res_file="save_results.dat", z = 2.358):
	'''
	Plotting traces for Redshift, N_H2, B and T
	'''

	redshift, nh2, temp, broad = [], [], [], []

	file = open(res_file, "r")
	for line in file:
		if not "TEMP" in line:
			s = line.split(",")
			redshift.append(float(s[0])+float(z))
			nh2.append(float(s[1]))
			temp.append(float(s[2]))
			broad.append(float(s[3]))
	file.close()

	fig = figure(figsize=(10, 8))
	
	ax1 = fig.add_axes([0.10, 0.12, 0.85, 0.15])
	ax2 = fig.add_axes([0.10, 0.34, 0.85, 0.15])
	ax3 = fig.add_axes([0.10, 0.56, 0.85, 0.15])
	ax4 = fig.add_axes([0.10, 0.78, 0.85, 0.15])

	trials = []
	for i in np.arange(0, len(redshift), 1):
		trials.append(i)

	ax1.plot(trials, redshift)
	ax2.plot(trials, nh2)
	ax3.plot(trials, temp)
	ax4.plot(trials, broad)

	ax1.set_xlabel(r"$\sf Redshift$", fontsize=18)
	ax2.set_xlabel(r"$\sf N_{H2}$", fontsize=18)
	ax3.set_xlabel(r"$\sf Temperature$", fontsize=18)
	ax4.set_xlabel(r"$\sf b$", fontsize=18)

	fig.savefig("traces.pdf")
	show()

#========================================================================
#========================================================================

def bokeh_plt(wav_aa_pl, n_flux_pl, y_min, y_max, y_min2, y_max2, y_fit, redshift, ignore_lst, \
	a_name, a_wav, ai_name, ai_wav, aex_name, aex_wav, h2_name, h2_wav):

	output_file("GRB120815A_spec_bokeh.html", title="specrum_bokeh_html.py", mode="cdn")

	
	source = ColumnDataSource(data={"wav_aa_pl":wav_aa_pl, "n_flux_pl":n_flux_pl,
		"y_fit":y_fit})

	hover = HoverTool(tooltips=[
		("(wav_aa_pl, n_flux_pl)", "($wav_aa_pl, $n_flux_pl)"),
	])

	p = bokeh_fig(title="X-shooter spectrum of GRB120815A", x_axis_label='Observed Wavelength', tools="hover",
		y_axis_label='Normalized Flux', y_range=[-0.8, 2.2], x_range=[990.0*(1+redshift), 990.0*(1+redshift)+40],
		plot_height=300, plot_width=1200, toolbar_location="above")


	for i in np.arange(0, len(h2_name), 1):

		vline = Span(location=h2_wav[i], dimension='height', line_color='red', \
			line_width=0.8, line_dash='dashed')

		if i%2 == 0:
			H2_label = Label(x=h2_wav[i]+0.2, y=1.70, text=h2_name[i], text_font_size="8pt",
				text_color="red", text_font="helvetica")
		else:
			H2_label = Label(x=h2_wav[i]+0.2, y=1.58, text=h2_name[i], text_font_size="8pt",
				text_color="red", text_font="helvetica")		

		p.renderers.extend([vline])
		p.add_layout(H2_label)


	for i in np.arange(0, len(a_name), 1):

		vline = Span(location=a_wav[i], dimension='height', line_color='green', \
			line_width=1, line_dash='dashed')

		atom_label = Label(x=a_wav[i]+0.2, y=1.4, text=a_name[i], text_font_size="12pt",
			text_color="green", text_font="helvetica")

		p.renderers.extend([vline])
		p.add_layout(atom_label)


	for i in np.arange(0, len(ai_name), 1):

		vline_int = Span(location=ai_wav[i], dimension='height', line_color='cyan', \
			line_width=1, line_dash='dashed')

		atom_label_int = Label(x=ai_wav[i]+0.2, y=1.4, text=ai_name[i], text_font_size="12pt",
			text_color="cyan", text_font="helvetica")

		p.renderers.extend([vline_int])
		p.add_layout(atom_label_int)

	p.line(x="wav_aa_pl", y="n_flux_pl", source=source, legend="Data", line_width=2, color="black")

	#p.circle(x="wav_aa_pl", y="n_flux_pl", source=source, legend="Data", radius=0.2, color="black")
	p.line(x="wav_aa_pl", y="y_fit", source=source, legend="Fit", line_width=4, color="#2171b5")
	
	callback = CustomJS(args=dict(x_range=p.x_range), code="""
	var start = cb_obj.get("value");
	x_range.set("start", start);
	x_range.set("end", start+40);
	""")
	
	slider = Slider(start=990.0*(1+redshift), end=1122.0*(1+redshift)-40, value=1, \
		step=.1, title="Scroll", callback=callback)
	
	inputs = widgetbox(slider)
	
	bokeh_show(row(inputs, p, width=800), browser="safari")




def bokeh_H2vib_plt(wav_aa_pl, n_flux_pl, y_min, y_max, y_min2, y_max2, y_fit, redshift, ignore_lst, \
	a_name, a_wav, w1, w2):

	output_file("H2vib_spec_bokeh.html", title="H2vib_spec_bokeh_html.py", mode="cdn")

	source = ColumnDataSource(data={"wav_aa_pl":wav_aa_pl, "n_flux_pl":n_flux_pl,
		"y_fit":y_fit})

	hover = HoverTool(tooltips=[
		("(wav_aa_pl, n_flux_pl)", "($wav_aa_pl, $n_flux_pl)"),
	])

	p = bokeh_fig(title="X-shooter spectrum of GRB120815A", x_axis_label='Observed Wavelength', tools="hover",
		y_axis_label='Normalized Flux', y_range=[-0.6, 1.5], x_range=[w1*(1+redshift), w1*(1+redshift)+40],
		plot_height=400, plot_width=1200, toolbar_location="above")


	for i in np.arange(0, len(a_name), 1):

		vline = Span(location=a_wav[i], dimension='height', line_color='green', \
			line_width=1, line_dash='dashed')

		atom_label = Label(x=a_wav[i]+0.2, y=0.2, text=a_name[i], text_font_size="12pt",
			text_color="green", text_font="helvetica")

		p.renderers.extend([vline])
		p.add_layout(atom_label)

	p.line(x="wav_aa_pl", y="n_flux_pl", source=source, legend="Data", line_width=2, color="black")

	#p.circle(x="wav_aa_pl", y="n_flux_pl", source=source, legend="Data", radius=0.2, color="black")
	p.line(x="wav_aa_pl", y="y_fit", source=source, legend="Fit", line_width=4, color="#2171b5")
	
	callback = CustomJS(args=dict(x_range=p.x_range), code="""
	var start = cb_obj.get("value");
	x_range.set("start", start);
	x_range.set("end", start+40);
	""")
	
	slider = Slider(start=w1*(1+redshift), end=w2*(1+redshift)-40, value=1, \
		step=.1, title="Scroll", callback=callback)
	
	inputs = widgetbox(slider)
	
	bokeh_show(row(inputs, p, width=800), browser="safari")


#========================================================================
#======================================================================== 


def plot_H2vib(wav_aa, n_flux, y_min, y_max, y_min2, y_max2, \
	y_fit, a_name, a_wav, aex_name, aex_wav, target):

	sns.set_style("white", {'legend.frameon': True})

	fig = figure(figsize=(9, 4))
	
	ax1 = fig.add_axes([0.16, 0.22, 0.82, 0.76])

	y_fill = [1 for wav in wav_aa]

	ax1.fill_between(wav_aa, y_fit, y_fill, color="red", alpha=0.5)

	ax1.errorbar(wav_aa, n_flux, linestyle='-', color="black", linewidth=1, \
		drawstyle = 'steps-mid', label=r"$\sf Data$")
	ax1.axhline(1, color="red", linewidth=1)
	
	ax1.plot(wav_aa, y_fit, label=r"$\sf Fit$", color="red", linewidth=2, alpha=0.9)
	ax1.fill_between(wav_aa, y_min, y_max, color='black', alpha=0.4)
	ax1.fill_between(wav_aa, y_min2, y_max2, color='black', alpha=0.6)
	
	lg = ax1.legend(numpoints=1, fontsize=16, loc=3)
	lg.get_frame().set_edgecolor("white")
	lg.get_frame().set_facecolor('#f0f0f0')
	
	ax1.set_xlabel(r"$\sf Observed\, Wavelength (\AA)$", fontsize=24)
	ax1.set_ylabel(r"$\sf Normalized\, Flux$", fontsize=24)
	ax1.set_ylim([-0.5, 1.6])
	ax1.set_yticks([-0.2, 0.0, 0.5, 1.0, 1.5])

	#for i in np.arange(0, len(a_name), 1):
	#	if min(wav_aa) < a_wav[i] < (max(wav_aa)):
	#		ax1.text(a_wav[i]+0.2, -0.1, a_name[i], fontsize=8)
	#		ax1.axvline(a_wav[i], linestyle="dashed", color="black", linewidth=0.6)

	for i in np.arange(0, len(aex_name), 1):
		if min(wav_aa) < aex_wav[i] < (max(wav_aa)):
			if not aex_name[i].startswith("CO"):
				if i%2 == 0:
					ax1.text(aex_wav[i]+0.2, 1.45, aex_name[i], fontsize=8)
				else:
					ax1.text(aex_wav[i]+0.2, 1.35, aex_name[i], fontsize=8)
				ax1.axvline(aex_wav[i], linestyle="dashed", color="gray", linewidth=0.6)

	for axis in ['top','bottom','left','right']:
	  ax1.spines[axis].set_linewidth(2)
	ax1.tick_params(which='major', length=8, width=2)
	ax1.tick_params(which='minor', length=4, width=1.5)
	
	for tick in ax1.xaxis.get_major_ticks():
	    tick.label.set_fontsize(18)
	for tick in ax1.yaxis.get_major_ticks():
		tick.label.set_fontsize(18)

	fig.savefig(target + "_H2vib_fit_spec.pdf")


def get_co(redshift):

	co_name, co_wav = [], []
	co_file = open("atoms/co.dat", "r")
	for line in co_file:
		ss = line.split()			
		co_name.append(str(ss[0]).strip("CO"))
		co_wav.append(float(ss[1])*(1+redshift))
	co_file.close()

	return co_name, co_wav

def get_a(redshift):

	a_name, a_wav = [], []
	a_file = open("atoms/atom.dat", "r")
	for line in a_file:
		if not line.startswith("#"):
			ss = line.split()
			a_name.append((ss[0]))	
			a_wav.append(float(ss[1])*(1+redshift))
	a_file.close()

	return a_name, a_wav

def plot_CO(wav_aa_pl,n_flux_pl,y_min,y_max,y_min2,y_max2,y_fit,redshift,target,fb):

	sns.set_style("white", {'legend.frameon': True})

	co_name, co_wav = get_co(redshift)
	a_name, a_wav = get_a(redshift)

	fig = figure(figsize=(7, 4))
	ax = fig.add_axes([0.15, 0.22, 0.83, 0.66])

	ax.errorbar(wav_aa_pl, n_flux_pl, linestyle='-', color="black", linewidth=0.5, \
		drawstyle='steps-mid', label=r"$\sf Data$")

	ax.plot(wav_aa_pl, y_fit, label=r"$\sf Fit$", color="#2171b5", linewidth=1.8, alpha=0.9)

	for side in ['top','bottom','left','right']:
	  	ax.spines[side].set_linewidth(2)
	ax.tick_params(which='major', length=8, width=2)
	ax.tick_params(which='minor', length=6, width=1)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(18)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(18)

	for i in np.arange(0, len(a_name), 1):
		#print a_wav[i], a_name[i]
		if i%2 == 0 and not i%3 == 0:
			ax.text(a_wav[i]+0.05, 0.0, a_name[i], fontsize=6, color="black")
			ax.axvline(a_wav[i], ymin=0.0, ymax=0.1, linestyle="-", color="black", linewidth=1.0)
		elif i%3 == 0 and not i%2 == 0:
			ax.text(a_wav[i]+0.05, 0.1, a_name[i], fontsize=6, color="black")
			ax.axvline(a_wav[i], ymin=0.0, ymax=0.1, linestyle="-", color="black", linewidth=1.0)
		else:
			ax.text(a_wav[i]+0.05, 0.2, a_name[i], fontsize=6, color="black")
			ax.axvline(a_wav[i],  ymin=0.0, ymax=0.1, linestyle="-", color="black", linewidth=1.0)	

	for i in np.arange(0, len(co_name), 1):
		if i%2 == 0 and not i%3 == 0:
			ax.text(co_wav[i]+0.05, 1.25, co_name[i], fontsize=8, color="#41ae76")
			ax.axvline(co_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)
		elif i%3 == 0 and not i%2 == 0:
			ax.text(co_wav[i]+0.05, 1.35, co_name[i], fontsize=8, color="#41ae76")
			ax.axvline(co_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)
		else:
			ax.text(co_wav[i]+0.05, 1.45, co_name[i], fontsize=8, color="#41ae76")
			ax.axvline(co_wav[i],  ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)

	if min(wav_aa_pl) < 1544.31*(redshift+1) < max(wav_aa_pl):
		plt.title(r"$\sf CO\, AX(0-0)$", fontsize=24)

	if min(wav_aa_pl) < 1509.72*(redshift+1) < max(wav_aa_pl):
		plt.title(r"$\sf CO\, AX(1-0)$", fontsize=24)

	if min(wav_aa_pl) < 1477.54*(redshift+1) < max(wav_aa_pl):
		plt.title(r"$\sf CO\, AX(2-0)$", fontsize=24)

	if min(wav_aa_pl) < 1447.41*(redshift+1) < max(wav_aa_pl):
		plt.title(r"$\sf CO\, AX(3-0)$", fontsize=24)

	lg = ax.legend(numpoints=1, fontsize=10, loc=4)
	ax.set_xlim([min(wav_aa_pl), max(wav_aa_pl)])
	ax.set_ylim([-0.05, 1.55])
	ax.set_xlabel(r"$\sf Observed\, Wavelength (\AA)$", fontsize=24)
	ax.set_ylabel(r"$\sf Normalized\, Flux$", fontsize=24)
	#ax.axhline(1,color="#2171b5",linewidth=2)
	if fb == None:
		fig.savefig(target+"_CO_fit_spec.pdf")
	else:
		fig.savefig(target+"_b_"+str(fb)+"_CO_fit_spec.pdf")


def plot_trace(trace):
	'''
	Plotting a trace
	'''

	fig = figure(figsize=(10, 4))
	ax = fig.add_axes([0.13, 0.13, 0.85, 0.85])

	step = []

	for i in np.arange(0, len(trace), 1):
		step.append(i)

	ax.errorbar(step, trace)
	ax.set_ylim([-0.5, 11])
	show()

def plot_hist(trace):
	'''
	Plotting a histogram
	'''

	fig = figure(figsize=(6, 6))
	ax = fig.add_axes([0.13, 0.13, 0.85, 0.85])

	ax.hist(trace)
	ax.set_xlim([0.5, 10.5])
	#plt.title("Number of Absorption lines")
	show()


def writecmd(filename):
	now = datetime.datetime.now()

	with open(filename, "a") as f: 
		f.write(now.strftime("%Y-%m-%d %H:%M:%S"))
		f.write('python ')
		for arg in sys.argv:
			f.write(arg+' ')
		f.write('\n')


