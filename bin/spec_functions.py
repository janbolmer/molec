#! /usr/bin/python

import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *
import numpy as np
from scipy.ndimage import gaussian_filter1d as gauss1d
from scipy.interpolate import splrep, splev
from scipy.special import wofz

import bokeh
from bokeh.plotting import figure as bokeh_fig
from bokeh.plotting import show as bokeh_show
from bokeh.plotting import output_file
from bokeh.io import output_file
from bokeh.layouts import widgetbox, row
from bokeh.models import *

import seaborn as sns

m_e = 9.1095e-28 
e = 4.8032e-10
c = 2.998e10

##################################################################################### 

##################################################################################### 

##################################################################################### 

def get_data(file, z, wl_range=False, wl1 = 3300, wl2 = 5000):

	wav_aa, n_flux, n_flux_err, flux, flux_err = [], [], [], [], []
	data = open(file, "r")
	if wl_range==False:
		for line in data:
			if not "GRB" in line:
				wav_aa 		= np.append(wav_aa, float(line.split()[0]))
				flux 		= np.append(flux, float(line.split()[1]))
				flux_err 	= np.append(flux_err, float(line.split()[2]))
				n_flux 		= np.append(n_flux, float(line.split()[6]))
				n_flux_err 	= np.append(n_flux_err, float(line.split()[7]))
	if wl_range==True: 
		for line in data:
			if not "GRB" in line:
				if (wl1*(1+z)) <= float(line.split()[0]) <= (wl2*(1+z)):
					wav_aa 		= np.append(wav_aa, float(line.split()[0]))
					flux 		= np.append(flux, float(line.split()[1]))
					flux_err 	= np.append(flux_err, float(line.split()[2]))
					n_flux 		= np.append(n_flux, float(line.split()[6]))
					n_flux_err 	= np.append(n_flux_err, float(line.split()[7]))
	data.close()
	return wav_aa, n_flux, n_flux_err, flux, flux_err

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
		if not "GRB" in line:
			if wl1 <= float(line.split()[0]) <= wl2:
				wav = float(line.split()[0])
				tester = 0.0
				for i in np.arange(0, len(wl_low), 1):
					if wl_low[i] < wav < wl_up[i]:
						wav_aa 		= np.append(wav_aa, float(line.split()[0]))
						n_flux 		= np.append(n_flux, 1.0)
						n_flux_err 	= np.append(n_flux_err, 0.01)
						tester += 1.0
				if tester == 0.0:
					wav_aa 		= np.append(wav_aa, float(line.split()[0]))
					n_flux 		= np.append(n_flux, float(line.split()[6]))
					n_flux_err 	= np.append(n_flux_err, float(line.split()[7]))

	data.close()
	return wav_aa, n_flux, n_flux_err


##################################################################################### 

##################################################################################### 

##################################################################################### 

def get_lines(redshift):

	a_name, a_wav = [], []
	atom_file = open("atoms/atom.dat", "r")
	for line in atom_file:
		if not line.startswith("#") and not line.startswith("H2") and not line.startswith("HD"):
			s = line.split()
			if float(s[2]) > 0.000001:
				a_name.append(str(s[0]))
				a_wav.append(float(s[1])*(1+redshift))
	atom_file.close()
	
	ai_name, ai_wav = [], []
	atomi_file = open("atoms/atom.dat", "r")
	for line in atomi_file:
		if not line.startswith("#") and not line.startswith("H2") and not line.startswith("HD"):
			if line.startswith("Mg"):
				s = line.split()
				ai_name.append(str(s[0]))
				ai_wav.append(float(s[1])*(1+1.539))
	atomi_file.close()
	
	aex_name, aex_wav = [], []
	atomex_file = open("atoms/atom_excited.dat", "r")
	for line in atomex_file:
		if not line.startswith("#") and not line.startswith("H2") and not line.startswith("HD"):
			s = line.split()
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


##################################################################################### 

##################################################################################### 

##################################################################################### 


def quick_plot(x, y, redshift):

	wav_aa = x

	wav_range = (max(x)-min(x))/4.0

	a_name, a_wav = [], []
	atom_file = open("atoms/atom.dat", "r")
	for line in atom_file:
		if not line.startswith("#") and not line.startswith("H2") and not line.startswith("HD"):
			s = line.split()
			a_name.append(str(s[0]))
			a_wav.append(float(s[1])*(1+redshift))
	atom_file.close()

	h2_name, h2_wav = [], []
	h2_file = open("atoms/h2.dat", "r")
	for line in h2_file:
		s = line.split()
		h2_name.append(str(s[0]).strip("H2"))
		h2_wav.append(float(s[1])*(1+redshift))
	h2_file.close()


	fig = figure(figsize=(12, 10))
	
	ax1 = fig.add_axes([0.07, 0.09, 0.92, 0.13])
	ax2 = fig.add_axes([0.07, 0.29, 0.92, 0.13])
	ax3 = fig.add_axes([0.07, 0.48, 0.92, 0.13])
	ax4 = fig.add_axes([0.07, 0.67, 0.92, 0.13])
	ax5 = fig.add_axes([0.07, 0.86, 0.92, 0.13])


	ax1.errorbar(x, y, linestyle='-', color="black", linewidth=0.5, drawstyle='steps-mid', label=r"$\sf Data$")
	ax2.errorbar(x, y, linestyle='-', color="black", linewidth=0.5, drawstyle='steps-mid', label=r"$\sf Data$")
	ax3.errorbar(x, y, linestyle='-', color="black", linewidth=0.5, drawstyle='steps-mid', label=r"$\sf Data$")
	ax4.errorbar(x, y, linestyle='-', color="black", linewidth=0.5, drawstyle='steps-mid', label=r"$\sf Data$")
	ax5.errorbar(x, y, linestyle='-', color="black", linewidth=0.5, drawstyle='steps-mid', label=r"$\sf Data$")


	for i in np.arange(0, len(a_name), 1):
		if min(wav_aa) < a_wav[i] < (max(wav_aa)-wav_range*3):
			ax5.text(a_wav[i]+0.2, 1.5, a_name[i], fontsize=6)
			ax5.axvline(a_wav[i], linestyle="dashed", color="black", linewidth=0.6)
		if (max(wav_aa)-wav_range*3) < a_wav[i] < (max(wav_aa)-wav_range*2):
			ax4.text(a_wav[i]+0.2, 1.5, a_name[i], fontsize=6)
			ax4.axvline(a_wav[i], linestyle="dashed", color="black", linewidth=0.6)
		if (max(wav_aa)-wav_range*2) < a_wav[i] < (max(wav_aa)-wav_range*1):
			ax3.text(a_wav[i]+0.2, 1.5, a_name[i], fontsize=6)
			ax3.axvline(a_wav[i], linestyle="dashed", color="black", linewidth=0.6)
		if (max(wav_aa)-wav_range*1) < a_wav[i] < max(wav_aa):
			ax2.text(a_wav[i]+0.2, 1.5, a_name[i], fontsize=6)
			ax2.axvline(a_wav[i], linestyle="dashed", color="black", linewidth=0.6)			

	for i in np.arange(0, len(h2_name), 1):
		if min(wav_aa) < h2_wav[i] < (max(wav_aa)-wav_range*3):
			ax5.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6, color="red")
			ax5.axvline(h2_wav[i], linestyle="dashed", color="red", linewidth=0.6)
		if (max(wav_aa)-wav_range*3) < h2_wav[i] < (max(wav_aa)-wav_range*2):
			ax4.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6, color="red")
			ax4.axvline(h2_wav[i], linestyle="dashed", color="red", linewidth=0.6)
		if (max(wav_aa)-wav_range*2) < h2_wav[i] < (max(wav_aa)-wav_range*1):
			ax3.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6, color="red")
			ax3.axvline(h2_wav[i], linestyle="dashed", color="red", linewidth=0.6)
		if (max(wav_aa)-wav_range*1) < h2_wav[i] < max(wav_aa):
			ax2.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6, color="red")
			ax2.axvline(h2_wav[i], linestyle="dashed", color="red", linewidth=0.6)

	ax1.set_xlabel(r"$\sf Observed\, Wavelength (\AA)$", fontsize=24)
	ax3.set_ylabel(r"$\sf Normalized\, Flux$", fontsize=24)
	ax1.set_ylim([-0.8, 1.8])
	ax2.set_ylim([-0.8, 1.8])
	ax3.set_ylim([-0.8, 1.8])
	ax4.set_ylim([-0.8, 1.8])
	ax5.set_ylim([-0.8, 1.8])

	
	ax5.set_xlim([min(wav_aa), max(wav_aa)-wav_range*3])
	ax4.set_xlim([max(wav_aa)-wav_range*3, max(wav_aa)-wav_range*2])
	ax3.set_xlim([max(wav_aa)-wav_range*2, max(wav_aa)-wav_range*1])
	ax2.set_xlim([max(wav_aa)-wav_range*1, max(wav_aa)])

	for axis in ['top','bottom','left','right']:
	  ax1.spines[axis].set_linewidth(2)
	ax1.tick_params(which='major', length=8, width=2)
	ax1.tick_params(which='minor', length=4, width=1.5)

	for axis in ['top','bottom','left','right']:
	  ax2.spines[axis].set_linewidth(2)
	ax2.tick_params(which='major', length=8, width=2)
	ax2.tick_params(which='minor', length=4, width=1.5)
	
	for axis in ['top','bottom','left','right']:
	  ax3.spines[axis].set_linewidth(2)
	ax3.tick_params(which='major', length=8, width=2)
	ax3.tick_params(which='minor', length=4, width=1.5)

	for axis in ['top','bottom','left','right']:
	  ax4.spines[axis].set_linewidth(2)
	ax4.tick_params(which='major', length=8, width=2)
	ax4.tick_params(which='minor', length=4, width=1.5)

	for axis in ['top','bottom','left','right']:
	  ax5.spines[axis].set_linewidth(2)
	ax5.tick_params(which='major', length=8, width=2)
	ax5.tick_params(which='minor', length=4, width=1.5)

	for tick in ax1.xaxis.get_major_ticks():
	    tick.label.set_fontsize(18)
	for tick in ax1.yaxis.get_major_ticks():
		tick.label.set_fontsize(18)

	for tick in ax2.xaxis.get_major_ticks():
	    tick.label.set_fontsize(18)
	for tick in ax2.yaxis.get_major_ticks():
		tick.label.set_fontsize(18)

	for tick in ax3.xaxis.get_major_ticks():
	    tick.label.set_fontsize(18)
	for tick in ax3.yaxis.get_major_ticks():
		tick.label.set_fontsize(18)

	for tick in ax4.xaxis.get_major_ticks():
	    tick.label.set_fontsize(18)
	for tick in ax4.yaxis.get_major_ticks():
		tick.label.set_fontsize(18)

	for tick in ax5.xaxis.get_major_ticks():
	    tick.label.set_fontsize(18)
	for tick in ax5.yaxis.get_major_ticks():
		tick.label.set_fontsize(18)

	fig.savefig("quick_plot.pdf")
	show()


##################################################################################### 

##################################################################################### 

##################################################################################### 

def voigt(x, y):
	z = x + 1j*y
	I = wofz(z).real
	return I

def H(a,x):
	P = x**2 
	H0 = np.exp(-x**2)
	Q = 1.5/x**2
	return H0 - a / np.sqrt(np.pi) / P * ( H0 ** 2 * (4. * P**2 + 7. * P + 4. + Q) - Q - 1.0)

##################################################################################### 

##################################################################################### 

##################################################################################### 


def addAbs(wls, N_ion, lamb, f, gamma, broad, redshift):
	C_a = np.sqrt(np.pi) * e**2 * f * lamb * 1E-8 / m_e / c / broad
	a = lamb * 1E-8 * gamma / (4.*np.pi * broad)
	dl_D = broad/c * lamb
	x = (wls/(redshift+1.0) - lamb)/dl_D+0.01
	tau = C_a * N_ion * H(a, x)
	return np.exp(-tau)

##################################################################################### 

##################################################################################### 

##################################################################################### 

def fNHII(T, J):
	# Para molecular hydrogen
	if J % 2 == 0:
		I = 0
	# Ortho molecular hydrogen
	else:
		I = 1
	# Statistical weights
	gj = (2*J + 1) * (2*I + 1)
	# Energy difference between the different states / from Dabrowski, I. 1984, Can. J. Phys., 62, 1639
	dE0J = {0:0, 1:170.5, 2:509.9, 3:1015.2, 4:1681.7, 5:2503.9, 6:3474.4, 7:4586.4}
	nj = gj * np.exp(-dE0J[J]/T)
	return nj


##################################################################################### 

##################################################################################### 

##################################################################################### 


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

##################################################################################### 

##################################################################################### 

##################################################################################### 


def tell_lines(intensity=-0.2, file = 'atoms/tel_lines_uves.dat'):
	file = open(file, 'r')
	lines = [line.strip().split() for line in file.readlines() if not line.strip().startswith('#')]
	file.close()
	tell = []
	y_val = []
	for line in lines:
		if line != []:
			if float(line[-2]) < intensity:
				tell.append(float(line[3]))
				y_val.append(1.3)
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

##################################################################################### 

##################################################################################### 

##################################################################################### 

def plot_spec(wav_aa, n_flux, y_min, y_max, y_min2, y_max2, y_fit, redshift, ignore_lst, \
		a_name, a_wav, ai_name, ai_wav, aex_name, aex_wav, h2_name, h2_wav):

	sns.set_style("white")

	wav_range = (max(wav_aa)-min(wav_aa))/4.0

	fig = figure(figsize=(12, 10))
	
	ax1 = fig.add_axes([0.07, 0.09, 0.92, 0.13])
	ax2 = fig.add_axes([0.07, 0.29, 0.92, 0.13])
	ax3 = fig.add_axes([0.07, 0.48, 0.92, 0.13])
	ax4 = fig.add_axes([0.07, 0.67, 0.92, 0.13])
	ax5 = fig.add_axes([0.07, 0.86, 0.92, 0.13])

	for axis in [ax1, ax2, ax3, ax4, ax5]:

		axis.errorbar(wav_aa, n_flux, linestyle='-', color="black", linewidth=0.5, \
			drawstyle='steps-mid', label=r"$\sf Data$")

		axis.plot(wav_aa, y_fit, label=r"$\sf Fit$", color="#2171b5", linewidth=2, alpha=0.9)
		axis.fill_between(wav_aa, y_min, y_max, color='#2171b5', alpha=0.2)
		axis.fill_between(wav_aa, y_min2, y_max2, color='#2171b5', alpha=0.4)
		axis.set_ylim([-0.8, 1.8])

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
		if min(wav_aa) < a_wav[i] < (max(wav_aa)-wav_range*3):
			ax5.text(a_wav[i]+0.2, 1.5, a_name[i], fontsize=6)
			ax5.axvline(a_wav[i], linestyle="dashed", color="black", linewidth=0.6)
		if (max(wav_aa)-wav_range*3) < a_wav[i] < (max(wav_aa)-wav_range*2):
			ax4.text(a_wav[i]+0.2, 1.5, a_name[i], fontsize=6)
			ax4.axvline(a_wav[i], linestyle="dashed", color="black", linewidth=0.6)
		if (max(wav_aa)-wav_range*2) < a_wav[i] < (max(wav_aa)-wav_range*1):
			ax3.text(a_wav[i]+0.2, 1.5, a_name[i], fontsize=6)
			ax3.axvline(a_wav[i], linestyle="dashed", color="black", linewidth=0.6)
		if (max(wav_aa)-wav_range*1) < a_wav[i] < max(wav_aa):
			ax2.text(a_wav[i]+0.2, 1.5, a_name[i], fontsize=6)
			ax2.axvline(a_wav[i], linestyle="dashed", color="black", linewidth=0.6)			
	
	for i in np.arange(0, len(h2_name), 1):
		if min(wav_aa) < h2_wav[i] < (max(wav_aa)-wav_range*3):
			ax5.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=8, color="#a50f15")
			ax5.axvline(h2_wav[i], linestyle="dashed", color="#a50f15", linewidth=0.6)
		if (max(wav_aa)-wav_range*3) < h2_wav[i] < (max(wav_aa)-wav_range*2):
			ax4.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=8, color="#a50f15")
			ax4.axvline(h2_wav[i], linestyle="dashed", color="#a50f15", linewidth=0.6)
		if (max(wav_aa)-wav_range*2) < h2_wav[i] < (max(wav_aa)-wav_range*1):
			ax3.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=8, color="#a50f15")
			ax3.axvline(h2_wav[i], linestyle="dashed", color="#a50f15", linewidth=0.6)
		if (max(wav_aa)-wav_range*1) < h2_wav[i] < max(wav_aa):
			ax2.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=8, color="#a50f15")
			ax2.axvline(h2_wav[i], linestyle="dashed", color="#a50f15", linewidth=0.6)	

	lg = ax1.legend(numpoints=1, fontsize=12, loc=4)
	lg.get_frame().set_edgecolor("white")
	lg.get_frame().set_facecolor('#f0f0f0')
	
	ax1.set_xlabel(r"$\sf Observed\, Wavelength (\AA)$", fontsize=24)
	ax3.set_ylabel(r"$\sf Normalized\, Flux$", fontsize=24)
	ax1.axhline(1, color="#2171b5", linewidth=2)
	ax5.set_xlim([min(wav_aa), max(wav_aa)-wav_range*3])
	ax4.set_xlim([max(wav_aa)-wav_range*3, max(wav_aa)-wav_range*2])
	ax3.set_xlim([max(wav_aa)-wav_range*2, max(wav_aa)-wav_range*1])
	ax2.set_xlim([max(wav_aa)-wav_range*1, max(wav_aa)])
	
	fig.savefig("fit_spec.pdf")
	
##################################################################################### 

##################################################################################### 

##################################################################################### 

def plot_H2_hist(res_file="save_results.dat", z = 2.358):

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


##################################################################################### 

##################################################################################### 

##################################################################################### 

def plot_H2_trace(res_file="save_results.dat", z = 2.358):

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

##################################################################################### 

############################## CREATE BOKEH PLOT #################################### 

##################################################################################### 

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


##################################################################################### 

################################# PLOT H2 VIB #######################################

##################################################################################### 


def plot_H2vib(wav_aa, n_flux, y_min, y_max, y_min2, y_max2, y_fit, a_name, a_wav):

	sns.set_style("white")

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
	ax1.set_ylim([-0.6, 1.5])

	for i in np.arange(0, len(a_name), 1):
		if min(wav_aa) < a_wav[i] < (max(wav_aa)):
			ax1.text(a_wav[i]+0.2, 0.0, a_name[i], fontsize=8)
			ax1.axvline(a_wav[i], linestyle="dashed", color="black", linewidth=0.6)

	for axis in ['top','bottom','left','right']:
	  ax1.spines[axis].set_linewidth(2)
	ax1.tick_params(which='major', length=8, width=2)
	ax1.tick_params(which='minor', length=4, width=1.5)
	
	for tick in ax1.xaxis.get_major_ticks():
	    tick.label.set_fontsize(18)
	for tick in ax1.yaxis.get_major_ticks():
		tick.label.set_fontsize(18)

	fig.savefig("H2vib_fit_spec.pdf")







