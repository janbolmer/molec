#! /usr/bin/python

import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *
import numpy as np
import mpmath
from mpmath import *
from scipy.special import wofz

from astropy.convolution import Gaussian1DKernel

import importlib

import seaborn as sns

e = 4.8032E-10  # cm 3/2 g 1/2 s-1
c = 2.998e10 # speed of light cm/s
m_e = 9.10938291e-28 # electron mass g

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

def add_abs_velo(v, N, b, gamma, f, l0):
	'''
	Add absorption line l0 in velocity space v, given the oscillator strength,
	the damping constant gamma, column density N, and broadening b
	'''

	A = (((np.pi*e**2)/(m_e*c))*f*l0*1E-13) * (10**N)
	tau = A * voigt(v,b/np.sqrt(2.0),gamma)

	return np.exp(-tau)

def gauss(x, mu, sig):

	return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

def draw_new(old_val, scale=2.0):
	new_val = np.random.normal(loc=old_val, scale=scale)
	return new_val

def likelihood(v, data_y, data_y_err, v0, b, N, t, gamma, f, l0):
	p = 1.0
	model_y = model_voigt(v, v0, b, N, t, gamma, f, l0)
	for i in np.arange(0, len(data_y), 1):
		#print mpmath.exp(-0.5*((data_y[i]-model_y[i])/data_y_err[i])**2)
		p *= 1./(data_y_err[i]*np.sqrt(2*np.pi)) * \
			mpmath.exp(-0.5*((data_y[i]-model_y[i])/data_y_err[i])**2)

	return p

def t_prior(value=9e-17, mu=9e-17, sig=1e-17):
	pp = gauss(value, mu, sig)
	return pp

def N_prior(value=13.0, mu=13.0, sig=8):
	if 0 <= value < 55.0:
		pp = gauss(value, mu, sig)
	else:
		pp = -np.inf
	return pp

def b_prior(value=35.0, mu=35.0, sig=15):
	if value > 0.0:
		pp = gauss(value, mu, sig)
	else:
		pp = -np.inf
	return pp

def v0_prior(value=0.0, mu=0.0, sig=50):
	pp = gauss(value, mu, sig)
	return pp

def model_voigt(v, v0, b, N, t, gamma, f, l0):

	vv = []
	F = []

	transform = 12.0012125737
	RES = 30.0
	#gauss_k = Gaussian1DKernel(stddev=RES/(2*np.sqrt(2*np.log(2))*transform),
	#	mode="oversample")

	for i in np.arange(0, len(v), 1):
		vv.append(v[i]-v0)
		F.append(t)

	vv = np.array(vv)
	F = np.array(F)
	F *= add_abs_velo(vv,N,b,gamma,f,l0)

	return F # np.convolve(F, gauss_k, mode='same')

def mcmc(iterations, burn_in, t_guess, N_guess, b_guess, v0_guess,f,gamma,l0):

	sns.set_style("white")

	t_trace = [t_guess]
	N_trace = [N_guess]
	b_trace = [b_guess]
	v0_trace = [v0_guess]
	
	t_trace_all = [t_guess]
	N_trace_all = [N_guess]
	b_trace_all = [b_guess]
	v0_trace_all = [v0_guess]

	for i in np.arange(0, iterations, 1):

		t_prop = draw_new(t_trace[i], scale=0.5e-17)
		N_prop = draw_new(N_trace[i], scale=3.0)
		b_prop = draw_new(b_trace[i], scale=8.0)
		v0_prop = draw_new(v0_trace[i], scale=10.0)

		v0_trace_all.append(v0_prop)
		N_trace_all.append(N_prop)
		b_trace_all.append(b_prop)
		t_trace_all.append(t_prop)

		############## v0 ##############

		l_old = likelihood(velocity, fluxv, fluxv_err, v0_trace[i],
			b_trace[i], N_trace[i], t_trace[i], gamma, f, l0)
		l_prop = likelihood(velocity, fluxv, fluxv_err, v0_prop,
			b_prop, N_prop, t_prop, gamma, f, l0)

		Av0 = v0_prior(v0_prop)*l_prop
		Bv0 = v0_prior(v0_trace[i])*l_old
		av0 = float(Av0/Bv0)

		if av0 >= 1:
			v0_trace.append(v0_prop)
		else:
			v0_trace.append(v0_trace[i])

		############## N ##############

		l_old = likelihood(velocity, fluxv, fluxv_err, v0_trace[i+1],
			b_trace[i], N_trace[i], t_trace[i], gamma, f, l0)
		l_prop = likelihood(velocity, fluxv, fluxv_err, v0_prop,
			b_prop, N_prop, t_prop, gamma, f, l0)

		AN = N_prior(N_prop)*l_prop
		BN = N_prior(N_trace[i])*l_old
		aN = float(AN/BN)

		if aN >= 1:
			N_trace.append(N_prop)
		else:
			N_trace.append(N_trace[i])

		############## b ##############

		l_old = likelihood(velocity, fluxv, fluxv_err, v0_trace[i+1],
			b_trace[i], N_trace[i+1], t_trace[i], gamma, f, l0)
		l_prop = likelihood(velocity, fluxv, fluxv_err, v0_prop,
			b_prop, N_prop, t_prop, gamma, f, l0)

		Ab = b_prior(b_prop)*l_prop
		Bb = b_prior(b_trace[i])*l_old
		ab = float(Ab/Bb)

		if ab >= 1:
			b_trace.append(b_prop)
		else:
			b_trace.append(b_trace[i])


		############## t ##############

		l_old = likelihood(velocity, fluxv, fluxv_err, v0_trace[i+1],
			b_trace[i+1], N_trace[i+1], t_trace[i], gamma, f, l0)
		l_prop = likelihood(velocity, fluxv, fluxv_err, v0_prop,
			b_prop, N_prop, t_prop, gamma, f, l0)

		At = t_prior(t_prop)*l_prop
		Bt = t_prior(t_trace[i])*l_old
		at = float(At/Bt)

		if at >= 1:
			t_trace.append(t_prop)
		else:
			t_trace.append(t_trace[i])

		############# Progress #############
		print i, "from", iterations
		if round(float(i)/float(iterations), 2) in np.arange(0, 1.1, 0.1):
				print str(i) + " from " + str(iterations), \
					str(round(float(i)/float(iterations),1)*100) + "%"

		############# Plot #############

		if i+1 == i+1:
		#if i+1 == iterations:
			fig = figure(figsize=(10, 8))
			
			ax = fig.add_axes([0.13, 0.08, 0.37, 0.14])
			ax.set_xlabel("Velocity", fontsize=12)
			ax.set_ylabel("Flux", fontsize=12)
			ax.set_yscale("log")
			
			ax.set_ylim([min(fluxv)-0.2*min(fluxv), max(fluxv)+0.1*max(fluxv)])

			ax.errorbar(velocity, fluxv, yerr=fluxv_err, fmt="o", color="#FF6A48")

			ax1 = fig.add_axes([0.56, 0.28, 0.40, 0.20])
			ax2 = fig.add_axes([0.56, 0.76, 0.40, 0.20])
			ax7 = fig.add_axes([0.08, 0.76, 0.40, 0.20])
			
			ax1.set_xlabel("N", fontsize=12)
			ax1.set_ylabel("Prior / Posterior", fontsize=12)
			ax2.set_xlabel("b", fontsize=12)
			ax2.set_ylabel("Prior / Posterior", fontsize=12)
	
			ax7.set_xlabel("v0", fontsize=12)
			ax7.set_ylabel("Prior / Posterior", fontsize=12)		

			ax3 = fig.add_axes([0.56, 0.08, 0.40, 0.14])
			ax4 = fig.add_axes([0.56, 0.56, 0.40, 0.14])
			
			ax3.set_xlabel("Iteration", fontsize=12)
			ax3.set_ylabel("N", fontsize=12)
			ax3.set_ylim([0.0, 30.0])
			
			ax4.set_xlabel("Iteration", fontsize=12)
			ax4.set_ylabel("b", fontsize=12)
			ax4.set_ylim([0.0, 85.0])
			
			ax8 = fig.add_axes([0.08, 0.28, 0.17, 0.20])
			ax8.set_xlabel("N", fontsize=12)
			ax8.set_ylabel("b", fontsize=12)

			ax9 = fig.add_axes([0.31, 0.28, 0.17, 0.20])
			ax9.set_xlabel("N", fontsize=12)
			ax9.set_ylabel("v0", fontsize=12)


			ax5 = fig.add_axes([0.08, 0.56, 0.40, 0.14])
			ax5.set_xlabel("Iteration", fontsize=12)
			ax5.set_ylabel("v0", fontsize=12)
			ax5.set_ylim([-100.0, 100.0])


			weights1 = np.ones_like(N_trace_all[burn_in:-1])/float(len(N_trace_all[burn_in:-1]))
			weights2 = np.ones_like(b_trace_all[burn_in:-1])/float(len(b_trace_all[burn_in:-1]))
			weights3 = np.ones_like(v0_trace_all[burn_in:-1])/float(len(v0_trace_all[burn_in:-1]))
			
			weights4= np.ones_like(N_trace_all)/float(len(N_trace_all))
			weights5 = np.ones_like(b_trace_all)/float(len(b_trace_all))
			weights6 = np.ones_like(v0_trace_all)/float(len(v0_trace_all))


			ax1.hist(N_trace_all[burn_in:-1],weights=weights1,bins=np.arange(0.0, 50.0, 1.00))
			ax2.hist(b_trace_all[burn_in:-1],weights=weights2,bins=np.arange(0.0, 120.0, 2.00))
			ax7.hist(v0_trace_all[burn_in:-1],weights=weights3,bins=np.arange(-120., 120.0, 4.00))
			

			ax1.hist(N_trace_all,weights=weights4,bins=np.arange(0.0, 40.0, 1.00), color="black", alpha=0.33)
			ax2.hist(b_trace_all,weights=weights5,bins=np.arange(0.0, 80.0, 1.00), color="black", alpha=0.33)
			ax7.hist(v0_trace_all,weights=weights6,bins=np.arange(-120., 120.0, 2.00), color="black", alpha=0.33)

			ax8.errorbar(N_trace_all, b_trace_all, fmt="o", markersize=2, color="#294E86", alpha=0.66)
			ax8.errorbar(N_trace_all[burn_in:-1], b_trace_all[burn_in:-1], color="#294E86", fmt="o", markersize=2)
			ax8.plot(N_trace, b_trace) #, fmt="o")

			ax9.errorbar(N_trace_all, v0_trace_all, fmt="o", markersize=2, color="#294E86", alpha=0.66)
			ax9.errorbar(N_trace_all[burn_in:-1], b_trace_all[burn_in:-1], color="#294E86", fmt="o", markersize=2)
			ax9.plot(N_trace, v0_trace) #, fmt="o")
			
			#x_trace = np.arange(0, iterations+1, 1)
			x_trace = np.arange(0, i+2, 1)
			ax3.plot(x_trace, N_trace_all)
			ax4.plot(x_trace, b_trace_all)
			ax5.plot(x_trace, v0_trace_all)
			
			ax3.plot(x_trace, N_trace)
			ax4.plot(x_trace, b_trace)
			ax5.plot(x_trace, v0_trace)

			ax3.axvspan(-2, burn_in, facecolor='grey', alpha=0.5)
			ax4.axvspan(-2, burn_in, facecolor='grey', alpha=0.5)
			ax5.axvspan(-2, burn_in, facecolor='grey', alpha=0.5)
			
			if i+1 < 50000:
				ax3.set_xlim([-1, i+10])
				ax4.set_xlim([-1, i+10])
				ax5.set_xlim([-1, i+10])
			else:
				ax3.set_xlim([-1, i+100])
				ax4.set_xlim([-1, i+100])
				ax5.set_xlim([-1, i+100])				
			
			y_model = model_voigt(velocity, v0_trace_all[-1], b_trace_all[-1],
				N_trace_all[-1],t_trace_all[-1], gamma, f, l0)

			ax.plot(velocity, y_model, color="#294E86", alpha=0.500)
			if i > 5:
				y_line1 = model_voigt(velocity, v0_trace_all[-2], b_trace_all[-2],
				N_trace_all[-2],t_trace_all[-2], gamma, f, l0)
				ax.plot(velocity, y_line1, color="#294E86", alpha=0.375)
				y_line2 = model_voigt(velocity, v0_trace_all[-3], b_trace_all[-3],
				N_trace_all[-3],t_trace_all[-3], gamma, f, l0)
				ax.plot(velocity, y_line2, color="#294E86", alpha=0.250)
				y_line3 = model_voigt(velocity, v0_trace_all[-4], b_trace_all[-4],
				N_trace_all[-4],t_trace_all[-4], gamma, f, l0)
				ax.plot(velocity, y_line3, color="#294E86", alpha=0.125)
			
			y_fit = model_voigt(velocity, v0_trace[-1], b_trace[-1], N_trace[-1],
				t_trace[-1], gamma, f, l0)
			ax.plot(velocity, y_fit, color="#294E86", linewidth=3.0)
			
			#show()
			fig.savefig("mcmc_voigt_fit_" + str(i+1) + ".jpeg", transparent=True)

wav_aa, n_flux, n_flux_err, flux, flux_err, grb_name, \
res, psf_fwhm = get_data("../spectra/GRB090926_OB1UVB.txt",
	2.1068, wl_range=False)

velocity, fluxv, fluxv_err = aa_to_velo(wav_aa, flux, flux_err, 1608.4509, 2.1068, 5.0)


f = 0.05399224
gamma = 0.0623086746483
l0 = 1608.4509

mcmc(2000, 1500, t_guess=8.0e-17,N_guess=11.0,b_guess=22.0,v0_guess=0.0,f=f,gamma=gamma,l0=l0)

