#! usr/bin/env python

import pymc, math, argparse, os, sys, time

import numpy as np
import pandas as pd

import matplotlib as plt

from scipy.special import wofz
from scipy.interpolate import interp1d
from scipy import integrate

sys.path.append('bin/')

from spec_functions import * # spec_functions.py
from syn_spec import * # syn_spec.py
from sns_plots import * # sns_plots.py

from astropy.convolution import Gaussian1DKernel, convolve

e = 4.8032e-10
c = 2.998e10
m_e = 9.10938291e-28

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

def plot_spec(wav_aa, flux, flux_err, grb_name, y_min, y_max, y_min2, y_max2, y_fit):

	sns.set_style("white", {'legend.frameon': True})

	fig = figure(figsize=(10, 6))
	ax = fig.add_axes([0.13, 0.15, 0.83, 0.78])

	ax.errorbar(wav_aa,flux,yerr=flux_err,color='gray',marker='o',
		ls='None',label='Observed')
	ax.plot(wav_aa,flux,drawstyle='steps-mid',color='gray',alpha=0.66)
	ax.plot(wav_aa,y_fit,label='Fit',color="black",linewidth=1.5,linestyle="dashed")
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

#def H(a, u):
#    I = integrate.quad(lambda y: np.exp(-y**2)/(a**2 + (u - y)**2),-np.inf, np.inf)[0]
#    return (a/np.pi)*I

def H(a,u):

	P = u**2 
	H0 = np.exp(-u**2)
	Q = 1.5/u**2
	return H0 - a / np.sqrt(np.pi) / P * ( H0 ** 2 * \
		(4. * P**2 + 7. * P + 4. + Q) - Q - 1.0)

def mult_voigts(wav_aa,flux,flux_err,redshift,l0,f,gamma,nvoigts,RES,CSV_LST):
	'''
	Fitting a number of Voigt profiles to a normalized spectrum in
	velocity space
	'''

	l0 = l0 * (1+redshift)

	tau = 1 / np.array(flux_err)**2
	bg = pymc.Uniform('bg',lower=1E-18,upper=1E-14, doc='bg') # constant Background

	vars_dic = {}

	for i in range(1, nvoigts+1):

		b = pymc.Uniform('b'+str(i),lower=0.,upper=40.,doc='b'+str(i))
		l = pymc.Uniform('l'+str(i),lower=l0-20,upper=l0+20,doc='l'+str(i))
		N = pymc.Uniform('N'+str(i),lower=0.0,upper=22.0,doc='N'+str(i))
	
		CSV_LST.extend(('b'+str(i),'l'+str(i),'N'+str(i)))
	
		vars_dic['b'+str(i)] = b
		vars_dic['l'+str(i)] = l
		vars_dic['N'+str(i)] = N

		CSV_LST.extend(('b'+str(i),'l'+str(i),'N'+str(i)))
	
	@pymc.deterministic(plot=False)
	def multVoigt(x=wav_aa,bg=bg,gamma=gamma,f=f,nvoigts=nvoigts,vars_dic=vars_dic):

		flux = bg

		for i in range(1, nvoigts + 1):

			a = vars_dic["l"+str(i)] * gamma / vars_dic["b"+str(i)]
			u = c * (wav_aa - vars_dic["l"+str(i)]) / vars_dic["b"+str(i)]
			tau = 10**(vars_dic["N"+str(i)]) * (wav_aa**2 * np.sqrt(np.pi) * e**2 * f)/(c**2 * vars_dic["b"+str(i)] * m_e)*H(a,u)
			flux = flux * np.exp(-tau)

		return flux

	y_val = pymc.Normal('y_val',mu=multVoigt,tau=tau,value=flux,observed=True)
	return locals()

def do_mcmc(grb,redshift,l0,wav_aa,flux,flux_err,grb_name,f,gamma,nvoigts,iterations,burn_in,RES):
	'''
	MCMC sample 
	Reading and writing Results
	'''
	CSV_LST = ["bg"]

	pymc.np.random.seed(1)

	MDL = pymc.MCMC(mult_voigts(wav_aa,flux,flux_err,redshift,l0,f,gamma,nvoigts,RES,CSV_LST), db='pickle',dbname='velo_fit.pickle')

	MDL.db
	#MDL.use_step_method(pymc.AdaptiveMetropolis, MDL.velo_pred)
	MDL.sample(iterations, burn_in)
	MDL.db.close()

	y_min = MDL.stats()['multVoigt']['quantiles'][2.5]
	y_max = MDL.stats()['multVoigt']['quantiles'][97.5]
	y_min2 = MDL.stats()['multVoigt']['quantiles'][25]
	y_max2 = MDL.stats()['multVoigt']['quantiles'][75]
	y_fit = MDL.stats()['multVoigt']['mean']
	
	MDL.write_csv(grb_name+"_"+str(nvoigts)+"_voigt_res.csv",variables=CSV_LST)

	csv_f = open(grb_name+"_"+str(nvoigts)+"_voigt_res.csv", "a")
	csv_f.write("Osc, " + str(f) + "\n")
	csv_f.write("GAMMA, " + str(gamma))
	csv_f.close()

	return y_min, y_max, y_min2, y_max2, y_fit

l0 = 1808.0129
redshift = 2.710
f, gamma = get_osc_gamma(l0)
RES = 33

nvoigts=3
iterations = 100
burn_in = 20

wav_aa, flux, flux_err, grb_name, res, psf_fwhm = read_data("spectra/GRB161023A_OB1VIS.txt", redshift, l0, 8)

y_min, y_max, y_min2, y_max2, y_fit = do_mcmc(grb_name,redshift,l0,wav_aa,flux,flux_err,grb_name,f,gamma,
			nvoigts,iterations+(nvoigts*400),burn_in+(nvoigts*400),RES)


plot_spec(wav_aa, flux, flux_err, grb_name, y_min, y_max, y_min2, y_max2, y_fit)



#def gauss(x, mu, sig):
#	return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#
#sns.set_style("white", {'legend.frameon': True})
#
#fig = figure(figsize=(12, 5))
#ax1 = fig.add_axes([0.03, 0.08, 0.45, 0.90])
#ax2 = fig.add_axes([0.53, 0.08, 0.45, 0.90])
#
#x_vals = np.arange(-100, 1000, 1)
#x_vals2 = np.arange(0, 1000, 1)
#TEMP = gauss(x_vals2, mu=100, sig=280)
#B = gauss(x_vals2, mu=2.0, sig=7)
#A_Z = gauss(x_vals, mu=0, sig=25)
#
#ax2.plot(x_vals2, TEMP, color="red")
#ax1.plot(x_vals2, B)
#ax1.plot(x_vals, A_Z)
#
#ax1.set_ylim([-0.01, 1.01])
#ax1.set_xlim([-80, 80])
#
#ax2.set_ylim([-0.01, 1.01])
#ax2.set_xlim([-80, 950])
#
#show()













