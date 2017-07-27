#! /usr/bin/python

import pymc, math, time, sys, os
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot  as pyplot
from pylab import *
from scipy.special import wofz

sys.path.append('bin/')

from spec_functions import get_data, aa_to_velo, voigt # spec_functions.py
from syn_spec import * # syn_spec.py
from sns_plots import * # sns_plots.py


colors = ["#a6cee3", "#1f78b4",
"#b2df8a", "#33a02c", "#fb9a99",
"#e31a1c", "#fdbf6f", "#ff7f00",
"#cab2d6", "#6a3d9a"]

def get_results(para_file="voigt_results.csv"):
	'''
	Reads the Results from the voigt_results.csv file
	'''

	par_dic = {}
	par = pd.read_csv(para_file, delimiter=', ', engine='python')
	for i in np.arange(0, len(par), 1):
		par_dic[par['Parameter'][i]] = par['Mean'][i]

	return par_dic

def mult_voigts(velocity, fluxv, fluxv_err):

	tau = 1 / np.array(fluxv_err)**2

	#n_voigts = pymc.DiscreteUniform('n_voigts', lower=0.1, upper=10, value=10, doc='n_voigts')
	a = pymc.Uniform('a', lower=0.75, upper=1.25, doc='a')
	velo_pred = pymc.Normal('velo_pred', mu=velocity, tau=2)

	@pymc.stochastic(dtype=int)
	def n_voigts(value=10, lower=1, upper=10):
		if value in range(lower, upper+1):
			return 1./(value+2)
		else:
			return -np.inf

	# setting initial values
	vars_dic = {}
	#"nu01": array(0.1), "alphaD1": array(0.1), "alphaL1": array(0.1), "A1":array(0.1),
	#"nu02": array(0.1), "alphaD2": array(0.1), "alphaL2": array(0.1), "A2":array(0.1),
	#"nu03": array(0.1), "alphaD3": array(0.1), "alphaL3": array(0.1), "A3":array(0.1),
	#"nu04": array(0.1), "alphaD4": array(0.1), "alphaL4": array(0.1), "A4":array(0.1),
	#"nu05": array(0.1), "alphaD5": array(0.1), "alphaL5": array(0.1), "A5":array(0.1),
	#"nu06": array(0.1), "alphaD6": array(0.1), "alphaL6": array(0.1), "A6":array(0.1),
	#"nu07": array(0.1), "alphaD7": array(0.1), "alphaL7": array(0.1), "A7":array(0.1),
	#"nu08": array(0.1), "alphaD8": array(0.1), "alphaL8": array(0.1), "A8":array(0.1),
	#"nu09": array(0.1), "alphaD9": array(0.1), "alphaL9": array(0.1), "A9":array(0.1),
	#"nu010":array(0.1), "alphaD10":array(0.1), "alphaL10":array(0.1), "A10":array(0.1)
	#}

	for i in range(1, n_voigts+1):

		alphaD = pymc.Uniform('alphaD'+str(i),lower=0.,upper=100.,doc='AlphaD'+str(i))
		alphaL = pymc.Uniform('alphaL'+str(i),lower=0.,upper=100.,doc='AlphaL'+str(i))
		nu0 = pymc.Uniform('nu0'+str(i),lower=-450., upper=450.,doc='nu0'+str(i))
		A = pymc.Uniform('A'+str(i),lower=-120.,upper=0.0,doc='A'+str(i))

		CSV_LST.extend(('alphaD'+str(i),'alphaL'+str(i),'nu0'+str(i),'A'+str(i)))

		vars_dic['alphaD'+str(i)] = alphaD
		vars_dic['alphaL'+str(i)] = alphaL
		vars_dic['nu0'+str(i)] = nu0
		vars_dic['A'+str(i)] = A

	@pymc.deterministic(plot=False)
	def multVoigt(nu=velo_pred, a=a, n_voigts=n_voigts, vars_dic=vars_dic):

		voigts = 0


		for i in range(1, n_voigts+1):
			f = np.sqrt(math.log(2))
			x = (nu-vars_dic["nu0"+str(i)])/vars_dic["alphaD"+str(i)] * f
			y = vars_dic["alphaL"+str(i)]/vars_dic["alphaD"+str(i)] * f
			V = vars_dic["A"+str(i)]*f/(vars_dic["alphaD"+str(i)]*np.sqrt(np.pi)) * voigt(x, y)
			voigts += V

		voigts += a

		return voigts

	y_val = pymc.Normal('y_val', mu=multVoigt, tau=tau, value=fluxv, observed=True)
	return locals()

def do_mcmc(grb, redshift, my_line, velocity, fluxv, fluxv_err):

	MDL = pymc.MCMC(mult_voigts(velocity, fluxv, fluxv_err))
	MDL.use_step_method(pymc.AdaptiveMetropolis, MDL.velo_pred)
	MDL.sample(10000, 1, 1)

	y_min = MDL.stats()['multVoigt']['quantiles'][2.5]
	y_max = MDL.stats()['multVoigt']['quantiles'][97.5]
	y_min2 = MDL.stats()['multVoigt']['quantiles'][25]
	y_max2 = MDL.stats()['multVoigt']['quantiles'][75]
	y_fit = MDL.stats()['multVoigt']['mean']
	
	MDL.write_csv("voigt_results.csv", variables=CSV_LST)

	return y_min, y_max, y_min2, y_max2, y_fit, MDL.trace('n_voigts')
	

def plot_results(grb, redshift, my_line, velocity, fluxv, fluxv_err, 
	y_min, y_max, y_min2, y_max2, y_fit, nv_trace, element="SiII"): 

	par_dic = get_results("voigt_results.csv")
	
	n_list = []
	n1, n2, n3, n4, n5 = 0, 0, 0, 0,0
	n6, n7, n8, n9, n10 = 0, 0, 0, 0,0

	# this must go better
	for i in nv_trace[:]:
		if i == 1:
			n1 += 1
		if i == 2:
			n2 += 1
		if i == 3:
			n3 += 1
		if i == 4:
			n4 += 1
		if i == 5:
			n5 += 1
		if i == 6:
			n6 += 1
		if i == 7:
			n7 += 1
		if i == 8:
			n8 += 1
		if i == 9:
			n9 += 1
		if i == 10:
			n9 += 1
		if i == 10:
			n10+= 1

	n_list.extend((n1, n2, n3, n4, n5, n6, n7, n8, n9, n10))

	m = max(n_list)

	n_max = [i for i, j in enumerate(n_list) if j == m]
	nvs = n_max[0] + 1 

	#nvs = float((par_dic['n_voigts']))
	#nvs = int(round(nvs, 0))

	fig = figure(figsize=(10, 6))
	ax = fig.add_axes([0.13, 0.13, 0.85, 0.85])

	for i in range(1, nvs+1):
		print i
		ff = []
		for vv in velocity:
			f = np.sqrt(math.log(2))
			x = (vv-par_dic["nu0"+str(i)])/par_dic["alphaD"+str(i)] * f
			y = par_dic["alphaL"+str(i)]/par_dic["alphaD"+str(i)] * f
			V = par_dic["A"+str(i)]*f/(par_dic["alphaD"+str(i)]*np.sqrt(np.pi)) * voigt(x, y)
			ff.append(V + par_dic["a"])

		ax.plot(velocity, ff, label='Voigt'+str(i), color=colors[i-1], linewidth=2)

	ax.errorbar(velocity, fluxv, yerr=fluxv_err, color='gray', marker='o', ls='None', label='Observed')
	ax.plot(velocity, y_fit, label='Fit', color="black", linewidth=1.5, linestyle="dashed")
	ax.fill_between(velocity, y_min, y_max, color='black', alpha=0.3)
	ax.fill_between(velocity, y_min2, y_max2, color='black', alpha=0.5)
	
	ax.text(-200, 1.4, element + " " + str(my_line), fontsize=24)
	
	ylim([-0.5, 1.55])
	
	lg = legend(numpoints=1, fontsize=16, loc=3)
	lg.get_frame().set_edgecolor("white")
	lg.get_frame().set_facecolor('#f0f0f0')
	
	ax.set_xlabel(r"$\sf Velocity\, (km/s)$", fontsize=24)
	ax.set_ylabel(r"$\sf Normalized\, Flux$", fontsize=24)
	
	for axis in ['top','bottom','left','right']:
	  ax.spines[axis].set_linewidth(2)
	ax.tick_params(which='major', length=8, width=2)
	ax.tick_params(which='minor', length=4, width=1.5)
	
	for tick in ax.xaxis.get_major_ticks():
	    tick.label.set_fontsize(18)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(18)
	
	fig.savefig(grb + "_" + str(my_line) + ".pdf")
	show()

def plot_nv_trace(nv_trace):

	fig = figure(figsize=(10, 4))
	ax = fig.add_axes([0.13, 0.13, 0.85, 0.85])

	step = []

	for i in np.arange(0, len(nv_trace), 1):
		step.append(i)

	ax.errorbar(step, nv_trace)
	ax.set_ylim([-0.5, 11])
	show()

def plot_nv_hist(nv_trace):

	fig = figure(figsize=(6, 6))
	ax = fig.add_axes([0.13, 0.13, 0.85, 0.85])

	ax.hist(nv_trace)
	ax.set_xlim([0.5, 10.5])
	plt.title("Number of Absorption lines")
	show()

if __name__ == "__main__":

	CSV_LST = ["n_voigts", "a"]
	
	redshift = 2.707
	my_line = 1808.0129
	
	wav_aa, n_flux, n_flux_err, flux, flux_err, grb_name, res, psf_fwhm = get_data(
		"spectra/GRB121229Avis.txt", redshift, wl_range=False)
	
	velocity, fluxv, fluxv_err = aa_to_velo(wav_aa, n_flux, n_flux_err, my_line,
		redshift, 4.0)
	
	y_min, y_max, y_min2, y_max2, y_fit, nv_trace = do_mcmc(grb_name, redshift, my_line,
		velocity,fluxv, fluxv_err)
	
	time.sleep(5.0)

	plot_results(grb_name, redshift, my_line, velocity, 
		fluxv, fluxv_err, y_min, y_max, y_min2, y_max2, y_fit, nv_trace,
		element="SiII")

	plot_nv_trace(nv_trace[:])
	plot_nv_hist(nv_trace[:])
















