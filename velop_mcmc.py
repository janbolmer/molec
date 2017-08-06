#! /usr/bin/python

__author__ = "Jan Bolmer"
__copyright__ = "Copyright 2017"
__version__ = "0.1"
__maintainer__ = "Jan Bolmer"
__email__ = "jbolmer@eso.org"
__status__ = "Production"

'''
Suggestes lines to fit:
ClI     1347.2396  0.15300000
SiII    1526.7070  0.13300000 
SiII    1808.0129  0.00208000
FeII    1608.4509  0.05399224
FeII    2344.2129  0.12523167
'''

import pymc, math, time, sys, os, argparse
import numpy as np
import pandas as pd

import matplotlib as plt
import matplotlib.pyplot  as pyplot
from pylab import *
from scipy.special import wofz

from astropy.convolution import Gaussian1DKernel, convolve

sys.path.append('bin/')

from spec_functions import * # spec_functions.py
from syn_spec import * # syn_spec.py
from sns_plots import * # sns_plots.py

m_e = 9.10938291e-28 # g
hbar = 1.054571726e-27 # erg * s
alpha = 1 / 137.035999139 # dimensionless
K = np.pi * alpha * hbar / m_e # cm^2 / s

colors = ["#a6cee3", "#1f78b4",
"#b2df8a", "#33a02c", "#fb9a99",
"#e31a1c", "#fdbf6f", "#ff7f00",
"#cab2d6", "#6a3d9a", "#a6cee3",
"#1f78b4", "#b2df8a", "#33a02c",
"#fb9a99", "#e31a1c", "#fdbf6f",
"#ff7f00", "#cab2d6", "#6a3d9a"]

#def voigt(x, sigma, gamma):
#	z = (x + 1j*gamma) / (sigma * math.sqrt(2))
#	V = wofz(z).real / (sigma * math.sqrt(2*math.pi))
#	return V

def get_results(para_file):
	'''
	Reads the Results from the .csv file created
	from PyMC
	'''

	par_dic = {}
	par = pd.read_csv(para_file, delimiter=', ', engine='python')
	for i in np.arange(0, len(par), 1):
		par_dic[par['Parameter'][i]] = par['Mean'][i]

	return par_dic

def mult_voigts(velocity, fluxv, fluxv_err, gamma, nvoigts, CSV_LST):
	'''
	Fitting a number of Voigt profiles to a normalized spectrum in
	velocity space
	'''

	tau = 1 / np.array(fluxv_err)**2

	#n_voigts = pymc.DiscreteUniform('n_voigts', lower=nvoigts,
	 #upper=nvoigts, value=nvoigts, doc='n_voigts')
	a = pymc.Uniform('a', lower=0.98, upper=1.02, doc='a')
	velo_pred = pymc.Normal('velo_pred', mu=velocity, tau=1.2)

	vars_dic = {}

	for i in range(1, nvoigts+1):

		sigma = pymc.Uniform('sigma'+str(i),lower=0., upper=80.,
			doc='sigma'+str(i))
		v0 = pymc.Uniform('v0'+str(i),lower=-350., upper=350.,
			doc='v0'+str(i))
		A = pymc.Uniform('A'+str(i),lower=-600.,upper=0.0,
			doc='A'+str(i))

		CSV_LST.extend(('sigma'+str(i),'v0'+str(i),'A'+str(i)))

		vars_dic['sigma'+str(i)] = sigma
		vars_dic['v0'+str(i)] = v0
		vars_dic['A'+str(i)] = A

	@pymc.deterministic(plot=False)
	def multVoigt(vv=velo_pred, a=a, gamma=gamma,nvoigts=nvoigts,
		vars_dic=vars_dic):

		voigts = 0

		for i in range(1, nvoigts + 1):
			
			# use interpolation?

			x = vv-vars_dic["v0"+str(i)]
			V = vars_dic["A"+str(i)]*voigt(x, vars_dic["sigma"+str(i)], gamma)
			gauss_kernel = Gaussian1DKernel(stddev=28.0/((2*np.sqrt(2*np.log(2)))*transform),
				x_size=1, mode="oversample")
			V = convolve(V, gauss_kernel)
			voigts += V

		voigts += a

		return voigts

	y_val = pymc.Normal('y_val', mu=multVoigt, tau=tau, value=fluxv, observed=True)
	return locals()

def do_mcmc(grb, redshift, my_line, velocity, fluxv, fluxv_err, grb_name,
			gamma, nvoigts, iterations, burn_in):
	'''
	MCMC sample 
	Reading and writing Results
	'''
	CSV_LST = ["a"]

	pymc.np.random.seed(1)

	MDL = pymc.MCMC(mult_voigts(velocity, fluxv, fluxv_err, gamma, nvoigts, CSV_LST))
	MDL.use_step_method(pymc.AdaptiveMetropolis, MDL.velo_pred)
	MDL.sample(iterations, burn_in)

	y_min = MDL.stats()['multVoigt']['quantiles'][2.5]
	y_max = MDL.stats()['multVoigt']['quantiles'][97.5]
	y_min2 = MDL.stats()['multVoigt']['quantiles'][25]
	y_max2 = MDL.stats()['multVoigt']['quantiles'][75]
	y_fit = MDL.stats()['multVoigt']['mean']
	
	MDL.write_csv(grb_name+"_"+str(nvoigts)+"_voigt_res.csv",variables=CSV_LST)

	csv_f = open(grb_name+"_"+str(nvoigts)+"_voigt_res.csv", "a")
	csv_f.write("Osc, " + str(osc_line) + "\n")
	csv_f.write("GAMMA, " + str(gamma_line))
	csv_f.close()

	return y_min, y_max, y_min2, y_max2, y_fit
	

def plot_results(grb, redshift, my_line, velocity, fluxv, fluxv_err, 
	y_min, y_max, y_min2, y_max2, y_fit, para_file, gamma, nvoigts,
	velo_range, element="SiII"): 
	'''
	Plotting the Spectrum including the individual Voigt Profiles
	'''

	par_dic = get_results(para_file)
	
	fig = figure(figsize=(10, 6))
	ax = fig.add_axes([0.13, 0.13, 0.85, 0.80])

	for i in range(1, nvoigts+1):
		ff = []
		for vv in velocity:
			x = vv-par_dic["v0"+str(i)]
			V = par_dic["A"+str(i)]*voigt(x, par_dic["sigma"+str(i)], gamma) #par_dic["gamma"+str(i)])
			#gauss_kernel = Gaussian1DKernel(stddev=28.0/((2*np.sqrt(2*np.log(2)))*transform),
			#	x_size=1, mode="oversample")
			#V = convolve(V, gauss_kernel)
			#ff.append(V + par_dic["a"])
			ff.append(V)
		gauss_kernel = Gaussian1DKernel(stddev=23.0/((2*np.sqrt(2*np.log(2)))*transform),
				x_size=1, mode="oversample")
		V = convolve(ff, gauss_kernel)
		ff = V + par_dic["a"]

		# sigma * 2*sqrt(ln(2))
		# broad = round(par_dic["sigma"+str(i)]*2*math.sqrt(math.log(2)),1)

		broad = round(par_dic["sigma"+str(i)]/2,1)

		ax.axvline(par_dic["v0"+str(i)], linestyle="dashed", color="black", linewidth=1.2)
		ax.plot(velocity, ff, label='Voigt'+str(i), color=colors[i-1], linewidth=2)
		ax.text(par_dic["v0"+str(i)], 1.3, "b = " + str(broad), rotation=55)


	ax.errorbar(velocity,fluxv,yerr=fluxv_err,color='gray',marker='o',
		ls='None',label='Observed')
	ax.plot(velocity,fluxv, drawstyle='steps-mid', color='gray', alpha=0.66)

	ax.plot(velocity,y_fit,label='Fit',color="black",linewidth=1.5,
		linestyle="dashed")
	ax.fill_between(velocity,y_min, y_max,color='black',alpha=0.3)
	ax.fill_between(velocity,y_min2, y_max2,color='black',alpha=0.5)
	
	plt.title(element + " " + str(my_line) + " at z = " + str(redshift),
		fontsize=24)
	
	ylim([-0.5, 1.55])
	xlim([-velo_range, velo_range])
	
	lg = legend(numpoints=1, fontsize=12, loc=3, ncol=2)
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
	
	fig.savefig(grb + "_" + element + "_" + str(my_line) + ".pdf")
	#show()

def plt_nv_chi2(chi2_list, min_n, max_n, grb_name):

	fig = figure(figsize=(12, 6))
	ax = fig.add_axes([0.10, 0.14, 0.86, 0.85])

	ax.errorbar(range(min_n, max_n+1),chi2_list,linewidth=5)
	ax.errorbar(range(min_n, max_n+1),chi2_list,fmt="o",color="black",
		markersize=15)
	ax.set_xlabel(r"Number of Components",fontsize=24)
	ax.set_ylabel(r"${\chi}^2_{red}$",fontsize=24)
	ax.set_yscale("log")
	ax.set_ylim([0.1, 600])
	ax.set_xlim([min_n-0.5, max_n+0.5])
	ax.set_xticks(range(min_n, max_n+1))
	ax.set_yticks([0.2, 0.5, 1.0, 2.0, 5.0, 10, 20, 50, 100, 200, 500])
	ax.set_yticklabels(["0.2", "0.5", "1.0", "2.0", "5.0", "10",
		"20", "50", "100", "200", "500"])
	ax.axhline(1,linewidth=2,linestyle="dashed",color="black")

	for axis in ['top','bottom','left','right']:
	  ax.spines[axis].set_linewidth(2)
	ax.tick_params(which='major',length=8,width=2)
	ax.tick_params(which='minor',length=4,width=1.5)
	
	for tick in ax.xaxis.get_major_ticks():
	    tick.label.set_fontsize(18)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(18)

	show()
	fig.savefig(grb_name + "_Chi2red.pdf")

if __name__ == "__main__":

	#print __doc__

	start = time.time()
	print "\n Parsing Arguments \n"

	parser = argparse.ArgumentParser(usage=__doc__)
	parser.add_argument('-f','--file',dest="file",
		default="spectra/GRB120815Auvb.txt",type=str)
	parser.add_argument('-z','--z', dest="z",default=2.358,type=float)
	parser.add_argument('-e','--element',dest="element",
		default="FeII",type=str)
	parser.add_argument('-line','--line',dest="line",
		default=1608.4509,type=float)
	parser.add_argument('-w','--wav_range',dest="wav_range",
		default=10.0,type=float)
	parser.add_argument('-it','--it',dest="it",default=10000,type=int)
	parser.add_argument('-bi','--bi',dest="bi",default=5000,type=int)
	parser.add_argument('-min','--min',dest="min",default=1,type=int)
	parser.add_argument('-max','--max',dest="max",default=6,type=int)
	parser.add_argument('-vr','--velo_range',dest="velo_range",
		default=410.0,type=float)

	args = parser.parse_args()

	spec_file = args.file
	element = args.element
	redshift = args.z
	wav_range = args.wav_range
	line = args.line
	iterations = args.it
	burn_in = args.bi
	min_n = args.min
	max_n = args.max
	velo_range = args.velo_range
	
	# Read Oscillator strength f and decay rate gamma
	# for given line
	osc_line, gamma_line = get_osc_gamma(line)

	# converting gamma to km/s
	gamma = (gamma_line * line * 10e-13) / (2 * math.pi)

	print "Gamma in km/s: ", gamma

	# Read data, GRB-Name, Resolution and PSF_FWHM from file
	wav_aa, n_flux, n_flux_err, flux, flux_err, grb_name, \
	res, psf_fwhm = get_data(spec_file, redshift, wl_range=False)

	if not min(wav_aa) < line*(1+redshift) < max(wav_aa):
		print "Line is at:", line*(1+redshift), "Spectrum covers: ", \
			min(wav_aa), "to", max(wav_aa)
		sys.exit("ERROR: Chosen line must fall within the wavelength \
			range of the given file")

	if burn_in >= iterations:
		sys.exit("ERROR: Burn-In cannot be bigger than Iterations")

	velocity, fluxv, fluxv_err = aa_to_velo(wav_aa, n_flux,
		n_flux_err, line, redshift, wav_range)


	transform = np.median(np.diff(velocity))
	print "transform: ", transform
	print "Res (sigma): ", 28.0/(2*np.sqrt(2*np.log(2)))

	chi2_list = []

	for nvoigts in range(min_n, max_n+1):

		print "\n Using", nvoigts, "Voigt Profiles \n"

		y_min, y_max, y_min2, y_max2, y_fit = do_mcmc(grb_name,
			redshift, line, velocity, fluxv, fluxv_err, grb_name, gamma,
			nvoigts, iterations+(nvoigts*400), burn_in+(nvoigts*400))
		
		chi2 = 0
		for i in range(0, len(y_fit), 1):
			chi2_tmp = (fluxv[i] - y_fit[i])**2 / (fluxv_err[i])**2
			chi2 += (chi2_tmp / (len(fluxv)+(4*nvoigts)))
		chi2_list.append(chi2)

		print "\n Chi^2_red:", chi2, "\n"

		time.sleep(0.5)

		para_file = grb_name+"_"+str(nvoigts)+"_voigt_res.csv"
	
		plot_results(grb_name+str(nvoigts), redshift, line, velocity,
			fluxv, fluxv_err, y_min, y_max, y_min2, y_max2, y_fit,
			para_file, gamma, nvoigts, velo_range, element)

	plt_nv_chi2(chi2_list, min_n, max_n, grb_name)


 	dur = str(round((time.time() - start)/60, 1))
	sys.exit("\n Script finished after " + dur + " minutes")

#========================================================================
#========================================================================












