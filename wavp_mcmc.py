#! /usr/bin/python

"""
MCMC sampler for fitting X-shooter spectra with voigt profiles
=========================================================================
e.g.: velop_mcmc.py -f spectra/GRB120815A_OB1UVB.txt -z 2.358
    -line 1347.2396 -e Cl -vr 400 -w 12 -min 3 -max 4 -it 200
    -bi 80 -res 28 -par velo_para.csv
=========================================================================
-f 			path to spectrum data file
-line 		line in AA, eg. 1808.0129
-e 			Name of the line/element / e.g.: FeII, SiII
-z 			redshift for centering
-vr 		velocity range (-vr to +vr)
-min 		minimum number of voigt profiles to fit
-max 		maximum number of voigt profiles to fit
-res 		spectral resolution km/s (fwhm)
-par 		Parameter file with velocity components
-it 		number of iterations
-bi 		burn-in
-w 			walength range (in AA) to be extracted from spectrum
=========================================================================
ClI     1347.2396  0.15300000
SiII    1526.7070  0.13300000 
SiII    1808.0129  0.00208000
FeII    1608.4509  0.05399224
FeII    2344.2129  0.12523167
=========================================================================
"""

__author__ = "Jan Bolmer"
__copyright__ = "Copyright 2017"
__version__ = "1.0"
__maintainer__ = "Jan Bolmer"
__email__ = "jbolmer@eso.org"
__status__ = "stable"


import pymc, math, time, sys, os, argparse
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *
from scipy.special import wofz

from astropy.convolution import Gaussian1DKernel, convolve

sys.path.append('bin/')

from spec_functions import * # spec_functions.py
from syn_spec import * # syn_spec.py
from sns_plots import * # sns_plots.py

# constants to calculate the cloumn density
e = 4.8032E-10  # cm 3/2 g 1/2 s-1
c = 2.998e10 # speed of light cm/s
m_e = 9.10938291e-28 # electron mass g

colors = ["#a6cee3", "#1f78b4",
"#b2df8a", "#33a02c", "#fb9a99",
"#e31a1c", "#fdbf6f", "#ff7f00",
"#cab2d6", "#6a3d9a", "#a6cee3",
"#1f78b4", "#b2df8a", "#33a02c",
"#fb9a99", "#e31a1c", "#fdbf6f",
"#ff7f00", "#cab2d6", "#6a3d9a"]

#https://www.pantone.com/color-of-the-year-2017
pt_analogous = ["#86af49", "#817397", "#b88bac",
"#d57f70","#dcb967","#ac9897","#ac898d","#f0e1ce",
"#86af49", "#817397","#b88bac", "#d57f70", "#dcb967"]

#========================================================================
#========================================================================

def get_results(para_file):
	'''
	Reads the Results from the .csv file created
	from PyMC
	'''

	par_dic = {}
	par = pd.read_csv(para_file, delimiter=', ', engine='python')
	for i in np.arange(0, len(par), 1):
		par_dic[par['Parameter'][i]] = [par['Mean'][i], par['SD'][i]]

	#print par_dic
	return par_dic

def print_results(res_file, redshift):
	'''
	Prints the velocity components and corresponding redshift
	'''
	comp_str = ""
	red_str = ""
	with open(res_file, "r") as f:
		for line in f:
			s = line.split(",")
			if s[0].startswith("v"):
				comp_str += str(s[1]) + " "
				nz = v_to_dz(float(s[1]), redshift)
				red_str += str(nz) + " "
	print comp_str
	print red_str

#========================================================================
#========================================================================

def gauss(x, mu, sig):
	'''
	Normal distribution used to create prior probability distributions
	'''
	return np.exp(-np.power(x-mu,2.0)/(2.0*np.power(sig,2.0)))

#========================================================================
#========================================================================

def add_abs_velo(v, N, b, gamma, f, l0):
	'''
	Add absorption line l0 in velocity space v, given the oscillator strength,
	the damping constant gamma, column density N, and broadening b
	'''
	A = (((np.pi*e**2)/(m_e*c))*f*l0*1E-13) * (10**N)
	tau = A * voigt(v,b/np.sqrt(2.0),gamma)

	return np.exp(-tau)

#========================================================================
#========================================================================

def add_abs_wave(wav, N, b, gamma, f, l0):
	'''
	Add absorption line l0 in wavlength space, given the oscillator strength,
	the damping constant gamma, column density N, and broadening b
	'''

	# check units
	A = ((np.pi*(e**2))/(m_e*(c**2)))*f*(l0**2)*(10**N)
	tau = A * voigt(wav,b/np.sqrt(2.0),gamma)

	return np.exp(-tau)

#========================================================================
#========================================================================

def mult_voigts(velocity, fluxv, fluxv_err, f, gamma, l0, nvoigts, RES,
				CSV_LST, velo_range, para_dic):
	'''
	Fitting a number of Voigt profiles to a normalized spectrum in
	velocity space
	'''

	tau = 1 / np.array(fluxv_err)**2

	@pymc.stochastic(dtype=float)
	def a(value=1.0, mu=1.0, sig=0.05, doc="B"):
		pp = 0.0
		#if 0.85 <= value < 1.15:
		pp = gauss(value, mu, sig)
		#else:
		#	pp = -np.inf
		return pp

	vars_dic = {}

	for i in range(1, nvoigts+1):

		if not "v0" + str(i) in para_dic:
			v0 = pymc.Uniform('v0'+str(i),lower=-velo_range, upper=velo_range,
				doc='v0'+str(i))
		else:
			if para_dic["v0"+str(i)][0] == 0:
				v0 = pymc.Uniform('v0'+str(i),lower=para_dic["v0"+str(i)][2],
					upper=para_dic["v0"+str(i)][3], doc='v0'+str(i))

			if para_dic["v0"+str(i)][0] == 1:
				v0 = pymc.Uniform('v0'+str(i),lower=para_dic["v0"+str(i)][1]-0.5,
					upper=para_dic["v0"+str(i)][1]+0.5, doc='v0'+str(i))

		if not "b" + str(i) in para_dic:
			b = pymc.Uniform('b'+str(i),lower=5,upper=100,value=15,doc='b'+str(i))

		else:
			if para_dic["b"+str(i)][0] == 0:

				b = pymc.Normal('b'+str(i),mu=para_dic["b"+str(i)][1],
					tau=1.0/(para_dic["b"+str(i)][3]-para_dic["b"+str(i)][1])**2,
					doc='b'+str(i))

			if para_dic["b"+str(i)][0] == 1:
				print "b fixed to:", para_dic["b"+str(i)][1] 
				b = pymc.Normal('b'+str(i),mu=para_dic["b"+str(i)][1],
					tau=1.0/(para_dic["b"+str(i)][3]-para_dic["b"+str(i)][1])**6,
					doc='b'+str(i))

		N = pymc.Uniform('N'+str(i),lower=0,upper=24.,value=15,doc='N'+str(i))

		CSV_LST.extend(('v0'+str(i),'b'+str(i),'N'+str(i)))

		vars_dic['v0'+str(i)] = v0
		vars_dic['b'+str(i)] = b
		vars_dic['N'+str(i)] = N

	@pymc.deterministic(plot=False)
	def multVoigt(vv=velocity,a=a,f=f,gamma=gamma,l0=l0,
				  nvoigts=nvoigts,vars_dic=vars_dic):

		gauss_k = Gaussian1DKernel(stddev=RES/(2*np.sqrt(2*np.log(2))*transform),
			mode="oversample")

		flux = np.ones(len(vv))*a

		for i in range(1, nvoigts + 1):
			
			v = vv-vars_dic["v0"+str(i)]
			flux *= add_abs_velo(v, vars_dic["N"+str(i)],
				vars_dic["b"+str(i)], gamma, f, l0)

		return np.convolve(flux, gauss_k, mode='same')

	y_val = pymc.Normal('y_val',mu=multVoigt,tau=tau,value=fluxv,observed=True)
	return locals()

def do_mcmc(grb,redshift,velocity,fluxv,fluxv_err,grb_name,f,gamma,
			l0,nvoigts,iterations,burn_in,RES,velo_range,para_dic):
	'''
	MCMC sample 
	Reading and writing Results
	'''

	CSV_LST = ["a"]

	pymc.np.random.seed(1)

	MDL = pymc.MCMC(mult_voigts(velocity,fluxv,fluxv_err,
		f,gamma,l0,nvoigts,RES,CSV_LST,velo_range,para_dic),
		db='pickle',dbname='velo_fit.pickle')

	MDL.db
	MDL.sample(iterations, burn_in)
	MDL.db.close()

	y_min = MDL.stats()['multVoigt']['quantiles'][2.5]
	y_max = MDL.stats()['multVoigt']['quantiles'][97.5]
	y_min2 = MDL.stats()['multVoigt']['quantiles'][25]
	y_max2 = MDL.stats()['multVoigt']['quantiles'][75]
	y_fit = MDL.stats()['multVoigt']['mean']
	
	MDL.write_csv(grb_name+"_"+str(nvoigts)+"_"+str(l0)+
		"_voigt_res.csv",variables=CSV_LST)

	csv_f = open(grb_name+"_"+str(nvoigts)+"_"+str(l0)+
		"_voigt_res.csv", "a")
	csv_f.write("Osc, " + str(f) + "\n")
	csv_f.write("GAMMA, " + str(gamma_line))
	csv_f.close()
	
	return y_min, y_max, y_min2, y_max2, y_fit

#========================================================================
#========================================================================
	
def plot_results(grb, redshift, velocity, fluxv, fluxv_err, y_min,
	y_max, y_min2, y_max2, y_fit, res_file, f, gamma, l0, nvoigts,
	velo_range, grb_name, RES, element="SiII"):
	'''
	Plotting the Spectrum including the individual Voigt Profiles
	'''
	sns.set_style("white")

	par_dic = get_results(res_file)
	
	fig = figure(figsize=(10, 6))
	ax = fig.add_axes([0.13, 0.15, 0.85, 0.78])

	gauss_k = Gaussian1DKernel(stddev=RES/(2*np.sqrt(2*np.log(2))*transform),
		mode="oversample")

	for i in range(1, nvoigts+1):

		ff = np.ones(len(velocity))*par_dic["a"][0]

		v = velocity-par_dic["v0"+str(i)][0]

		ff *= np.convolve(add_abs_velo(v, par_dic["N"+str(i)][0],
				par_dic["b"+str(i)][0],gamma,f,l0),gauss_k,mode='same')

		b_C = round(par_dic["b"+str(i)][0],2)
		b_Cerr = round(par_dic["b"+str(i)][1],2)
		N_C = round(par_dic["N"+str(i)][0],2)
		N_Cerr = round(par_dic["N"+str(i)][1],2)

		print "Component",i,":","b:",b_C,"+/-",b_Cerr,"N:",N_C,"+/-",N_Cerr

		ax.axvline(par_dic["v0"+str(i)][0],linestyle="dashed",color="black",
			linewidth=1.2)

		ax.plot(velocity[8:-8],ff[8:-8]/par_dic["a"][0],label='Comp. '+str(i),
			color=pt_analogous[i-1],linewidth=2)

		ax.text(par_dic["v0"+str(i)][0],1.45,"b = "+str(b_C)+"+/-"+str(b_Cerr),
			rotation=55,color=pt_analogous[i-1])
		ax.text(par_dic["v0"+str(i)][0],1.65,"N = "+str(N_C)+"+/-"+str(N_Cerr),
			rotation=55,color=pt_analogous[i-1])
	
	ylim([-0.5, 1.75])
	xlim([-velo_range, velo_range])
	
	# Transparent for Presentations
	trans = False

	if trans == False:

		ax.axhline(0.0,xmin=0.0, xmax=1.0, linewidth=2,
			linestyle="dotted",color="black")
		ax.axhline(1.0,xmin=0.0, xmax=1.0, linewidth=2,
			linestyle="-",color="black")

		ax.errorbar(velocity,fluxv/par_dic["a"][0],yerr=fluxv_err,
			color='gray',marker='o',
			ls='None',label='Observed')
		ax.plot(velocity,fluxv/par_dic["a"][0], drawstyle='steps-mid',
			color='gray', alpha=0.66)

		ax.plot(velocity,y_fit/par_dic["a"][0],label='Fit',color="black",
			linewidth=1.5,
			linestyle="dashed")
		ax.fill_between(velocity,y_min/par_dic["a"][0],y_max/par_dic["a"][0],
			color='black',alpha=0.3)
		ax.fill_between(velocity,y_min2/par_dic["a"][0],y_max2/par_dic["a"][0],
			color='black',alpha=0.5)

		lg = legend(numpoints=1, fontsize=12, loc=3, ncol=2)
		lg.get_frame().set_edgecolor("white")
		lg.get_frame().set_facecolor('#f0f0f0')

		plt.title(str(grb)+" "+element+" "+str(l0)+
			" at z = "+str(redshift),fontsize=24)
		
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
		
		fig.savefig(grb_name+"_"+element+"_"+str(l0)+"_"
			+str(nvoigts)+".pdf",transparent=trans)

	else:

		ax.axhline(0.0,xmin=0.0, xmax=1.0, linewidth=2,
			linestyle="dotted",color="white")
		ax.axhline(1.0,xmin=0.0, xmax=1.0, linewidth=2,
			linestyle="-",color="white")

		ax.errorbar(velocity,fluxv/par_dic["a"][0],yerr=fluxv_err,
			color="#C56A4D",marker='o',
			ls='None',label='Observed')
		ax.plot(velocity,fluxv/par_dic["a"][0],drawstyle='steps-mid',
			color="#C56A4D", alpha=0.66)

		ax.plot(velocity,y_fit/par_dic["a"][0],label='Fit',color="white",
			linewidth=1.5,
			linestyle="dashed")
		ax.fill_between(velocity,y_min/par_dic["a"][0], y_max/par_dic["a"][0],
			color='white',alpha=0.5)
		ax.fill_between(velocity,y_min2/par_dic["a"][0], y_max2/par_dic["a"][0],
			color='white',alpha=0.8)
	
		lg = legend(numpoints=1, fontsize=12, loc=3, ncol=2)
		lg.get_frame().set_edgecolor('white')
		lg.get_frame().set_facecolor('white')
		for text in lg.get_texts():
			text.set_color("#7187A6")

		plt.title(str(grb)+" "+element+" "+str(l0)+" at z = "
			+str(redshift),fontsize=24, color="#7187A6")
	
		ax.set_xlabel(r"$\sf Velocity\, (km/s)$",fontsize=24,color="#7187A6")
		ax.set_ylabel(r"$\sf Normalized\, Flux$",fontsize=24,color="#7187A6")
		
		for axis in ['top','bottom','left','right']:
		  ax.spines[axis].set_linewidth(2)
		  ax.spines[axis].set_color("#7187A6")
		ax.tick_params(which='major',length=8,width=2,colors="#7187A6")
		ax.tick_params(which='minor',length=4,width=1.5,colors="#7187A6")

		for tick in ax.xaxis.get_major_ticks():
		    tick.label.set_fontsize(18)
		for tick in ax.yaxis.get_major_ticks():
			tick.label.set_fontsize(18)
		
		fig.savefig(grb_name+"_"+element+"_"+str(l0)+"_"
			+str(nvoigts)+".pdf",transparent=trans)

def plt_nv_chi2(chi2_list, min_n, max_n, grb_name):
	'''
	Plot reduced Chi^2 vs. number of components 
	'''

	fig = figure(figsize=(12, 6))
	ax = fig.add_axes([0.10, 0.14, 0.86, 0.85])

	ax.errorbar(range(min_n,max_n+1),chi2_list,linewidth=5)
	ax.errorbar(range(min_n,max_n+1),chi2_list,fmt="o",color="black",
		markersize=15)
	ax.set_xlabel(r"Number of Components",fontsize=24)
	ax.set_ylabel(r"${\chi}^2_{red}$",fontsize=24)
	ax.set_yscale("log")
	ax.set_ylim([0.1, 600])
	ax.set_xlim([min_n-0.5, max_n+0.5])
	ax.set_xticks(range(min_n, max_n+1))
	ax.set_yticks([0.2,0.5,1.0,2.0,5.0,10,20,50,100,200,500])
	ax.set_yticklabels(["0.2", "0.5", "1.0", "2.0", "5.0", "10",
		"20", "50", "100", "200", "500"])
	ax.axhline(1,xmin=0.0, xmax=1.0, linewidth=2,linestyle="dashed",
		color="black")

	for axis in ['top','bottom','left','right']:
	  ax.spines[axis].set_linewidth(2)
	ax.tick_params(which='major',length=8,width=2)
	ax.tick_params(which='minor',length=4,width=1.5)
	
	for tick in ax.xaxis.get_major_ticks():
	    tick.label.set_fontsize(18)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(18)

	fig.savefig(grb_name + "_Chi2red.pdf")

if __name__ == "__main__":

	writecmd("velo_cmd_hist.dat")

	start = time.time()
	print "\n Parsing Arguments \n"

	parser = argparse.ArgumentParser(usage=__doc__)
	parser.add_argument('-f','--file',dest="file",
		default="spectra/GRB120815Auvb.txt",type=str)
	parser.add_argument('-z','--z', dest="z",default=None,type=float)
	parser.add_argument('-e','--element',dest="element",
		default="FeII",type=str)
	parser.add_argument('-line','--line',dest="line",
		default=1608.4509,type=float)
	parser.add_argument('-w','--wav_range',dest="wav_range",
		default=10.0,type=float)
	parser.add_argument('-res','--res',dest="resolution",
		default=30,type=float)
	parser.add_argument('-it','--it',dest="it",default=10000,type=int)
	parser.add_argument('-bi','--bi',dest="bi",default=5000,type=int)
	parser.add_argument('-min','--min',dest="min",default=1,type=int)
	parser.add_argument('-max','--max',dest="max",default=6,type=int)
	parser.add_argument('-vr','--velo_range',dest="velo_range",
		default=410.0,type=float)
	parser.add_argument('-par','--par',dest="par",default=None,type=str)

	args = parser.parse_args()

	spec_file = args.file
	element = args.element
	redshift = args.z
	wav_range = args.wav_range
	l0 = args.line
	iterations = args.it
	burn_in = args.bi
	min_n = args.min
	max_n = args.max
	velo_range = args.velo_range
	para_file = args.par
	RES = args.resolution
	
	# Read Oscillator strength f and decay rate gamma
	# for given line
	f, gamma_line = get_osc_gamma(l0)

	# converting gamma to km/s
	gamma = (gamma_line * l0 * 10e-13) / (2 * math.pi)

	print "Gamma in km/s: ", gamma
	print "f: ", f

	# Read data, GRB-Name, Resolution and PSF_FWHM from file
	wav_aa, n_flux, n_flux_err, flux, flux_err, grb_name, \
	res, psf_fwhm = get_data(spec_file, redshift, wl_range=False)


	if not min(wav_aa) < l0*(1+redshift) < max(wav_aa):
		print "Line is at:", l0*(1+redshift), "Spectrum covers: ", \
			min(wav_aa), "to", max(wav_aa)
		sys.exit("ERROR: Chosen line must fall within the wavelength \
			range of the given file")

	if burn_in >= iterations:
		sys.exit("ERROR: Burn-In cannot be bigger than Iterations")

	if redshift == None:
		sys.exit("ERROR: please specify input redshift: e.g. -z 2.358")

	para_dic = {}

	if para_file != None:
		para_dic = get_paras_velo(para_file)
		print "\n Using parameters given in:", para_file

	velocity, fluxv, fluxv_err = aa_to_velo(wav_aa, n_flux,
		n_flux_err, l0, redshift, wav_range)


	transform = np.median(np.diff(velocity))
	print "transform", transform
	print "Resoluton in km/s:", RES
	print "Res (sigma):", RES/(2*np.sqrt(2*np.log(2)))
	print "stddev gauss_k:", RES/(2*np.sqrt(2*np.log(2))*transform)

	chi2_list = []

	for nvoigts in range(min_n, max_n+1):

		print "\n Using", nvoigts, "Voigt Profiles \n"

		y_min, y_max, y_min2, y_max2, y_fit = do_mcmc(grb_name,
			redshift,velocity,fluxv,fluxv_err,grb_name,f,gamma,
			l0,nvoigts,iterations+(nvoigts*400),burn_in+(nvoigts*400),RES,
			velo_range,para_dic)
		
		chi2 = 0
		for i in range(8, len(y_fit)-8, 1):
			chi2_tmp = (fluxv[i] - y_fit[i])**2 / (fluxv_err[i])**2
			chi2 += (chi2_tmp / (len(fluxv)+(4*nvoigts)))
		chi2_list.append(chi2)

		print "\n Chi^2_red:", chi2, "\n"

		time.sleep(0.5)

		res_file = grb_name+"_"+str(nvoigts)+"_"+str(l0)+"_voigt_res.csv"
	
		plot_results(grb_name+str(nvoigts),redshift,velocity[8:-8],
			fluxv[8:-8], fluxv_err[8:-8], y_min[8:-8], y_max[8:-8],
			y_min2[8:-8], y_max2[8:-8], y_fit[8:-8], res_file,
			f,gamma,l0,nvoigts,velo_range,grb_name,RES,element)

		print "Components:", print_results(res_file, redshift)

		sns_velo_pair_plot(grb_name,l0,file='velo_fit.pickle',nvoigts=nvoigts)

	print "\n Plotting Chi2"
	plt_nv_chi2(chi2_list, min_n, max_n, grb_name)

	os.system("mv *.pdf plots")
	print "\n Plots Moved to plots directory"

	os.system("mv *.csv results")
	print "\n Result .csv files moved to results directory"

 	dur = str(round((time.time() - start)/60, 1))
	sys.exit("\n Script finished after " + dur + " minutes")

#========================================================================
#========================================================================
