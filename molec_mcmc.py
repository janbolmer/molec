#! /usr/bin/env python

"""
MCMC sampler for fitting X-shooter spectra (using PyMC2) with HI, H2 and
other lines
=========================================================================
e.g.: molec_mcmc.py -f spectra/GRB120815Auvb.txt -m H2vib -w1 1566
		-w2 1611 -red 2.358 -it 40 -bi 10 -t GRB120815A -e GeII FeII
=========================================================================
-target    	target name
-file 		path to spectrum data file
-model  	Model to use: H2, H2vib or CO
-elements 	list of additional elements to be included, e.g.: ["FeII"]
-redshift 	redshift
-nrot 		number of H2 rotational levels, 1-7
-w1 		Wavelength start - restframe
-wl 		Wavelength end - restframe
-ignore 	list of intervals (restframe) to be ignored when fitting the
			data, e.g. [[1090., 1092.0], [1140, 1180]]
-res 		spectral resolution for rebinning the H2vib model from Draine

=========================================================================
.....
=========================================================================
"""

import pymc, math, argparse, sys, time

import numpy as np
import pandas as pd

import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *

from scipy.special import wofz
from scipy.interpolate import interp1d

sys.path.append('bin/')

from spec_functions import * # spec_functions.py
from syn_spec import * # syn_spec.py
from sns_plots import * # sns_plots.py

# physical constants
m_e = 9.1095e-28
e = 4.8032e-10
c = 2.998e10

def model_csv_file(model):

	if model == "H2":
		CSV_LST = ['NTOTH2', 'TEMP', 'B', 'A_Z', 'N_H', 'B_H', 'A_Z_H']
	elif model == "H2vib":
		CSV_LST = ['MH2S', 'A_Z']
	else:
		sys.exit("ERROR: Choose one of the following models: H2, H2vib \
			and CO")

	return CSV_LST

def model_H2(wav_aa, n_flux, n_flux_err, redshift, line_lst): 

	tau = 1 / np.array(n_flux_err)**2

	NTOTH2 = pymc.Uniform('NTOTH2',lower=10., upper=22.0, doc='NTOTH2')
	TEMP = pymc.Uniform('TEMP', lower=1., upper=800, doc='TEMP')
	B = pymc.Uniform('B', lower=1., upper=80.0, doc='B')	
	A_Z = pymc.Uniform('A_Z', lower=-100, upper=+100, doc='A_Z')
	
	N_H = pymc.Uniform('N_H', lower=17.0, upper=23.0, doc='N_H')
	B_H = pymc.Uniform('B_H', lower=1, upper=40, doc='B_H')
	A_Z_H = pymc.Uniform('A_Z_H', lower=-100, upper=+100, doc='A_Z_H')
	
	vars_dic = {}

	for elmt in line_lst: 
		N_E = pymc.Uniform('N_' + elmt, lower=16.0, upper=19.0, doc='N_' + elmt)
		B_E = pymc.Uniform('B_' + elmt, lower=1.0, upper=30, doc='B_' + elmt)
		A_Z_E = pymc.Uniform('A_Z_' + elmt, lower=-50, upper=+50, doc='A_Z_' + elmt)
		CSV_LST.extend(('N_' + elmt, 'B_' + elmt, 'A_Z_' + elmt))
		vars_dic[elmt] = N_E, B_E, A_Z_E

	@pymc.deterministic(plot=False)
	def H2(wav_aa=wav_aa, A_REDSHIFT=A_Z, NTOTH2=NTOTH2, TEMP=TEMP, BROAD=B, \
			redshift=redshift, N_H=N_H, BROAD_H=B_H, A_REDSHIFT_H=A_Z_H, vars_dic=vars_dic):

		A_REDSHIFT = float(A_REDSHIFT)/100000.0
		A_REDSHIFT_H = float(A_REDSHIFT_H)/100000.0

		norm_spec = np.ones(len(wav_aa))
		synspec = SynSpec()

		# add H2 1
		h2spec = synspec.add_H2(wav_aa, norm_spec, redshift, broad=BROAD, NTOTH2=NTOTH2, \
			TEMP=TEMP, A_REDSHIFT=A_REDSHIFT, NROT=NROT)
		# add HI

		h2spec_HI = synspec.add_ion(wav_aa, h2spec, redshift, "HI", broad=BROAD_H, Natom=N_H, \
			A_REDSHIFT=A_REDSHIFT_H)

		# add lines
		for key in vars_dic:
			A_Z_E = vars_dic[key][2]/100000.00
			h2spec_HI_lines = synspec.add_ion(wav_aa, h2spec_HI, redshift, key, broad=vars_dic[key][1], \
				Natom=vars_dic[key][0], A_REDSHIFT=A_Z_E)

		return h2spec_HI_lines

	y_val = pymc.Normal('y_val', mu=H2, tau=tau, value=n_flux, observed=True)
	return locals()


def model_H2vib(wav_aa, n_flux, n_flux_err, redshift, line_lst):

	tau = 1 / np.array(n_flux_err)**2

	MH2S = pymc.Uniform('MH2S', lower=0.000, upper=0.40, doc='MH2S')
	A_Z = pymc.Uniform('A_Z', lower=-150, upper=+150, value=10, doc='A_Z')

	vars_dic = {}

	for elmt in line_lst: 
		N_E = pymc.Uniform('N_' + elmt, lower=5.0, upper=20.0, doc='N_' + elmt)
		B_E = pymc.Uniform('B_' + elmt, lower=1.0, 	upper=30, doc='B_' + elmt)
		A_Z_E = pymc.Uniform('A_Z_' + elmt,	lower=-150,	upper=+150, doc='A_Z_' + elmt)
		CSV_LST.extend(('N_' + elmt, 'B_' + elmt, 'A_Z_' + elmt))
		vars_dic[elmt] = N_E, B_E, A_Z_E

	@pymc.deterministic(plot=False)
	def H2vib(wav_aa=wav_aa, redshift=redshift, A_REDSHIFT=A_Z, MH2S=MH2S, vars_dic=vars_dic):

		A_REDSHIFT = float(A_REDSHIFT)/100000.0

		norm_spec = np.ones(len(wav_aa))
		synspec = SynSpec()
	
		h2vib_spec = synspec.add_vibH2(wav_aa, norm_spec, redshift, h2swl, modspec, tauspec, \
			RES=6000, MH2S=MH2S, A_REDSHIFT=A_REDSHIFT)

		if len(vars_dic) > 1:
			for key in vars_dic:
				A_Z_E = vars_dic[key][2]/100000.0
				h2vib_spec = synspec.add_ion(wav_aa, h2vib_spec, redshift, key, \
					broad=vars_dic[key][1], Natom=vars_dic[key][0], A_REDSHIFT=A_Z_E)

		return h2vib_spec

	y_val = pymc.Normal('y_val', mu=H2vib, tau=tau, value=n_flux, observed=True)
	return locals()


def makeMCMC(wav_aa, n_flux, n_flux_err, trials, burn_in, n_thin, model_used, line_lst):

	if model == "H2":

		MDL = pymc.MCMC(model_H2(wav_aa, n_flux, n_flux_err, redshift, line_lst), \
			db='pickle', dbname='H2_fit.pickle')
		MDL.db
		MDL.sample(trials, burn_in, n_thin)
		MDL.db.close()
	
		MDL.write_csv("120815A_H2_results.csv", variables=CSV_LST)
	
		y_min 	= MDL.stats()[model_used]['quantiles'][2.5]
		y_max 	= MDL.stats()[model_used]['quantiles'][97.5]
		y_min2 	= MDL.stats()[model_used]['quantiles'][25]
		y_max2 	= MDL.stats()[model_used]['quantiles'][75]
		y_fit 	= MDL.stats()[model_used]['mean']
	
		return y_min, y_max, y_min2, y_max2, y_fit

	if model == "H2vib":

		MDL = pymc.MCMC(model_H2vib(wav_aa, n_flux, n_flux_err, redshift, line_lst), \
			db='pickle', dbname='H2vib_fit.pickle')
		MDL.db
		MDL.sample(trials, burn_in, n_thin)
		MDL.db.close()
	
		MDL.write_csv("120815A_H2vib_results.csv", variables=CSV_LST)
	
		y_min 	= MDL.stats()[model_used]['quantiles'][2.5]
		y_max 	= MDL.stats()[model_used]['quantiles'][97.5]
		y_min2 	= MDL.stats()[model_used]['quantiles'][25]
		y_max2 	= MDL.stats()[model_used]['quantiles'][75]
		y_fit 	= MDL.stats()[model_used]['mean']
	
		return y_min, y_max, y_min2, y_max2, y_fit


if __name__ == "__main__":

	print "\n Parsing Arguments \n"

	parser = argparse.ArgumentParser(usage=__doc__)
	parser.add_argument('-t','--target',dest="target",nargs=1,default="GRB", type=str)
	parser.add_argument('-f','--file',dest="file",nargs=1,default="spectra/GRB120815Auvb.txt")
	parser.add_argument('-red','--redshift',dest="redshift",nargs=1, default=[2.358], type=float)
	parser.add_argument('-m','--model',dest="model", nargs=1, default="H2", type=str)
	parser.add_argument('-nrot','--nrot',dest="nrot",nargs=1, default=[3], type=int)
	parser.add_argument('-e','--elements',dest="elements",nargs='+', default=["FeII", "SiII"])
	parser.add_argument('-w1','--w1',dest="w1", nargs=1, default=[980.0])
	parser.add_argument('-w2','--w2',dest="w2", nargs=1, default=[1120.0])
	parser.add_argument('-ign','--ignore',dest="ignore",nargs=1, default=[])
	parser.add_argument('-res','--resolution',dest="resolution",nargs=1, default=6000)
	parser.add_argument('-it','--iterations',dest="iterations",nargs=1, default=1000, type=int)
	parser.add_argument('-bi','--burn_in',dest="burn_in",nargs=1, default=100, type=int)
	args = parser.parse_args()

	target = args.target[0]
	spec_file = str(args.file[0])
	model = args.model[0]
	elements = args.elements
	redshift = args.redshift[0]
	nrot = int(args.nrot[0])
	w1 = float(args.w1[0])
	w2 = float(args.w2[0])
	ignore_lst = args.ignore
	res = args.resolution
	iterations = args.iterations[0]
	burn_in = args.burn_in[0]

	if burn_in >= iterations:
		sys.exit("ERROR: Burn-In cannot be bigger than Iterations")

	NROT = []
	for i in np.arange(0, nrot+1, 1):
		NROT.append(i)

	CSV_LST = model_csv_file(model)

	time.sleep(1.0)

	print "\n Fitting", target, "at redshift", redshift, "\n"

	time.sleep(1.0)

	print "\n Starting MCMC \n"

	time.sleep(1.0)

	a_name, a_wav, ai_name, ai_wav, aex_name, aex_wav, h2_name, h2_wav = get_lines(redshift)

	wav_aa_pl, n_flux_pl, n_flux_err_pl, flux_pl, flux_err_pl = get_data(spec_file, \
		redshift, wl_range=True, wl1=w1, wl2=w2)

	wav_aa, n_flux, n_flux_err = get_data_ign(spec_file, redshift, ignore_lst, wl1=w1, wl2=w2)
	
	y_min, y_max, y_min2, y_max2, y_fit = makeMCMC(wav_aa, n_flux, n_flux_err, iterations, burn_in, 1, \
		model_used=model, line_lst=elements)
	
	print "\n MCMC finished \n"

	time.sleep(1.0)

	print "Model used:", model, "with J =", NROT, "and", elements
	print "using", iterations, "iterations", "and a burn-in of", burn_in
	print "Wavelenth range:", w1, "to", w2, "/ ignored are", ignore_lst

	time.sleep(1.0)

	print "\n Plotting Results \n"


	if model == "H2":

		plot_spec(wav_aa_pl, n_flux_pl, y_min, y_max, y_min2, y_max2, y_fit, redshift, ignore_lst, \
			a_name, a_wav, ai_name, ai_wav, aex_name, aex_wav, h2_name, h2_wav)


	if model == "H2vib":

		plot_H2vib(wav_aa_pl, n_flux_pl, y_min, y_max, y_min2, y_max2, y_fit, a_name, a_wav)

		sns_H2vib_plot(var_list=CSV_LST, file="H2vib_fit.pickle", redshift=redshift)

		bokeh_H2vib_plt(wav_aa_pl, n_flux_pl, y_min, y_max, y_min2, y_max2, y_fit, redshift, ignore_lst, \
			a_name, a_wav, w1, w2)

	sys.exit("Script Finished")



	#sns_pair_plot(CSV_LST, "H2_fit.pickle")

	#plot_H2_hist(res_file="save_results.dat", z = redshift)
	#plot_H2_trace(res_file="save_results.dat", z = redshift)

	#bokeh_plt(wav_aa_pl, n_flux_pl, y_min, y_max, y_min2, y_max2, y_fit, redshift, ignore_lst, \
	#	a_name, a_wav, ai_name, ai_wav, aex_name, aex_wav, h2_name, h2_wav)






