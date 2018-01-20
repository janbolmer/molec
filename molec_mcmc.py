#! /usr/bin/python

"""
MCMC sampler for fitting X-shooter spectra (using PyMC) with HI, H2, CO
and other lines
=========================================================================
e.g.: molec_mcmc.py -f spectra/GRB120815Auvb.txt -m H2vib -w1 1566
		-w2 1611 -red 2.358 -it 40 -bi 10 -t GRB120815A -e SiII FeII
=========================================================================
-target    	target name
-file 		path to spectrum data file
-model  	Model to use: H2, H2s, H2all, H2vib or CO
-elements 	list of additional elements to be included, e.g.: -e HI FeII
-redshift 	redshift
-nrot 		number of H2 rotational levels to be fitted, e.g.: -nrot 0 7
			for only J1: -nrot 1 1, J0-J3: -nrot 0 3
-w1 		Wavelength start - restframe
-wl 		Wavelength end - restframe
-ignore 	list of intervals (restframe) to be ignored when fitting the
			data, e.g. -ignore 1090.0,1092.0 1140,1180
-res 		spectral resolution in AA for rebinning the H2vib model from
			Draine
-par 		Parameter file with Column Densities, b and redshift for each
			given element [.csv format]. Values can be fixed or given as
			a Uniform Distribution for PyMC (see the para.csv example)
			(first line: element,fixed,N_val,N_low,N_up,B_val,B_low,B_up,
			R_val,R_low,R_up)
-rl 		redshifts for additional H2 components, e.g.: -rl 2.03 2.01
-fb 		to fix the broadening parameter for H2, e.g.: -fb 3.0
-j 			rotaional level to use for H2s
-intv 		redshift of intervening systems, e.g. -intv 1.245 1.748
=========================================================================
Typical are: 	HI, FeII, MnII, NV, SiII, SII, CIV, OI, CII, NiII, SiIV,
				AlII, AlIII, CI, ZnII, CrII, MgII, MgI

and:			FeIIa, FeIIb, OIa, SiIIa, NiIIa
=========================================================================
CO lines:
1544.0 - 1545.5 v' = 0
1509.4 - 1510.3 v' = 1 	-w1 1509.6 -w2 1510.1
1477.0 - 1478.1 v' = 2
1447.0 - 1448.0 v' = 3
1418.6 - 1419.4 v' = 4
=========================================================================
"""

__author__ = "Jan Bolmer"
__copyright__ = "Copyright 2017"
__credits__ = "Thomas Kruehler" # https://github.com/Kruehlio/H2sim
__version__ = "0.1"
__maintainer__ = "Jan Bolmer"
__email__ = "jbolmer@eso.org"
__status__ = "Production"

import pymc, math, argparse, os, sys, time
import random
import numpy as np
import pandas as pd

import matplotlib as plt

from scipy.special import wofz
from scipy.interpolate import interp1d

sys.path.append('bin/')

from spec_functions import * # spec_functions.py
from syn_spec import * # syn_spec.py
from sns_plots import * # sns_plots.py

# physical constants
m_e = 9.1095e-28
e = 4.8032e-10
c = 2.998e10 # speed of light in m/s

#========================================================================
#========================================================================

def usage():
	'''
	Printing Documentation to terminal
	'''
	print __doc__

#========================================================================
#========================================================================

def model_csv_file(model, fixed_b):
	'''
	Creates a list of variable names for the given model
	(and checks if the model is either H2, H2vib or CO)
	The lists are used for plotting the results.
	'''

	if model == "H2" and fixed_b == None:
		CSV_LST = ['NTOTH2', 'TEMP', 'B', 'A_Z']
	elif model == "H2" and fixed_b != None:
		CSV_LST = ['NTOTH2', 'TEMP', 'A_Z']
	elif model == "H2s" and fixed_b == None:
		CSV_LST = ['NH2', 'B', 'A_Z']
	elif model == "H2s" and fixed_b != None:
		CSV_LST = ['NH2', 'A_Z']
	elif model == "H2all" and fixed_b == None:
		CSV_LST = ['B', 'A_Z']
	elif model == "H2all" and fixed_b != None:
		CSV_LST = ['A_Z']
	elif model == "H2vib":
		CSV_LST = ['MH2S', 'A_Z']
	elif model == "CO" and fixed_b == None:
		CSV_LST = ['NTOTCO', 'TEMP', 'B', 'A_Z', 'BG']
	elif model == "CO" and fixed_b != None:
		CSV_LST = ['NTOTCO', 'TEMP', 'A_Z']
	else:
		sys.exit("ERROR: Choose one of the following models: H2, H2vib, \
			or CO")

	return CSV_LST

#========================================================================
#========================================================================

def gauss(x, mu, sig):
	'''
	Normal distribution used to create prior probability distributions
	'''
	return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#========================================================================
#========================================================================


















def model_H2(wav_aa, n_flux, n_flux_err, redshift, res, line_lst,
	redshift_lst, ignore_lst, par_dic, CSV_LST, NROT, fixed_b):
	'''
	Defines the model for H2
	'''

	tau = 1 / np.array(n_flux_err)**2

#======================broadening parameter B============================
	# Checks if the b is fixed; if not the following prior is assumed:
	if fixed_b == None:

		@pymc.stochastic(dtype=float)
		def B(value=2.0, mu=2.0, sig=0.5, doc="B"):
			'''
			bla
			'''
			pp = 0.0
			if 0 <= value < 25:
				pp = gauss(value, mu, sig)
			else:
				#invalid values
				pp = -np.inf
			#print value, pp
			return pp

	if fixed_b != None:
		B = fixed_b

#=======================Redshift freedom A_Z=============================
	# Redshift is allowed to vary between:
	@pymc.stochastic(dtype=float)
	def A_Z(value=0.0, mu=0.0, sig=25.0, doc="A_Z"):
		'''
		bla
		'''
		pp = gauss(value, mu, sig)
		return pp

#===========================Temperature==================================
	@pymc.stochastic(dtype=float)
	def TEMP(value=100.0, mu=100.0, sig=280.0, doc="TEMP"):
		'''
		bla
		'''
		if 0 <= value < 1000:
			pp = gauss(value, mu, sig)
		else:
			pp = -np.inf
		return pp

#=========================H2 Column Density==============================
	# Uniform Prior for the total H2 column density
	NTOTH2 = pymc.Uniform('NTOTH2',lower=0.0,upper=21.0,doc='NTOTH2')

#======================Additional H2 Components==========================
	add_h2_dic = {} # dictionary to collect variables
	if len(redshift_lst) == 0:
		print "\n no addtional H2 component\n"

	if not len(redshift_lst) == 0:
		print "\n additonal H2 component(s) at", redshift_lst, "\n" 

		for add_comp in redshift_lst:

			H2_c = pymc.Uniform('H2'+add_comp,lower=0.0,upper=21.0,
				doc='H2'+add_comp)

			T_c = pymc.Uniform('T'+add_comp,lower=0.,upper=800,
				doc='T'+add_comp)

			B_c = pymc.Normal('B'+add_comp,mu=3.4,tau=1/(3.4**2),value=3.0,
				doc='B'+add_comp)# USE B AS ABOVE

			Z_c = float(add_comp) # ALLOW SOME VARIATION

			CSV_LST.extend(('H2'+add_comp,'T'+add_comp,'B'+add_comp))
			add_h2_dic[add_comp] = H2_c, T_c, B_c, Z_c

#==========================Absoprtion Lines==============================
	# Adding other absoprtion lines for major component:
	vars_dic = {} # dictionary to collect variables

	for elmt in line_lst:
		if not elmt in par_dic:
			if elmt == "HI":
				N_E = pymc.Uniform('N_'+elmt,lower=18.0,upper=23.0,
					value=21.8,doc='N_'+elmt)
				B_E = pymc.Uniform('B_'+elmt,lower=8.,upper=150.,
					value=8.,doc='B_'+elmt)
				A_Z_E = pymc.Uniform('A_Z_'+elmt,lower=-150.,upper=+150.,
					value=0.,doc='A_Z_'+elmt)
			else:
				N_E=pymc.Uniform('N_'+elmt,lower=0.,upper=20.0,
					value=15.0,doc='N_'+elmt)
				B_E=pymc.Uniform('B_'+elmt,lower=8.,upper=100.,
					value=15.,doc='B_'+elmt)
				A_Z_E=pymc.Uniform('A_Z_'+elmt,lower=-100.,upper=+100.,
					value=0.,doc='A_Z_'+elmt)

			CSV_LST.extend(('N_'+elmt,'B_'+elmt,'A_Z_'+elmt))

		# Reading data from parameter file:
		else:
			# Parameters free
			if par_dic[elmt][0] == 0:
				if elmt != "HI":
					N_E = pymc.Uniform('N_' + elmt,lower=par_dic[elmt][2],
						upper=par_dic[elmt][3],doc='N_'+elmt)

				if elmt == "HI":

					N_E = pymc.Normal('N_'+elmt,mu=par_dic[elmt][1],
						tau=1/((par_dic[elmt][1]-par_dic[elmt][2])**2),
						doc='N_'+elmt)

				B_E = pymc.Uniform('B_'+elmt,lower=par_dic[elmt][5],
					upper=par_dic[elmt][6],doc='B_'+elmt)
				A_Z_E = pymc.Uniform('A_Z_'+elmt,lower=par_dic[elmt][8],
					upper=par_dic[elmt][9],doc='A_Z_'+elmt)

				CSV_LST.extend(('N_'+elmt,'B_'+elmt,'A_Z_'+elmt))

			# Parameters fixed
			if par_dic[elmt][0] == 1:
				N_E = par_dic[elmt][1]
				B_E = par_dic[elmt][4]
				A_Z_E = par_dic[elmt][7]

		vars_dic[elmt] = N_E, B_E, A_Z_E


#======================Addiotnal Absoprtion Lines==========================
 	# Adding other absoprtion lines for other components
	vars_dic_add = {}

	if not len(redshift_lst) == 0:
		print "additonal line component(s) at", redshift_lst, "\n"
		for elmt in line_lst:
			for add_comp in redshift_lst:
				if not elmt+add_comp in par_dic:

					if not elmt.startswith("HI"):
						print elmt+add_comp
						N_EC=pymc.Uniform('N_'+elmt+add_comp,lower=0.,upper=20.0,
							value=16.0,doc='N_'+elmt+add_comp)
						B_EC=pymc.Uniform('B_'+elmt+add_comp,lower=0.,upper=80.,
							value=8.,doc='B_'+elmt+add_comp)
						Z_EC=float(add_comp)
	
						CSV_LST.extend(('N_'+elmt+add_comp,'B_'+elmt+add_comp,))		
						vars_dic_add[elmt+add_comp] = N_EC, B_EC, Z_EC, elmt

				if elmt+add_comp in par_dic:
					if par_dic[elmt][0] == 1:
						N_EC = par_dic[elmt][1]
						B_EC = par_dic[elmt][4]
						Z_EC = float(add_comp)

						vars_dic_add[elmt+add_comp] = N_EC, B_EC, Z_EC, elmt		

	# Defining the model:
	@pymc.deterministic(plot=False) 
	def H2(wav_aa=wav_aa,A_REDSHIFT=A_Z,NTOTH2=NTOTH2,TEMP=TEMP,BROAD=B,
		redshift=redshift,vars_dic=vars_dic,vars_dic_add=vars_dic_add,
		add_h2_dic=add_h2_dic,res=res):

		A_REDSHIFT = float(A_REDSHIFT)/100000.0
		norm_spec = np.ones(len(wav_aa)) # add Background multiplier
		synspec = SynSpec(wav_aa,redshift,res,ignore_lst=ignore_lst)

		# add H2
		h2spec = synspec.add_H2(norm_spec,broad=BROAD,NTOTH2=NTOTH2,
			TEMP=TEMP,A_REDSHIFT=A_REDSHIFT,NROT=NROT)

		# add H2 for components
		for key in add_h2_dic:

			A_REDSHIFT = add_h2_dic[key][3]-redshift
			h2spec = synspec.add_H2(h2spec,broad=add_h2_dic[key][2],
				NTOTH2=add_h2_dic[key][0],TEMP=add_h2_dic[key][1],
				A_REDSHIFT=A_REDSHIFT,NROT=NROT)

		# add other absorption lines
		for key in vars_dic:
			A_Z_E = vars_dic[key][2]/100000.00
			h2spec = synspec.add_ion(h2spec, key,
				broad=vars_dic[key][1], Natom=vars_dic[key][0],
				A_REDSHIFT=A_Z_E)

		# add other absorption lines for other components
		for key in vars_dic_add:
			A_REDSHIFT = vars_dic_add[key][2]-redshift
			h2spec = synspec.add_ion(h2spec, vars_dic_add[key][3],
				broad=vars_dic_add[key][1], Natom=vars_dic_add[key][0],
				A_REDSHIFT=A_REDSHIFT)

		# convolve with spectral resolution
		#h2spec = synspec.convolve_spec(h2spec)

		return h2spec

	# Data:
	y_val = pymc.Normal('y_val',mu=H2,tau=tau,value=n_flux,observed=True)

	return locals()

















#========================================================================
#========================================================================

def model_single_H2(wav_aa, n_flux, n_flux_err, redshift, res, line_lst,
	redshift_lst, ignore_lst, par_dic, CSV_LST, J, fixed_b):
	'''
	Defines the model for H2
	'''
	tau = 1 / np.array(n_flux_err)**2

#======================broadening parameter B============================
	# Checks if the b is fixed; if not the following prior is assumed:
	if fixed_b == None:

		@pymc.stochastic(dtype=float)
		def B(value=2.0, mu=2.0, sig=0.5, doc="B"):
			'''
			bla
			'''
			pp = 0.0
			if 0 <= value < 25:
				pp = gauss(value, mu, sig)
			else:
				#invalid values
				pp = -np.inf
			#print value, pp
			return pp

	if fixed_b != None:
		B = fixed_b

#=======================Redshift freedom A_Z=============================
	# Redshift is allowed to vary between:
	@pymc.stochastic(dtype=float)
	def A_Z(value=0.0, mu=0.0, sig=5.0, doc="A_Z"):
		'''
		bla
		'''
		pp = gauss(value, mu, sig)
		return pp

#=========================H2 Column Density==============================
	# Uniform Prior for the total H2 column density
	NH2 = pymc.Uniform('NH2',lower=0.0,upper=21.0,doc='NH2')

#==========================Absoprtion Lines==============================
	# Adding other absoprtion lines for major component:
	vars_dic = {} # dictionary to collect variables

	for elmt in line_lst:
		if not elmt in par_dic:
			if elmt == "HI":
				N_E = pymc.Uniform('N_'+elmt,lower=18.0,upper=23.0,
					value=21.8,doc='N_'+elmt)
				B_E = pymc.Uniform('B_'+elmt,lower=0.,upper=150.,
					value=8.,doc='B_'+elmt)
				A_Z_E = pymc.Uniform('A_Z_'+elmt,lower=-150.,upper=+150.,
					value=0.,doc='A_Z_'+elmt)
			else:
				N_E=pymc.Uniform('N_'+elmt,lower=0.,upper=20.0,
					value=16.0,doc='N_'+elmt)
				B_E=pymc.Uniform('B_'+elmt,lower=0.,upper=30.,
					value=8.,doc='B_'+elmt)
				A_Z_E=pymc.Uniform('A_Z_'+elmt,lower=-100.,upper=+100.,
					value=0.,doc='A_Z_'+elmt)

			CSV_LST.extend(('N_'+elmt,'B_'+elmt,'A_Z_'+elmt))

		# Reading data from parameter file:
		else:
			# Parameters free
			if par_dic[elmt][0] == 0:
				if elmt != "HI":
					N_E = pymc.Uniform('N_' + elmt,lower=par_dic[elmt][2],
						upper=par_dic[elmt][3],doc='N_'+elmt)

				if elmt == "HI":

					N_E = pymc.Normal('N_'+elmt,mu=par_dic[elmt][1],
						tau=1/((par_dic[elmt][1]-par_dic[elmt][2])**6),
						doc='N_'+elmt)

				B_E = pymc.Uniform('B_'+elmt,lower=par_dic[elmt][5],
					upper=par_dic[elmt][6],doc='B_'+elmt)
				A_Z_E = pymc.Uniform('A_Z_'+elmt,lower=par_dic[elmt][8],
					upper=par_dic[elmt][9],doc='A_Z_'+elmt)

				CSV_LST.extend(('N_'+elmt,'B_'+elmt,'A_Z_'+elmt))

			# Parameters fixed
			if par_dic[elmt][0] == 1:
				N_E = par_dic[elmt][1]
				B_E = par_dic[elmt][4]
				A_Z_E = par_dic[elmt][7]

		vars_dic[elmt] = N_E, B_E, A_Z_E	

	# Defining the model:
	@pymc.deterministic(plot=False) 
	def H2s(wav_aa=wav_aa,A_REDSHIFT=A_Z,NH2=NH2,BROAD=B,
		redshift=redshift,vars_dic=vars_dic,res=res):

		A_REDSHIFT = float(A_REDSHIFT)/100000.0
		norm_spec = np.ones(len(wav_aa)) # add Background multiplier
		synspec = SynSpec(wav_aa,redshift,res)

		# add H2
		h2spec = synspec.add_single_H2(norm_spec,broad=BROAD,NH2=NH2,
			A_REDSHIFT=A_REDSHIFT,J=J)

		# add other absorption lines
		for key in vars_dic:
			A_Z_E = vars_dic[key][2]/100000.00
			h2spec = synspec.add_ion(h2spec, key,
				broad=vars_dic[key][1], Natom=vars_dic[key][0],
				A_REDSHIFT=A_Z_E)

		return h2spec

	# Data:
	y_val = pymc.Normal('y_val',mu=H2s,tau=tau,value=n_flux,observed=True)

	return locals()
















#========================================================================
#========================================================================

def model_all_H2(wav_aa, n_flux, n_flux_err, redshift, res, line_lst,
	redshift_lst, ignore_lst, par_dic, CSV_LST, NROT, fixed_b):
	'''
	Defines the model for H2
	'''
	tau = 1 / np.array(n_flux_err)**2

#======================broadening parameter B============================
	# Checks if the b is fixed; if not the following prior is assumed:
	if fixed_b == None:
		B = pymc.Exponential('B',beta=0.02,value=2,doc='B')

	if fixed_b != None:
		B = fixed_b

#=======================Redshift freedom A_Z=============================
	# Redshift is allowed to vary between:
	@pymc.stochastic(dtype=float)
	def A_Z(value=0.0, mu=0.0, sig=0.01, doc="A_Z"):

		pp = gauss(value, mu, sig)
		return pp

#=========================H2 Column Densities==============================
	# Uniform Prior for the individual J - H2 column density

	nhj_dic = {}

	for J in NROT:
		#NH2J = pymc.Uniform('NH2J'+str(J),lower=0.0,upper=21.0,doc='NH2J'+str(J))
		NH2J = pymc.Exponential('NH2J'+str(J),beta=0.1,value=2,doc='NH2J'+str(J))

		nhj_dic[J] = NH2J
		CSV_LST.append('NH2J'+str(J))

	print CSV_LST
#==========================Absoprtion Lines==============================
	# Adding other absoprtion lines for major component:
	vars_dic = {} # dictionary to collect variables

	for elmt in line_lst:
		if not elmt in par_dic:
			if elmt == "HI":
				N_E = pymc.Uniform('N_'+elmt,lower=18.0,upper=23.0,
					value=21.8,doc='N_'+elmt)
				B_E = pymc.Uniform('B_'+elmt,lower=0.,upper=150.,
					value=8.,doc='B_'+elmt)
				A_Z_E = pymc.Uniform('A_Z_'+elmt,lower=-150.,upper=+150.,
					value=0.,doc='A_Z_'+elmt)
			else:
				N_E = pymc.Exponential('N_'+elmt,beta=0.1,value=2,doc='N_'+elmt)
				B_E = pymc.Exponential('B_'+elmt,beta=0.02,value=5,doc='B_'+elmt)
				A_Z_E = pymc.Uniform('A_Z_'+elmt,lower=-100.,upper=+100.,
					value=0.,doc='A_Z_'+elmt)

			CSV_LST.extend(('N_'+elmt,'B_'+elmt,'A_Z_'+elmt))

		# Reading data from parameter file:
		else:
			# Parameters free
			if par_dic[elmt][0] == 0:
				if elmt != "HI":
					N_E = pymc.Uniform('N_' + elmt,lower=par_dic[elmt][2],
						upper=par_dic[elmt][3],doc='N_'+elmt)

				if elmt == "HI":

					N_E = pymc.Normal('N_'+elmt,mu=par_dic[elmt][1],
						tau=1/((par_dic[elmt][1]-par_dic[elmt][2])**6),
						doc='N_'+elmt)

				B_E = pymc.Uniform('B_'+elmt,lower=par_dic[elmt][5],
					upper=par_dic[elmt][6],doc='B_'+elmt)
				A_Z_E = pymc.Uniform('A_Z_'+elmt,lower=par_dic[elmt][8],
					upper=par_dic[elmt][9],doc='A_Z_'+elmt)

				CSV_LST.extend(('N_'+elmt,'B_'+elmt,'A_Z_'+elmt))

			# Parameters fixed
			if par_dic[elmt][0] == 1:
				N_E = par_dic[elmt][1]
				B_E = par_dic[elmt][4]
				A_Z_E = par_dic[elmt][7]

		vars_dic[elmt] = N_E, B_E, A_Z_E	



#======================Additional H2 Components==========================
	add_h2_dic = {} # dictionary to collect variables
	if len(redshift_lst) == 0:
		print "\n no addtional H2 component\n"

	if not len(redshift_lst) == 0:
		print "\n additonal H2 component(s) at", redshift_lst, "\n"

#		for add_comp in redshift_lst:
#
#			B_c = pymc.Exponential('B'+add_comp,beta=0.02,value=2,doc='B'+add_comp)
#			Z_c = float(add_comp)
#
#			for J in NROT:
#
#			CSV_LST.extend(('H2'+add_comp,'T'+add_comp,'B'+add_comp))
#			add_h2_dic[add_comp] = H2_c, T_c, B_c, Z_c

#======================Additional H2 Components==========================




	# Defining the model:
	@pymc.deterministic(plot=False) 
	def H2all(wav_aa=wav_aa,A_REDSHIFT=A_Z,BROAD=B,redshift=redshift,
		vars_dic=vars_dic,nhj_dic=nhj_dic,res=res,NROT=NROT,
		ignore_lst=ignore_lst):

		A_REDSHIFT = float(A_REDSHIFT)/100000.0
		norm_spec = np.ones(len(wav_aa)) # add Background multiplier
		synspec = SynSpec(wav_aa,redshift,res,ignore_lst=ignore_lst)

		# add H2
		for J in NROT:
			h2spec = synspec.add_single_H2(norm_spec,broad=BROAD,NH2=nhj_dic[J],
				A_REDSHIFT=A_REDSHIFT,J=J)

		# add other absorption lines
		for key in vars_dic:
			A_Z_E = vars_dic[key][2]/100000.00
			h2spec = synspec.add_ion(h2spec, key,
				broad=vars_dic[key][1], Natom=vars_dic[key][0],
				A_REDSHIFT=A_Z_E)
		return h2spec

	# Data:
	y_val = pymc.Normal('y_val',mu=H2all,tau=tau,value=n_flux,observed=True)

	return locals()


#========================================================================
#========================================================================

def model_H2vib(wav_aa, n_flux, n_flux_err, redshift, res, line_lst, \
	par_dic, CSV_LST):
	'''
	Defines the model for H2vib
	'''

	tau = 1 / np.array(n_flux_err)**2
	MH2S =pymc.Uniform('MH2S',value=0.15,lower=0.,upper=0.40,doc='MH2S')
	A_Z = pymc.Uniform('A_Z',value=0.,lower=-100.,upper=+100.,doc='A_Z')

	vars_dic = {}

	for elmt in line_lst:

		N_E = pymc.Uniform('N_'+elmt,lower=0.,upper=20.,
			value=16., doc='N_' + elmt)
		B_E = pymc.Uniform('B_'+elmt,lower=0.,upper=30.,
			value=8., doc='B_' + elmt)
		A_Z_E = pymc.Uniform('A_Z_'+elmt,lower=-150.,upper=+150.,
			value=0., doc='A_Z_'+elmt)

		CSV_LST.extend(('N_'+elmt,'B_'+elmt,'A_Z_'+elmt))
		vars_dic[elmt] = N_E, B_E, A_Z_E

	@pymc.deterministic(plot=False)
	def H2vib(wav_aa=wav_aa, redshift=redshift, A_REDSHIFT=A_Z,
		MH2S=MH2S, vars_dic=vars_dic, res=res):

		A_REDSHIFT = float(A_REDSHIFT)/100000.0
		norm_spec = np.ones(len(wav_aa))
		synspec = SynSpec(wav_aa,redshift,res)

		res_aa = np.median(np.diff(wav_aa))

		# add H2*
		h2vib_spec = synspec.add_vibH2(norm_spec,h2swl,modspec,
			tauspec,RES=res_aa,MH2S=MH2S,A_REDSHIFT=A_REDSHIFT)

		# add other lines
		if len(vars_dic) >= 1:
			for key in vars_dic:
				A_Z_E = vars_dic[key][2]/100000.0
				h2vib_spec = synspec.add_ion(h2vib_spec, key, 
					broad=vars_dic[key][1], Natom=vars_dic[key][0],
					A_REDSHIFT=A_Z_E)

		return h2vib_spec

	y_val = pymc.Normal('y_val',mu=H2vib,tau=tau,value=n_flux,
		observed=True)

	return locals()














#========================================================================
#========================================================================

def model_CO(wav_aa, n_flux, n_flux_err, redshift, res, line_lst,
	par_dic, CSV_LST, NROT, fixed_b):
	'''
	Defines the model for CO
	'''

	tau = 1 / np.array(n_flux_err)**2

#=========================== Background ================================
	@pymc.stochastic(dtype=float)
	def BG(value=1.0, mu=1.0, sig=0.05, doc="BG"):
		pp = gauss(value, mu, sig)
		return pp

#======================broadening parameter B============================
	# Checks if the b is fixed; if not the following prior is assumed:
	if fixed_b == None:

		@pymc.stochastic(dtype=float)
		def B(value=2.0, mu=2.0, sig=2.0, doc="B"):

			if 0 <= value < 25:
				pp = gauss(value, mu, sig)
			else:
				#invalid values
				pp = -np.inf
			#print value, pp
			return pp

	if fixed_b != None:
		B = fixed_b

#=======================Redshift freedom A_Z=============================
	# Redshift is allowed to vary between:
	@pymc.stochastic(dtype=float)
	def A_Z(value=0.0, mu=0.0, sig=5.0, doc="A_Z"):
		if 0 <= value < 25:
			pp = gauss(value, mu, sig)
		else:
			pp = -np.inf
		return pp

#===========================Temperature==================================
	@pymc.stochastic(dtype=float)
	def TEMP(value=20.0, mu=20.0, sig=180.0, doc="TEMP"):

		if 0 <= value < 1000.0:
			pp = gauss(value, mu, sig)
		else:
			pp = -np.inf
		return pp

#=========================CO Column Density==============================
	# Uniform Prior for the total H2 column density
	NTOTCO = pymc.Uniform('NTOTCO',lower=0.0,upper=21.0,doc='NTOTCO')

	# Defining the model:
	@pymc.deterministic(plot=False) 
	def CO(wav_aa=wav_aa,BG=BG, A_REDSHIFT=A_Z,NTOTCO=NTOTCO,TEMP=TEMP,BROAD=B,
		redshift=redshift,res=res):

		A_REDSHIFT = float(A_REDSHIFT)/100000.0
		norm_spec = np.ones(len(wav_aa))*BG # add Background multiplier
		synspec = SynSpec(wav_aa,redshift,res)

		# add CO
		co_spec = synspec.add_CO(norm_spec,broad=BROAD,NTOTCO=NTOTCO,
			TEMP=TEMP,A_REDSHIFT=A_REDSHIFT,NROT=NROT)

		return co_spec

	# Data:
	y_val = pymc.Normal('y_val',mu=CO,tau=tau,value=n_flux,observed=True)

	return locals()

#========================================================================
#========================================================================


def makeMCMC(wav_aa, n_flux, n_flux_err, trials, burn_in, n_thin,
	fixed_b, model_used, line_lst, redshift_lst, ignore_lst, target,
	par_dic, redshift, CSV_LST, NROT, J, res):
	'''
	Performing the MCMC
	'''

	if model_used == "H2":

		MDL = pymc.MCMC(model_H2(wav_aa, n_flux, n_flux_err, redshift,
			res, line_lst, redshift_lst, ignore_lst, par_dic, CSV_LST,
			NROT, fixed_b), db='pickle', dbname='H2_fit.pickle')
		
		MDL.db
		MDL.sample(trials, burn_in, n_thin)
		MDL.db.close()

		MDL.write_csv(target + "_H2_results.csv", variables=CSV_LST)

		y_min 	= MDL.stats()[model_used]['quantiles'][2.5]
		y_max 	= MDL.stats()[model_used]['quantiles'][97.5]
		y_min2 	= MDL.stats()[model_used]['quantiles'][25]
		y_max2 	= MDL.stats()[model_used]['quantiles'][75]
		y_fit 	= MDL.stats()[model_used]['mean']

		return y_min, y_max, y_min2, y_max2, y_fit

	if model_used == "H2s":

		MDL = pymc.MCMC(model_single_H2(wav_aa, n_flux, n_flux_err, redshift,
			res, line_lst, redshift_lst, par_dic, CSV_LST, J, fixed_b),
			db='pickle', dbname='H2s_fit.pickle')
		
		MDL.db
		MDL.sample(trials, burn_in, n_thin)
		MDL.db.close()

		MDL.write_csv(target + "_H2s_results.csv", variables=CSV_LST)

		y_min 	= MDL.stats()[model_used]['quantiles'][2.5]
		y_max 	= MDL.stats()[model_used]['quantiles'][97.5]
		y_min2 	= MDL.stats()[model_used]['quantiles'][25]
		y_max2 	= MDL.stats()[model_used]['quantiles'][75]
		y_fit 	= MDL.stats()[model_used]['mean']

		return y_min, y_max, y_min2, y_max2, y_fit


	if model_used == "H2all":

		MDL = pymc.MCMC(model_all_H2(wav_aa, n_flux, n_flux_err, redshift,
			res, line_lst, redshift_lst, ignore_lst, par_dic, CSV_LST,
			NROT, fixed_b), db='pickle', dbname='H2all_fit.pickle')
		
		MDL.db
		MDL.sample(trials, burn_in, n_thin)
		MDL.db.close()

		MDL.write_csv(target + "_H2all_results.csv", variables=CSV_LST)

		y_min 	= MDL.stats()[model_used]['quantiles'][2.5]
		y_max 	= MDL.stats()[model_used]['quantiles'][97.5]
		y_min2 	= MDL.stats()[model_used]['quantiles'][25]
		y_max2 	= MDL.stats()[model_used]['quantiles'][75]
		y_fit 	= MDL.stats()[model_used]['mean']

		return y_min, y_max, y_min2, y_max2, y_fit

	if model_used == "H2vib":

		MDL = pymc.MCMC(model_H2vib(wav_aa, n_flux, n_flux_err, redshift,
			res, line_lst, par_dic, CSV_LST),db='pickle',
			dbname='H2vib_fit.pickle')

		MDL.db
		MDL.sample(trials, burn_in, n_thin)
		MDL.db.close()

		MDL.write_csv(target+"_H2vib_results.csv",variables=CSV_LST)

		y_min 	= MDL.stats()[model_used]['quantiles'][2.5]
		y_max 	= MDL.stats()[model_used]['quantiles'][97.5]
		y_min2 	= MDL.stats()[model_used]['quantiles'][25]
		y_max2 	= MDL.stats()[model_used]['quantiles'][75]
		y_fit 	= MDL.stats()[model_used]['mean']

		return y_min, y_max, y_min2, y_max2, y_fit


	if model_used == "CO":

		MDL = pymc.MCMC(model_CO(wav_aa, n_flux, n_flux_err, redshift,
			res,line_lst, par_dic, CSV_LST, NROT, fixed_b),db='pickle',
			dbname='CO_fit.pickle')

		MDL.db
		MDL.sample(trials, burn_in, n_thin)
		MDL.db.close()

		MDL.write_csv(target+"_CO_results.csv",variables=CSV_LST)

		y_min 	= MDL.stats()[model_used]['quantiles'][2.5]
		y_max 	= MDL.stats()[model_used]['quantiles'][97.5]
		y_min2 	= MDL.stats()[model_used]['quantiles'][25]
		y_max2 	= MDL.stats()[model_used]['quantiles'][75]
		y_fit 	= MDL.stats()[model_used]['mean']

		return y_min, y_max, y_min2, y_max2, y_fit

#========================================================================
#========================================================================

def main():

	writecmd("H2_cmd_hist.dat")

	start = time.time()
	print "\n Parsing Arguments \n"

	parser = argparse.ArgumentParser(usage=__doc__)
	parser.add_argument('-t','--target',dest="target",default="GRB",
						type=str)
	parser.add_argument('-f','--file',dest="file",
		default="spectra/GRB120815Auvb.txt", type=str)
	parser.add_argument('-red','--redshift',dest="redshift",default=2.358,
		type=float)
	parser.add_argument('-m','--model',dest="model",default="H2",type=str)
	parser.add_argument('-nrot','--nrot',dest="nrot",nargs=2,
		default=[0, 1, 2, 3, 4, 5, 6, 7], type=int)
	parser.add_argument('-e','--elements',dest="elements", nargs='+',
						default=["FeII", "SiII"])
	parser.add_argument('-w1','--w1',dest="w1",default=980.,type=float)
	parser.add_argument('-w2','--w2',dest="w2",default=1120.,type=float)
	parser.add_argument('-ign','--ignore',dest="ignore", nargs='+',
						default=[])
	parser.add_argument('-res','--resolution',dest="resolution",
						default=5000,type=float)
	parser.add_argument('-it','--iterations',dest="iterations",
						default=1000, type=int)
	parser.add_argument('-bi','--burn_in',dest="burn_in",
						default=100, type=int)
	parser.add_argument('-sp','--save_pickle',dest="save_pickle",
						default=True, type=bool)
	parser.add_argument('-par','--par',dest="par",default=None,type=str)
	parser.add_argument('-rl','--red_lst',dest="red_lst",nargs='+',default=[])
	parser.add_argument('-fb','--fixed_b',dest="fixed_b",default=None,type=float)
	parser.add_argument('-j','--j',dest="j",default=None,type=int)
	parser.add_argument('-intv','--intv',dest="intv",nargs='+',default=[])

	args = parser.parse_args()


	target = args.target
	spec_file = args.file
	model = args.model
	elements = args.elements
	redshift = args.redshift
	nrot = args.nrot
	w1 = args.w1
	w2 = args.w2
	ignore = args.ignore
	res = args.resolution
	iterations = args.iterations
	burn_in = args.burn_in
	save_pickle = args.save_pickle
	para_file = args.par
	red_lst = args.red_lst
	fixed_b = args.fixed_b
	J = args.j
	intv = args.intv

	ignore_lst = []
	for itrvl in ignore:
		tmp_lst = []
		s = itrvl.split(",")
		tmp_lst.extend((float(s[0]), float(s[1])))
		ignore_lst.append(tmp_lst)

	redshift_lst = []
	for rds in red_lst:
		redshift_lst.append(rds)

	intv_lst = []
	for i in intv:
		intv_lst.append(float(i))

	if burn_in >= iterations:
		sys.exit("ERROR: Burn-In cannot be bigger than Iterations")

	par_dic = {}

	NROT = []

	if not len(nrot) >= 8:
		for i in np.arange(nrot[0], nrot[1]+1, 1):
			NROT.append(i)
	else:
		NROT = nrot

	CSV_LST = model_csv_file(model, fixed_b)

	time.sleep(1.0)
	print "\n Fitting", target, "at redshift", redshift, \
		"with a spectral resolution of R =", res
	time.sleep(1.0)

	if fixed_b != None:
		print "\n b fixed to", fixed_b

	if para_file != None:
		par_dic = get_paras(para_file)
		print "\n Using parameters given in:", para_file

	time.sleep(1.0)

	if model == "H2s" and J == None:
		sys.exit("Error: Please specify the rotational level, e.g. -j 0")

	if model == "H2s" and J != None:
		print "\n fitting H2, J =", J
		NROT = [J]


	print "\n Starting MCMC " + '(pymc version:',pymc.__version__,")"
	print "\n This might take a while ..."

	a_name, a_wav, ai_name, ai_wav, aex_name, \
	aex_wav, h2_name, h2_wav = get_lines(redshift)

	try:
		wav_aa_pl, n_flux_pl, n_flux_err_pl, flux_pl, flux_err_pl, \
		grb_name, resolution, psf_fwhm = get_data(spec_file, redshift,
			wl_range=True, wl1=w1, wl2=w2)
	except:
		sys.exit("ERROR: File not found (-f)")

	wav_aa, n_flux, n_flux_err = get_data_ign(spec_file, redshift,
		ignore_lst, wl1=w1, wl2=w2)

	y_min, y_max, y_min2, y_max2, y_fit = makeMCMC(wav_aa, n_flux,
		n_flux_err, iterations, burn_in, 1, fixed_b, model_used=model,
		line_lst=elements, redshift_lst=redshift_lst, ignore_lst=ignore_lst,
		target=target, par_dic=par_dic, redshift=redshift, CSV_LST=CSV_LST,
		NROT=NROT, J=J, res=res)

	print "\n MCMC finished \n"
	time.sleep(1.0)
	print "Model used:",model,"with J =",NROT,"and",elements
	print "using",iterations,"iterations","and a burn-in of",burn_in
	print "Wavelenth range:",w1,"to",w2,"/ ignored are",ignore_lst
	time.sleep(1.0)
	print "\n Plotting (Markers are plotted for the input redshift!)"

	if model == "H2":

		plot_spec(wav_aa_pl,n_flux_pl,n_flux_err_pl,y_min,y_max,y_min2,y_max2,y_fit,
			redshift,ignore_lst,a_name,a_wav,ai_name,ai_wav,aex_name,
			aex_wav,h2_name,h2_wav,target=target,fb=fixed_b,intv_lst=intv_lst)

		if fixed_b == None:
			sns_pair_plot(target, var_list=CSV_LST,file="H2_fit.pickle",
				redshift=redshift)
		else:
			sns_pair_plot_fb(target,var_list=CSV_LST,file="H2_fit.pickle",
				redshift=redshift,fb=fixed_b)

		if "NV" in elements:
			plot_H2fit_elmt(target,element="NV",file="H2_fit.pickle")

	if model == "H2s":

		plot_spec(wav_aa_pl,n_flux_pl,n_flux_err_pl,y_min,y_max,y_min2,y_max2,y_fit,
			redshift,ignore_lst,a_name,a_wav,ai_name,ai_wav,aex_name,
			aex_wav,h2_name,h2_wav,target=target,fb=fixed_b,intv_lst=intv_lst)

		if fixed_b == None:
			sns_H2s_pair_plot(target, var_list=CSV_LST,file="H2s_fit.pickle",
				redshift=redshift)
		#else:
		#	sns_pair_plot_fb(target,var_list=CSV_LST,file="H2_fit.pickle",
		#		redshift=redshift,fb=fixed_b)

	if model == "H2all":

		plot_spec(wav_aa_pl,n_flux_pl,n_flux_err_pl,y_min,y_max,y_min2,y_max2,y_fit,
			redshift,ignore_lst,a_name,a_wav,ai_name,ai_wav,aex_name,
			aex_wav,h2_name,h2_wav,target=target,fb=fixed_b,intv_lst=intv_lst)

		if fixed_b == None:
			sns_H2all_pair_plot(target, var_list=CSV_LST,file="H2all_fit.pickle",
				redshift=redshift)
		#else:
		#	sns_pair_plot_fb(target,var_list=CSV_LST,file="H2_fit.pickle",
		#		redshift=redshift,fb=fixed_b)

	if model == "H2vib":

		plot_H2vib(wav_aa_pl,n_flux_pl,n_flux_err_pl,y_min,y_max,y_min2,y_max2,y_fit,
			a_name,a_wav,aex_name, aex_wav,target=target,intv_lst=intv_lst)

		sns_H2vib_plot(target,var_list=CSV_LST,file="H2vib_fit.pickle",
			redshift=redshift)

	if model == "CO":

		plot_CO(wav_aa_pl,n_flux_pl,n_flux_err_pl,y_min,y_max,y_min2,y_max2,y_fit,
			redshift=redshift,target=target,fb=fixed_b,intv_lst=intv_lst)

		if fixed_b == None:
			sns_pair_plot_CO(target,var_list=CSV_LST,file="CO_fit.pickle",
				redshift=redshift)
		else:
			sns_pair_plot_CO_fb(target,var_list=CSV_LST,file="CO_fit.pickle",
				redshift=redshift,fb=fixed_b)	

	if save_pickle != False:
		os.system("rm -r *.pickle")
		print "\n Pickle Files Deleted"
	if save_pickle != True:
		print "\n Pickle Files Saved"

	os.system("mv *.pdf plots")
	print "\n Plots Moved to plots directory"

	os.system("mv *ts.csv results")
	print "\n Result .csv files moved to results directory"

 	dur = str(round((time.time() - start)/60, 1))
	sys.exit("\n Script finished after " + dur + " minutes")

#========================================================================
#========================================================================

if __name__ == "__main__":
	
	main()


