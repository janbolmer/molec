import math
import numpy as np
import pandas as pd

import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *

import seaborn as sns

def sns_pair_plot(target, var_list, file, redshift):

	sns.set_style("white")
	sns.set_style("ticks")
	sns.set_context("talk")

	data = pd.read_pickle(file)
	
	plt.figure()

	data_f = {}

	for var in var_list:
		if var in ['N_HI', 'NTOTH2', 'TEMP', 'B']:
			data_f[var] = data[var][0]
		if var == 'A_Z':
			data_f[var] = data[var][0]/100000.0 + redshift

	df = pd.DataFrame(data_f)

	g = sns.PairGrid(df) #, diag_kws=dict(color="blue", shade=True))
	g.map_upper(sns.kdeplot, cmap="bone_r",n_levels=10,shade=True,
		shade_lowest=False)
	g.map_lower(sns.kdeplot, cmap="bone_r",n_levels=10,shade=True,
		shade_lowest=False)
	g.map_diag(plt.hist) #, lw=2);
	
	#g.axes[0,0].set_ylim(redshift-0.0002,redshift+0.00005)
	g.axes[0,0].set_ylabel(r"$z$")
	g.axes[1,0].set_ylabel(r"$b$")
	g.axes[2,0].set_ylabel(r"$N(H_2)$")
	g.axes[3,0].set_ylabel(r"$N(HI)$")
	g.axes[4,0].set_ylabel(r"$T$")

	#g.axes[3,0].set_xlim(redshift-0.0002,redshift+0.00005)
	g.axes[4,0].set_xlabel(r"$z$")
	g.axes[4,1].set_xlabel(r"$b$")
	g.axes[4,2].set_xlabel(r"$N(H_2)$")
	g.axes[4,3].set_xlabel(r"$N(HI)$")
	g.axes[4,4].set_xlabel(r"$T$")

	g.savefig(target + "H2_pairplot.pdf")


def sns_pair_plot_fb(target, var_list, file, redshift, fb):
	'''
	pair plot for fixed broadening parameter
	'''

	sns.set_style("white")
	sns.set_style("ticks")
	sns.set_context("talk")

	data = pd.read_pickle(file)
	
	plt.figure()

	data_f = {}

	for var in var_list:
		if var in ['N_HI', 'NTOTH2', 'TEMP']:
			data_f[var] = data[var][0]
		if var == 'A_Z':
			data_f[var] = data[var][0]/100000.0 + redshift

	df = pd.DataFrame(data_f)

	g = sns.PairGrid(df) #, diag_kws=dict(color="blue", shade=True))
	g.map_upper(sns.kdeplot, cmap="bone_r",n_levels=10,shade=True,
		shade_lowest=False)
	g.map_lower(sns.kdeplot, cmap="bone_r",n_levels=10,shade=True,
		shade_lowest=False)
	g.map_diag(plt.hist) #, lw=2);
	
	#g.axes[0,0].set_ylim(redshift-0.0002,redshift+0.00005)
	g.axes[0,0].set_ylabel(r"$z$")
	g.axes[1,0].set_ylabel(r"$N(H_2)$")
	g.axes[2,0].set_ylabel(r"$N(HI)$")
	g.axes[3,0].set_ylabel(r"$T$")

	#g.axes[3,0].set_xlim(redshift-0.0002,redshift+0.00005)
	g.axes[3,0].set_xlabel(r"$z$")
	g.axes[3,1].set_xlabel(r"$N(H_2)$")
	g.axes[3,2].set_xlabel(r"$N(HI)$")
	g.axes[3,3].set_xlabel(r"$T$")

	g.savefig(target+"_b_"+str(fb)+"_H2_pairplot.pdf")


def sns_velo_pair_plot(target, file, nvoigts):

	sns.set_style("white")
	sns.set_style("ticks")
	sns.set_context("talk")

	for i in np.arange(1, nvoigts + 1, 1):

		data = pd.read_pickle(file)
		plt.figure()

		data_f = {}

		for key in data:
			if key.endswith(str(i)):
				data_f[key] = data[key][0]
		
		df = pd.DataFrame(data_f)
		#g = sns.pairplot(df, diag_kind="kde")

		g = sns.PairGrid(df) #, diag_kws=dict(color="blue", shade=True))
		g.map_upper(sns.kdeplot, cmap="bone_r",n_levels=10,shade=True,
			shade_lowest=False)
		g.map_lower(sns.kdeplot, cmap="bone_r",n_levels=10,shade=True,
			shade_lowest=False)
		g.map_diag(sns.kdeplot, lw=2);
		g.savefig(target + "_" + str(nvoigts) + "_" +
			str(i) + "_" + "velo_pairplot.pdf")

def sns_H2vib_plot(target, var_list, file, redshift):

	var_list = var_list

	data = pd.read_pickle(file)
	
	plt.figure()

	data_f = {}

	for var in var_list:
		if not var.startswith(("B_", "A_")):
			data_f[var] = data[var][0]
		if var == "A_Z":
			data_f[var] = data[var][0]/100000.0 + redshift

	df = pd.DataFrame(data_f)
	#g = sns.pairplot(df)

	g = sns.PairGrid(df)
	g.map_upper(sns.kdeplot, cmap="bone_r",n_levels=10,shade=True,
		shade_lowest=False)
	g.map_lower(sns.kdeplot, cmap="bone_r",n_levels=10,shade=True,
		shade_lowest=False)
	g.map_diag(plt.hist)

	g.axes[0,0].set_ylabel(r"$z$")
	g.axes[2,0].set_ylabel(r"$z$")

	g.savefig(target + "H2vib_pairplot.pdf")










