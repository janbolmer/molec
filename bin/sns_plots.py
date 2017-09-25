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
		if var in ['NTOTH2', 'TEMP', 'B']:
			data_f[var] = data[var][0]
		if var == 'A_Z':
			data_f[var] = data[var][0]/100000.0 + redshift

	df = pd.DataFrame(data_f)

	#g = sns.pairplot(df)

	#g.axes[0,0].set_ylim(redshift-0.005,redshift+0.05)
	#g.axes[1,0].set_ylim(0,20)
	#g.axes[2,0].set_ylim(0,24)
	#g.axes[3,0].set_ylim(0,1000)
#
#
	#g.axes[0,0].set_xlim(redshift-0.005,redshift+0.05)
	#g.axes[0,1].set_xlim(0,20)
	#g.axes[0,2].set_xlim(0,24)
	#g.axes[0,3].set_xlim(0,1000)
#
	#g.savefig(target + "H2_pairplot.pdf")

	g = sns.PairGrid(df) #, diag_kws=dict(color="blue", shade=True))
	g.map_upper(sns.kdeplot, cmap="bone_r",n_levels=10,shade=True,
		shade_lowest=False)
	g.map_lower(sns.kdeplot, cmap="bone_r",n_levels=10,shade=True,
		shade_lowest=False)
	g.map_diag(plt.hist) #, lw=2);
	#g[0,1].set(xlabel=r"$\sigma_1$")
	g.savefig(target + "H2_pairplot.pdf")

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
		#g[0,1].set(xlabel=r"$\sigma_1$")
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

	g = sns.pairplot(df)

	g.savefig(target + "H2vib_pairplot.pdf")

