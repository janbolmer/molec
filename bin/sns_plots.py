import math
import numpy as np
import pandas as pd

import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *

import seaborn as sns

def sns_pair_plot(target, var_list, file, redshift):

	data = pd.read_pickle(file)
	
	plt.figure()

	data_f = {}

	for var in var_list:
		if var in ['NTOTH2', 'TEMP', 'B']:
			data_f[var] = data[var][0]
		if var == 'A_Z':
			data_f[var] = data[var][0]/100000.0 + redshift

	df = pd.DataFrame(data_f)

	g = sns.pairplot(df)

	g.savefig(target + "H2_pairplot.pdf")


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

