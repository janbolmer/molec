import pymc, math, time, sys, os, argparse
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *

def gauss(x, mu, sig):
	'''
	Normal distribution used to create prior probability distributions
	'''
	return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

def s_line(x, m=1.0, t=0.0):

	y = []
	for i in x:
		y.append(m*i + t)
	return y

def s_line_err(x, m=1.0, t=0.0):

	y = []
	yerr = []
	for i in x:
		yerr.append(abs(np.random.normal()/5.0))
		y.append(m*i + t + np.random.normal()/2.0)
	return y, yerr

def model(x, y, yerr):

	tau = 1 / np.array(yerr)**2

	#m = pymc.Normal('m',mu=1.0,tau=1./0.25,doc='m')
	#t = pymc.Normal('t',mu=1.0,tau=1./0.25,doc='t')

	@pymc.stochastic(dtype=float)
	def m(value=2.0, mu=2.0, sig=5.0, doc="m"):
		pp = gauss(value, mu, sig)
		return pp

	@pymc.stochastic(dtype=float)
	def t(value=-0.5, mu=2.0, sig=5.0, doc="t"):
		pp = gauss(value, mu, sig)
		return pp

	@pymc.deterministic(plot=False)
	def model_line(x=x,y=y,yerr=yerr,m=m,t=t):

		return s_line(x, m, t)

	y_val = pymc.Normal('y_val',mu=model_line,tau=tau,
		value=y,observed=True)

	return locals()


def do_mcmc(x, y, yerr, iterations, burn_in):
	'''
	MCMC sample 
	Reading and writing Results
	'''

	pymc.np.random.seed(1)

	MDL = pymc.MCMC(model(x, y, yerr), db='pickle',dbname='mt.pickle')

	MDL.db
	MDL.sample(iterations, burn_in)
	MDL.db.close()

	y_min = MDL.stats()['model_line']['quantiles'][2.5]
	y_max = MDL.stats()['model_line']['quantiles'][97.5]
	y_min2 = MDL.stats()['model_line']['quantiles'][25]
	y_max2 = MDL.stats()['model_line']['quantiles'][75]
	y_fit = MDL.stats()['model_line']['mean']
	
	MDL.write_csv("test.csv",variables=["m", "t"])

	return y_min, y_max, y_min2, y_max2, y_fit

x = np.arange(0, 10, 0.5)
y, yerr = s_line_err(x, 1.0, 0.5)

iterations = 12000
burn_in = 0

y_min, y_max, y_min2, y_max2, y_fit = do_mcmc(x, y, yerr, iterations, burn_in)

df = pd.read_pickle("mt.pickle")

fig = figure(figsize=(10, 8))

ax = fig.add_axes([0.08, 0.08, 0.40, 0.40])


ax1 = fig.add_axes([0.56, 0.28, 0.40, 0.20])
ax2 = fig.add_axes([0.56, 0.76, 0.40, 0.20])

ax3 = fig.add_axes([0.56, 0.08, 0.40, 0.14])
ax4 = fig.add_axes([0.56, 0.56, 0.40, 0.14])

ax5 = fig.add_axes([0.08, 0.56, 0.40, 0.40])

x_gauss = np.linspace(-25, 25)
y_gauss_m = gauss(x_gauss, mu=2.0, sig=5.0)
y_gauss_t = gauss(x_gauss, mu=2.0, sig=5.0)

ax1.plot(x_gauss, y_gauss_m)
ax2.plot(x_gauss, y_gauss_t)

ax5.errorbar(df["m"][0], df["t"][0], fmt="o")

x_trace = np.arange(0, iterations-burn_in, 1)
ax3.plot(x_trace, df["m"][0])
ax4.plot(x_trace, df["t"][0])

#for i in np.arange(0, len(df["m"][0]), 1):
#	ax.plot(x, df["model_line"][0][i], linestyle="-", linewidth=1.0)

ax.errorbar(x, y, yerr=yerr, fmt="o")
ax.plot(x, y_fit)
ax.fill_between(x, y_min, y_max, color="black", alpha=0.66)
ax.fill_between(x, y_min2, y_max2, color="black", alpha=0.33)

show()
fig.savefig("test.pdf")




