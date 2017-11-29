#! /usr/bin/python

import math, time, sys, os, argparse
import numpy as np
import pandas as pd
import seaborn as sns
import decimal
import mpmath
from mpmath import *
import matplotlib.pyplot as plt
import matplotlib.pyplot as pyplot
from pylab import *

def gauss(x, mu, sig):
	return np.exp(-np.power(x-mu,2.0)/(2*np.power(sig,2.0)))

def s_line(x, m=1.0, t=0.0):
	y = []
	for i in x:
		y.append(m*i + t)
	return y

def s_line_err(x, m=1.0, t=0.0):
	y = []
	yerr = []
	for i in x:
		yerr.append(round(abs(np.random.normal()/5.0),2))
		y.append(round(m*i + t + np.random.normal()/2.0, 2))
	return y, yerr

def draw_new(old_val, scale=1.0):
	new_val = np.random.normal(loc=old_val, scale=scale)
	return new_val

def likelihood(x, data_y, m, t, sigma):

	p = 1.0

	model_y = s_line(x, m, t)
	for i in np.arange(0, len(data_y), 1):
		p *= (1./(sigma[i]*np.sqrt(2*np.pi))) * \
			mpmath.exp(-0.5*(((data_y[i]-model_y[i])/sigma[i])**2))
	return p

def m_prior(value=1.0, mu=1.0, sig=2.0):
	pp = gauss(value, mu, sig)
	return pp

def t_prior(value=0.0, mu=0.0, sig=2.0):
	pp = gauss(value, mu, sig)
	return pp

x = np.arange(-2, 11, 0.5)
#y, yerr = s_line_err(x, m=0.8, t=0.2)

y = [-1.11, -1.36, -0.57, -0.59, -0.53, 0.33, 1.65, 1.53, 1.36, 1.69,
2.24, 2.6, 2.36, 3.42, 4.51, 5.32, 4.86, 6.32, 6.41, 4.81, 7.23, 7.45,
7.75, 7.43, 8.68, 9.11]

yerr = [0.17, 0.12, 0.1, 0.26, 0.4, 0.01, 0.18, 0.15, 0.09, 0.38, 0.34,
0.13, 0.06, 0.17, 0.24, 0.22, 0.26, 0.19, 0.09, 0.01, 0.01, 0.27, 0.28,
0.01, 0.17, 0.11]


def mcmc_jpegs(iterations, m_guess, t_guess):

	sns.set_style("white")

	plt.rc('axes', linewidth=2, labelcolor='white', edgecolor="white")
	plt.rc('xtick', color="white")
	plt.rc('ytick', color="white")

	m_trace = [m_guess]
	t_trace = [t_guess]
	
	m_trace_all = [m_guess]
	t_trace_all = [t_guess]

	for i in np.arange(0, iterations, 1):
	
		m_prop = draw_new(m_trace[i], scale=3.0)
		t_prop = draw_new(t_trace[i], scale=3.0)
		
		l_old = likelihood(x, y, m_trace[i], t_trace[i], yerr)
		l_prop = likelihood(x, y, m_prop, t_prop, yerr)
		
		print l_old, l_prop

		Am = m_prior(m_prop, 1.0, 2.0)*l_prop
		Bm = m_prior(m_trace[i], 1.0, 2.0)*l_old
		am = float(Am/Bm)

		At = t_prior(t_prop, 0.0, 2.0)*l_prop
		Bt = t_prior(t_trace[i], 0.0, 2.0)*l_old
		at = float(At/Bt)

		if am >= 1:
			m_new = m_prop
			m_trace.append(m_new)
			m_trace_all.append(m_prop)
		else:
			m_new = m_trace[i]
			m_trace.append(m_new)	
			m_trace_all.append(m_prop)

		if at >= 1:
			t_new = t_prop
			t_trace.append(t_new)
			t_trace_all.append(t_prop)
	
		else:
			t_new = t_trace[i]
			t_trace.append(t_new)
			t_trace_all.append(t_prop)	
	
		fig = figure(figsize=(10, 8))
		
		ax = fig.add_axes([0.08, 0.08, 0.40, 0.40])
		ax.set_xlabel("x", fontsize=12)
		ax.set_ylabel("y", fontsize=12)
		
		ax.set_xlim([-3, 12])
		ax.set_ylim([-5, 11.2])
	
		ax.errorbar(x, y, yerr=yerr, fmt="o", color="#FF6A48")

		ax1 = fig.add_axes([0.56, 0.28, 0.40, 0.20])
		ax2 = fig.add_axes([0.56, 0.76, 0.40, 0.20])
		
		ax1.set_xlabel("m", fontsize=12)
		ax1.set_ylabel("Prior / Posterior", fontsize=12)
		ax2.set_xlabel("t", fontsize=12)
		ax2.set_ylabel("Prior / Posterior", fontsize=12)
		
		ax3 = fig.add_axes([0.56, 0.08, 0.40, 0.14])
		ax4 = fig.add_axes([0.56, 0.56, 0.40, 0.14])
		
		ax3.set_xlabel("Iteration", fontsize=12)
		ax3.set_ylabel("m", fontsize=12)
		ax3.set_ylim([-5.1, 5.1])
		
		ax4.set_xlabel("Iteration", fontsize=12)
		ax4.set_ylabel("t", fontsize=12)
		ax4.set_ylim([-5.1, 5.1])
		
		ax5 = fig.add_axes([0.08, 0.56, 0.40, 0.40])
		ax5.set_xlabel("m", fontsize=12)
		ax5.set_ylabel("t", fontsize=12)
		ax5.set_xlim([-5.1, 5.1])
		ax5.set_ylim([-5.1, 5.1])
		
		x_gauss = np.linspace(-20.0, 20.0, 200)
		y_gauss_m = gauss(x_gauss, mu=1.0, sig=2.0)
		y_gauss_t = gauss(x_gauss, mu=0.0, sig=2.0)

		ax1.plot(x_gauss, y_gauss_m)
		ax2.plot(x_gauss, y_gauss_t)
		
		#ax1.set_yscale("log")
		#ax2.set_yscale("log")

		ax1.set_xlim([-5.1, 5.1])
		ax2.set_xlim([-5.1, 5.1])

		ax1.set_ylim([-0.01, 1.05])
		ax2.set_ylim([-0.01, 1.05])
		
		weights1 = np.ones_like(m_trace_all)/float(len(m_trace_all))
		weights2 = np.ones_like(t_trace_all)/float(len(t_trace_all))
		
		ax1.hist(m_trace_all,weights=weights1,bins=np.arange(-6.33, 6.33, 0.66))
		ax2.hist(t_trace_all,weights=weights2,bins=np.arange(-6.33, 6.33, 0.66))
		
		ax5.errorbar(m_trace_all, t_trace_all, fmt="o")
		ax5.plot(m_trace, t_trace) #, fmt="o")
		
		#x_trace = np.arange(0, iterations+1, 1)
		x_trace = np.arange(0, i+2, 1)
		ax3.plot(x_trace, m_trace_all)
		ax4.plot(x_trace, t_trace_all)
		
		ax3.plot(x_trace, m_trace)
		ax4.plot(x_trace, t_trace)
		
		ax3.set_xlim([-1, iterations+1])
		ax4.set_xlim([-1, iterations+1])
		
		y_line = s_line(x, m_trace_all[-1], t_trace_all[-1])
		ax.plot(x, y_line, color="#294E86", alpha=0.500)
		if i > 5:
			y_line1 = s_line(x, m_trace_all[-2], t_trace_all[-2])
			ax.plot(x, y_line1, color="#294E86", alpha=0.375)
			y_line2 = s_line(x, m_trace_all[-3], t_trace_all[-3])
			ax.plot(x, y_line2, color="#294E86", alpha=0.250)
			y_line3 = s_line(x, m_trace_all[-4], t_trace_all[-4])
			ax.plot(x, y_line3, color="#294E86", alpha=0.125)
	
		y_line_fit = s_line(x, m_trace[-1], t_trace[-1])
		ax.plot(x, y_line_fit, color="#294E86", linewidth=3.0)
		
		fig.savefig("mcmc_test" + str(i+1) + ".png", transparent=True)


def mcmc_fit(iterations, m_guess, t_guess):

	sns.set_style("white")

	#plt.rc('axes', linewidth=2, labelcolor='white', edgecolor="white")
	#plt.rc('xtick', color="white")
	#plt.rc('ytick', color="white")

	m_trace = [m_guess]
	t_trace = [t_guess]
	
	m_trace_all = [m_guess]
	t_trace_all = [t_guess]

	for i in np.arange(0, iterations, 1):
	
		m_prop = draw_new(m_trace[i], scale=1.0)
		t_prop = draw_new(t_trace[i], scale=1.0)
		
		l_old = likelihood(x, y, m_trace[i], t_trace[i], yerr)
		l_prop = likelihood(x, y, m_prop, t_prop, yerr)

		print l_old, l_prop

		Am = m_prior(m_prop, 1.0, 1.0)*l_prop
		Bm = m_prior(m_trace[i], 1.0, 1.0)*l_old
		am = Am/Bm

		At = t_prior(t_prop, 0.0, 1.0)*l_prop
		Bt = t_prior(t_trace[i], 0.0, 1.0)*l_old
		at = At/Bt

		m_trace_all.append(m_prop)
		t_trace_all.append(t_prop)

		if am >= 1:
			m_trace.append(m_prop)
		else:
			m_trace.append(m_trace[i])	

		if at >= 1:
			t_trace.append(t_prop)
		else:
			t_trace.append(t_trace[i])

	fig = figure(figsize=(10, 8))
	
	ax = fig.add_axes([0.08, 0.08, 0.40, 0.40])
	ax.set_xlabel("x", fontsize=12)
	ax.set_ylabel("y", fontsize=12)
	
	ax.set_xlim([-3, 12])
	ax.set_ylim([-5, 11.2])
	
	ax.errorbar(x, y, yerr=yerr, fmt="o", color="#FF6A48")

	ax1 = fig.add_axes([0.56, 0.28, 0.40, 0.20])
	ax2 = fig.add_axes([0.56, 0.76, 0.40, 0.20])
	
	ax1.set_xlabel("m", fontsize=12)
	ax1.set_ylabel("Prior / Posterior", fontsize=12)
	ax2.set_xlabel("t", fontsize=12)
	ax2.set_ylabel("Prior / Posterior", fontsize=12)
	
	ax3 = fig.add_axes([0.56, 0.08, 0.40, 0.14])
	ax4 = fig.add_axes([0.56, 0.56, 0.40, 0.14])
	
	ax3.set_xlabel("Iteration", fontsize=12)
	ax3.set_ylabel("m", fontsize=12)
	ax3.set_ylim([-5.1, 5.1])
	
	ax4.set_xlabel("Iteration", fontsize=12)
	ax4.set_ylabel("t", fontsize=12)
	ax4.set_ylim([-5.1, 5.1])
	
	ax5 = fig.add_axes([0.08, 0.56, 0.40, 0.40])
	ax5.set_xlabel("m", fontsize=12)
	ax5.set_ylabel("t", fontsize=12)
	ax5.set_xlim([-5.1, 5.1])
	ax5.set_ylim([-5.1, 5.1])
	
	x_gauss = np.linspace(-20.0, 20.0, 200)
	y_gauss_m = gauss(x_gauss, mu=1.0, sig=2.0)
	y_gauss_t = gauss(x_gauss, mu=0.0, sig=2.0)

	ax1.plot(x_gauss, y_gauss_m)
	ax2.plot(x_gauss, y_gauss_t)
	
	ax1.set_xlim([-5.1, 5.1])
	ax2.set_xlim([-5.1, 5.1])

	ax1.set_ylim([-0.01, 1.05])
	ax2.set_ylim([-0.01, 1.05])
	
	weights1 = np.ones_like(m_trace_all)/float(len(m_trace_all))
	weights2 = np.ones_like(t_trace_all)/float(len(t_trace_all))
	
	ax1.hist(m_trace_all,weights=weights1,bins=np.arange(-6.33, 6.33, 0.66))
	ax2.hist(t_trace_all,weights=weights2,bins=np.arange(-6.33, 6.33, 0.66))
	
	ax5.errorbar(m_trace_all, t_trace_all, fmt="o")
	ax5.plot(m_trace, t_trace)
	
	x_trace = np.arange(0, iterations+1, 1)
	ax3.plot(x_trace, m_trace_all)
	ax4.plot(x_trace, t_trace_all)
	
	ax3.plot(x_trace, m_trace)
	ax4.plot(x_trace, t_trace)
	
	ax3.set_xlim([-1, iterations+1])
	ax4.set_xlim([-1, iterations+1])
	
	y_line_fit = s_line(x, m_trace[-1], t_trace[-1])
	ax.text(0, 9.0, "m:" + str(round(m_trace[-1], 2)))
	ax.text(0, 8.0, "t:" + str(round(t_trace[-1], 2)))
	ax.plot(x, y_line_fit, color="#294E86", linewidth=3.0)
	
	fig.savefig("mcmc_line_fit_" + str(i+1) + ".pdf", transparent=True)


#mcmc_jpegs(10, -1.0, -0.5)
mcmc_fit(400, -3.0, 2.0)
