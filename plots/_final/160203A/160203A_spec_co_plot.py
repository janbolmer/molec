#! usr/bin/python

#https://www.pantone.com/color-of-the-year-2017
pt_analogous = ["#86af49", "#817397", "#b88bac",
"#d57f70","#dcb967","#ac9897","#ac898d","#f0e1ce",
"#86af49", "#817397","#b88bac", "#d57f70", "#dcb967"]

target = "GRB161023A"
redshift = 2.710681
redshift2 = 2.709807
redshift3 = 2.708428

DR1 = redshift - redshift2
DR2 = redshift - redshift3

w1 = 1060.0
w2 = 1646.0

co_intervals = [[1544.0, 1545.0], [1509.0, 1510.0],
				[1477.0, 1478.0]]

ignore_lst = [[1072.59, 1074.2], [1077.85, 1078.41], [1061.94, 1062.65], [1057.5, 1059.38]]

import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *
import numpy as np

sys.path.append('../../../bin/')

from spec_functions import * # spec_functions.py
from syn_spec import * # syn_spec.py
from sns_plots import * # sns_plots.py

sns.set_style("white", {'legend.frameon': True})

def get_co(redshift):

	co_name, co_wav = [], []
	co_file = open("atoms/co.dat", "r")
	for line in co_file:
		ss = line.split()			
		co_name.append(str(ss[0]).strip("CO"))
		co_wav.append(float(ss[1])*(1+redshift))
	co_file.close()

	return co_name, co_wav

co_name, co_wav = get_co(redshift)

wav_aa, n_flux, n_flux_err, flux, flux_err, grb_name, res, psf_fwhm \
= get_data("../../../spectra/GRB161023A_OB1VIS.txt",
	redshift, wl_range=True, wl1=w1, wl2=w2)

norm_spec = np.ones(len(wav_aa)) # add Background multiplier
synspec = SynSpec(wav_aa,redshift,9733)

h2spec = synspec.add_CO(norm_spec,broad=2,NTOTCO=0.0,TEMP=30,A_REDSHIFT=0.0,NROT=[0, 1, 2, 3, 4, 5, 6])

wav_range = (max(wav_aa)-min(wav_aa))/2.0

fig = figure(figsize=(6, 7))

ax = fig.add_axes([0.15, 0.10, 0.83, 0.18*3+0.10])
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

ax1 = fig.add_axes([0.15, 0.13, 0.83, 0.18])
ax2 = fig.add_axes([0.15, 0.45, 0.83, 0.18])
ax3 = fig.add_axes([0.15, 0.77, 0.83, 0.18])


ax1.set_title(r"$\sf CO\, AX(0-0)$", fontsize=20)
ax2.set_title(r"$\sf CO\, AX(1-0)$", fontsize=20)
ax3.set_title(r"$\sf CO\, AX(2-0)$", fontsize=20)

ax1.set_xlabel(r"$\sf Observed\, Wavelength (\AA)$", fontsize=24)
ax2.set_ylabel(r"$\sf Normalized\, Flux$", fontsize=24)

for axis in [ax1, ax2, ax3]:

	axis.errorbar(wav_aa, n_flux, linestyle='-', color="black", linewidth=1.0, \
		drawstyle='steps-mid', label=r"$\sf Data$")
	axis.errorbar(wav_aa, n_flux, yerr=n_flux_err, fmt="o", color="gray", capsize=0, markersize=4)

	#axis.plot(wav_aa, n_flux_err, color="gray", linewidth=0.8)
	axis.plot(wav_aa, h2spec, label=r"$\sf Fit$", color="#2171b5", linewidth=2)


	axis.set_ylim([-0.25, 2.10])
	axis.set_yticks([0, 1, 2])


	axis.axhline(0.0, linestyle="dashed", color="black", linewidth=2)

	y_fill = [1 for wav in wav_aa]

	for wav_rng in ignore_lst:
		axis.axvspan(wav_rng[0]*(1+redshift), wav_rng[1]*(1+redshift), \
			facecolor='black', alpha=0.25)

	for side in ['top','bottom','left','right']:
	  	axis.spines[side].set_linewidth(2)
	axis.tick_params(which='major', length=8, width=2)
	axis.tick_params(which='minor', length=6, width=1)
	for tick in axis.xaxis.get_major_ticks():
		tick.label.set_fontsize(18)
	for tick in axis.yaxis.get_major_ticks():
		tick.label.set_fontsize(18)

	if axis == ax1:
		axis.set_xlim([1544.0*(1+redshift), 1545.6*(1+redshift)])
		axis.set_xticks([5731, 5733, 5735])
		for i in np.arange(0, len(co_name), 1):
			if (1544.2*(1+redshift)) < co_wav[i] < (1545.5*(1+redshift)):
				if i%2 == 0 and not i%3 == 0:
					axis.text(co_wav[i]+0.05, 1.4, co_name[i], fontsize=8, color="#41ae76")
					axis.axvline(co_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)
				elif i%3 == 0 and not i%2 == 0:
					axis.text(co_wav[i]+0.05, 1.6, co_name[i], fontsize=8, color="#41ae76")
					axis.axvline(co_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)
				else:
					axis.text(co_wav[i]+0.05, 1.8, co_name[i], fontsize=8, color="#41ae76")
					axis.axvline(co_wav[i],  ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)

	if axis == ax2:
		axis.set_xlim([1509.4*(1+redshift), 1510.2*(1+redshift)])
		axis.set_xticks([5601, 5602, 5603])
		for i in np.arange(0, len(co_name), 1):
			if (1509.6*(1+redshift)) < co_wav[i] < (1510.1*(1+redshift)):
				if i%2 == 0 and not i%3 == 0:
					axis.text(co_wav[i]+0.03, 1.4, co_name[i], fontsize=8, color="#41ae76")
					axis.axvline(co_wav[i], ymin=0.7, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)
				elif i%3 == 0 and not i%2 == 0:
					axis.text(co_wav[i]+0.03, 1.6, co_name[i], fontsize=8, color="#41ae76")
					axis.axvline(co_wav[i], ymin=0.7, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)
				else:
					axis.text(co_wav[i]+0.03, 1.8, co_name[i], fontsize=8, color="#41ae76")
					axis.axvline(co_wav[i],  ymin=0.7, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)

	if axis == ax3:
		axis.set_xlim([1477.2*(1+redshift), 1478.1*(1+redshift)])
		axis.set_xticks([5482, 5483, 5484])
		for i in np.arange(0, len(co_name), 1):
			if (1477.0*(1+redshift)) < co_wav[i] < (1478.0*(1+redshift)):
				if i%2 == 0 and not i%3 == 0:
					axis.text(co_wav[i]+0.03, 1.4, co_name[i], fontsize=8, color="#41ae76")
					axis.axvline(co_wav[i], ymin=0.7, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)
				elif i%3 == 0 and not i%2 == 0:
					axis.text(co_wav[i]+0.03, 1.6, co_name[i], fontsize=8, color="#41ae76")
					axis.axvline(co_wav[i], ymin=0.7, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)
				else:
					axis.text(co_wav[i]+0.03, 1.8, co_name[i], fontsize=8, color="#41ae76")
					axis.axvline(co_wav[i],  ymin=0.7, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)

show()
fig.savefig(target + "_CO_fit_spec_pub.pdf")
