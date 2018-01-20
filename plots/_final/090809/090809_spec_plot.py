#! usr/bin/python

target = "GRB090809"
redshift = 2.7373

w1 = 990.
w2 = 1120.

import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *
import numpy as np

sys.path.append('../../../bin/')

from spec_functions import * # spec_functions.py
from syn_spec import * # syn_spec.py
from sns_plots import * # sns_plots.py

sns.set_style("white", {'legend.frameon': True})

wav_aa, n_flux, n_flux_err, flux, flux_err, grb_name, res, psf_fwhm \
= get_data("../../../spectra/GRB090809_OB1UVB.txt",
	redshift, wl_range=True, wl1=w1, wl2=w2)


ignore_lst = [[1057.10,1058.40], [1071.98,1073.05], [1105.16,1107.84]]

norm_spec = np.ones(len(wav_aa)) # add Background multiplier

synspec = SynSpec(wav_aa,redshift,4000,ignore_lst=ignore_lst) # resolution is not correct
h2spec = synspec.add_ion(norm_spec,"HI",broad=23,Natom=21.50,A_REDSHIFT=-50/100000.0)
h2spec = synspec.add_ion(h2spec,"ArI",broad=32,Natom=16.75,A_REDSHIFT=29/100000.0)
h2spec = synspec.add_ion(h2spec,"FeII",broad=84.12,Natom=15.86,A_REDSHIFT=36/100000.0)
h2spec = synspec.add_ion(h2spec,"OI",broad=36,Natom=17.36,A_REDSHIFT=27/100000.0)
h2spec = synspec.add_ion(h2spec,"OVI",broad=58,Natom=16.06,A_REDSHIFT=-79/100000.0)
h2spec = synspec.add_ion(h2spec,"CII",broad=96,Natom=9.91,A_REDSHIFT=10/100000.0)


norm_spec = np.ones(len(wav_aa))
synspec2 = SynSpec(wav_aa,redshift,4000,ignore_lst=ignore_lst)
h2spec_up = synspec2.add_ion(norm_spec,"HI",broad=3,Natom=21.30,A_REDSHIFT=-50/100000.0)
h2spec_up = synspec2.add_ion(h2spec_up,"ArI",broad=16,Natom=15.32,A_REDSHIFT=29/100000.0)
h2spec_up = synspec2.add_ion(h2spec_up,"FeII",broad=67,Natom=15.65,A_REDSHIFT=36/100000.0)
h2spec_up = synspec2.add_ion(h2spec_up,"OI",broad=31,Natom=15.43,A_REDSHIFT=27/100000.0)
h2spec_up = synspec2.add_ion(h2spec_up,"OVI",broad=42,Natom=15.46,A_REDSHIFT=-79/100000.0)
h2spec_up = synspec2.add_ion(h2spec_up,"CII",broad=36,Natom=3.6,A_REDSHIFT=10/100000.0)

norm_spec = np.ones(len(wav_aa))
synspec3 = SynSpec(wav_aa,redshift,4000,ignore_lst=ignore_lst)
h2spec_low = synspec3.add_ion(norm_spec,"HI",broad=46,Natom=21.70,A_REDSHIFT=-50/100000.0)
h2spec_low = synspec3.add_ion(h2spec_low,"ArI",broad=48,Natom=18.18,A_REDSHIFT=29/100000.0)
h2spec_low = synspec3.add_ion(h2spec_low,"FeII",broad=101,Natom=16.07,A_REDSHIFT=36/100000.0)
h2spec_low = synspec3.add_ion(h2spec_low,"OI",broad=51,Natom=18.59,A_REDSHIFT=36/100000.0)
h2spec_low = synspec3.add_ion(h2spec_low,"OVI",broad=74,Natom=16.66,A_REDSHIFT=-79/100000.0)
h2spec_low = synspec3.add_ion(h2spec_low,"CII",broad=156,Natom=16.6,A_REDSHIFT=10/100000.0)

wav_range = (max(wav_aa)-min(wav_aa))/5.0

fig = figure(figsize=(10, 12))

ax1 = fig.add_axes([0.08, 0.08, 0.90, 0.11])
ax2 = fig.add_axes([0.08, 0.25, 0.90, 0.11])
ax3 = fig.add_axes([0.08, 0.41, 0.90, 0.11])
ax4 = fig.add_axes([0.08, 0.58, 0.90, 0.11])
ax5 = fig.add_axes([0.08, 0.73, 0.90, 0.11])
ax6 = fig.add_axes([0.08, 0.88, 0.90, 0.11])

for axis in [ax1, ax2, ax3, ax4, ax5, ax6]:


	axis.errorbar(wav_aa, n_flux, linestyle='-', color="black", linewidth=0.5, \
		drawstyle='steps-mid', label=r"$\sf Data$")

	axis.errorbar(wav_aa, n_flux_err, linestyle='-', color="black", linewidth=0.5, \
		drawstyle='steps-mid', alpha=0.6)

	axis.plot(wav_aa, h2spec, label=r"$\sf Fit$", color="#2171b5", linewidth=2.0, alpha=0.9)

	axis.plot(wav_aa, h2spec_up, color="#2171b5", linewidth=1.0, alpha=0.9, linestyle="dashed")

	axis.plot(wav_aa, h2spec_low, color="#2171b5", linewidth=1.0, alpha=0.9, linestyle="dashed")

	# fill space between quantiles
	axis.fill_between(wav_aa, h2spec_low, h2spec_up, color='#2171b5', alpha=0.2)
	#axis.fill_between(wav_aa, y_min2, y_max2, color='#2171b5', alpha=0.4)

	# plot 25% and 75% quantiles
	#axis.plot(wav_aa, y_max2, color="#2171b5", linewidth=1, alpha=0.9, linestyle="dashed")
	#axis.plot(wav_aa, y_min2, color="#2171b5", linewidth=1, alpha=0.9, linestyle="dashed")

	# plot 2.5& and 97.5% quantiles
	#axis.plot(wav_aa, y_max, color="#2171b5", linewidth=1, alpha=0.9, linestyle=":")
	#axis.plot(wav_aa, y_min, color="#2171b5", linewidth=1, alpha=0.9, linestyle=":")

	axis.set_ylim([-0.85, 2.25])

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

a_name, a_wav, ai_name, ai_wav, aex_name, \
aex_wav, h2_name, h2_wav = get_lines(redshift)

for i in np.arange(0, len(a_name), 1):

	# Better Solution than this mess?
	if min(wav_aa) < a_wav[i] < (max(wav_aa)-wav_range*4):
		if i%2 == 0:
			ax6.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=5, color="#41ae76")
			ax6.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
		elif i%3 == 0:
			ax6.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=5, color="#41ae76")
			ax6.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
		else:
			ax6.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=5, color="#41ae76")
			ax6.axvline(a_wav[i],  ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)				

	if (max(wav_aa)-wav_range*4) < a_wav[i] < (max(wav_aa)-wav_range*3):
		if i%2 == 0:
			ax5.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=5, color="#41ae76")
			ax5.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
		elif i%3 == 0:
			ax5.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=5, color="#41ae76")
			ax5.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
		else:
			ax5.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=5, color="#41ae76")
			ax5.axvline(a_wav[i],  ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)	

	if (max(wav_aa)-wav_range*3) < a_wav[i] < (max(wav_aa)-wav_range*2):
		if i%2 == 0:
			ax4.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=5, color="#41ae76")
			ax4.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
		elif i%3 == 0:
			ax4.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=5, color="#41ae76")
			ax4.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
		else:
			ax4.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=5, color="#41ae76")
			ax4.axvline(a_wav[i],  ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)	

	if (max(wav_aa)-wav_range*2) < a_wav[i] < (max(wav_aa)-wav_range*1):
		if i%2 == 0:
			ax3.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=5, color="#41ae76")
			ax3.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
		elif i%3 == 0:
			ax3.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=5, color="#41ae76")
			ax3.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
		else:
			ax3.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=5, color="#41ae76")
			ax3.axvline(a_wav[i],  ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)	

	if (max(wav_aa)-wav_range*1) < a_wav[i] < max(wav_aa):
		if i%2 == 0:
			ax2.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=5, color="#41ae76")
			ax2.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
		elif i%3 == 0:
			ax2.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=5, color="#41ae76")
			ax2.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)
		else:
			ax2.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=5, color="#41ae76")
			ax2.axvline(a_wav[i],  ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=0.4)			

for i in np.arange(0, len(h2_name), 1):
	if min(wav_aa) < h2_wav[i] < (max(wav_aa)-wav_range*4):
		if i%2 == 0:
			ax6.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6,color="#a50f15")
			ax6.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
		else:
			ax6.text(h2_wav[i]+0.2, -0.8, h2_name[i], fontsize=6,color="#a50f15")
			ax6.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)

	if (max(wav_aa)-wav_range*4) < h2_wav[i] < (max(wav_aa)-wav_range*3):
		if i%2 == 0:
			ax5.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6,color="#a50f15")
			ax5.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
		else:
			ax5.text(h2_wav[i]+0.2, -0.8, h2_name[i], fontsize=6,color="#a50f15")
			ax5.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)

	if (max(wav_aa)-wav_range*3) < h2_wav[i] < (max(wav_aa)-wav_range*2):
		if i%2 == 0:
			ax4.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6,color="#a50f15")
			ax4.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
		else:
			ax4.text(h2_wav[i]+0.2, -0.8, h2_name[i], fontsize=6,color="#a50f15")
			ax4.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)

	if (max(wav_aa)-wav_range*2) < h2_wav[i] < (max(wav_aa)-wav_range*1):
		if i%2 == 0:
			ax3.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6,color="#a50f15")
			ax3.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
		else:
			ax3.text(h2_wav[i]+0.2, -0.8, h2_name[i], fontsize=6,color="#a50f15")
			ax3.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
	if (max(wav_aa)-wav_range*1) < h2_wav[i] < max(wav_aa):
		if i%2 == 0:
			ax2.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6,color="#a50f15")
			ax2.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
		else:
			ax2.text(h2_wav[i]+0.2, -0.8, h2_name[i], fontsize=6,color="#a50f15")
			ax2.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)

lg = ax1.legend(numpoints=1, fontsize=10, loc=1)

ax1.set_xlabel(r"$\sf Observed\, Wavelength (\AA)$", fontsize=24)
ax3.set_ylabel(r"$\sf Normalized\, Flux$", fontsize=24)
ax1.axhline(1, color="#2171b5", linewidth=2)

ax6.set_xlim([min(wav_aa), max(wav_aa)-wav_range*4])
ax5.set_xlim([max(wav_aa)-wav_range*4, max(wav_aa)-wav_range*3])
ax4.set_xlim([max(wav_aa)-wav_range*3, max(wav_aa)-wav_range*2])
ax3.set_xlim([max(wav_aa)-wav_range*2, max(wav_aa)-wav_range*1])
ax2.set_xlim([max(wav_aa)-wav_range*1, max(wav_aa)])

show()
fig.savefig(target + "_H2_fit_spec_pub.pdf")
