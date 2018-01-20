#! usr/bin/python

target = "GRB090926A"
redshift = 2.1068

red1 = 2.10702387241
red2 = 2.10650568989

dr1 = red1-redshift
dr2 = red2-redshift


w1 = 920.0
#w2 = 1116.54
w2 = 1300

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
= get_data("../../../spectra/GRB090926_OB1UVB.txt",
	redshift, wl_range=True, wl1=w1, wl2=w2)


ignore_lst = [[1116.7,1117.5], [1046.1,1047.0], [1055.4,1056.3], [1058.8,1059.4],
 [1083.1,1084.4], [1074.7,1076.0], [1100.8,1103.7], [1110.8,1112.3], [1097.59,1098.87]]

norm_spec = np.ones(len(wav_aa))*1.05 # add Background multiplier
synspec = SynSpec(wav_aa,redshift,5200.0,ignore_lst=ignore_lst)


h2spec = synspec.add_ion(norm_spec,"HI",broad=96,Natom=21.55,A_REDSHIFT=-24.5/100000)

h2spec = synspec.add_ion(h2spec,"FeII",broad=12,Natom=14.24,A_REDSHIFT=dr1)
h2spec = synspec.add_ion(h2spec,"FeII",broad=18,Natom=13.4,A_REDSHIFT=dr2)

h2spec = synspec.add_ion(h2spec,"SiII",broad=12,Natom=15.32,A_REDSHIFT=dr1)
h2spec = synspec.add_ion(h2spec,"SiII",broad=19,Natom=13.8,A_REDSHIFT=dr2)

h2spec = synspec.add_ion(h2spec,"SiIV",broad=12,Natom=15.85,A_REDSHIFT=dr1)
h2spec = synspec.add_ion(h2spec,"SiIV",broad=19,Natom=14.0,A_REDSHIFT=dr2)

h2spec = synspec.add_ion(h2spec,"OI",broad=12,Natom=16.19,A_REDSHIFT=dr1)

h2spec = synspec.add_ion(h2spec,"ArI",broad=12,Natom=13.76,A_REDSHIFT=dr1)

#h2spec = synspec.add_single_H2(h2spec,broad=2,NH2=15.0,A_REDSHIFT=0.0,J=0)
#h2spec = synspec.add_single_H2(h2spec,broad=2,NH2=15.0,A_REDSHIFT=0.0,J=1)
#h2spec = synspec.add_single_H2(h2spec,broad=2,NH2=15.0,A_REDSHIFT=0.0,J=2)
#h2spec = synspec.add_single_H2(h2spec,broad=2,NH2=15.0,A_REDSHIFT=0.0,J=3)


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

	axis.errorbar(wav_aa, n_flux_err, linestyle='-', color="gray", linewidth=0.5, \
		drawstyle='steps-mid')


	axis.plot(wav_aa, h2spec, label=r"$\sf Fit$", color="#2171b5", linewidth=2.0, alpha=0.9)

	# fill space between quantiles
	#axis.fill_between(wav_aa, y_min, y_max, color='#2171b5', alpha=0.2)
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
