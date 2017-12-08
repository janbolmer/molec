#! usr/bin/python

target = "GRB161023A"
redshift = 2.710681
redshift2 = 2.709807
redshift3 = 2.708428

DR1 = redshift - redshift2
DR2 = redshift - redshift3

w1 = 940.
w2 = 1240.

ignore_lst = [[1072.59, 1074.2], [1077.85, 1078.41], [1061.94, 1062.65]]# , [1057.5, 1059.38]]

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
= get_data("../../../spectra/GRB161023A_OB1UVB.txt",
	redshift, wl_range=True, wl1=w1, wl2=w2)

norm_spec = np.ones(len(wav_aa)) # add Background multiplier
synspec = SynSpec(wav_aa,redshift,7000)

#h2spec = synspec.add_H2(norm_spec,broad=0.19,NTOTH2=14.17,TEMP=1221.,A_REDSHIFT=-5.764/100000.0,NROT=[0, 1, 2, 3, 4, 5])
#h2spec = synspec.add_H2(norm_spec,broad=4.578,NTOTH2=13.86,TEMP=536.35,A_REDSHIFT=-DR1,NROT=[0, 1, 2, 3, 4, 5])
#h2spec = synspec.add_H2(norm_spec,broad=47.34,NTOTH2=14.91,TEMP=396.3,A_REDSHIFT=-DR2,NROT=[0, 1, 2, 3, 4, 5])

h2spec = synspec.add_H2(norm_spec,broad=0.19,NTOTH2=0,TEMP=1221.,A_REDSHIFT=-5.764/100000.0,NROT=[0, 1, 2, 3, 4, 5])
h2spec = synspec.add_H2(norm_spec,broad=4.578,NTOTH2=0,TEMP=536.35,A_REDSHIFT=-DR1,NROT=[0, 1, 2, 3, 4, 5])
h2spec = synspec.add_H2(norm_spec,broad=47.34,NTOTH2=0,TEMP=396.3,A_REDSHIFT=-DR2,NROT=[0, 1, 2, 3, 4, 5])


h2spec = synspec.add_ion(h2spec,"ArI",broad=29.45,Natom=13.29,A_REDSHIFT=-7.63/100000.0)
h2spec = synspec.add_ion(h2spec,"ArI",broad=29.45,Natom=13.00,A_REDSHIFT=-DR1)
h2spec = synspec.add_ion(h2spec,"HI",broad=22.45,Natom=18.155,A_REDSHIFT=-(redshift-2.2311))
h2spec = synspec.add_ion(h2spec,"HI",broad=0.669,Natom=17.359,A_REDSHIFT=-(redshift-2.2292))

wav_range = (max(wav_aa)-min(wav_aa))/4.0

fig = figure(figsize=(14, 10))

ax = fig.add_axes([0.07, 0.15, 0.92, 4*0.18])
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

ax2 = fig.add_axes([0.07, 0.12, 0.92, 0.18])
ax3 = fig.add_axes([0.07, 0.35, 0.92, 0.18])
ax4 = fig.add_axes([0.07, 0.58, 0.92, 0.18])
ax5 = fig.add_axes([0.07, 0.81, 0.92, 0.18])

for axis in [ax2, ax3, ax4, ax5]:

	axis.errorbar(wav_aa, n_flux, linestyle='-', color="black", linewidth=0.5, \
		drawstyle='steps-mid', label=r"$\sf Data$")

	axis.plot(wav_aa[2:-2], h2spec[2:-2], label=r"$\sf Fit$", color="#2171b5", linewidth=1.8, alpha=0.9)

	axis.set_ylim([-0.55, 2.05])

	axis.axhline(0.0, linestyle="dashed", color="black", linewidth=2)

	y_fill = [1 for wav in wav_aa]

	axis.set_yticks([0, 1, 2])
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

	#for side in ['top','bottom','left','right']:
	#	axis.spines[side].set_linewidth(2)
	#	axis.spines[side].set_color("#7187A6")
	#	axis.tick_params(which='major',length=8,width=2,colors="#7187A6")
	#	axis.tick_params(which='minor',length=4,width=1.5,colors="#7187A6")

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(18)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(18)

a_name, a_wav, ai_name, ai_wav, aex_name, \
aex_wav, h2_name, h2_wav = get_lines(redshift)

a_name2, a_wav2, ai_name2, ai_wav2, aex_name2, \
aex_wav2, h2_name2, h2_wav2 = get_lines(redshift2)

a_name3, a_wav3, ai_name3, ai_wav3, aex_name3, \
aex_wav3, h2_name3, h2_wav3 = get_lines(redshift3)


for i in np.arange(0, len(a_name), 1):

	if (max(wav_aa)-wav_range*4) < a_wav[i] < (max(wav_aa)-wav_range*3):
		if i%2 == 0:
			ax5.text(a_wav[i]+0.2, 1.4, a_name[i], fontsize=8, color="#41ae76")
			ax5.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)
		elif i%3 == 0:
			ax5.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=8, color="#41ae76")
			ax5.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)
		else:
			ax5.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=8, color="#41ae76")
			ax5.axvline(a_wav[i],  ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)	

	if (max(wav_aa)-wav_range*3) < a_wav[i] < (max(wav_aa)-wav_range*2):
		if i%2 == 0:
			ax4.text(a_wav[i]+0.2, 1.4, a_name[i], fontsize=8, color="#41ae76")
			ax4.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)
		elif i%3 == 0:
			ax4.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=8, color="#41ae76")
			ax4.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)
		else:
			ax4.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=8, color="#41ae76")
			ax4.axvline(a_wav[i],  ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)

	if (max(wav_aa)-wav_range*2) < a_wav[i] < (max(wav_aa)-wav_range*1):
		if i%2 == 0:
			ax3.text(a_wav[i]+0.2, 1.4, a_name[i], fontsize=8, color="#41ae76")
			ax3.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)
		elif i%3 == 0:
			ax3.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=8, color="#41ae76")
			ax3.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)
		else:
			ax3.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=8, color="#41ae76")
			ax3.axvline(a_wav[i],  ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)	

	if (max(wav_aa)-wav_range*1) < a_wav[i] < max(wav_aa):
		if i%2 == 0:
			ax2.text(a_wav[i]+0.2, 1.4, a_name[i], fontsize=8, color="#41ae76")
			ax2.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)
		elif i%3 == 0:
			ax2.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=8, color="#41ae76")
			ax2.axvline(a_wav[i], ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)
		else:
			ax2.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=8, color="#41ae76")
			ax2.axvline(a_wav[i],  ymin=0.8, ymax=1.0, linestyle="-", color="#41ae76", linewidth=1.0)			


for i in np.arange(0, len(a_name2), 1):

	if (max(wav_aa)-wav_range*4) < a_wav2[i] < (max(wav_aa)-wav_range*3):
		if i%2 == 0:
			#ax3.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=8, color="#41ae76")
			ax5.axvline(a_wav2[i], ymin=0.8, ymax=1.0, linestyle="dashed", color="#41ae76", linewidth=1.0)
		elif i%3 == 0:
			#ax3.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=8, color="#41ae76")
			ax5.axvline(a_wav2[i], ymin=0.8, ymax=1.0, linestyle="dashed", color="#41ae76", linewidth=1.0)
		else:
			#ax3.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=8, color="#41ae76")
			ax5.axvline(a_wav2[i],  ymin=0.8, ymax=1.0, linestyle="dashed", color="#41ae76", linewidth=1.0)	

	if (max(wav_aa)-wav_range*3) < a_wav2[i] < (max(wav_aa)-wav_range*2):
		if i%2 == 0:
			#ax3.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=8, color="#41ae76")
			ax4.axvline(a_wav2[i], ymin=0.8, ymax=1.0, linestyle="dashed", color="#41ae76", linewidth=1.0)
		elif i%3 == 0:
			#ax3.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=8, color="#41ae76")
			ax4.axvline(a_wav2[i], ymin=0.8, ymax=1.0, linestyle="dashed", color="#41ae76", linewidth=1.0)
		else:
			#ax3.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=8, color="#41ae76")
			ax4.axvline(a_wav2[i],  ymin=0.8, ymax=1.0, linestyle="dashed", color="#41ae76", linewidth=1.0)	

	if (max(wav_aa)-wav_range*2) < a_wav2[i] < (max(wav_aa)-wav_range*1):
		if i%2 == 0:
			#ax3.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=8, color="#41ae76")
			ax3.axvline(a_wav2[i], ymin=0.8, ymax=1.0, linestyle="dashed", color="#41ae76", linewidth=1.0)
		elif i%3 == 0:
			#ax3.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=8, color="#41ae76")
			ax3.axvline(a_wav2[i], ymin=0.8, ymax=1.0, linestyle="dashed", color="#41ae76", linewidth=1.0)
		else:
			#ax3.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=8, color="#41ae76")
			ax3.axvline(a_wav2[i],  ymin=0.8, ymax=1.0, linestyle="dashed", color="#41ae76", linewidth=1.0)	

	if (max(wav_aa)-wav_range*1) < a_wav2[i] < max(wav_aa):
		if i%2 == 0:
			#ax2.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=8, color="#41ae76")
			ax2.axvline(a_wav2[i], ymin=0.8, ymax=1.0, linestyle="dashed", color="#41ae76", linewidth=1.0)
		elif i%3 == 0:
			#ax2.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=8, color="#41ae76")
			ax2.axvline(a_wav2[i], ymin=0.8, ymax=1.0, linestyle="dashed", color="#41ae76", linewidth=1.0)
		else:
			#ax2.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=8, color="#41ae76")
			ax2.axvline(a_wav2[i],  ymin=0.8, ymax=1.0, linestyle="dashed", color="#41ae76", linewidth=1.0)	


for i in np.arange(0, len(a_name3), 1):

	if (max(wav_aa)-wav_range*4) < a_wav3[i] < (max(wav_aa)-wav_range*3):
		if i%2 == 0:
			#ax3.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=8, color="#41ae76")
			ax5.axvline(a_wav3[i], ymin=0.8, ymax=1.0, linestyle="dotted", color="#41ae76", linewidth=1.0)
		elif i%3 == 0:
			#ax3.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=8, color="#41ae76")
			ax5.axvline(a_wav3[i], ymin=0.8, ymax=1.0, linestyle="dotted", color="#41ae76", linewidth=1.0)
		else:
			#ax3.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=8, color="#41ae76")
			ax5.axvline(a_wav3[i],  ymin=0.8, ymax=1.0, linestyle="dotted", color="#41ae76", linewidth=1.0)	

	if (max(wav_aa)-wav_range*3) < a_wav3[i] < (max(wav_aa)-wav_range*2):
		if i%2 == 0:
			#ax3.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=8, color="#41ae76")
			ax4.axvline(a_wav3[i], ymin=0.8, ymax=1.0, linestyle="dotted", color="#41ae76", linewidth=1.0)
		elif i%3 == 0:
			#ax3.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=8, color="#41ae76")
			ax4.axvline(a_wav3[i], ymin=0.8, ymax=1.0, linestyle="dotted", color="#41ae76", linewidth=1.0)
		else:
			#ax3.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=8, color="#41ae76")
			ax4.axvline(a_wav3[i],  ymin=0.8, ymax=1.0, linestyle="dotted", color="#41ae76", linewidth=1.0)	

	if (max(wav_aa)-wav_range*2) < a_wav3[i] < (max(wav_aa)-wav_range*1):
		if i%2 == 0:
			#ax3.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=8, color="#41ae76")
			ax3.axvline(a_wav3[i], ymin=0.8, ymax=1.0, linestyle="dotted", color="#41ae76", linewidth=1.0)
		elif i%3 == 0:
			#ax3.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=8, color="#41ae76")
			ax3.axvline(a_wav3[i], ymin=0.8, ymax=1.0, linestyle="dotted", color="#41ae76", linewidth=1.0)
		else:
			#ax3.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=8, color="#41ae76")
			ax3.axvline(a_wav3[i],  ymin=0.8, ymax=1.0, linestyle="dotted", color="#41ae76", linewidth=1.0)	

	if (max(wav_aa)-wav_range*1) < a_wav3[i] < max(wav_aa):
		if i%2 == 0:
			#ax2.text(a_wav[i]+0.2, 1.6, a_name[i], fontsize=8, color="#41ae76")
			ax2.axvline(a_wav3[i], ymin=0.8, ymax=1.0, linestyle="dotted", color="#41ae76", linewidth=1.0)
		elif i%3 == 0:
			#ax2.text(a_wav[i]+0.2, 1.8, a_name[i], fontsize=8, color="#41ae76")
			ax2.axvline(a_wav3[i], ymin=0.8, ymax=1.0, linestyle="dotted", color="#41ae76", linewidth=1.0)
		else:
			#ax2.text(a_wav[i]+0.2, 2.0, a_name[i], fontsize=8, color="#41ae76")
			ax2.axvline(a_wav3[i],  ymin=0.8, ymax=1.0, linestyle="dotted", color="#41ae76", linewidth=1.0)	

for i in np.arange(0, len(h2_name), 1):

	if not "6" in h2_name[i]:
		if not "7" in h2_name[i]:

			if (max(wav_aa)-wav_range*4) < h2_wav[i] < (max(wav_aa)-wav_range*3):
				if i%2 == 0:
					ax5.text(h2_wav[i]+0.2, -0.2, h2_name[i], fontsize=10,color="#a50f15")
					ax5.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=1.0)
				else:
					ax5.text(h2_wav[i]+0.2, -0.4, h2_name[i], fontsize=10,color="#a50f15")
					ax5.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=1.0)
			if (max(wav_aa)-wav_range*3) < h2_wav[i] < (max(wav_aa)-wav_range*2):
				if i%2 == 0:
					ax4.text(h2_wav[i]+0.2, -0.2, h2_name[i], fontsize=10,color="#a50f15")
					ax4.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=1.0)
				else:
					ax4.text(h2_wav[i]+0.2, -0.4, h2_name[i], fontsize=10,color="#a50f15")
					ax4.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=1.0)
			if (max(wav_aa)-wav_range*2) < h2_wav[i] < (max(wav_aa)-wav_range*1):
				if i%2 == 0:
					ax3.text(h2_wav[i]+0.2, -0.2, h2_name[i], fontsize=10,color="#a50f15")
					ax3.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=1.0)
				else:
					ax3.text(h2_wav[i]+0.2, -0.4, h2_name[i], fontsize=10,color="#a50f15")
					ax3.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=1.0)
			if (max(wav_aa)-wav_range*1) < h2_wav[i] < max(wav_aa):
				if i%2 == 0:
					ax2.text(h2_wav[i]+0.2, -0.2, h2_name[i], fontsize=10,color="#a50f15")
					ax2.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=1.0)
				else:
					ax2.text(h2_wav[i]+0.2, -0.4, h2_name[i], fontsize=10,color="#a50f15")
					ax2.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=1.0)

for i in np.arange(0, len(h2_name2), 1):

	if not "6" in h2_name[i]:
		if not "7" in h2_name[i]:
			if (max(wav_aa)-wav_range*4) < h2_wav2[i] < (max(wav_aa)-wav_range*3):
				if i%2 == 0:
					#ax3.text(h2_wav2[i]+0.2, -0.5, h2_name2[i], fontsize=10,color="#a50f15")
					ax5.axvline(h2_wav2[i], ymin=0.0, ymax=0.2, linestyle="dashed", color="#a50f15", linewidth=1.0)
				else:
					#ax3.text(h2_wav2[i]+0.2, -0.8, h2_name2[i], fontsize=10,color="#a50f15")
					ax5.axvline(h2_wav2[i], ymin=0.0, ymax=0.2, linestyle="dashed", color="#a50f15", linewidth=1.0)
			if (max(wav_aa)-wav_range*3) < h2_wav2[i] < (max(wav_aa)-wav_range*2):
				if i%2 == 0:
					#ax3.text(h2_wav2[i]+0.2, -0.5, h2_name2[i], fontsize=10,color="#a50f15")
					ax4.axvline(h2_wav2[i], ymin=0.0, ymax=0.2, linestyle="dashed", color="#a50f15", linewidth=1.0)
				else:
					#ax3.text(h2_wav2[i]+0.2, -0.8, h2_name2[i], fontsize=10,color="#a50f15")
					ax4.axvline(h2_wav2[i], ymin=0.0, ymax=0.2, linestyle="dashed", color="#a50f15", linewidth=1.0)
			if (max(wav_aa)-wav_range*2) < h2_wav2[i] < (max(wav_aa)-wav_range*1):
				if i%2 == 0:
					#ax3.text(h2_wav2[i]+0.2, -0.5, h2_name2[i], fontsize=10,color="#a50f15")
					ax3.axvline(h2_wav2[i], ymin=0.0, ymax=0.2, linestyle="dashed", color="#a50f15", linewidth=1.0)
				else:
					#ax3.text(h2_wav2[i]+0.2, -0.8, h2_name2[i], fontsize=10,color="#a50f15")
					ax3.axvline(h2_wav2[i], ymin=0.0, ymax=0.2, linestyle="dashed", color="#a50f15", linewidth=1.0)
			if (max(wav_aa)-wav_range*1) < h2_wav2[i] < max(wav_aa):
				if i%2 == 0:
					#ax2.text(h2_wav2[i]+0.2, -0.5, h2_name2[i], fontsize=10,color="#a50f15")
					ax2.axvline(h2_wav2[i], ymin=0.0, ymax=0.2, linestyle="dashed", color="#a50f15", linewidth=1.0)
				else:
					#ax2.text(h2_wav2[i]+0.2, -0.8, h2_name2[i], fontsize=10,color="#a50f15")
					ax2.axvline(h2_wav2[i], ymin=0.0, ymax=0.2, linestyle="dashed", color="#a50f15", linewidth=1.0)

for i in np.arange(0, len(h2_name3), 1):

	if not "6" in h2_name[i]:
		if not "7" in h2_name[i]:
			if (max(wav_aa)-wav_range*4) < h2_wav3[i] < (max(wav_aa)-wav_range*3):
				if i%2 == 0:
					#ax3.text(h2_wav2[i]+0.2, -0.5, h2_name2[i], fontsize=10,color="#a50f15")
					ax5.axvline(h2_wav3[i], ymin=0.0, ymax=0.2, linestyle="dotted", color="#a50f15", linewidth=1.0)
				else:
					#ax3.text(h2_wav2[i]+0.2, -0.8, h2_name2[i], fontsize=10,color="#a50f15")
					ax5.axvline(h2_wav3[i], ymin=0.0, ymax=0.2, linestyle="dotted", color="#a50f15", linewidth=1.0)
			if (max(wav_aa)-wav_range*3) < h2_wav3[i] < (max(wav_aa)-wav_range*3):
				if i%2 == 0:
					#ax3.text(h2_wav2[i]+0.2, -0.5, h2_name2[i], fontsize=10,color="#a50f15")
					ax4.axvline(h2_wav3[i], ymin=0.0, ymax=0.2, linestyle="dotted", color="#a50f15", linewidth=1.0)
				else:
					#ax3.text(h2_wav2[i]+0.2, -0.8, h2_name2[i], fontsize=10,color="#a50f15")
					ax4.axvline(h2_wav3[i], ymin=0.0, ymax=0.2, linestyle="dotted", color="#a50f15", linewidth=1.0)	
			if (max(wav_aa)-wav_range*2) < h2_wav3[i] < (max(wav_aa)-wav_range*1):
				if i%2 == 0:
					#ax3.text(h2_wav2[i]+0.2, -0.5, h2_name2[i], fontsize=10,color="#a50f15")
					ax3.axvline(h2_wav3[i], ymin=0.0, ymax=0.2, linestyle="dotted", color="#a50f15", linewidth=1.0)
				else:
					#ax3.text(h2_wav2[i]+0.2, -0.8, h2_name2[i], fontsize=10,color="#a50f15")
					ax3.axvline(h2_wav3[i], ymin=0.0, ymax=0.2, linestyle="dotted", color="#a50f15", linewidth=1.0)
			if (max(wav_aa)-wav_range*1) < h2_wav3[i] < max(wav_aa):
				if i%2 == 0:
					#ax2.text(h2_wav2[i]+0.2, -0.5, h2_name2[i], fontsize=10,color="#a50f15")
					ax2.axvline(h2_wav3[i], ymin=0.0, ymax=0.2, linestyle="dotted", color="#a50f15", linewidth=1.0)
				else:
					#ax2.text(h2_wav2[i]+0.2, -0.8, h2_name2[i], fontsize=10,color="#a50f15")
					ax2.axvline(h2_wav3[i], ymin=0.0, ymax=0.2, linestyle="dotted", color="#a50f15", linewidth=1.0)

ax3.axvline(1215.67*(2.2310+1), ymin=0.0, ymax=0.2, linestyle="-", color="blue", linewidth=2.0)
ax3.axvline(1215.67*(2.2292+1), ymin=0.0, ymax=0.2, linestyle="-", color="red", linewidth=2.0)

ax3.text(1215.67*(2.2310+1)+0.2, 0.7, r"$\sf Ly\alpha,\, 2.231$", fontsize=10,color="blue", rotation=55)
#ax3.text(1215.67*(2.2310+1)+0.2, -0.4, r"$\sf 2.231$", fontsize=10,color="blue", rotation=45)


ax3.text(1215.67*(2.229+1)+0.2, 0.7, r"$\sf Ly\alpha,\, 2.229$", fontsize=10,color="red", rotation=55)
#ax3.text(1215.67*(2.229+1)+0.2, -0.4, r"$\sf 2.229$", fontsize=10,color="red", rotation=45)


ax3.axhline(2.00, xmin=0.92, xmax=0.97,linestyle="-", color="#a50f15", linewidth=1.2)
ax3.axhline(1.75, xmin=0.92, xmax=0.97,linestyle="dashed", color="#a50f15", linewidth=1.2)
ax3.axhline(1.50, xmin=0.92, xmax=0.97,linestyle="dotted", color="#a50f15", linewidth=1.2)

ax3.text(4348, 1.9,  r"$\sf I$",fontsize=11,color="#a50f15")
ax3.text(4348, 1.63, r"$\sf II$",fontsize=11,color="#a50f15")
ax3.text(4348, 1.36, r"$\sf III$",fontsize=11,color="#a50f15")

lg = ax2.legend(numpoints=1, fontsize=10, loc=1)

ax2.set_xlabel(r"$\sf Observed\, Wavelength (\AA)$", fontsize=24, color="black")
ax.set_ylabel(r"$\sf Normalized\, Flux$", fontsize=24, color="black")
#ax1.axhline(1, color="#2171b5", linewidth=2)

#ax6.set_xlim([min(wav_aa), max(wav_aa)-wav_range*4])
ax5.set_xlim([max(wav_aa)-wav_range*4, max(wav_aa)-wav_range*3])
ax4.set_xlim([max(wav_aa)-wav_range*3, max(wav_aa)-wav_range*2])
ax3.set_xlim([max(wav_aa)-wav_range*2, max(wav_aa)-wav_range*1])
ax2.set_xlim([max(wav_aa)-wav_range*1, max(wav_aa)])

show()
fig.savefig(target + "_H2_fit_spec_pub.pdf", transparent=True)
