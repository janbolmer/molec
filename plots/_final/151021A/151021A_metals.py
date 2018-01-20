#! usr/bin/python

target = "GRB151021A"
redshift = 2.330

w1 = 920.0
w2 = 1116.54

import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *
import numpy as np

sys.path.append('../../../bin/')

from spec_functions import * # spec_functions.py
from syn_spec import * # syn_spec.py
from sns_plots import * # sns_plots.py

from astropy.convolution import Gaussian1DKernel, convolve

#pt_analogous = ["#86af49", "#817397", "#b88bac",
#"#d57f70","#dcb967","#ac9897","#ac898d","#f0e1ce",
#"#86af49", "#817397","#b88bac", "#d57f70", "#dcb967"]


pt_analogous = ["#2b8cbe", "#2b8cbe", "#2b8cbe", "#2b8cbe", "#2b8cbe"]

# constants to calculate the cloumn density
e = 4.8032E-10  # cm 3/2 g 1/2 s-1
c = 2.998e10 # speed of light cm/s
m_e = 9.10938291e-28 # electron mass g

RES_vis = 29.1
RES_uvb = 49.8

def get_line(file, a=1.00):
	y_fit, y_min, y_max, y_min2, y_max2 = [], [], [], [], []
	file = open(file, "r")
	for line in file:
		if not line.startswith("y"):
			s = line.split(",")
			y_fit.append(float(s[0])/a)
			y_min.append(float(s[1])/a)
			y_max.append(float(s[2])/a)
			y_min2.append(float(s[3])/a)
			y_max2.append(float(s[4])/a)
	file.close()

	return y_fit, y_min, y_max, y_min2, y_max2

def add_abs_velo(v, N, b, gamma, f, l0):
	'''
	Add absorption line l0 in velocity space v, given the oscillator strength,
	the damping constant gamma, column density N, and broadening b
	'''
	A = (((np.pi*e**2)/(m_e*c))*f*l0*1E-13) * (10**N)
	tau = A * voigt(v,b/np.sqrt(2.0),gamma)

	return np.exp(-tau)


sns.set_style("white", {'legend.frameon': True})

wav_aa_vis, n_flux_vis, n_flux_err_vis, flux_vis, flux_err_vis, grb_name, res, psf_fwhm = get_data("../../../spectra/GRB151021A_comVIS.txt", redshift)
wav_aa_uvb, n_flux_uvb, n_flux_err_uvb, flux_uvb, flux_err_vis, grb_name, res, psf_fwhm = get_data("../../../spectra/GRB151021A_comUVB.txt", redshift)

l0_1 = 1611.2004
gamma1 =  0.0715441117241
f1 = 0.00135754
transform_1 = 11.1761541809
wav_range_1 = 7
velocity_1, fluxv_1, fluxv_err_1 = aa_to_velo(wav_aa_uvb, n_flux_uvb/1.04870776634, n_flux_err_uvb, l0_1, redshift, wav_range_1)
y_fit_1, y_min_1, y_max_1, y_min2_1, y_max2_1 = get_line("lines/GRB151021A_1_1611.2004_fit.dat", a=1.04870776634)


l0_2 = 1709.6001
gamma2 =  0.0777909045642
f2 = 0.03238247
transform_2 = 10.5328866597
wav_range_2 = 8
velocity_2, fluxv_2, fluxv_err_2 = aa_to_velo(wav_aa_vis, n_flux_vis/0.996034030858, n_flux_err_vis, l0_2, redshift, wav_range_2)
y_fit_2, y_min_2, y_max_2, y_min2_2, y_max2_2 = get_line("lines/GRB151021A_1_1709.6001_fit.dat", a=0.996034030858)


l0_3 = 1741.5486
gamma3 =  0.0879479664794
f3 = 0.04269857
transform_3 = 10.3396621184
wav_range_3 = 8
velocity_3, fluxv_3, fluxv_err_3 = aa_to_velo(wav_aa_vis, n_flux_vis/1.04587070419, n_flux_err_vis, l0_3, redshift, wav_range_3)
y_fit_3, y_min_3, y_max_3, y_min2_3, y_max2_3 = get_line("lines/GRB151021A_1_1741.5486_fit.dat", a=1.04587070419)


l0_4 = 1808.0129
gamma4 =  0.000684854972697
f4 = 0.00208
transform_4 = 9.95956615503
wav_range_4 = 8
velocity_4, fluxv_4, fluxv_err_4 = aa_to_velo(wav_aa_vis, n_flux_vis/1.02228921287, n_flux_err_vis, l0_4, redshift, wav_range_4)
y_fit_4, y_min_4, y_max_4, y_min2_4, y_max2_4 = get_line("lines/GRB151021A_1_1808.0129_fit.dat", a=1.02228921287)


l0_5 = 2056.2568
gamma5 =  0.133196217632
f5 = 0.103
transform_5 = 8.75718640138
wav_range_5 = 9
velocity_5, fluxv_5, fluxv_err_5 = aa_to_velo(wav_aa_vis, n_flux_vis/1.02529737601, n_flux_err_vis, l0_5, redshift, wav_range_5)
y_fit_5, y_min_5, y_max_5, y_min2_5, y_max2_5 = get_line("lines/GRB151021A_1_2056.2568_fit.dat", a=1.02529737601)


l0_6 = 2249.8754
gamma6 =  0.111362504079
f6 = 0.00218568
transform_6 = 8.00356503596
wav_range_6 = 10
velocity_6, fluxv_6, fluxv_err_6 = aa_to_velo(wav_aa_vis, n_flux_vis/1.0094434451, n_flux_err_vis, l0_6, redshift, wav_range_6)
y_fit_6, y_min_6, y_max_6, y_min2_6, y_max2_6 = get_line("lines/GRB151021A_1_2249.8754_fit.dat", a=1.0094434451)


l0_7 = 2260.7793
gamma7 =  0.0917526212129
f7 = 0.00262071
transform_7 = 7.96496327028
wav_range_7 = 10
velocity_7, fluxv_7, fluxv_err_7 = aa_to_velo(wav_aa_vis, n_flux_vis/0.973008363872, n_flux_err_vis, l0_7, redshift, wav_range_7)
y_fit_7, y_min_7, y_max_7, y_min2_7, y_max2_7 = get_line("lines/GRB151021A_1_2260.7793_fit.dat", a=0.973008363872)


l0_8 = 2576.8753
gamma8 =  0.115654528567
f8 = 0.361
transform_8 = 6.98792995013
wav_range_8 = 10
velocity_8, fluxv_8, fluxv_err_8 = aa_to_velo(wav_aa_vis, n_flux_vis/1.03753479931, n_flux_err_vis, l0_8, redshift, wav_range_8)
y_fit_8, y_min_8, y_max_8, y_min2_8, y_max2_8 = get_line("lines/GRB151021A_1_2576.8753_fit.dat", a=1.03753479931)


l0_9 = 2594.4967
gamma9 =  0.11479369895
f9 = 0.28
transform_9 = 6.94046906539
wav_range_9 = 10
velocity_9, fluxv_9, fluxv_err_9 = aa_to_velo(wav_aa_vis, n_flux_vis/1.00620132192, n_flux_err_vis, l0_9, redshift, wav_range_9)
y_fit_9, y_min_9, y_max_9, y_min2_9, y_max2_9 = get_line("lines/GRB151021A_1_2594.4967_fit.dat", a=1.00620132192)


l0_10 = 2606.4588
gamma10 =  0.11283397814
f10 = 0.198
transform_10 = 6.90861642878
wav_range_10 = 10
velocity_10, fluxv_10, fluxv_err_10 = aa_to_velo(wav_aa_vis, n_flux_vis/0.992815977609, n_flux_err_vis, l0_10, redshift, wav_range_10)
y_fit_10, y_min_10, y_max_10, y_min2_10, y_max2_10 = get_line("lines/GRB151021A_1_2606.4588_fit.dat", a=0.992815977609)


l0_11 = 2612.6536
gamma11 =  0.112353681804
f11 = 0.12485329
transform_11 = 6.89223557483
wav_range_11 = 10
velocity_11, fluxv_11, fluxv_err_11 = aa_to_velo(wav_aa_vis, n_flux_vis/1.0175381799, n_flux_err_vis, l0_11, redshift, wav_range_11)
y_fit_11, y_min_11, y_max_11, y_min2_11, y_max2_11 = get_line("lines/GRB151021A_1_2612.6536_fit.dat", a=1.0175381799)



l0_12 = 2026.1369
gamma12 =  0.131245169127
f12 = 0.501
transform_12 = 8.88736792005
wav_range_12 = 9
velocity_12, fluxv_12, fluxv_err_12 = aa_to_velo(wav_aa_vis, n_flux_vis/1.00018352338, n_flux_err_vis, l0_12, redshift, wav_range_12)
y_fit_12, y_min_12, y_max_12, y_min2_12, y_max2_12 = get_line("lines/GRB151021A_1_2026.1369_fit.dat", a=1.00018352338)



fig = figure(figsize=(8, 12))

ax = fig.add_axes([0.10, 0.09, 0.88, 0.15*6])
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

ax.set_ylabel("Normalized Flux", fontsize=20, labelpad=10)
ax.set_xlabel("Relative Velocity (km/s)", fontsize=20, labelpad=20)

ax1 = fig.add_axes([0.10, 0.09, 0.44, 0.15])
ax2 = fig.add_axes([0.54, 0.09, 0.44, 0.15])
ax3 = fig.add_axes([0.10, 0.24, 0.44, 0.15])
ax4 = fig.add_axes([0.54, 0.24, 0.44, 0.15])
ax5 = fig.add_axes([0.10, 0.39, 0.44, 0.15])
ax6 = fig.add_axes([0.54, 0.39, 0.44, 0.15])
ax7 = fig.add_axes([0.10, 0.54, 0.44, 0.15])
ax8 = fig.add_axes([0.54, 0.54, 0.44, 0.15])
ax9 = fig.add_axes([0.10, 0.69, 0.44, 0.15])
ax10 = fig.add_axes([0.54, 0.69, 0.44, 0.15])
ax11 = fig.add_axes([0.10, 0.84, 0.44, 0.15])
ax12 = fig.add_axes([0.54, 0.84, 0.44, 0.15])



ax1.errorbar(velocity_1,fluxv_1,yerr=fluxv_err_1, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax1.plot(velocity_1,fluxv_1, drawstyle='steps-mid',color='gray', alpha=0.66)
ax1.plot(velocity_1, y_fit_1,label='Fit',color=pt_analogous[0],linewidth=2)
ax1.fill_between(velocity_1, y_min_1, y_max_1, color=pt_analogous[0],alpha=0.3)
ax1.fill_between(velocity_1, y_min2_1, y_max2_1, color=pt_analogous[0],alpha=0.05)
ax1.text(145, 0.05, "Fe 1611.2004", fontsize=10)
#Component 1 : b: 29.52 +/- 2.62 N: 13.7 +/- 0.02
#Components:  -60.1746310694 
#2.70925532522 
N1 = [15.85]
b1 = [35.59]
b1_err = [8.01]
N1_err = [0.10]
v_01 = [-3.88698768467]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_uvb/(2*np.sqrt(2*np.log(2))*transform_1),
			mode="oversample")
	ff = np.ones(len(velocity_1))
	v = np.array(velocity_1) - v_01[i]
	ff *= np.convolve(add_abs_velo(v,N1[i],b1[i],gamma1,f1,l0_1),gauss_k,mode='same')

	#ax1.plot(velocity_1,ff,color=pt_analogous[i+2],linewidth=2)
	#ax1.text(-310,0.05-float(i)/8,"b: "+str(b1[i]) + " +/- " + str(b1_err[i]) + 
	#	", N: "+str(N1[i]) + " +/- " + str(N1_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax1.axvline(v_01[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)


ax2.errorbar(velocity_2,fluxv_2,yerr=fluxv_err_2, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax2.plot(velocity_2,fluxv_2, drawstyle='steps-mid',color='gray', alpha=0.66)
ax2.plot(velocity_2, y_fit_2,label='Fit',color=pt_analogous[0],linewidth=2)
ax2.fill_between(velocity_2, y_min_2, y_max_2, color=pt_analogous[0],alpha=0.3)
ax2.fill_between(velocity_2, y_min2_2, y_max2_2, color=pt_analogous[0],alpha=0.05)
ax2.text(145, 0.05, "NiII 1709.6011", fontsize=10)
#Component 1 : b: 29.52 +/- 2.62 N: 13.7 +/- 0.02
#Components:  -60.1746310694 
#2.70925532522 
N2 = [14.68]
b2 = [32.48]
b2_err = [9.32]
N2_err = [0.22]
v_02 = [4.61002873681]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_2),
			mode="oversample")
	ff = np.ones(len(velocity_2))
	v = np.array(velocity_2) - v_02[i]
	ff *= np.convolve(add_abs_velo(v,N2[i],b2[i],gamma2,f2,l0_2),gauss_k,mode='same')

	#ax2.plot(velocity_2,ff,color=pt_analogous[i+2],linewidth=2)
	#ax2.text(-310,0.05-float(i)/8,"b: "+str(b2[i]) + " +/- " + str(b2_err[i]) + 
	#	", N: "+str(N2[i]) + " +/- " + str(N2_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax2.axvline(v_02[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)




ax3.errorbar(velocity_3,fluxv_3,yerr=fluxv_err_3, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax3.plot(velocity_3,fluxv_3, drawstyle='steps-mid',color='gray', alpha=0.66)
ax3.plot(velocity_3, y_fit_3,label='Fit',color=pt_analogous[0],linewidth=2)
ax3.fill_between(velocity_3, y_min_3, y_max_3, color=pt_analogous[0],alpha=0.3)
ax3.fill_between(velocity_3, y_min2_3, y_max2_3, color=pt_analogous[0],alpha=0.03)
ax3.text(145, 0.05, "NiII 1741.5486", fontsize=10)
#Component 1 : b: 27.89 +/- 9.91 N: 14.82 +/- 0.38
#Background a = 1.04587070419
#Components:  0.597958801047 
N3 = [14.82]
b3 = [27.89]
b3_err = [9.91]
N3_err = [0.38]
v_03 = [0.597958801047]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_3),
			mode="oversample")
	ff = np.ones(len(velocity_3))
	v = np.array(velocity_3) - v_03[i]
	ff *= np.convolve(add_abs_velo(v,N2[i],b2[i],gamma3,f3,l0_3),gauss_k,mode='same')

	#ax3.plot(velocity_3,ff,color=pt_analogous[i+2],linewidth=2)
	#ax3.text(-310,0.05-float(i)/8,"b: "+str(b3[i]) + " +/- " + str(b3_err[i]) + 
	#	", N: "+str(N3[i]) + " +/- " + str(N3_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax3.axvline(v_03[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)




ax4.errorbar(velocity_4,fluxv_4,yerr=fluxv_err_4, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax4.plot(velocity_4,fluxv_4, drawstyle='steps-mid',color='gray', alpha=0.66)
ax4.plot(velocity_4, y_fit_4,label='Fit',color=pt_analogous[0],linewidth=2)
ax4.fill_between(velocity_4, y_min_4, y_max_4, color=pt_analogous[0],alpha=0.3)
ax4.fill_between(velocity_4, y_min2_4, y_max2_4, color=pt_analogous[0],alpha=0.05)
ax4.text(145, 0.05, "SiII 1808.0129", fontsize=10)
#Component 1 : b: 30.53 +/- 10.85 N: 17.35 +/- 1.23
#Background a = 1.02228921287
#Components:  2.15362622359 
N4 = [17.35]
b4 = [30.53]
b4_err = [10.81]
N4_err = [1.23]
v_04 = [2.15362622359]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_4),
			mode="oversample")
	ff = np.ones(len(velocity_4))
	v = np.array(velocity_4) - v_04[i]
	ff *= np.convolve(add_abs_velo(v,N4[i],b4[i],gamma4,f4,l0_4),gauss_k,mode='same')

	#ax4.plot(velocity_4,ff,color=pt_analogous[i+2],linewidth=2)
	#ax4.text(-310,0.05-float(i)/8,"b: "+str(b4[i]) + " +/- " + str(b4_err[i]) + 
	#	", N: "+str(N4[i]) + " +/- " + str(N4_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax4.axvline(v_04[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)


ax5.errorbar(velocity_5,fluxv_5,yerr=fluxv_err_5, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax5.plot(velocity_5,fluxv_5, drawstyle='steps-mid',color='gray', alpha=0.66)
ax5.plot(velocity_5, y_fit_5,label='Fit',color=pt_analogous[0],linewidth=2)
ax5.fill_between(velocity_5, y_min_5, y_max_5, color=pt_analogous[0],alpha=0.3)
ax5.fill_between(velocity_5, y_min2_5, y_max2_5, color=pt_analogous[0],alpha=0.05)
ax5.text(145, 0.05, "CrII 2056.2568", fontsize=10)
#Component 1 : b: 29.11 +/- 5.39 N: 14.27 +/- 0.11
#Background a = 1.02529737601
#Components:  -2.68079744451 
N5 = [14.27]
b5 = [29.11]
b5_err = [5.39]
N5_err = [0.11]
v_05 = [-2.68079744451]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_5),
			mode="oversample")
	ff = np.ones(len(velocity_5))
	v = np.array(velocity_5) - v_05[i]
	ff *= np.convolve(add_abs_velo(v,N5[i],b5[i],gamma5,f5,l0_5),gauss_k,mode='same')

	#ax5.plot(velocity_5,ff,color=pt_analogous[i+2],linewidth=2)
	#ax5.text(-310,0.05-float(i)/8,"b: "+str(b5[i]) + " +/- " + str(b5_err[i]) + 
	#	", N: "+str(N5[i]) + " +/- " + str(N5_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax5.axvline(v_05[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)




ax6.errorbar(velocity_6,fluxv_6,yerr=fluxv_err_6, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax6.plot(velocity_6,fluxv_6, drawstyle='steps-mid',color='gray', alpha=0.66)
ax6.plot(velocity_6, y_fit_6,label='Fit',color=pt_analogous[0],linewidth=2)
ax6.fill_between(velocity_6, y_min_6, y_max_6, color=pt_analogous[0],alpha=0.3)
ax6.fill_between(velocity_6, y_min2_6, y_max2_6, color=pt_analogous[0],alpha=0.05)
ax6.text(145, 0.05, "FeII 2249.8754", fontsize=10)
#Component 1 : b: 28.19 +/- 4.41 N: 15.8 +/- 0.06
#Background a = 1.0094434451
#Components:  3.03362056965 
N6 = [15.80]
b6 = [28.19]
b6_err = [4.41]
N6_err = [0.06]
v_06 = [3.03362056965]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_6),
			mode="oversample")
	ff = np.ones(len(velocity_6))
	v = np.array(velocity_6) - v_06[i]
	ff *= np.convolve(add_abs_velo(v,N6[i],b6[i],gamma6,f6,l0_6),gauss_k,mode='same')

	#ax6.plot(velocity_6,ff,color=pt_analogous[i+2],linewidth=2)
	#ax6.text(-310,0.05-float(i)/8,"b: "+str(b6[i]) + " +/- " + str(b6_err[i]) + 
	#	", N: "+str(N6[i]) + " +/- " + str(N6_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax6.axvline(v_06[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)






ax7.errorbar(velocity_7,fluxv_7,yerr=fluxv_err_7, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax7.plot(velocity_7,fluxv_7, drawstyle='steps-mid',color='gray', alpha=0.66)
ax7.plot(velocity_7, y_fit_7,label='Fit',color=pt_analogous[0],linewidth=2)
ax7.fill_between(velocity_7, y_min_7, y_max_7, color=pt_analogous[0],alpha=0.3)
ax7.fill_between(velocity_7, y_min2_7, y_max2_7, color=pt_analogous[0],alpha=0.05)
ax7.text(145, 0.05, "FeII 2260.7793", fontsize=10)
#Component 1 : b: 25.55 +/- 5.17 N: 16.0 +/- 0.25
#Background a = 0.973008363872
#Components:  1.21277322194 
N7 = [16.00]
b7 = [25.55]
b7_err = [5.17]
N7_err = [0.25]
v_07 = [1.21277322194]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_7),
			mode="oversample")
	ff = np.ones(len(velocity_7))
	v = np.array(velocity_7) - v_07[i]
	ff *= np.convolve(add_abs_velo(v,N7[i],b7[i],gamma7,f7,l0_7),gauss_k,mode='same')

	#ax7.plot(velocity_7,ff,color=pt_analogous[i+2],linewidth=2)
	#ax7.text(-310,0.05-float(i)/8,"b: "+str(b7[i]) + " +/- " + str(b7_err[i]) + 
	#	", N: "+str(N7[i]) + " +/- " + str(N7_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax7.axvline(v_07[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)


ax8.errorbar(velocity_8,fluxv_8,yerr=fluxv_err_8, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax8.plot(velocity_8,fluxv_8, drawstyle='steps-mid',color='gray', alpha=0.66)
ax8.plot(velocity_8, y_fit_8,label='Fit',color=pt_analogous[0],linewidth=2)
ax8.fill_between(velocity_8, y_min_8, y_max_8, color=pt_analogous[0],alpha=0.3)
ax8.fill_between(velocity_8, y_min2_8, y_max2_8, color=pt_analogous[0],alpha=0.05)
ax8.text(145, 0.05, "MnII 2576.8753", fontsize=10)
#Component 1 : b: 16.71 +/- 4.02 N: 14.91 +/- 0.53
#Background a = 1.03753479931
#Components:  3.08136832674 
N8 = [14.91]
b8 = [16.71]
b8_err = [4.02]
N8_err = [0.53]
v_08 = [3.08136832674]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_8),
			mode="oversample")
	ff = np.ones(len(velocity_8))
	v = np.array(velocity_8) - v_08[i]
	ff *= np.convolve(add_abs_velo(v,N8[i],b8[i],gamma8,f8,l0_8),gauss_k,mode='same')

	#ax7.plot(velocity_7,ff,color=pt_analogous[i+2],linewidth=2)
	#ax8.text(-310,0.05-float(i)/8,"b: "+str(b8[i]) + " +/- " + str(b8_err[i]) + 
	#	", N: "+str(N8[i]) + " +/- " + str(N8_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax8.axvline(v_08[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)



ax9.errorbar(velocity_9,fluxv_9,yerr=fluxv_err_9, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax9.plot(velocity_9,fluxv_9, drawstyle='steps-mid',color='gray', alpha=0.66)
ax9.plot(velocity_9, y_fit_9,label='Fit',color=pt_analogous[0],linewidth=2)
ax9.fill_between(velocity_9, y_min_9, y_max_9, color=pt_analogous[0],alpha=0.3)
ax9.fill_between(velocity_9, y_min2_9, y_max2_9, color=pt_analogous[0],alpha=0.05)
ax9.text(145, 0.05, "MnII 2594.4967", fontsize=10)
#Component 1 : b: 23.19 +/- 4.77 N: 14.12 +/- 0.37
#Background a = 1.00620132192
#Components:  2.06884974304 
N9 = [14.12]
b9 = [23.19]
b9_err = [4.77]
N9_err = [0.37]
v_09 = [2.06884974304]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_9),
			mode="oversample")
	ff = np.ones(len(velocity_9))
	v = np.array(velocity_9) - v_09[i]
	ff *= np.convolve(add_abs_velo(v,N9[i],b9[i],gamma9,f9,l0_9),gauss_k,mode='same')

	#ax7.plot(velocity_7,ff,color=pt_analogous[i+2],linewidth=2)
	#ax9.text(-310,0.05-float(i)/8,"b: "+str(b9[i]) + " +/- " + str(b9_err[i]) + 
	#	", N: "+str(N9[i]) + " +/- " + str(N9_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax9.axvline(v_09[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)


ax10.errorbar(velocity_10,fluxv_10,yerr=fluxv_err_10, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax10.plot(velocity_10,fluxv_10, drawstyle='steps-mid',color='gray', alpha=0.66)
ax10.plot(velocity_10, y_fit_10,label='Fit',color=pt_analogous[0],linewidth=2)
ax10.fill_between(velocity_10, y_min_10, y_max_10, color=pt_analogous[0],alpha=0.3)
ax10.fill_between(velocity_10, y_min2_10, y_max2_10, color=pt_analogous[0],alpha=0.05)
ax10.text(145, 0.05, "MnII 2606.4588", fontsize=10)
#Component 1 : b: 19.53 +/- 4.25 N: 14.3 +/- 0.36
#Background a = 0.992815977609
#Components:  2.92405720578 
N10 = [14.30]
b10 = [19.53]
b10_err = [4.25]
N10_err = [0.36]
v_010 = [2.92405720578]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_10),
			mode="oversample")
	ff = np.ones(len(velocity_10))
	v = np.array(velocity_10) - v_010[i]
	ff *= np.convolve(add_abs_velo(v,N10[i],b10[i],gamma10,f10,l0_10),gauss_k,mode='same')

	#ax7.plot(velocity_7,ff,color=pt_analogous[i+2],linewidth=2)
	#ax10.text(-310,0.05-float(i)/8,"b: "+str(b10[i]) + " +/- " + str(b10_err[i]) + 
	#	", N: "+str(N10[i]) + " +/- " + str(N10_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax10.axvline(v_010[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)



ax11.errorbar(velocity_11,fluxv_11,yerr=fluxv_err_11, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax11.plot(velocity_11,fluxv_11, drawstyle='steps-mid',color='gray', alpha=0.66)
ax11.plot(velocity_11, y_fit_11,label='Fit',color=pt_analogous[0],linewidth=2)
ax11.fill_between(velocity_11, y_min_11, y_max_11, color=pt_analogous[0],alpha=0.3)
ax11.fill_between(velocity_11, y_min2_11, y_max2_11, color=pt_analogous[0],alpha=0.05)
ax11.text(140, 0.05, "FeIIa 2612.6536", fontsize=10)
#Component 1 : b: 29.89 +/- 10.86 N: 13.5 +/- 0.08
#Background a = 1.0175381799
#Components:  -10.5641688687  
N11 = [13.60]
b11 = [29.89]
b11_err = [10.86]
N11_err = [0.08]
v_011 = [-10.5641688687]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_11),
			mode="oversample")
	ff = np.ones(len(velocity_11))
	v = np.array(velocity_11) - v_011[i]
	ff *= np.convolve(add_abs_velo(v,N11[i],b11[i],gamma11,f11,l0_11),gauss_k,mode='same')

	#ax7.plot(velocity_7,ff,color=pt_analogous[i+2],linewidth=2)
	#ax11.text(-310,0.05-float(i)/8,"b: "+str(b11[i]) + " +/- " + str(b11_err[i]) + 
	#	", N: "+str(N11[i]) + " +/- " + str(N11_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax11.axvline(v_011[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)



ax12.errorbar(velocity_12,fluxv_12,yerr=fluxv_err_12, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax12.plot(velocity_12,fluxv_12, drawstyle='steps-mid',color='gray', alpha=0.66)
ax12.plot(velocity_12, y_fit_12,label='Fit',color=pt_analogous[0],linewidth=2)
ax12.fill_between(velocity_12, y_min_12, y_max_12, color=pt_analogous[0],alpha=0.3)
ax12.fill_between(velocity_12, y_min2_12, y_max2_12, color=pt_analogous[0],alpha=0.05)
ax12.text(145, 0.05, "ZnII 2026.1369", fontsize=10)
#Component 1 : b: 39.0 +/- 5.13 N: 13.87 +/- 0.12
#Background a = 1.00018352338
#Components:  9.10824901378 
N12 = [13.87]
b12 = [39.0]
b12_err = [5.13]
N12_err = [0.12]
v_012 = [9.10824901378]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_12),
			mode="oversample")
	ff = np.ones(len(velocity_12))
	v = np.array(velocity_12) - v_012[i]
	ff *= np.convolve(add_abs_velo(v,N12[i],b12[i],gamma12,f12,l0_12),gauss_k,mode='same')

	#ax7.plot(velocity_7,ff,color=pt_analogous[i+2],linewidth=2)
	#ax11.text(-310,0.05-float(i)/8,"b: "+str(b11[i]) + " +/- " + str(b11_err[i]) + 
	#	", N: "+str(N11[i]) + " +/- " + str(N11_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax12.axvline(v_012[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)




for axis in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8,
	ax9, ax10, ax11, ax12]:

	axis.set_xlim([-330, 330])
	axis.set_ylim([-0.15, 1.45])


	#axis.axhline(1.0,xmin=0.0, xmax=1.0, linewidth=1,linestyle="dotted",color="black", alpha=0.5)
	axis.axhline(0.0,xmin=0.0, xmax=1.0, linewidth=1,linestyle="dotted",color="black")
	
	if not axis in [ax1, ax2]:
		for tick in axis.xaxis.get_major_ticks():
			tick.label.set_fontsize(0)
	else:
		axis.set_xticks([-300, -200, -100, 0, 100, 200, 300])
		for side in ['bottom']:
		  	axis.spines[side].axis.axes.tick_params(which='major', length=6, width=1, axis='x')
			#axis.tick_params[side](which='major', length=6, width=1)


	if not axis in [ax1, ax3, ax5, ax7, ax9, ax11]:
		for tick in axis.yaxis.get_major_ticks():
			tick.label.set_fontsize(0)
	else:
		axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
		for side in ['right']:
		  	axis.spines[side].axis.axes.tick_params(which='major', length=4, width=1, axis='y')


show()
fig.savefig(target + "_metals.pdf")
