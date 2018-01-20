#! usr/bin/python

target = "GRB161023A"
redshift = 2.710

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

pt_analogous = ["#86af49", "#817397", "#b88bac",
"#d57f70","#dcb967","#ac9897","#ac898d","#f0e1ce",
"#86af49", "#817397","#b88bac", "#d57f70", "#dcb967"]

# constants to calculate the cloumn density
e = 4.8032E-10  # cm 3/2 g 1/2 s-1
c = 2.998e10 # speed of light cm/s
m_e = 9.10938291e-28 # electron mass g

RES_vis = 26.5
RES_uvb = 46.2

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

wav_aa_vis, n_flux_vis, n_flux_err_vis, flux_vis, flux_err_vis, grb_name, res, psf_fwhm = get_data("../../../spectra/GRB161023A_OB1VIS.txt", redshift)
wav_aa_uvb, n_flux_uvb, n_flux_err_uvb, flux_uvb, flux_err_vis, grb_name, res, psf_fwhm = get_data("../../../spectra/GRB161023A_OB1UVB.txt", redshift)

l0_1 = 1238.8210
gamma1 =  0.0670359251571
f1 = 0.1563
transform_1 = 13.0457253232
wav_range_1 = 6
velocity_1, fluxv_1, fluxv_err_1 = aa_to_velo(wav_aa_uvb, n_flux_uvb/0.945446452374, n_flux_err_uvb, l0_1, redshift, wav_range_1)
y_fit_1, y_min_1, y_max_1, y_min2_1, y_max2_1 = get_line("lines/GRB161023A_1_1238.821_fit.dat", a=0.945446452374)

l0_2 = 1242.8040
gamma2 =  0.0664602623645
f2 = 0.07783
transform_2 = 13.0039157345
wav_range_2 = 6
velocity_2, fluxv_2, fluxv_err_2 = aa_to_velo(wav_aa_uvb, n_flux_uvb/0.968870812394, n_flux_err_uvb, l0_2, redshift, wav_range_2)
y_fit_2, y_min_2, y_max_2, y_min2_2, y_max2_2 = get_line("lines/GRB161023A_1_1242.804_fit.dat", a=0.968870812394)

l0_3 = 1253.8110
gamma3 =  0.00921922008791
f3 = 0.0109
transform_3 = 12.8897565028
wav_range_3 = 7
velocity_3, fluxv_3, fluxv_err_3 = aa_to_velo(wav_aa_uvb, n_flux_uvb/1.00159974692, n_flux_err_uvb, l0_3, redshift, wav_range_3)
y_fit_3, y_min_3, y_max_3, y_min2_3, y_max2_3 = get_line("lines/GRB161023A_3_1253.811_fit.dat", a=1.00159974692)

l0_4 = 1302.1685
gamma4 =  0.117094302735
f4 = 0.048
transform_4 = 12.4110808168
wav_range_4 = 6
velocity_4, fluxv_4, fluxv_err_4 = aa_to_velo(wav_aa_uvb, n_flux_uvb/1.00434513073, n_flux_err_uvb, l0_4, redshift, wav_range_4)
y_fit_4, y_min_4, y_max_4, y_min2_4, y_max2_4 = get_line("lines/GRB161023A_3_1302.1685_fit.dat", a=1.00434513073)

l0_5 = 1393.7602
gamma5 =  0.195204966277
f5 = 0.513
transform_5 = 11.5954799761
wav_range_5 = 12
velocity_5, fluxv_5, fluxv_err_5 = aa_to_velo(wav_aa_uvb, n_flux_uvb/1.00193254846, n_flux_err_uvb, l0_5, redshift, wav_range_5)
y_fit_5, y_min_5, y_max_5, y_min2_5, y_max2_5 = get_line("lines/GRB161023A_3_1393.7602_fit.dat", a=1.00193254846)

l0_6 = 1402.7729
gamma6 =   0.192448603803
f6 = 0.254
transform_6 = 11.5209799751
wav_range_6 = 7
velocity_6, fluxv_6, fluxv_err_6 = aa_to_velo(wav_aa_uvb, n_flux_uvb/1.01582040056, n_flux_err_uvb, l0_6, redshift, wav_range_6)
y_fit_6, y_min_6, y_max_6, y_min2_6, y_max2_6 = get_line("lines/GRB161023A_3_1402.7729_fit.dat", a=1.01582040056)

l0_7 = 1608.4509
gamma7 =  0.0623086746483
f7 = 0.05399224
transform_7 = 10.0477537055
wav_range_7 = 8
velocity_7, fluxv_7, fluxv_err_7 = aa_to_velo(wav_aa_vis, n_flux_vis/1.03351245535, n_flux_err_vis, l0_7, redshift, wav_range_7)
y_fit_7, y_min_7, y_max_7, y_min2_7, y_max2_7 = get_line("lines/GRB161023A_3_1608.4509_fit.dat", a=1.03351245535)

l0_8 = 1808.0129
gamma8 =  0.000684854972697
f8 = 0.00208
transform_8 = 8.93871857361
wav_range_8 = 9
velocity_8, fluxv_8, fluxv_err_8 = aa_to_velo(wav_aa_vis, n_flux_vis/1.01073062115, n_flux_err_vis, l0_8, redshift, wav_range_8)
y_fit_8, y_min_8, y_max_8, y_min2_8, y_max2_8 = get_line("lines/GRB161023A_3_1808.0129_fit.dat", a=1.01073062115)

l0_9 = 2260.7793
gamma9 =  0.0917526212129
f9 = 0.00262071
transform_9 = 7.14856089253
wav_range_9 = 9
velocity_9, fluxv_9, fluxv_err_9 = aa_to_velo(wav_aa_vis, n_flux_vis/1.02211810752, n_flux_err_vis, l0_9, redshift, wav_range_9)
y_fit_9, y_min_9, y_max_9, y_min2_9, y_max2_9 = get_line("lines/GRB161023A_2_2260.7793_fit.dat", a=1.02211810752)

l0_10 = 2333.5147
gamma10 =  0.10855741371
f10 = 0.07776113
transform_10 = 6.92574102517
wav_range_10 = 10
velocity_10, fluxv_10, fluxv_err_10 = aa_to_velo(wav_aa_vis, n_flux_vis/1.00875215253, n_flux_err_vis, l0_10, redshift, wav_range_10)
y_fit_10, y_min_10, y_max_10, y_min2_10, y_max2_10 = get_line("lines/GRB161023A_3_2333.5147_fit.dat", a=1.00875215253)

l0_11 = 2344.2129 
gamma11 =  0.108532774265
f11 = 0.12523167
transform_11 = 6.89413427024
wav_range_11 = 11
velocity_11, fluxv_11, fluxv_err_11 = aa_to_velo(wav_aa_vis, n_flux_vis/1.00976670827, n_flux_err_vis, l0_11, redshift, wav_range_11)
y_fit_11, y_min_11, y_max_11, y_min2_11, y_max2_11 = get_line("lines/GRB161023A_3_2344.2129_fit.dat", a=1.00976670827)

l0_12 = 2374.4604 
gamma12 =  0.12448260198
f12 = 0.03296636
transform_12 = 6.80631207437
wav_range_12 = 11
velocity_12, fluxv_12, fluxv_err_12 = aa_to_velo(wav_aa_vis, n_flux_vis/1.018371361, n_flux_err_vis, l0_12, redshift, wav_range_12)
y_fit_12, y_min_12, y_max_12, y_min2_12, y_max2_12 = get_line("lines/GRB161023A_3_2374.4604_fit.dat", a=1.018371361)


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
ax1.fill_between(velocity_1, y_min_1, y_max_1, color=pt_analogous[0],alpha=0.5)
ax1.fill_between(velocity_1, y_min2_1, y_max2_1, color=pt_analogous[0],alpha=0.2)
ax1.text(120, 1.3, "NV 1238.8210", fontsize=10)
#Component 1 : b: 29.52 +/- 2.62 N: 13.7 +/- 0.02
#Components:  -60.1746310694 
#2.70925532522 
N1 = [13.70]
b1 = [29.52]
b1_err = [2.62]
N1_err = [0.02]
v_01 = [-60.1746310694]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_uvb/(2*np.sqrt(2*np.log(2))*transform_1),
			mode="oversample")
	ff = np.ones(len(velocity_1))
	v = np.array(velocity_1) - v_01[i]
	ff *= np.convolve(add_abs_velo(v,N1[i],b1[i],gamma1,f1,l0_1),gauss_k,mode='same')

	ax1.plot(velocity_1,ff,color=pt_analogous[i+2],linewidth=2)
	ax1.text(-270,1.32-float(i)/8,"b: "+str(b1[i]) + " +/- " + str(b1_err[i]) + 
		", N: "+str(N1[i]) + " +/- " + str(N1_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax1.axvline(v_01[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)




ax2.errorbar(velocity_2,fluxv_2,yerr=fluxv_err_2, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax2.plot(velocity_2,fluxv_2, drawstyle='steps-mid',color='gray', alpha=0.66)
ax2.plot(velocity_2, y_fit_2,label='Fit',color=pt_analogous[0],linewidth=2)
ax2.fill_between(velocity_2, y_min_2, y_max_2, color=pt_analogous[0],alpha=0.5)
ax2.fill_between(velocity_2, y_min2_2, y_max2_2, color=pt_analogous[0],alpha=0.2)
ax2.text(120, 1.3, "NV 1242.8040", fontsize=10)
#Component 1 : b: 26.51 +/- 5.25 N: 13.75 +/- 0.03
#Components:  -58.9519673103 
#2.70927045597 
N2 = [13.75]
b2 = [26.51]
N2_err = [0.03]
b2_err = [5.25]
v_02 = [-58.9519673103]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_uvb/(2*np.sqrt(2*np.log(2))*transform_2),
			mode="oversample")
	ff = np.ones(len(velocity_2))
	v = np.array(velocity_2) - v_02[i]
	ff *= np.convolve(add_abs_velo(v,N2[i],b2[i],gamma2,f2,l0_2),gauss_k,mode='same')

	ax2.plot(velocity_2,ff,color=pt_analogous[i+2],linewidth=2)
	ax2.text(-270,1.32-float(i)/8,"b: "+str(b2[i]) + " +/- " + str(b2_err[i]) + 
		", N: "+str(N2[i]) + " +/- " + str(N2_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax2.axvline(v_02[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)





ax3.errorbar(velocity_3,fluxv_3,yerr=fluxv_err_3, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax3.plot(velocity_3,fluxv_3, drawstyle='steps-mid',color='gray', alpha=0.66)
ax3.plot(velocity_3, y_fit_3,label='Fit',color=pt_analogous[0],linewidth=2)
ax3.fill_between(velocity_3, y_min_3, y_max_3, color=pt_analogous[0],alpha=0.5)
ax3.fill_between(velocity_3, y_min2_3, y_max2_3, color=pt_analogous[0],alpha=0.2)
ax3.text(120, 1.3, "SII 1253.8110 ", fontsize=10)
#Component 1 : b: 23.16 +/- 4.17 N: 14.71 +/- 0.03
#Component 2 : b: 24.37 +/- 9.78 N: 14.4 +/- 0.06
#Component 3 : b: 34.56 +/- 11.38 N: 14.29 +/- 0.07
#Components:  53.2551793133  -22.6658681703  -136.787360114 
#2.71065904498 2.70971950472 2.70830722524 
N3 = [14.71, 14.4, 14.29]
b3 = [23.16, 24.37, 34.56]
v_03 = [53.2551793133, -22.6658681703, -136.787360114]
for i in [0, 1, 2]:
	gauss_k = Gaussian1DKernel(stddev=RES_uvb/(2*np.sqrt(2*np.log(2))*transform_3),
			mode="oversample")
	ff = np.ones(len(velocity_3))
	v = np.array(velocity_3) - v_03[i]
	ff *= np.convolve(add_abs_velo(v,N3[i],b3[i],gamma3,f3,l0_3),gauss_k,mode='same')

	ax3.plot(velocity_3,ff,color=pt_analogous[i+2],linewidth=2)
	ax3.text(-270,1.32-float(i)/8,"b = "+str(b3[i]) + ", N = "+str(N3[i]),color=pt_analogous[i+2], fontsize=8)
	ax3.axvline(v_03[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)




ax4.errorbar(velocity_4,fluxv_4,yerr=fluxv_err_4, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax4.plot(velocity_4,fluxv_4, drawstyle='steps-mid',color='gray', alpha=0.66)
ax4.plot(velocity_4, y_fit_4,label='Fit',color=pt_analogous[0],linewidth=2)
ax4.fill_between(velocity_4, y_min_4, y_max_4, color=pt_analogous[0],alpha=0.5)
ax4.fill_between(velocity_4, y_min2_4, y_max2_4, color=pt_analogous[0],alpha=0.2)
ax4.text(120, 1.3, "OI 1302.1685 ", fontsize=10)
#Component 1 : b: 24.89 +/- 2.08 N: 15.35 +/- 0.11
#Component 2 : b: 24.9 +/- 2.44 N: 15.18 +/- 0.07
#Component 3 : b: 22.9 +/- 1.44 N: 14.63 +/- 0.02
#Components:  46.7844574086  -9.84773157653  -131.16582556 
#2.71057896832 2.70987813208 2.70837679301 
N4 = [15.35, 15.18, 14.63]
b4 = [24.89, 24.90, 22.90]
v_04 = [46.7844574086, -9.84773157653, -131.16582556]
for i in [0, 1, 2]:
	gauss_k = Gaussian1DKernel(stddev=RES_uvb/(2*np.sqrt(2*np.log(2))*transform_4),
			mode="oversample")
	ff = np.ones(len(velocity_4))
	v = np.array(velocity_4) - v_04[i]
	ff *= np.convolve(add_abs_velo(v,N4[i],b4[i],gamma4,f4,l0_4),gauss_k,mode='same')

	ax4.plot(velocity_4,ff,color=pt_analogous[i+2],linewidth=2)
	ax4.text(-270,1.32-float(i)/8,"b = "+str(b4[i]) + ", N = "+str(N4[i]),color=pt_analogous[i+2], fontsize=8)
	ax4.axvline(v_04[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)


ax5.errorbar(velocity_5,fluxv_5,yerr=fluxv_err_5, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax5.plot(velocity_5,fluxv_5, drawstyle='steps-mid',color='gray', alpha=0.66)
ax5.plot(velocity_5, y_fit_5,label='Fit',color=pt_analogous[0],linewidth=2)
ax5.fill_between(velocity_5, y_min_5, y_max_5, color=pt_analogous[0],alpha=0.5)
ax5.fill_between(velocity_5, y_min2_5, y_max2_5, color=pt_analogous[0],alpha=0.2)
ax5.text(120, 1.3, "SiIV 1393.7602 ", fontsize=10)
#Component 1 : b: 17.59 +/- 0.91 N: 15.82 +/- 0.12
#Component 2 : b: 28.63 +/- 4.16 N: 13.11 +/- 0.05
#Component 3 : b: 14.46 +/- 6.62 N: 13.0 +/- 0.1
#Components:  -66.9542844297  36.9682379599  -149.333963074 
#2.70917142547 2.71045749037 2.70815195817 
N5 = [15.82,13.11, 13.00]
b5 = [17.59, 28.63, 14.46]
v_05 = [-66.9542844297, 36.9682379599, -149.333963074]
for i in [0, 1, 2]:
	gauss_k = Gaussian1DKernel(stddev=RES_uvb/(2*np.sqrt(2*np.log(2))*transform_5),
			mode="oversample")
	ff = np.ones(len(velocity_5))
	v = np.array(velocity_5) - v_05[i]
	ff *= np.convolve(add_abs_velo(v,N5[i],b5[i],gamma5,f5,l0_5),gauss_k,mode='same')

	ax5.plot(velocity_5,ff,color=pt_analogous[i+2],linewidth=2)
	ax5.text(-270,1.32-float(i)/8,"b = "+str(b5[i]) + ", N = "+str(N5[i]),color=pt_analogous[i+2], fontsize=8)
	ax5.axvline(v_05[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)



ax6.errorbar(velocity_6,fluxv_6,yerr=fluxv_err_6, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax6.plot(velocity_6,fluxv_6, drawstyle='steps-mid',color='gray', alpha=0.66)
ax6.plot(velocity_6, y_fit_6,label='Fit',color=pt_analogous[0],linewidth=2)
ax6.fill_between(velocity_6, y_min_6, y_max_6, color=pt_analogous[0],alpha=0.5)
ax6.fill_between(velocity_6, y_min2_6, y_max2_6, color=pt_analogous[0],alpha=0.2)
ax6.text(120, 1.3, "SiIV 1402.7729 ", fontsize=10)
#Component 1 : b: 11.6 +/- 0.63 N: 16.27 +/- 0.03
#Component 2 : b: 14.77 +/- 5.94 N: 13.01 +/- 0.05
#Component 3 : b: 13.06 +/- 7.29 N: 12.94 +/- 0.1
#Components:  -64.4658078234  54.893234007  -138.197440554 
#2.70920222093 2.71067931628 2.70828977517 

#Component 1 : b: 13.93 +/- 0.83 N: 16.13 +/- 0.06
#Component 2 : b: 31.11 +/- 6.32 N: 13.12 +/- 0.05
#Component 3 : b: 11.04 +/- 4.3 N: 13.1 +/- 0.08
#Components:  -64.5807931749  44.0870921581  -141.628399232 
#2.70920079796 2.71054558781 2.70824731628 
N6 = [16.13, 13.12, 13.1]
b6 = [13.93, 31.11, 11.04]
v_06 = [-64.5807931749, 44.0870921581, -141.628399232]
for i in [0, 1, 2]:
	gauss_k = Gaussian1DKernel(stddev=RES_uvb/(2*np.sqrt(2*np.log(2))*transform_6),
			mode="oversample")
	ff = np.ones(len(velocity_6))
	v = np.array(velocity_6) - v_06[i]
	ff *= np.convolve(add_abs_velo(v,N6[i],b6[i],gamma6,f6,l0_6),gauss_k,mode='same')

	ax6.plot(velocity_6,ff,color=pt_analogous[i+2],linewidth=2)
	ax6.text(-270,1.32-float(i)/8,"b = "+str(b6[i]) + ", N = "+str(N6[i]),color=pt_analogous[i+2], fontsize=8)
	ax6.axvline(v_06[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)



ax7.errorbar(velocity_7,fluxv_7,yerr=fluxv_err_7, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax7.plot(velocity_7,fluxv_7, drawstyle='steps-mid',color='gray', alpha=0.66)
ax7.plot(velocity_7, y_fit_7,label='Fit',color=pt_analogous[0],linewidth=2)
ax7.fill_between(velocity_7, y_min_7, y_max_7, color=pt_analogous[0],alpha=0.5)
ax7.fill_between(velocity_7, y_min2_7, y_max2_7, color=pt_analogous[0],alpha=0.2)
ax7.text(120, 1.3, "FeII 1608.4509 ", fontsize=10)
#Component 1 : b: 17.79 +/- 1.11 N: 14.48 +/- 0.02
#Component 2 : b: 16.85 +/- 1.54 N: 14.22 +/- 0.02
#Component 3 : b: 18.65 +/- 4.48 N: 13.65 +/- 0.05
#Components:  51.3508241361  -23.2072620753  -135.413299056 
#2.71063547815 2.70971280484 2.70832422956 
N7 = [14.48, 14.22, 13.65]
b7 = [17.79, 16.85, 18.65]
N7_err = [0.02, 0.02, 0.05]
b7_err = [1.11, 1.54, 4.48]
v_07 = [51.3508241361, -23.2072620753, -135.413299056 ]
for i in [0, 1, 2]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_7),
			mode="oversample")
	ff = np.ones(len(velocity_7))
	v = np.array(velocity_7) - v_07[i]
	ff *= np.convolve(add_abs_velo(v,N7[i],b7[i],gamma7,f7,l0_7),gauss_k,mode='same')

	ax7.plot(velocity_7,ff,color=pt_analogous[i+2],linewidth=2)
	ax7.text(-270,1.32-float(i)/8,"b: "+str(b7[i]) + " +/- " + str(b7_err[i]) + 
		", N: "+str(N7[i]) + " +/- " + str(N7_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax7.axvline(v_07[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)


ax8.errorbar(velocity_8,fluxv_8,yerr=fluxv_err_8, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax8.plot(velocity_8,fluxv_8, drawstyle='steps-mid',color='gray', alpha=0.66)
ax8.plot(velocity_8, y_fit_8,label='Fit',color=pt_analogous[0],linewidth=2)
ax8.fill_between(velocity_8, y_min_8, y_max_8, color=pt_analogous[0],alpha=0.5)
ax8.fill_between(velocity_8, y_min2_8, y_max2_8, color=pt_analogous[0],alpha=0.2)
ax8.text(120, 1.3, "SiII 1808.0129", fontsize=10)
#Component 1 : b: 16.56 +/- 2.74 N: 15.02 +/- 0.03
#Component 2 : b: 17.76 +/- 5.01 N: 14.7 +/- 0.07
#Component 3 : b: 19.39 +/- 9.93 N: 14.4 +/- 0.14
#Components:  53.3968242436  -21.1585901625  -133.469984434 
#2.71066079787 2.70973815762 2.70834827852 
N8 = [15.02, 14.7, 14.4]
b8 = [16.56, 17.76, 19.39]
v_08 = [53.3968242436, -21.1585901625, -133.469984434]
for i in [0, 1, 2]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_8),
			mode="oversample")
	ff = np.ones(len(velocity_8))
	v = np.array(velocity_8) - v_08[i]
	ff *= np.convolve(add_abs_velo(v,N8[i],b8[i],gamma8,f8,l0_8),gauss_k,mode='same')

	ax8.plot(velocity_8,ff,color=pt_analogous[i+2],linewidth=2)
	ax8.text(-270,1.32-float(i)/8,"b = "+str(b8[i]) + ", N = "+str(N8[i]),color=pt_analogous[i+2], fontsize=8)
	ax8.axvline(v_08[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)



ax9.errorbar(velocity_9,fluxv_9,yerr=fluxv_err_9, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax9.plot(velocity_9,fluxv_9, drawstyle='steps-mid',color='gray', alpha=0.66)
ax9.plot(velocity_9, y_fit_9,label='Fit',color=pt_analogous[0],linewidth=2)
ax9.fill_between(velocity_9, y_min_9, y_max_9, color=pt_analogous[0],alpha=0.5)
ax9.fill_between(velocity_9, y_min2_9, y_max2_9, color=pt_analogous[0],alpha=0.2)
ax9.text(120, 1.3, "FeII 2260.7793", fontsize=10)
#Component 1 : b: 12.39 +/- 4.03 N: 14.61 +/- 0.04
#Component 2 : b: 35.97 +/- 6.82 N: 14.74 +/- 0.06
#Components:  57.6240629467  -17.2233122991 
#2.71071311091 2.70978685758 
N9 = [14.61, 14.74]
b9 = [12.39, 35.97]
v_09 = [57.6240629467, -17.2233122991]
for i in [0, 1]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_9),
			mode="oversample")
	ff = np.ones(len(velocity_9))
	v = np.array(velocity_9) - v_09[i]
	ff *= np.convolve(add_abs_velo(v,N9[i],b9[i],gamma9,f9,l0_9),gauss_k,mode='same')

	ax9.plot(velocity_9,ff,color=pt_analogous[i+2],linewidth=2)
	ax9.text(-270,1.32-float(i)/8,"b = "+str(b9[i]) + ", N = "+str(N9[i]),color=pt_analogous[i+2], fontsize=8)
	ax9.axvline(v_09[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)






ax10.errorbar(velocity_10,fluxv_10,yerr=fluxv_err_10, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax10.plot(velocity_10,fluxv_10, drawstyle='steps-mid',color='gray', alpha=0.66)
ax10.plot(velocity_10, y_fit_10,label='Fit',color=pt_analogous[0],linewidth=2)
ax10.fill_between(velocity_10, y_min_10, y_max_10, color=pt_analogous[0],alpha=0.5)
ax10.fill_between(velocity_10, y_min2_10, y_max2_10, color=pt_analogous[0],alpha=0.2)
ax10.text(115, 1.3, "FeII* 2333.5147", fontsize=10)
#Component 1 : b: 21.39 +/- 2.54 N: 13.23 +/- 0.03
#Component 2 : b: 14.23 +/- 2.05 N: 13.21 +/- 0.03
#Component 3 : b: 12.53 +/- 1.09 N: 13.16 +/- 0.03
#Background a = 1.00875215253
#Components:  55.853644401  -21.1923370345  -132.09544931 
#2.71069120158 2.70973774 2.70836528871 
N10 = [13.23, 13.21, 13.16]
b10 = [21.39, 14.23, 12.53]
N10_err = [0.03, 0.03, 0.03]
b10_err = [2.54, 2.05, 1.09]
v_010 = [55.853644401, -21.1923370345, -132.09544931]
for i in [0, 1, 2]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_10),
			mode="oversample")
	ff = np.ones(len(velocity_10))
	v = np.array(velocity_10) - v_010[i]
	ff *= np.convolve(add_abs_velo(v,N10[i],b10[i],gamma10,f10,l0_10),gauss_k,mode='same')

	ax10.plot(velocity_10,ff,color=pt_analogous[i+2],linewidth=2)
	ax10.text(-270,1.32-float(i)/8,"b: "+str(b10[i]) + " +/- " + str(b10_err[i]) + 
		", N: "+str(N10[i]) + " +/- " + str(N10_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax10.axvline(v_010[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)




ax11.errorbar(velocity_11,fluxv_11,yerr=fluxv_err_11, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.5)
ax11.plot(velocity_11,fluxv_11, drawstyle='steps-mid',color='gray', alpha=0.66)
ax11.plot(velocity_11, y_fit_11,label='Fit',color=pt_analogous[0],linewidth=2)
ax11.fill_between(velocity_11, y_min_11, y_max_11, color=pt_analogous[0],alpha=0.5)
ax11.fill_between(velocity_11, y_min2_11, y_max2_11, color=pt_analogous[0],alpha=0.2)
ax11.text(120, 1.3, "FeII 2344.2129", fontsize=10)
#Component 1 : b: 17.96 +/- 0.51 N: 14.29 +/- 0.02
#Component 2 : b: 21.04 +/- 0.56 N: 14.02 +/- 0.01
#Component 3 : b: 14.2 +/- 1.12 N: 13.49 +/- 0.01
#Components:  55.3135636155  -16.5832380916  -130.094995057 
#2.71068451796 2.70979477865 2.70839004478 
N11 = [14.29, 14.02, 13.49]
b11 = [17.96, 21.04, 14.20]
N11_err = [0.02, 0.01, 0.01]
b11_err = [0.51, 0.56, 1.12]
v_011 = [55.3135636155, -16.5832380916, -130.094995057]
for i in [0, 1, 2]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_11),
			mode="oversample")
	ff = np.ones(len(velocity_11))
	v = np.array(velocity_11) - v_011[i]
	ff *= np.convolve(add_abs_velo(v,N11[i],b11[i],gamma11,f11,l0_11),gauss_k,mode='same')

	ax11.plot(velocity_11,ff,color=pt_analogous[i+2],linewidth=2)
	ax11.text(-270,1.32-float(i)/8,"b: "+str(b11[i]) + " +/- " + str(b11_err[i]) + 
		", N: "+str(N11[i]) + " +/- " + str(N11_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax11.axvline(v_011[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)




ax12.errorbar(velocity_12,fluxv_12,yerr=fluxv_err_12, color='gray',marker='o', ls='None',label='Observed', markersize=4, alpha=0.66)
ax12.plot(velocity_12,fluxv_12, drawstyle='steps-mid',color='gray', alpha=0.66)
ax12.plot(velocity_12, y_fit_12,label='Fit',color=pt_analogous[0],linewidth=2)
ax12.fill_between(velocity_12, y_min_12, y_max_12, color=pt_analogous[0],alpha=0.5)
ax12.fill_between(velocity_12, y_min2_12, y_max2_12, color=pt_analogous[0],alpha=0.2)
ax12.text(120, 1.3, "FeII 2374.4604", fontsize=10)
#Component 1 : b: 15.2 +/- 0.49 N: 14.54 +/- 0.01
#Component 2 : b: 19.22 +/- 0.83 N: 14.21 +/- 0.01
#Component 3 : b: 21.03 +/- 2.52 N: 13.68 +/- 0.02
#Components:  56.9796814878  -15.5904001432  -131.80166005 
#2.71070513655 2.70980706524 2.70836892442 
N12 = [14.54, 14.21, 13.68]
b12 = [15.20, 19.22, 21.03]
N12_err = [0.01, 0.01, 0.02]
b12_err = [0.49, 0.83, 2.52]
v_012 = [56.9796814878, -15.5904001432, -131.80166005]
for i in [0, 1, 2]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_12),
			mode="oversample")
	ff = np.ones(len(velocity_12))
	v = np.array(velocity_12) - v_012[i]
	ff *= np.convolve(add_abs_velo(v,N12[i],b12[i],gamma12,f12,l0_12),gauss_k,mode='same')

	ax12.plot(velocity_12,ff,color=pt_analogous[i+2],linewidth=2)
	ax12.text(-270,1.32-float(i)/8,"b: "+str(b12[i]) + " +/- " + str(b12_err[i]) + 
		", N: "+str(N12[i]) + " +/- " + str(N12_err[i]) ,color=pt_analogous[i+2], fontsize=8)
	ax12.axvline(v_012[i], ymin=0, ymax=0.72, color='gray', linestyle="dashed", linewidth=0.8)




for axis in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8,
	ax9, ax10, ax11, ax12]:

	axis.set_xlim([-280, 280])
	axis.set_ylim([-0.15, 1.45])


	#axis.axhline(1.0,xmin=0.0, xmax=1.0, linewidth=1,linestyle="dotted",color="black", alpha=0.5)
	axis.axhline(0.0,xmin=0.0, xmax=1.0, linewidth=1,linestyle="dotted",color="black")
	
	if not axis in [ax1, ax2]:
		for tick in axis.xaxis.get_major_ticks():
			tick.label.set_fontsize(0)
	else:
		axis.set_xticks([-200, -100, 0, 100, 200])
		for side in ['bottom']:
		  	axis.spines[side].axis.axes.tick_params(which='major', length=6, width=1, axis='x')
			#axis.tick_params[side](which='major', length=6, width=1)


	if not axis in [ax1, ax3, ax5, ax7, ax9, ax11]:
		for tick in axis.yaxis.get_major_ticks():
			tick.label.set_fontsize(0)
	else:
		axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
		for side in ['right']:
		  	axis.spines[side].axis.axes.tick_params(which='major', length=4, width=1, axis='y')


show()
fig.savefig(target + "_metals.pdf")
