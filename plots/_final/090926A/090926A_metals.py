#! usr/bin/python

target = "GRB090926A"
redshift = 2.1068

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

RES_vis = 30.0
RES_uvb = 51.0


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



wav_aa_vis, n_flux_vis, n_flux_err_vis, flux_vis, flux_err_vis, grb_name, res, psf_fwhm = get_data("../../../spectra/GRB090926_OB1VIS.txt", redshift)
wav_aa_uvb, n_flux_uvb, n_flux_err_uvb, flux_uvb, flux_err_vis, grb_name, res, psf_fwhm = get_data("../../../spectra/GRB090926_OB1UVB.txt", redshift)


l0_1 = 1048.2199
gamma1 =  0.00530517423733
f1 = 0.263
transform_1 = 18.4151895743
wav_range_1 = 3.8
velocity_1, fluxv_1, fluxv_err_1 = aa_to_velo(wav_aa_uvb, n_flux_uvb, n_flux_err_uvb, l0_1, redshift, wav_range_1)
y_fit_1, y_min_1, y_max_1, y_min2_1, y_max2_1 = get_line("lines/GRB090926_1_1048.2199_fit.dat")


l0_2 = 1302.1685 #OI
gamma2 = 0.117094302735
f2 = 0.048
transform_2 = 14.8240117659
wav_range_2 = 5
velocity_2, fluxv_2, fluxv_err_2 = aa_to_velo(wav_aa_uvb, n_flux_uvb, n_flux_err_uvb, l0_2, redshift, wav_range_2)
y_fit_2, y_min_2, y_max_2, y_min2_2, y_max2_2 = get_line("lines/GRB090926_1_1302.1685_fit.dat")


l0_3 = 1402.7729
gamma3 = 0.192448603803
f3 = 0.254
transform_3 = 13.7608597694
wav_range_3 = 6
velocity_3, fluxv_3, fluxv_err_3 = aa_to_velo(wav_aa_uvb, n_flux_uvb, n_flux_err_uvb, l0_3, redshift, wav_range_3)
y_fit_3, y_min_3, y_max_3, y_min2_3, y_max2_3 = get_line("lines/GRB090926_3_1402.7729_fit.dat")



l0_4 = 1526.7070
gamma4 = 0.274570751244
f4 = 0.133
transform_4 = 12.643788995
wav_range_4 = 7
velocity_4, fluxv_4, fluxv_err_4 = aa_to_velo(wav_aa_uvb, n_flux_uvb, n_flux_err_uvb, l0_4, redshift, wav_range_4)
y_fit_4, y_min_4, y_max_4, y_min2_4, y_max2_4 = get_line("lines/GRB090926_2_1526.707_fit.dat")



l0_5 = 1550.7810
gamma5 = 0.0648628405618
f5 = 0.09475
transform_5 = 12.4475094583
wav_range_5 = 7
velocity_5, fluxv_5, fluxv_err_5 = aa_to_velo(wav_aa_uvb, n_flux_uvb, n_flux_err_uvb, l0_5, redshift, wav_range_5)
y_fit_5, y_min_5, y_max_5, y_min2_5, y_max2_5 = get_line("lines/GRB090926_3_1550.781_fit.dat")



l0_6 = 1608.4509
gamma6 = 0.0623086746483
f6 = 0.05399224
transform_6 = 12.0012125737
wav_range_6 = 7
velocity_6, fluxv_6, fluxv_err_6 = aa_to_velo(wav_aa_uvb, n_flux_uvb, n_flux_err_uvb, l0_6, redshift, wav_range_6)
y_fit_6, y_min_6, y_max_6, y_min2_6, y_max2_6 = get_line("lines/GRB090926_2_1608.4509_fit.dat")


l0_7 = 1808.0129
gamma7 = 0.000684854972697
f7 = 0.00208
transform_7 = 10.676721749
wav_range_7 = 8
velocity_7, fluxv_7, fluxv_err_7 = aa_to_velo(wav_aa_vis, n_flux_vis, n_flux_err_vis, l0_7, redshift, wav_range_7)
y_fit_7, y_min_7, y_max_7, y_min2_7, y_max2_7 = get_line("lines/GRB090926_1_1808.0129_fit.dat")



l0_8 = 1854.7183
gamma8 = 0.160345896437
f8 = 0.539
transform_8 = 10.407861211
wav_range_8 = 8
velocity_8, fluxv_8, fluxv_err_8 = aa_to_velo(wav_aa_vis, n_flux_vis, n_flux_err_vis, l0_8, redshift, wav_range_8)
y_fit_8, y_min_8, y_max_8, y_min2_8, y_max2_8 = get_line("lines/GRB090926_2_1854.7183_fit.dat")


l0_9 = 1862.7911
gamma9 = 0.158938859812
f9 = 0.268
transform_9 = 10.3627565388
wav_range_9 = 8
velocity_9, fluxv_9, fluxv_err_9 = aa_to_velo(wav_aa_vis, n_flux_vis, n_flux_err_vis, l0_9, redshift, wav_range_9)
y_fit_9, y_min_9, y_max_9, y_min2_9, y_max2_9 = get_line("lines/GRB090926_1_1862.7911_fit.dat")



l0_10 = 2344.2129
gamma10 = 0.108532774265
f10 = 0.12523167
transform_10 = 8.23459791216
wav_range_10 = 8
velocity_10, fluxv_10, fluxv_err_10 = aa_to_velo(wav_aa_vis, n_flux_vis, n_flux_err_vis, l0_10, redshift, wav_range_10)
y_fit_10, y_min_10, y_max_10, y_min2_10, y_max2_10 = get_line("lines/GRB090926_2_2344.2129_fit.dat")


l0_11 = 2374.4604
gamma11 = 0.12448260198
f11 = 0.03296636
transform_11 = 8.12969997394
wav_range_11 = 8
velocity_11, fluxv_11, fluxv_err_11 = aa_to_velo(wav_aa_vis, n_flux_vis, n_flux_err_vis, l0_11, redshift, wav_range_11)
y_fit_11, y_min_11, y_max_11, y_min2_11, y_max2_11 = get_line("lines/GRB090926_2_2374.4604_fit.dat", a=1.01708771097)


l0_12 = 2382.7641
gamma12 = 0.127420838072
f12 = 0.34320935
transform_12 = 8.10136876411
wav_range_12 = 8
velocity_12, fluxv_12, fluxv_err_12 = aa_to_velo(wav_aa_vis, n_flux_vis, n_flux_err_vis, l0_12, redshift, wav_range_12)
y_fit_12, y_min_12, y_max_12, y_min2_12, y_max2_12 = get_line("lines/GRB090926_3_2382.7641_fit.dat")


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



ax1.errorbar(velocity_1,fluxv_1,yerr=fluxv_err_1, color='gray',marker='o', ls='None',label='Observed', markersize=5)
ax1.plot(velocity_1,fluxv_1, drawstyle='steps-mid',color='gray', alpha=0.66)
ax1.plot(velocity_1, y_fit_1,label='Fit',color=pt_analogous[0],linewidth=2)
ax1.fill_between(velocity_1, y_min_1, y_max_1, color=pt_analogous[0],alpha=0.5)
ax1.fill_between(velocity_1, y_min2_1, y_max2_1, color=pt_analogous[0],alpha=0.2)
ax1.text(40, 1.3, "ArI 1048.2199", fontsize=10)


#Component 1 : b: 11.96 +/- 0.58 N: 13.76 +/- 0.15
#Components:  22.5122828734 
#2.1070332986 

N1 = [13.76]
b1 = [11.96]
v_01 = [22.5122828734]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_uvb/(2*np.sqrt(2*np.log(2))*transform_1),
			mode="oversample")
	ff = np.ones(len(velocity_1))
	v = np.array(velocity_1) - v_01[i]
	ff *= np.convolve(add_abs_velo(v,N1[i],b1[i],gamma1,f1,l0_1),gauss_k,mode='same')

	ax1.plot(velocity_1,ff,color=pt_analogous[i+2],linewidth=2)
	ax1.text(-220,0.50-float(i)/2.5,"b = "+str(b1[i]),color=pt_analogous[i+2])
	ax1.text(-220,0.70-float(i)/2.5,"N = "+str(N1[i]),color=pt_analogous[i+2])



ax2.errorbar(velocity_2,fluxv_2,yerr=fluxv_err_2, color='gray',marker='o', ls='None',label='Observed', markersize=5)
ax2.plot(velocity_2,fluxv_2, drawstyle='steps-mid',color='gray', alpha=0.66)
ax2.plot(velocity_2, y_fit_2,label='Fit',color=pt_analogous[0],linewidth=2)
ax2.fill_between(velocity_2, y_min_2, y_max_2, color=pt_analogous[0],alpha=0.5)
ax2.fill_between(velocity_2, y_min2_2, y_max2_2, color=pt_analogous[0],alpha=0.2)
ax2.text(40, 1.3, "OI 1302.1685", fontsize=10)

#Component 1 : b: 11.83 +/- 0.54 N: 16.19 +/- 0.13
#Components:  19.2556091395 
#2.10699954914 

N2 = [16.19]
b2 = [11.83]
v_02 = [19.2556091395]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_uvb/(2*np.sqrt(2*np.log(2))*transform_2),
			mode="oversample")
	ff = np.ones(len(velocity_2))
	v = np.array(velocity_2) - v_02[i]
	ff *= np.convolve(add_abs_velo(v,N2[i],b2[i],gamma2,f2,l0_2),gauss_k,mode='same')

	ax2.plot(velocity_2,ff,color=pt_analogous[i+2],linewidth=2)
	ax2.text(-220,0.50-float(i)/2.5,"b = "+str(b2[i]),color=pt_analogous[i+2])
	ax2.text(-220,0.70-float(i)/2.5,"N = "+str(N2[i]),color=pt_analogous[i+2])



ax3.errorbar(velocity_3,fluxv_3,yerr=fluxv_err_3, color='gray',marker='o', ls='None',label='Observed', markersize=5)
ax3.plot(velocity_3,fluxv_3, drawstyle='steps-mid',color='gray', alpha=0.66)
ax3.plot(velocity_3, y_fit_3,label='Fit',color=pt_analogous[0],linewidth=2)
ax3.fill_between(velocity_3, y_min_3, y_max_3, color=pt_analogous[0],alpha=0.5)
ax3.fill_between(velocity_3, y_min2_3, y_max2_3, color=pt_analogous[0],alpha=0.2)
ax3.axvline(-100, color='gray', linestyle="dashed", linewidth=0.8)
ax3.text(40, 1.3, "SiIV 1402.7729", fontsize=10)


#Component 1 : b: 12.25 +/- 0.49 N: 15.18 +/- 0.12
#Component 2 : b: 15.73 +/- 1.45 N: 13.57 +/- 0.05
#Component 3 : b: 13.83 +/- 2.31 N: 13.06 +/- 0.06
#Components:  18.2279273552  -24.749224825  -102.471452306 
#2.1069888991 2.10654351959 2.10573807099 

N3 = [15.18, 13.57, 13.06]
b3 = [12.25, 15.73, 13.83]
v_03 = [18.2279273552, -24.749224825, -102.471452306]
for i in [0, 1, 2]:
	gauss_k = Gaussian1DKernel(stddev=RES_uvb/(2*np.sqrt(2*np.log(2))*transform_3),
			mode="oversample")
	ff = np.ones(len(velocity_3))
	v = np.array(velocity_3) - v_03[i]
	ff *= np.convolve(add_abs_velo(v,N3[i],b3[i],gamma3,f3,l0_3),gauss_k,mode='same')

	ax3.plot(velocity_3,ff,color=pt_analogous[i+2],linewidth=2)
	ax3.text(-220,0.50-float(i)/2.5,"b = "+str(b3[i]),color=pt_analogous[i+2])
	ax3.text(-220,0.70-float(i)/2.5,"N = "+str(N3[i]),color=pt_analogous[i+2])


ax4.errorbar(velocity_4,fluxv_4,yerr=fluxv_err_4, color='gray',marker='o', ls='None',label='Observed', markersize=5)
ax4.plot(velocity_4,fluxv_4, drawstyle='steps-mid',color='gray', alpha=0.66)
ax4.plot(velocity_4, y_fit_4,label='Fit',color=pt_analogous[0],linewidth=2)
ax4.fill_between(velocity_4, y_min_4, y_max_4, color=pt_analogous[0],alpha=0.5)
ax4.fill_between(velocity_4, y_min2_4, y_max2_4, color=pt_analogous[0],alpha=0.2)
ax4.text(40, 1.3, "SiII 1526.7070", fontsize=10)


#Component 1 : b: 12.04 +/- 0.55 N: 14.86 +/- 0.11
#Component 2 : b: 19.24 +/- 1.97 N: 13.82 +/- 0.04
#Components:  18.3190116666  -25.445771356 
#2.10698984302 2.10653630116 

N4 = [14.86, 13.82]
b4 = [12.05, 19.24]
v_04 = [18.3190116666, -25.445771356]
for i in [0, 1]:
	gauss_k = Gaussian1DKernel(stddev=RES_uvb/(2*np.sqrt(2*np.log(2))*transform_4),
			mode="oversample")
	ff = np.ones(len(velocity_4))
	v = np.array(velocity_4) - v_04[i]
	ff *= np.convolve(add_abs_velo(v,N4[i],b4[i],gamma4,f4,l0_4),gauss_k,mode='same')

	ax4.plot(velocity_4,ff,color=pt_analogous[i+2],linewidth=2)
	ax4.text(-220,0.50-float(i)/2.5,"b = "+str(b4[i]),color=pt_analogous[i+2])
	ax4.text(-220,0.70-float(i)/2.5,"N = "+str(N4[i]),color=pt_analogous[i+2])



ax5.errorbar(velocity_5,fluxv_5,yerr=fluxv_err_5, color='gray',marker='o', ls='None',label='Observed', markersize=5)
ax5.plot(velocity_5,fluxv_5, drawstyle='steps-mid',color='gray', alpha=0.66)
ax5.plot(velocity_5, y_fit_5,label='Fit',color=pt_analogous[0],linewidth=2)
ax5.fill_between(velocity_5, y_min_5, y_max_5, color=pt_analogous[0],alpha=0.5)
ax5.fill_between(velocity_5, y_min2_5, y_max2_5, color=pt_analogous[0],alpha=0.2)
ax5.axvline(-100, color='gray', linestyle="dashed", linewidth=0.8)
ax5.text(40, 1.3, "CIV 1550.7810", fontsize=10)

#Component 1 : b: 12.05 +/- 0.56 N: 15.85 +/- 0.18
#Component 2 : b: 14.84 +/- 0.84 N: 14.0 +/- 0.04
#Component 3 : b: 13.96 +/- 2.29 N: 13.25 +/- 0.09
#Components:  18.6758334635  -24.3939596912  -101.234316387 
#2.10699354082 2.10654720127 2.10575089164 

N5 = [15.85, 14.00, 13.25]
b5 = [12.05, 14.84, 13.96]
v_05 = [18.6758334635, -24.3939596912, -101.234316387]
for i in [0, 1, 2]:
	gauss_k = Gaussian1DKernel(stddev=RES_uvb/(2*np.sqrt(2*np.log(2))*transform_5),
			mode="oversample")
	ff = np.ones(len(velocity_5))
	v = np.array(velocity_5) - v_05[i]
	ff *= np.convolve(add_abs_velo(v,N5[i],b5[i],gamma5,f5,l0_5),gauss_k,mode='same')

	ax5.plot(velocity_5,ff,color=pt_analogous[i+2],linewidth=2)
	ax5.text(-220,0.50-float(i)/2.5,"b = "+str(b5[i]),color=pt_analogous[i+2])
	ax5.text(-220,0.70-float(i)/2.5,"N = "+str(N5[i]),color=pt_analogous[i+2])





ax6.errorbar(velocity_6,fluxv_6,yerr=fluxv_err_6, color='gray',marker='o', ls='None',label='Observed', markersize=5)
ax6.plot(velocity_6,fluxv_6, drawstyle='steps-mid',color='gray', alpha=0.66)
ax6.plot(velocity_6, y_fit_6,label='Fit',color=pt_analogous[0],linewidth=2)
ax6.fill_between(velocity_6, y_min_6, y_max_6, color=pt_analogous[0],alpha=0.5)
ax6.fill_between(velocity_6, y_min2_6, y_max2_6, color=pt_analogous[0],alpha=0.2)
ax6.text(40, 1.3, "FeII 1608.4509", fontsize=10)


#Component 1 : b: 11.96 +/- 0.55 N: 14.22 +/- 0.04
#Component 2 : b: 17.51 +/- 2.26 N: 13.67 +/- 0.06
#Components:  18.619802189  -25.6599173947 
#2.10699296016 2.10653408193 

N6 = [14.22, 13.67]
b6 = [11.96, 17.51]
v_06 = [18.619802189, -25.6599173947]
for i in [0, 1]:
	gauss_k = Gaussian1DKernel(stddev=RES_uvb/(2*np.sqrt(2*np.log(2))*transform_6),
			mode="oversample")
	ff = np.ones(len(velocity_6))
	v = np.array(velocity_6) - v_06[i]
	ff *= np.convolve(add_abs_velo(v,N6[i],b6[i],gamma6,f6,l0_6),gauss_k,mode='same')

	ax6.plot(velocity_6,ff,color=pt_analogous[i+2],linewidth=2)
	ax6.text(-220,0.50-float(i)/2.5,"b = "+str(b6[i]),color=pt_analogous[i+2])
	ax6.text(-220,0.70-float(i)/2.5,"N = "+str(N6[i]),color=pt_analogous[i+2])



ax7.errorbar(velocity_7,fluxv_7,yerr=fluxv_err_8, color='gray',marker='o', ls='None',label='Observed', markersize=5)
ax7.plot(velocity_7,fluxv_7, drawstyle='steps-mid',color='gray', alpha=0.66)
ax7.plot(velocity_7, y_fit_7,label='Fit',color=pt_analogous[0],linewidth=2)
ax7.fill_between(velocity_7, y_min_7, y_max_7, color=pt_analogous[0],alpha=0.5)
ax7.fill_between(velocity_7, y_min2_7, y_max2_7, color=pt_analogous[0],alpha=0.2)
ax7.text(40, 1.3, "SiII 1808.0129", fontsize=10)

#Component 1 : b: 12.0 +/- 0.58 N: 15.32 +/- 0.05
#Components:  24.9002907895 
#2.10705804593 

N7 = [15.32]
b7 = [12.00]
v_07 = [24.9002907895]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_7),
			mode="oversample")
	ff = np.ones(len(velocity_7))
	v = np.array(velocity_7) - v_07[i]
	ff *= np.convolve(add_abs_velo(v,N7[i],b7[i],gamma7,f7,l0_7),gauss_k,mode='same')

	ax7.plot(velocity_7,ff,color=pt_analogous[i+2],linewidth=2)
	ax7.text(-220,0.50-float(i)/2.5,"b = "+str(b7[i]),color=pt_analogous[i+2])
	ax7.text(-220,0.70-float(i)/2.5,"N = "+str(N7[i]),color=pt_analogous[i+2])


ax8.errorbar(velocity_8,fluxv_8,yerr=fluxv_err_8, color='gray',marker='o', ls='None',label='Observed', markersize=5)
ax8.plot(velocity_8,fluxv_8, drawstyle='steps-mid',color='gray', alpha=0.66)
ax8.plot(velocity_8, y_fit_8,label='Fit',color=pt_analogous[0],linewidth=2)
ax8.fill_between(velocity_8, y_min_8, y_max_8, color=pt_analogous[0],alpha=0.5)
ax8.fill_between(velocity_8, y_min2_8, y_max2_8, color=pt_analogous[0],alpha=0.2)
ax8.text(40, 1.3, "AlIII 1854.7183", fontsize=10)



#Component 1 : b: 12.03 +/- 0.58 N: 13.32 +/- 0.06
#Component 2 : b: 17.8 +/- 2.28 N: 9.37 +/- 4.22
#Components:  25.1766331311  -27.841361999 
#2.10706090971 2.10651147525

N8 = [13.32, 9.37]
b8 = [12.03, 17.8]
v_08 = [25.1766331311, -27.841361999]
for i in [0, 1]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_8),
			mode="oversample")
	ff = np.ones(len(velocity_8))
	v = np.array(velocity_8) - v_08[i]
	ff *= np.convolve(add_abs_velo(v,N8[i],b8[i],gamma8,f8,l0_8),gauss_k,mode='same')

	ax8.plot(velocity_8,ff,color=pt_analogous[i+2],linewidth=2)
	ax8.text(-220,0.50-float(i)/2.5,"b = "+str(b8[i]),color=pt_analogous[i+2])
	ax8.text(-220,0.70-float(i)/2.5,"N = "+str(N8[i]),color=pt_analogous[i+2])




ax9.errorbar(velocity_9,fluxv_9,yerr=fluxv_err_9, color='gray',marker='o', ls='None',label='Observed', markersize=5)
ax9.plot(velocity_9,fluxv_9, drawstyle='steps-mid',color='gray', alpha=0.66)
ax9.plot(velocity_9, y_fit_9,label='Fit',color=pt_analogous[0],linewidth=2)
ax9.fill_between(velocity_9, y_min_9, y_max_9, color=pt_analogous[0],alpha=0.5)
ax9.fill_between(velocity_9, y_min2_9, y_max2_9, color=pt_analogous[0],alpha=0.2)
ax9.text(40, 1.3, "AlIII 1862.7911", fontsize=10)


#Component 1 : b: 12.1 +/- 0.56 N: 13.36 +/- 0.05
#Components:  23.1300202561 
#2.10703970032 

N9 = [13.36]
b9 = [12.10]
v_09 = [23.1300202561]
for i in [0]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_9),
			mode="oversample")
	ff = np.ones(len(velocity_9))
	v = np.array(velocity_9) - v_09[i]
	ff *= np.convolve(add_abs_velo(v,N9[i],b9[i],gamma9,f9,l0_9),gauss_k,mode='same')

	ax9.plot(velocity_9,ff,color=pt_analogous[i+2],linewidth=2)
	ax9.text(-220,0.50-float(i)/2.5,"b = "+str(b9[i]),color=pt_analogous[i+2])
	ax9.text(-220,0.70-float(i)/2.5,"N = "+str(N9[i]),color=pt_analogous[i+2])


ax10.errorbar(velocity_10,fluxv_10,yerr=fluxv_err_10, color='gray',marker='o', ls='None',label='Observed', markersize=5)
ax10.plot(velocity_10,fluxv_10, drawstyle='steps-mid',color='gray', alpha=0.66)
ax10.plot(velocity_10, y_fit_10,label='Fit',color=pt_analogous[0],linewidth=2)
ax10.fill_between(velocity_10, y_min_10, y_max_10, color=pt_analogous[0],alpha=0.5)
ax10.fill_between(velocity_10, y_min2_10, y_max2_10, color=pt_analogous[0],alpha=0.2)
ax10.text(40, 1.3, "FeII 2344.2129", fontsize=10)


#Component 1 : b: 12.0 +/- 0.55 N: 14.4 +/- 0.08
#Component 2 : b: 18.61 +/- 2.14 N: 13.13 +/- 0.05
#Components:  21.9986846233  -27.9389572475 
#2.10702797609 2.10651046386 

N10 = [14.40, 13.13]
b10 = [12.00, 18.61]
v0_10 = [21.9986846233, -27.9389572475]
for i in [0, 1]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_10),
			mode="oversample")
	ff = np.ones(len(velocity_10))
	v = np.array(velocity_10) - v0_10[i]
	ff *= np.convolve(add_abs_velo(v,N10[i],b10[i],gamma10,f10,l0_10),gauss_k,mode='same')

	ax10.plot(velocity_10,ff,color=pt_analogous[i+2],linewidth=2)
	ax10.text(-220,0.50-float(i)/2.5,"b = "+str(b10[i]),color=pt_analogous[i+2])
	ax10.text(-220,0.70-float(i)/2.5,"N = "+str(N10[i]),color=pt_analogous[i+2])


ax11.errorbar(velocity_11,fluxv_11,yerr=fluxv_err_11, color='gray',marker='o', ls='None',label='Observed', markersize=5)
ax11.plot(velocity_11,fluxv_11, drawstyle='steps-mid',color='gray', alpha=0.66)
ax11.plot(velocity_11, y_fit_11,label='Fit',color=pt_analogous[0],linewidth=2)
ax11.fill_between(velocity_11, y_min_11, y_max_11, color=pt_analogous[0],alpha=0.5)
ax11.fill_between(velocity_11, y_min2_11, y_max2_11, color=pt_analogous[0],alpha=0.2)
ax11.text(40, 1.3, "FeII 2374.4604", fontsize=10)

#Component 1 : b: 12.26 +/- 0.51 N: 14.24 +/- 0.02
#Component 2 : b: 19.5 +/- 2.05 N: 13.46 +/- 0.06
#Components:  21.6026970534  -28.3996237294 
#2.10702387241 2.10650568989 

N11 = [14.24, 13.46]
b11 = [12.26, 19.50]
v0_11 = [21.6351934025, -28.4779847018]
for i in [0, 1]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_11),
			mode="oversample")
	ff = np.ones(len(velocity_11))
	v = np.array(velocity_11) - v0_11[i]
	ff *= np.convolve(add_abs_velo(v,N11[i],b11[i],gamma11,f11,l0_11),gauss_k,mode='same')

	ax11.plot(velocity_11,ff,color=pt_analogous[i+2],linewidth=2)
	ax11.text(-220,0.50-float(i)/2.5,"b = "+str(b11[i]),color=pt_analogous[i+2])
	ax11.text(-220,0.70-float(i)/2.5,"N = "+str(N11[i]),color=pt_analogous[i+2])



ax12.errorbar(velocity_12,fluxv_12,yerr=fluxv_err_12, color='gray',marker='o', ls='None',label='Observed', markersize=5)
ax12.plot(velocity_12,fluxv_12, drawstyle='steps-mid',color='gray', alpha=0.66)
ax12.plot(velocity_12, y_fit_12,label='Fit',color=pt_analogous[0],linewidth=2)
ax12.fill_between(velocity_12, y_min_12, y_max_12, color=pt_analogous[0],alpha=0.5)
ax12.fill_between(velocity_12, y_min2_12, y_max2_12, color=pt_analogous[0],alpha=0.2)
ax12.axvline(-100, color='gray', linestyle="dashed", linewidth=0.8)
ax12.text(40, 1.3, "FeII 2382.7641", fontsize=10)

#Component 1 : b: 11.93 +/- 0.56 N: 14.84 +/- 0.14
#Component 2 : b: 21.0 +/- 0.95 N: 13.1 +/- 0.04
#Component 3 : b: 16.74 +/- 1.24 N: 12.5 +/- 0.05
#Components:  20.8145817248  -27.0641065676  -97.3039804236 
#2.10701570503 2.10651953008 2.10579162238

N12 = [14.84, 13.10, 12.50]
b12 = [11.93, 21.00, 16.74]
v0_12 = [20.8145817248, -27.0641065676, -97.3039804236]
for i in [0, 1, 2]:
	gauss_k = Gaussian1DKernel(stddev=RES_vis/(2*np.sqrt(2*np.log(2))*transform_12),
			mode="oversample")
	ff = np.ones(len(velocity_12))
	v = np.array(velocity_12) - v0_12[i]
	ff *= np.convolve(add_abs_velo(v,N12[i],b12[i],gamma12,f12,l0_12),gauss_k,mode='same')

	ax12.plot(velocity_11,ff,color=pt_analogous[i+2],linewidth=2)
	ax12.text(-220,0.50-float(i)/2.5,"b = "+str(b12[i]),color=pt_analogous[i+2])
	ax12.text(-220,0.70-float(i)/2.5,"N = "+str(N12[i]),color=pt_analogous[i+2])


for axis in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8,
	ax9, ax10, ax11, ax12]:

	axis.set_xlim([-250, 250])
	axis.set_ylim([-0.15, 1.45])

	axis.axvline(22, color='gray', linestyle="dashed", linewidth=0.8)
	axis.axvline(-28, color='gray', linestyle="dashed", linewidth=0.8)

	axis.axhline(1.0,xmin=0.0, xmax=1.0, linewidth=1,linestyle="dotted",color="black", alpha=0.5)
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
