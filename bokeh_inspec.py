#! /usr/bin/python

"""
Creating a BOKEH Plot to scroll through the given spectrum and
identify absoprtion lines
=========================================================================
e.g.: bokeh_inspec.py -f spectra/GRB120815Auvb.txt -t GRB120815A

=========================================================================
-file 		path to spectrum data file
-z			redshift of the GRB
=========================================================================
=========================================================================
"""
import time, math, os, sys, argparse

import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *
import numpy as np

import importlib

import bokeh
from bokeh.plotting import figure as bokeh_fig
from bokeh.plotting import show as bokeh_show
from bokeh.plotting import output_file
from bokeh.io import output_file
from bokeh.layouts import widgetbox, row
from bokeh.models import *

sys.path.append('bin/')

from spec_functions import *


def bokeh_inspec(file, redshift):

	a_name, a_wav, ai_name, ai_wav, \
	aex_name, aex_wav, h2_name, h2_wav \
	= get_lines(redshift)

	wav_aa, n_flux, n_flux_err, flux, \
	flux_err, grb_name, res, psf_fwhm \
	= get_data(file, redshift, wl_range=False)

	output_file(grb_name + "_spec_bokeh.html",
		title="specrum_bokeh_html.py", mode="cdn")

	source = ColumnDataSource(data={"wav_aa":wav_aa, "n_flux":n_flux})

	hover = HoverTool(tooltips=[
		("(wav_aa, n_flux)", "($wav_aa, $n_flux)")])

	p = bokeh_fig(title="X-shooter spectrum of " + grb_name + "at z = " +
		str(redshift), x_axis_label='Observed Wavelength',tools="hover",
		y_axis_label='Normalized Flux', y_range=[-0.8, 2.2],
		x_range=[min(wav_aa),max(wav_aa)+800],
		plot_height=400, plot_width=20000, toolbar_location="above")

	for i in np.arange(0, len(h2_name), 1):

		vline = Span(location=h2_wav[i], dimension='height',
			line_color='red', line_width=0.8, line_dash='dashed')

		if i%2 == 0:
			H2_label = Label(x=h2_wav[i]+0.2, y=-0.6, text=h2_name[i],
				text_font_size="9pt", text_color="red",text_font="helvetica")
		else:
			H2_label = Label(x=h2_wav[i]+0.2, y=-0.3, text=h2_name[i],
				text_font_size="9pt", text_color="red",text_font="helvetica")		

		p.renderers.extend([vline])
		p.add_layout(H2_label)

	for i in np.arange(0, len(a_name), 1):

		vline = Span(location=a_wav[i], dimension='height',
			line_color='green', line_width=1, line_dash='dashed')

		if i%2 == 0:
			atom_label = Label(x=a_wav[i]+0.2, y=1.4,
				text=a_name[i]+" " + str(round(a_wav[i]/(1+redshift), 2)),
				text_font_size="8pt", text_color="green",
				text_font="helvetica", angle=0.6)
		else:
			atom_label = Label(x=a_wav[i]+0.2, y=1.7,
				text=a_name[i]+" " + str(round(a_wav[i]/(1+redshift), 2)),
				text_font_size="8pt", text_color="green",
				text_font="helvetica", angle=0.6)			

		p.renderers.extend([vline])
		p.add_layout(atom_label)

	p.line(x="wav_aa", y="n_flux", source=source,
		legend=grb_name + "Data", line_width=2,
		color="black")

	callback = CustomJS(args=dict(x_range=p.x_range), code="""
	var start = cb_obj.get("value");
	x_range.set("start", start);
	x_range.set("end", start+800);
	""")
	
	slider = Slider(start=min(wav_aa),
		end=max(wav_aa)-800, value=1,
		step=.1, title="Scroll", callback=callback)
	
	inputs = widgetbox(slider)
	
	bokeh_show(row(inputs, p, width=800), browser="safari")


if __name__ == "__main__":

	start = time.time()
	print "\n Parsing Arguments \n"

	parser = argparse.ArgumentParser(usage=__doc__)
	parser.add_argument('-f','--file',dest="file",
		default="spectra/GRB120815Auvb.txt",type=str)
	parser.add_argument('-z','--z',dest="z",default=2.358,
		type=float)
	args = parser.parse_args()

	spec_file = args.file
	redshift = args.z

	print "\n Plotting Spectrum \n"

	bokeh_inspec(spec_file, redshift)	






