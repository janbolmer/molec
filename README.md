# MCMC sampler for fitting X-shooter spectra (using PyMC2) with HI, H2 and other lines

e.g.:
molec_mcmc.py -f spectra/GRB120815Auvb.txt -red 2.358 -t GRB120815A
-e HI FeII SiII NV SII MgII SiIIa SIV ZnII FeIIa ArI OVI
-w1 985 -w2 1120 -m H2 -nrot 5 -it 200 -bi 10 -par para.csv

# !!! Under Construction - I am learning Git here !!!

I am making use of the H2sim from Thomas Kruehler:
https://github.com/Kruehlio/H2sim

# Literature:

## Theory:

[[http://iopscience.iop.org/article/10.1086/308581/pdf|Draine 2000 - GRBs In Molecular Clouds: H2 Absorption Anf Fluorescence]]

[[http://iopscience.iop.org/article/10.1086/339394/pdf|Draine & Hao 2002 - GRBs in a Molecular Cloud: Destruction of Dust and H2 and the Emergent Spectrum]]

[[https://arxiv.org/pdf/astro-ph/0411259.pdf|Hirashita & Ferrara 2005 - Molecular hydrogen in damped Lyα systems: clues to interstellar physics at high redshift]]

## No detection of Molecules in GRBs:

[[http://iopscience.iop.org/article/10.1086/521294/pdf| Tumlinson et al. 2007 - Missing Molecular Hydrogen and the Physical Conditions of GRB Host Galaxies]]

[[https://www.aanda.org/articles/aa/pdf/2009/41/aa11572-08.pdf|Ledoux et al. 2009 - Physical conditions in high-redshift GRB-DLA absorbers observed with VLT/UVES: implications for molecular hydrogen searches]]

[[https://www.aanda.org/articles/aa/pdf/2010/15/aa15216-10.pdf|D'Elia et al. 2010 - VLT/X-shooter spectroscopy of the GRB 090926A afterglow]]

## Detection of Molecules in GRBs:

[[https://www.aanda.org/articles/aa/pdf/2006/21/aa5056-06.pdf|Fynbo et al. 2006 - Probing cosmic chemical evolution with gamma-ray bursts: GRB 060206 at z = 4.048]]

[[http://iopscience.iop.org/article/10.1088/0004-637X/701/2/L63/pdf| Sheffer et al. 2009 - GRB 080607]]

[[http://iopscience.iop.org/article/10.1088/0004-637X/691/1/L27/pdf| Prochaska et al. 2009 - GRB 080607]]

[[https://www.aanda.org/articles/aa/pdf/2013/09/aa21772-13.pdf| Krühler et al. 2013 - GRB 120815A]]

[[https://www.aanda.org/articles/aa/pdf/2014/04/aa23057-13.pdf| D’Elia et al. 2014 - GRB 120327A]]

[[http://mnras.oxfordjournals.org/content/451/1/167.full.pdf| Friis et al. 2015 - GRB 121024A]]

## Similar but on QSOs:

[Levshakov & Varshalovich 1985 - Molecular hydrogen in the z=2.811 absorbing material toward the quasar PKS 0528-250](http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1985MNRAS.212..517L&amp;data_type=PDF_HIGH&amp;whole_paper=YES&amp;type=PRINTER&amp;filetype=.pdf)

[Ledoux et al. 2002 - Detection of molecular hydrogen in a near Solar-metallicity damped Lyman-α system at zabs ≈ 2 toward Q 0551−366](https://www.aanda.org/articles/aa/pdf/2002/47/aah2875e.pdf)

[Petitjean et al. 2002 - Molecular hydrogen at z = 1.973 toward Q0013-004: dust depletion pattern in damped Lyman α systems](https://arxiv.org/pdf/astro-ph/0201477.pdf)

[Reimers et al. 2003 - Detection of molecular hydrogen at z = 1.15 toward HE 0515-4414](https://www.aanda.org/articles/aa/pdf/2003/42/aah4641.pdf)

[Ledoux et al. 2003 - The Very Large Telescope Ultraviolet and Visible Echelle Spectrograph survey for molecular hydrogen in high-redshift damped Lyman α systems](https://arxiv.org/pdf/astro-ph/0302582.pdf)

[Srianand et al. 2005 - The VLT-UVES survey for molecular hydrogen in high-redshift damped Lyman α systems: physical conditions in the neutral gas](https://arxiv.org/pdf/astro-ph/0506555.pdf)

[Cui et al. 2005 - Molecular Hydrogen in the Damped Lyα Absorber of Q1331+170](http://iopscience.iop.org/article/10.1086/444368/pdf)

[Ledoux et al. 2006 - Molecular hydrogen in a DLA Az z = 4.224](http://iopscience.iop.org/article/10.1086/503278/pdf)

[Gillmon et al. 2006 - A FUSE Survey Of Interstellar Molecular Hydrogen Toward High-Latitude AGNs](http://iopscience.iop.org/article/10.1086/498053/pdf)

[Petitjean et al. 2006 - Metallicity as a criterion to select H_2-bearing damped Lyman-α systems](https://www.aanda.org/articles/aa/pdf/2006/36/aa5769-06.pdf)

[Noterdaeme et al. 2007 - Excitation mechanisms in newly discovered H_2-bearing damped Lyman-α clouds: systems with low molecular fractions](https://www.aanda.org/articles/aa/pdf/2007/41/aa8021-07.pdf)

[Noterdaeme et al. 2007 - Physical conditions in the neutral interstellar medium at z = 2.43 toward Q 2348-011](https://www.aanda.org/articles/aa/pdf/2007/26/aa6897-06.pdf)

[Noterdaeme et al. 2008 - Molecular hydrogen in high-redshift damped Lyman-α systems: the VLT/UVES database](https://www.aanda.org/articles/aa/pdf/2008/14/aa8780-07.pdf)

[Fynbo et al. 2011 - Galaxy counterparts of metal-rich damped Lyα absorbers – II. A solar-metallicity and dusty DLA at z = 2.58](https://arxiv.org/pdf/1011.5312.pdf)

[Balashev et al. 2015 - Neutral chlorine and molecular hydrogen at high redshift](https://www.aanda.org/articles/aa/pdf/2015/03/aa25553-14.pdf)

## Other Related Papers:

[Tumlinson et al. 2002 - A Far Ultraviolet Spectroscopic Explorer Survey of Interstellar Molecular Hydrogen in the Small and Large Magellanic Clouds](http://iopscience.iop.org/article/10.1086/338112/pdf)

[De Cia et al. 2013 - Dust-to-metal ratios in damped Lyman-α absorbers](https://www.aanda.org/articles/aa/pdf/2013/12/aa21834-13.pdf)

[Cucchiara et al. 2015 - Unveiling the Secrets of Metallicity and Massive Star Formation Using DLAs along Gamma-Ray Bursts](http://iopscience.iop.org/article/10.1088/0004-637X/804/1/51/pdf)

[Wiseman et al. 2017 - Evolution of the dust-to-metals ratio in high-redshift galaxies probed by GRB-DLAs](https://www.aanda.org/articles/aa/pdf/2017/03/aa29228-16.pdf)

## Molecular Absorption From CH+:

[Fynbo et al. 2014 - The mysterious optical afterglow spectrum of GRB 140506A at z = 0.889](https://www.aanda.org/articles/aa/pdf/2014/12/aa24726-14.pdf)
