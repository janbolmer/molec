MCMC sampler for fitting X-shooter spectra (using PyMC2) with HI, H2 and
other lines

e.g.:
molec_mcmc.py -f spectra/GRB120815Auvb.txt -red 2.358 -t GRB120815A
-e HI FeII SiII NV SII MgII SiIIa SIV ZnII FeIIa ArI OVI
-w1 985 -w2 1120 -m H2 -nrot 5 -it 200 -bi 10 -par para.csv


!!! Under Construction - I am learning Git here !!!

I am making use of the H2sim from Thomas Kruehler: https://github.com/Kruehlio/H2sim

