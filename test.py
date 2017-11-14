import numpy as np 

e = 4.8032E-10  # cm 3/2 g 1/2 s-1
c = 2.998e10 # speed of light cm/s
m_e = 9.10938291e-28 # electron mass g

print np.log10(28.414/(((np.pi * e**2)/(m_e* c)) * 0.00208 * 1808.0129E-13))


