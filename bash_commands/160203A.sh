python velop_mcmc.py -f spectra/GRB160203A_comVIS.txt -z 3.5188 -e SiII  -line 1253.8110 -vr 350 -w 8 -min 3 -max 3 -it 22000 -bi 18000 -res 21.7 # HI
python velop_mcmc.py -f spectra/GRB160203A_comVIS.txt -z 3.5188 -e NiII  -line 1370.1323 -vr 350 -w 8 -min 3 -max 3 -it 22000 -bi 18000 -res 21.7 
python velop_mcmc.py -f spectra/GRB160203A_comVIS.txt -z 3.5188 -e CrIIa -line 1431.3812 -vr 350 -w 8 -min 2 -max 2 -it 22000 -bi 18000 -res 21.7  -par para_files/160203A_velo_para.csv
python velop_mcmc.py -f spectra/GRB160203A_comVIS.txt -z 3.5188 -e CIV 	 -line 1548.2040 -vr 350 -w 8 -min 2 -max 2 -it 22000 -bi 18000 -res 21.7 
python velop_mcmc.py -f spectra/GRB160203A_comVIS.txt -z 3.5188 -e CIV 	 -line 1550.7810 -vr 350 -w 8 -min 2 -max 2 -it 22000 -bi 18000 -res 21.7
python velop_mcmc.py -f spectra/GRB160203A_comVIS.txt -z 3.5188 -e FeII  -line 1608.4509 -vr 350 -w 8 -min 2 -max 2 -it 22000 -bi 18000 -res 21.7 
python velop_mcmc.py -f spectra/GRB160203A_comVIS.txt -z 3.5188 -e FeII  -line 1611.2004 -vr 350 -w 8 -min 2 -max 2 -it 22000 -bi 18000 -res 21.7 
python velop_mcmc.py -f spectra/GRB160203A_comVIS.txt -z 3.5188 -e AlII  -line 1670.7886 -vr 350 -w 8 -min 2 -max 2 -it 22000 -bi 18000 -res 21.7 
python velop_mcmc.py -f spectra/GRB160203A_comVIS.txt -z 3.5188 -e NiII  -line 1709.6001 -vr 350 -w 9 -min 2 -max 2 -it 22000 -bi 18000 -res 21.7 -ign 140,220
python velop_mcmc.py -f spectra/GRB160203A_comVIS.txt -z 3.5188 -e NiII  -line 1741.5486 -vr 350 -w 9 -min 2 -max 2 -it 22000 -bi 18000 -res 21.7 -ign 100,170 -par para_files/160203A_velo_para.csv
python velop_mcmc.py -f spectra/GRB160203A_comVIS.txt -z 3.5188 -e SiII  -line 1808.0120 -vr 350 -w 9 -min 2 -max 2 -it 22000 -bi 18000 -res 21.7 
python velop_mcmc.py -f spectra/GRB160203A_comVIS.txt -z 3.5188 -e AlIII -line 1862.7911 -vr 350 -w 9 -min 2 -max 2 -it 22000 -bi 18000 -res 21.7
python velop_mcmc.py -f spectra/GRB160203A_comVIS.txt -z 3.5188 -e NiIIb -line 2053.9511 -vr 350 -w 9 -min 2 -max 2 -it 22000 -bi 18000 -res 21.7