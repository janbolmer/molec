python velop_mcmc.py -f spectra/GRB161023A_OB1UVB.txt -z 2.710 -e NV    -line 1238.8210 -vr 380 -w 6  -min 1 -max 1 -it 40000 -bi 30000 -res 46.2
python velop_mcmc.py -f spectra/GRB161023A_OB1UVB.txt -z 2.710 -e NV    -line 1242.8040 -vr 380 -w 6  -min 1 -max 1 -it 40000 -bi 30000 -res 46.2
python velop_mcmc.py -f spectra/GRB161023A_OB1UVB.txt -z 2.710 -e SII   -line 1253.8110 -vr 380 -w 7  -min 3 -max 3 -it 40000 -bi 30000 -res 46.2 -par para_files/161023A_velo_para.csv
python velop_mcmc.py -f spectra/GRB161023A_OB1UVB.txt -z 2.710 -e OI    -line 1302.1685 -vr 450 -w 6  -min 3 -max 3 -it 40000 -bi 30000 -res 46.2 -par para_files/161023A_velo_para.csv
python velop_mcmc.py -f spectra/GRB161023A_OB1UVB.txt -z 2.710 -e SiIV  -line 1393.7602 -vr 380 -w 7  -min 3 -max 3 -it 40000 -bi 30000 -res 46.2
python velop_mcmc.py -f spectra/GRB161023A_OB1UVB.txt -z 2.710 -e SiIV  -line 1402.7729 -vr 480 -w 12  -min 3 -max 3 -it 40000 -bi 30000 -res 46.2 -par para_files/161023A_velo_para2.csv
python velop_mcmc.py -f spectra/GRB161023A_OB1VIS.txt -z 2.710 -e FeII  -line 1608.4509 -vr 380 -w 8  -min 3 -max 3 -it 40000 -bi 30000 -res 26.5 -par para_files/161023A_velo_para.csv
#python velop_mcmc.py -f spectra/GRB161023A_OB1VIS.txt -z 2.710 -e AlII  -line 1670.7886 -vr 380 -w 9  -min 3 -max 3 -it 20000 -bi 15000 -res 26.5 -par para_files/161023A_velo_para.csv
python velop_mcmc.py -f spectra/GRB161023A_OB1VIS.txt -z 2.710 -e SiII  -line 1808.0129 -vr 380 -w 9  -min 3 -max 3 -it 50000 -bi 40000 -res 26.5 -par para_files/161023A_velo_para.csv
python velop_mcmc.py -f spectra/GRB161023A_OB1VIS.txt -z 2.710 -e FeII  -line 2260.7793 -vr 380 -w 9  -min 2 -max 2 -it 50000 -bi 40000 -res 26.5 -par para_files/161023A_velo_para.csv
python velop_mcmc.py -f spectra/GRB161023A_OB1VIS.txt -z 2.710 -e FeII  -line 2344.2129 -vr 380 -w 11 -min 3 -max 3 -it 50000 -bi 40000 -res 26.5 -par para_files/161023A_velo_para.csv
python velop_mcmc.py -f spectra/GRB161023A_OB1VIS.txt -z 2.710 -e FeIIa -line 2333.5147 -vr 380 -w 10 -min 3 -max 3 -it 50000 -bi 40000 -res 26.5 -par para_files/161023A_velo_para.csv
python velop_mcmc.py -f spectra/GRB161023A_OB1VIS.txt -z 2.710 -e FeII  -line 2374.4604 -vr 380 -w 11 -min 3 -max 3 -it 50000 -bi 40000 -res 26.5 -par para_files/161023A_velo_para.csv
#python velop_mcmc.py -f spectra/GRB161023A_OB1VIS.txt -z 2.710 -e FeIIb -line 2405.6173 -vr 380 -w 11 -min 3 -max 3 -it 20000 -bi 15000 -res 26.5 -par para_files/161023A_velo_para.csv