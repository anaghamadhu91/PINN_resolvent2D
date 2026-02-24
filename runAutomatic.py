import numpy as np
import sys
import os

run_set = 1
run_exp_list = np.arange(1,108+1,5)
#run_exp_list = [1, 25, 43, 52, 70, 76, 79, 97, 103, 106]
#run_exp_list = [9, 71, 87, 99]
#run_exp_list = [13]
#run_exp_list = [99]
#run_exp_list = [43]

for run_exp in run_exp_list:
    os.system("python runAutomatic_resolvent2D.py %03d %03d"%(run_set,run_exp))
