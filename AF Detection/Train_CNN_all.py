
#!/usr/bin/env python
# coding: utf-8

import os
import sys
from utils import excute_command

if __name__ == "__main__":
	pool_factor = int(sys.argv[1])
	script_path = "Train_CNN.py"
	cross = 5
	repeat = 10
	for blur_size in [0,3,4,5,-1,6,7,1,2]:
		for cross_idx in range(cross):
			for repeat_idx in range(repeat):
				argvs = [cross_idx,repeat_idx,blur_size,pool_factor]
				excute_command(script_path,argvs)