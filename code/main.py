import os
import sys
sys.path.append(".")
sys.path.append("../../")

if __name__ == '__main__':
    os.system('python3 code/sr_gnn_main.py')
    os.system('python3 code/recall_main.py')
    os.system('python3 code/rank_main.py')
