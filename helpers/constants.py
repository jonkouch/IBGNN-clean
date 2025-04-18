import os

DECIMAL = 6
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD_PREFIX_STR = f"Results/"

MAX_NUM_COMMUNITIES = 2000
EPS = 1E-8
TIME_STEPS = 24
BASELINES = {'tolokers': 0.79, 'squirrel': 0.67, 'chameleon': 0.01}
