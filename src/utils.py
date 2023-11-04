import os
import math
import numpy as np
import pandas as pd
import plotly.express as px

import warnings
warnings.simplefilter("ignore")

from tqdm import tqdm
from tabulate import tabulate
from plotly import graph_objects as go
from multiprocessing import Pool
from datetime import datetime
from finclasses import addBaseIndicatorsToDf

import utils
import finclasses

from config import Config, Interval
cfg = Config()


def log(e, display=True):
    if display:
        print(e, flush=True)

    if not os.path.exists(cfg.LOG_DIR):
        os.makedirs(cfg.LOG_DIR)

    logPath = f'{cfg.LOG_DIR}/log.txt'
    if not os.path.exists(logPath):
        with open(logPath, 'w+') as wf:
            wf.write(f'{datetime.now()}: {e}\n')
    else:
        with open(logPath, 'a') as wf:
            wf.write(f'{datetime.now()}: {e}\n')
