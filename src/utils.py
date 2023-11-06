import os
import warnings
warnings.simplefilter("ignore")

from datetime import datetime
from config import Config
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
