import os
import time
from datetime import datetime


def log_message(text, display=True, startTime=None):
    if startTime:
        endTime = time.time()
        text = f'{str(round(endTime - startTime, 2)).rjust(5)}s: {text}'

    if display:
        print(f'{str(datetime.now())[:-3]}: {str(os.getpid()).rjust(3)}: {text}', flush=True)

    return time.time()
