import os


class Config:
    def __init__(self):
        self.docker = True if 'DOCKERCONTAINER' in os.environ else False

        self.env = parseEnvFile('../.env')

        self.THREADS   = 20
        self.PROCESSES = 8

        self.BASE_DIR = self.env['DB_DIR']
        self.TRAIN_DIR = self.BASE_DIR + '/train'

        self.DATA_DIR              = '/data' if self.docker else self.BASE_DIR + '/data'
        self.DATA_DIR_24_HOUR  = self.DATA_DIR + '/24_HOUR'
        self.DATA_DIR_5_MINUTE = self.DATA_DIR + '/5_MINUTE'

        self.PORT_DIR          = self.BASE_DIR + '/portfolios'

        self.LOG_DIR  = '/logs' if self.docker else self.BASE_DIR + '/logs'

        self.RISK_FREE_RATE = 0.0125


def parseEnvFile(envPath):
    if not os.path.exists(envPath):
        print(f'ERROR: No .env file found at {envPath}!')

    with open(envPath, 'r') as rf:
        envList = [val.split('=') for val in rf.read().split('\n')]
    envList = [e for e in envList if len(e) > 1]
    return {e[0] : e[1] for e in envList}
