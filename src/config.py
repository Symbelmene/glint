import os

class Config:
    def __init__(self):
        self.docker = True if 'DOCKERCONTAINER' in os.environ else False

        self.env = parseEnvFile('../.env')

        self.THREADS  = 8

        self.DATA_DIR = '/data' if self.docker else self.env['DB_DIR']
        self.LOG_DIR  = '/logs' if self.docker else self.env['LOG_DIR']

        self.DATA_DIR_RAW   = self.DATA_DIR + '/RAW'
        self.DATA_DIR_CLEAN = self.DATA_DIR + '/CLEAN'

        self.DATA_DIR_RAW_24H   = self.DATA_DIR_RAW + '/24H'
        self.DATA_DIR_CLEAN_24H = self.DATA_DIR_CLEAN + '/24H'

        self.DATA_DIR_RAW_5M    = self.DATA_DIR_RAW + '/5M'
        self.DATA_DIR_CLEAN_5M  = self.DATA_DIR_CLEAN + '/5M'


def parseEnvFile(envPath):
    if not os.path.exists(envPath):
        print(f'ERROR: No .env file found at {envPath}!')

    with open(envPath, 'r') as rf:
        envList = [val.split('=') for val in rf.read().split('\n')]
    envList = [e for e in envList if len(e) > 1]
    return {e[0] : e[1] for e in envList}
