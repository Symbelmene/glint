import os

class Config:
    def __init__(self):
        self.env = parseEnvFile('../.env')

        self.DATA_DIR = self.env['DB_DIR']
        self.LOG_DIR  = self.env['LOG_DIR']

        self.DATA_DIR_RAW = self.DATA_DIR + '/RAW'

        self.DATA_DIR_RAW_24H = self.DATA_DIR_RAW + '/24H'
        self.DATA_DIR_RAW_15M = self.DATA_DIR_RAW + '/15M'


def parseEnvFile(envPath):
    if not os.path.exists(envPath):
        print(f'ERROR: No .env file found at {envPath}!')

    with open(envPath, 'r') as rf:
        envList = [val.split('=') for val in rf.read().split('\n')]

    return {e[0] : e[1] for e in envList}