import os
import traceback


class Config:
    def __init__(self):
        self.docker = True if 'DOCKERCONTAINER' in os.environ else False
        self.env = parseEnvFile('../.env')

        # Postgres details
        self.STORER_DB_NAME  = self.env['STORER_NAME']
        self.STORER_HOST     = self.env['STORER_HOST']
        self.STORER_PORT     = self.env['STORER_PORT']
        self.STORER_USER     = self.env['STORER_USER']
        self.STORER_PASSWORD = self.env['STORER_PASSWORD']


def parseStrToBool(inputStr):
    if inputStr.lower() == 'true':
        return True
    if inputStr.lower() == 'false':
        return False
    raise TypeError


def parseEnvFile(path):
    if not os.path.exists(path):
        raise FileNotFoundError

    with open(path, 'r') as rf:
        contents = rf.read()

    lines = contents.split('\n')

    envDict = {}
    for idx, line in enumerate(lines):
        if not line:
            continue

        if line[0] == '#':
            continue

        invalidMessage = f'Invalid .config file on line {idx} ({line})'

        try:
            lineSplit = line.replace(' ', '')
            firstEquals = lineSplit.find('=')
            lineSplit = [lineSplit[:firstEquals], lineSplit[firstEquals + 1:]]
        except:
            print(invalidMessage)
            print(traceback.format_exc())
            raise Exception

        if len(lineSplit) != 2:
            print(invalidMessage)
            raise Exception

        envDict[lineSplit[0]] = lineSplit[1]

    return envDict
