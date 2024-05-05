from . import data_configs as dc
from pickle import load, dump
from os import remove
from os.path import exists

def loadCache(fileName, check_exists=True):
    file_path = dc.outputFolders()['pickles'] + '\\' + fileName
    # this is a new argument, there shouldn't be a case this is
    # needed. Remove in future
    if check_exists:
        if exists(file_path):
            return load(open(file_path,'rb'))
        else:
            print(f"Pickle file not found: {file_path}")
            return None
    else:
        return load(open(file_path,'rb'))


def saveCache(varToSave, fileName):
    dump(varToSave, open(dc.outputFolders()['pickles'] + '\\' + fileName,'wb'))

def deleteCache(fileName):
    filePath = dc.outputFolders()['pickles'] + '\\' + fileName
    if exists(filePath):
        remove(filePath)

def loadOnlyExps(experimentes):
    #modelsIn = loadCache('dataModels')
    models = []
    #for mod in modelsIn:
    #    if mod['experimentID'] in experimentes:
    #        models.append(mod)
        
    for exp in experimentes:
        models.append(loadCache('dataModel_' + exp))
    return models