
from os import path, mkdir
import numpy as np
import pandas as pd
from math import floor
from copy import deepcopy

figNames = {
    'normal': 'Intact',
    'cut': 'Transected',
    'scaffold': 'Implant'
}
figColours = {
    'normal':   "mediumseagreen",
    'cut':      "#fdae61",
    'scaffold': "#d7191c",
    'cut2':     "#2b83ba",#"#ffffbf",
    'scaffold2': "#1a9641",#"#a6611a",
    '0':        "green",
}

figColours = {
    'normal':   "mediumseagreen",
    'cut':      "#d7191c", # red
    'scaffold': "#2b83ba", # blue
    'cut2':     "#fdae61",#"#ffffbf",
    'scaffold2': "#1a9641",#"#a6611a",
    '0':        "green",
}
# 'christmas' colours
#color=['green']
#color=['mediumseagreen'], 
#color=['red']
#color=['lightcoral']

groupInfoFE = {
    'type': 'FE',
    'files': [
        r'\cut_fe_40_40\log1',
        r'\cut_fe_40_40\log2',
        r'\normal_fe_40_40\log1',
        r'\normal_fe_40_40\log2',
        r'\scaffold_fe_40_40\log1',
        r'\scaffold_fe_40_40\log2',
    ],
    'names': [
        'cut',
        'cut',
        'normal',
        'normal',
        'scaffold',
        'scaffold',
    ]
}
groupInfoUR = {
    'type': 'UR',
    'files': [
        r'\cut_ur_30_30\log1',
        r'\cut_ur_30_30\log2',
        r'\normal_ur_30_30\log1',
        r'\normal_ur_30_30\log2',
        r'\scaffold_ur_30_30\log1',
        r'\scaffold_ur_30_30\log2',
    ],
    'names': [
        'cut',
        'cut',
        'normal',
        'normal',
        'scaffold',
        'scaffold',
    ]
}

def createDataD(modelInfo, group):
    #fileExtBadpoints = r'_badPoints.sto'
    fileExtModelKinematics = r'_model_kinematics.sto'
    fileExtPosLunateBoneplug = r'_pos_lunate_boneplug.sto'
    fileExtPosScaphoidBoneplug = r'_pos_scaphoid_boneplug.sto'
    dataOutputDir = modelInfo['dataOutputDir']
    dataInputDir = modelInfo['dataInputDir']
    data = []
    axis = 'y'
    if ('UR' in group['type']):
        axis = 'x'
    for ind, session in enumerate(group['files']):
        testTemp = loadTest(dataInputDir + session)
        data.append({
            'file': session,
            'title': group['names'][ind],
            #'robotData': testTemp,
            'time': testTemp['time'].to_numpy(),
            'rot': testTemp['feedback/rot/' + axis].to_numpy() * (180.0/np.pi),
            'lunate': getData(dataOutputDir + session + fileExtPosLunateBoneplug).to_numpy(),
            'scaphoid': getData(dataOutputDir + session + fileExtPosScaphoidBoneplug).to_numpy(),
            'kinematics': getDataIK(dataOutputDir + session + fileExtModelKinematics),
            'type': group['type']
        })
        data[ind].update({'difference': calcDifNP(data[ind]['lunate'],data[ind]['scaphoid'])})
    return data

def createDataD_woutKinematics(modelInfo, group):
    #fileExtBadpoints = r'_badPoints.sto'
    #fileExtModelKinematics = r'_model_kinematics.sto'
    fileExtPosLunateBoneplug = r'_pos_lunate_boneplug.sto'
    fileExtPosScaphoidBoneplug = r'_pos_scaphoid_boneplug.sto'
    dataOutputDir = modelInfo['dataOutputDir']
    dataInputDir = modelInfo['dataInputDir']
    data = []
    axis = 'y'
    if ('UR' in group['type']):
        axis = 'x'
    for ind, session in enumerate(group['files']):
        testTemp = loadTest(dataInputDir + session)
        data.append({
            'file': session,
            'title': group['names'][ind],
            #'robotData': testTemp,
            'time': testTemp['time'].to_numpy(),
            'rot': testTemp['feedback/rot/' + axis].to_numpy() * (180.0/np.pi),
            'lunate': getData(dataOutputDir + session + fileExtPosLunateBoneplug).to_numpy(),
            'scaphoid': getData(dataOutputDir + session + fileExtPosScaphoidBoneplug).to_numpy(),
            #'kinematics': getDataIK(dataOutputDir + session + fileExtModelKinematics),
            'type': group['type']
        })
        data[ind].update({'difference': calcDifNP(data[ind]['lunate'],data[ind]['scaphoid'])})
    return data

def createDataD_woutLandS(modelInfo, group):
    #fileExtBadpoints = r'_badPoints.sto'
    fileExtModelKinematics = r'_model_kinematics.sto'
    #fileExtPosLunateBoneplug = r'_pos_lunate_boneplug.sto'
    #fileExtPosScaphoidBoneplug = r'_pos_scaphoid_boneplug.sto'
    dataOutputDir = modelInfo['dataOutputDir']
    dataInputDir = modelInfo['dataInputDir']
    data = []
    axis = 'y'
    if ('UR' in group['type']):
        axis = 'x'
    for ind, session in enumerate(group['files']):
        testTemp = loadTest(dataInputDir + session)
        data.append({
            'file': session,
            'title': group['names'][ind],
            #'robotData': testTemp,
            'time': testTemp['time'].to_numpy(),
            'rot': testTemp['feedback/rot/' + axis].to_numpy() * (180.0/np.pi),
            #'lunate': getData(dataOutputDir + session + fileExtPosLunateBoneplug).to_numpy(),
            #'scaphoid': getData(dataOutputDir + session + fileExtPosScaphoidBoneplug).to_numpy(),
            'kinematics': getDataIK(dataOutputDir + session + fileExtModelKinematics),
            'type': group['type']
        })
        #data[ind].update({'difference': calcDifNP(data[ind]['lunate'],data[ind]['scaphoid'])})
    return data

def selectOnlyExps(modelsIn, experimentes):
    models = []
    for mod in modelsIn:
        if mod['experimentID'] in experimentes:
            models.append(mod)
    return models

def loadSTO(file,srows, columns = []):
    if not path.exists(file) or path.getsize(file) <= 0:
        print('Error: File does not exist: ' + str(file))
        return pd.DataFrame()
    if (srows==0):
        with open(file, "r") as f:
            for line in f:
                if 'time\t' in line:
                    break
                srows += 1
    fileContents = pd.read_csv(file, skiprows = srows, sep='\t')
    if len(columns) == 0:
        data = pd.DataFrame(fileContents)
    else:
        data = pd.DataFrame(fileContents, columns=columns)
    return data

def getDataIK(name, skiprows = 0):
    columnNames = [
        "time",
        "uln_xrot",
        "uln_yrot",
        "uln_zrot",
        "uln_xtran",
        "uln_ytran",
        "uln_ztran",
        "sca_xrot",
        "sca_yrot",
        "sca_zrot",
        "sca_xtran",
        "sca_ytran",
        "sca_ztran",
        "lunate_flexion",
        "lunate_deviation",
        "lunate_rotation",
        "lunate_xtrans",
        "lunate_ytrans",
        "lunate_ztrans",
        "hand_flexion",
        "hand_deviation",
        "hand_rotation",
        "hand_xtrans",
        "hand_ytrans",
        "hand_ztrans"
    ]
    return loadSTO(str(name), skiprows, columnNames)

def getData(name, skiprows = 0):
    return loadSTO(str(name), skiprows)

def calcDif(a, b):
    differenceX = a['lunate_boneplug_X'].to_numpy() - b['scaphoid_boneplug_X'].to_numpy()
    differenceY = a['lunate_boneplug_Y'].to_numpy() - b['scaphoid_boneplug_Y'].to_numpy()
    differenceZ = a['lunate_boneplug_Z'].to_numpy() - b['scaphoid_boneplug_Z'].to_numpy()

    difference = (np.sqrt(np.power(differenceX,2)+np.power(differenceY,2)+np.power(differenceZ,2))*1000)
    return difference

def calcDifNP(a, b):
    differenceX = a[:,1] - b[:,1]
    differenceY = a[:,2] - b[:,2]
    differenceZ = a[:,3] - b[:,3]

    difference = (np.sqrt(np.power(differenceX,2)+np.power(differenceY,2)+np.power(differenceZ,2))*1000)
    return difference

def loadTest(file, skiprows = 0):
    fileContents = pd.read_csv(file + '.csv', skiprows = skiprows)
    markerNames = [s for s in fileContents.columns if "marker" in s]
    columnNames = []
    columnNames.extend(markerNames)
    columnNames.append('desired/time')
    columnNames.append('desired/rot/x')
    columnNames.append('desired/rot/y')
    data = pd.DataFrame(fileContents)#, columns=columnNames)
    data = data.rename(columns={'desired/time':'time'})
    data['time'] = data['time']/1000000 # ns to seconds
    return data

def loadTest2(file, skiprows=0):
    fileContents = pd.read_csv(file + '.csv', skiprows = skiprows)
    markerNames = [s for s in fileContents.columns if "marker" in s]
    columnNames = []
    columnNames.extend(markerNames)
    columnNames.append('desired/time')
    columnNames.append('desired/rot/x')
    columnNames.append('desired/rot/y')
    data = pd.DataFrame(fileContents, columns=columnNames)
    data = data.rename(columns={'desired/time':'time'})
    data['time'] = data['time']/1000000 # ns to seconds
    # fix for now
    for i in (range(len(markerNames))):
        if (np.isnan(data[markerNames[i]].iloc[-1])):
            data[markerNames[i]].iloc[-1] = data[markerNames[i]].iloc[-2]
    return data

def convertToNumpyArrays(dataIn):
    dataOut = {}

    dataOut['desired/rot/x'] = dataIn['desired/rot/x'].to_numpy()
    dataOut['desired/rot/y'] = dataIn['desired/rot/y'].to_numpy()
    dataOut['time'] = dataIn['time'].to_numpy()

    markerNames = [s for s in dataIn.columns if "marker" in s]
    for i, a in enumerate(markerNames):
        # This check is bad for arrays which begin with 0 but have values later on
        #if (not max(dataIn[a]) == 0):
        dataOut[a] = dataIn[a].to_numpy()

    return dataOut
    
def checkAndCreateFolderExists( pathIn ):
    # fileName is 'log'
    pathToTest, fileName = path.split( pathIn )
    if (not path.isdir(pathToTest)):
        print('Creating folder: ' + str(pathToTest))
        mkdir(pathToTest)
    
def convertRelativeToRadius(dataIn):
    for data in dataIn:
        data['kinematics']['lunate_flexion']   += data['kinematics']['sca_xrot']
        data['kinematics']['lunate_deviation'] += data['kinematics']['sca_yrot']
        data['kinematics']['lunate_rotation']  += data['kinematics']['sca_zrot']
        #data['kinematics']['lunate_xtrans'] -= data['kinematics']['sca_xtran']
        #data['kinematics']['lunate_ytrans'] -= data['kinematics']['sca_ytran']
        #data['kinematics']['lunate_ztrans'] -= data['kinematics']['sca_ztran']

        data['kinematics']['hand_flexion']   += data['kinematics']['sca_xrot']
        data['kinematics']['hand_deviation'] += data['kinematics']['sca_yrot']
        data['kinematics']['hand_rotation']  += data['kinematics']['sca_zrot']
        #data['kinematics']['hand_xtrans'] -= data['kinematics']['sca_xtran']
        #data['kinematics']['hand_ytrans'] -= data['kinematics']['sca_ytran']
        #data['kinematics']['hand_ztrans'] -= data['kinematics']['sca_ztran']
    return dataIn     

def cropData3(dataIn, offset = 2083, cycleLength = 8333):
    cl = cycleLength
    of = offset
    tmpCrop = []
    len1 = len(dataIn[0]['time'])
    for data in dataIn:

        # first run may not be good data
        for i in range(floor((len1-of)/cl)):
            j = i+1
            cl1 = of + (cl*i)
            cl2 = of + (cl*j)
            tmp = deepcopy(data)
            tmp['difference'] = tmp['difference'][cl1:cl2]
            tmp['time'] = tmp['time'][cl1:cl2]
            tmp['rot'] = tmp['rot'][cl1:cl2]
            tmp['kinematics'] = tmp['kinematics'][cl1:cl2]
            #tmp['kinematics']['time'] = tmp['kinematics']['time'][cl1:cl2]
            #tmp['kinematics']['time'] = tmp['kinematics']['time'] - tmp['kinematics']['time'][cl1]
            tmpCrop.append(tmp)

    return tmpCrop

def cropData3_noKinematics(dataIn, offset = 2083, cycleLength = 8333):
    cl = cycleLength
    of = offset
    tmpCrop = []
    len1 = len(dataIn[0]['time'])
    for data in dataIn:

        # first run may not be good data
        for i in range(floor((len1-of)/cl)):
            j = i+1
            cl1 = of + (cl*i)
            cl2 = of + (cl*j)
            tmp = deepcopy(data)
            del(tmp['lunate'])
            del(tmp['scaphoid'])
            tmp['difference'] = tmp['difference'][cl1:cl2]
            tmp['time'] = tmp['time'][cl1:cl2]
            tmp['rot'] = tmp['rot'][cl1:cl2]
            tmpCrop.append(tmp)

    return tmpCrop

def cropDataRemoveBeginning(dataIn, removedInds = 4165):
    tmpCrop = []
    for data in dataIn:
        tmp = deepcopy(data)
        if "difference" in tmp:
            tmp['difference'] = tmp['difference'][removedInds:]
        tmp['time'] = tmp['time'][removedInds:]
        tmp['rot'] = tmp['rot'][removedInds:]
        tmp['kinematics'] = tmp['kinematics'][removedInds:]
        #tmp['kinematics']['time'] = tmp['kinematics']['time']
        tmpCrop.append(tmp)
    return tmpCrop

def cropDataToWholeCycles(dataIn, cycleLength = 8333):
    # Crops data to second and third cycle
    cl = cycleLength
    tmpCrop = []
    for data in dataIn:
        
        # first run may not be good data
        tmp = deepcopy(data)
        if "difference" in tmp:
            tmp['difference'] = tmp['difference'][:cl]
        tmp['time'] = tmp['time'][:cl]
        tmp['rot'] = tmp['rot'][:cl]
        tmp['kinematics'] = tmp['kinematics'][:cl]
        tmp['kinematics']['time'] = tmp['kinematics']['time']
        tmpCrop.append(tmp)

        tmp = deepcopy(data)
        if "difference" in tmp:
            tmp['difference'] = tmp['difference'][cl:cl*2]
        tmp['time'] = tmp['time'][cl:cl*2] - tmp['time'][cl]
        tmp['rot'] = tmp['rot'][cl:cl*2]
        tmp['kinematics']['time'] = tmp['kinematics']['time'] - tmp['kinematics']['time'][cl]
        tmp['kinematics'] = tmp['kinematics'][cl:cl*2]
        tmpCrop.append(tmp)
        
        tmp = deepcopy(data)
        if (len(tmp['time'])>cl*3 and len(tmp['kinematics'])>cl*3):
            if "difference" in tmp:
                tmp['difference'] = tmp['difference'][cl*2:cl*3]
            tmp['time'] = tmp['time'][cl*2:cl*3] - tmp['time'][cl*2]
            tmp['rot'] = tmp['rot'][cl*2:cl*3]
            tmp['kinematics']['time'] = tmp['kinematics']['time'] - tmp['kinematics']['time'][cl*2]
            tmp['kinematics'] = tmp['kinematics'][cl*2:cl*3]
            tmpCrop.append(tmp)
        
        # never run unless a fourth cycle is captured
        tmp = deepcopy(data)
        if (len(tmp['time'])>cl*4 and len(tmp['kinematics'])>cl*4):
            if "difference" in tmp:
                tmp['difference'] = tmp['difference'][cl*3:cl*4]
            tmp['time'] = tmp['time'][cl*3:cl*4] - tmp['time'][cl*3]
            tmp['rot'] = tmp['rot'][cl*3:cl*4]
            tmp['kinematics']['time'] = tmp['kinematics']['time'] - tmp['kinematics']['time'][cl*3]
            tmp['kinematics'] = tmp['kinematics'][cl*3:cl*4]
            tmpCrop.append(tmp)
    return tmpCrop

def cropDataMinLoss(dataIn, cycleLength = 8333):
    cl = cycleLength
    tmpCrop = []
    for data in dataIn:
        
        # first run may not be good data
        tmp = deepcopy(data)
        tmp['difference'] = tmp['difference'][2083:cl]
        tmp['time'] = tmp['time'][2083:cl]
        tmp['rot'] = tmp['rot'][2083:cl]
        tmp['kinematics'] = tmp['kinematics'][2083:cl]
        tmp['kinematics']['time'] = tmp['kinematics']['time'] - tmp['kinematics']['time'][2083]
        tmpCrop.append(tmp)

        tmp = deepcopy(data)
        tmp['difference'] = tmp['difference'][cl:cl*2]
        tmp['time'] = tmp['time'][cl:cl*2] - tmp['time'][cl]
        tmp['rot'] = tmp['rot'][cl:cl*2]
        tmp['kinematics'] = tmp['kinematics'][cl:cl*2]
        tmp['kinematics']['time'] = tmp['kinematics']['time'] - tmp['kinematics']['time'][cl]
        tmpCrop.append(tmp)
        
        tmp = deepcopy(data)
        if (len(tmp['time'])>cl*3 and len(tmp['kinematics'])>cl*3):
            tmp['difference'] = tmp['difference'][cl*2:cl*3]
            tmp['time'] = tmp['time'][cl*2:cl*3] - tmp['time'][cl*2]
            tmp['rot'] = tmp['rot'][cl*2:cl*3]
            tmp['kinematics'] = tmp['kinematics'][cl*2:cl*3]
            tmp['kinematics']['time'] = tmp['kinematics']['time'] - tmp['kinematics']['time'][cl*2]
            tmpCrop.append(tmp)
        
        tmp = deepcopy(data)
        if (len(tmp['time'])>cl*4 and len(tmp['kinematics'])>cl*4):
            tmp['difference'] = tmp['difference'][cl*3:-400]
            tmp['time'] = tmp['time'][cl*3:-400] - tmp['time'][cl*3]
            tmp['rot'] = tmp['rot'][cl*3:-400]
            tmp['kinematics'] = tmp['kinematics'][cl*3:-400]
            tmp['kinematics']['time'] = tmp['kinematics']['time'] - tmp['kinematics']['time'][cl*3]
            tmpCrop.append(tmp)
    return tmpCrop

def cropDataMinLoss_woutKinematics(dataIn, cycleLength = 8333):
    cl = cycleLength
    tmpCrop = []
    for data in dataIn:
        
        # first run may not be good data
        tmp = deepcopy(data)
        tmp['difference'] = tmp['difference'][:cl]
        tmp['time'] = tmp['time'][:cl]
        tmp['rot'] = tmp['rot'][:cl]
        #tmp['kinematics'] = tmp['kinematics'][:cl]
        #tmp['kinematics']['time'] = tmp['kinematics']['time']
        tmpCrop.append(tmp)

        tmp = deepcopy(data)
        tmp['difference'] = tmp['difference'][cl:cl*2]
        tmp['time'] = tmp['time'][cl:cl*2] - tmp['time'][cl]
        tmp['rot'] = tmp['rot'][cl:cl*2]
        #tmp['kinematics'] = tmp['kinematics'][cl:cl*2]
        #tmp['kinematics']['time'] = tmp['kinematics']['time'] - tmp['kinematics']['time'][cl]
        tmpCrop.append(tmp)
        
        tmp = deepcopy(data)
        if (len(tmp['time'])>cl*3 and len(tmp['kinematics'])>cl*3):
            tmp['difference'] = tmp['difference'][cl*2:cl*3]
            tmp['time'] = tmp['time'][cl*2:cl*3] - tmp['time'][cl*2]
            tmp['rot'] = tmp['rot'][cl*2:cl*3]
            #tmp['kinematics'] = tmp['kinematics'][cl*2:cl*3]
            #tmp['kinematics']['time'] = tmp['kinematics']['time'] - tmp['kinematics']['time'][cl*2]
            tmpCrop.append(tmp)
        
        #tmp = deepcopy(data)
        #if (len(tmp['time'])>cl*4 and len(tmp['kinematics'])>cl*4):
        #    tmp['difference'] = tmp['difference'][cl*3:cl*4]
        #    tmp['time'] = tmp['time'][cl*3:cl*4] - tmp['time'][cl*3]
        #    tmp['rot'] = tmp['rot'][cl*3:cl*4]
        #    tmp['kinematics'] = tmp['kinematics'][cl*3:cl*4]
        #    tmp['kinematics']['time'] = tmp['kinematics']['time'] - tmp['kinematics']['time'][cl*3]
        #    tmpCrop.append(tmp)
    return tmpCrop

def generateSLPointKinematics(modelInfo, scaphoidPoints, lunatePoints, session):
    # Warning! This does not produce the exact same values as OpenSim's PointKinematics
    bones = ['lunate', 'scaphoid', 'radius', 'metacarp3']
    transMat = getInverseKinematicsTransformationMats(modelInfo, bones, session, coordSys='3-matic')
    totalFrames = len(transMat['lunate'])

    from slil.process.functions import getModelCache
    modelCache = getModelCache(modelInfo)
    lunateCOM = modelCache['lunateCOM']
    scaphoidCOM = modelCache['scaphoidCOM']
    radiusCOM = modelCache['radiusCOM']

    def to4x4(point):
        M = np.identity(4)
        M[:3,3] = point.T
        return M

    #initDist = fm.calcDist(relPosLunate + initialRelPos, relPosScaphoid)
    #bpLunate = to4x4(np.subtract(bpL, lunateCOM))
    #bpScaphoid = to4x4(np.subtract(bpS, scaphoidCOM))
    nLunPoints = len(lunatePoints)
    pointsLun = np.empty((nLunPoints, 4, 4))
    for i in range(nLunPoints):
        pointsLun[i,:,:] = to4x4(np.subtract(lunatePoints[i], lunateCOM))
    nScaPoints = len(scaphoidPoints)
    pointsSca = np.empty((nScaPoints, 4, 4))
    for i in range(nScaPoints):
        pointsSca[i,:,:] = to4x4(np.subtract(scaphoidPoints[i], scaphoidCOM))

    rad2ScaT = to4x4(np.subtract(scaphoidCOM, radiusCOM))
    initialRelPosT = to4x4(np.subtract(lunateCOM, scaphoidCOM))
    global2RadT = to4x4(np.subtract(radiusCOM,[0,0,0]))

    def dotAll(*argv):
        T = np.identity(4)
        for arg in argv:
            T = np.dot(T, arg)
        return T

    pointsTSca = np.empty((nScaPoints, totalFrames, 3))
    pointsTLun = np.empty((nLunPoints, totalFrames, 3))
    for i in range(totalFrames):
        for ii in range(nLunPoints):
            tLun= dotAll(initialRelPosT, transMat['lunate'][i])
            pointsTLun[ii,i,:] = np.dot(tLun, pointsLun[ii,:,:])[:3, 3]
        for ii in range(nScaPoints):
            tSca= np.identity(4)
            pointsTSca[ii,i,:] = np.dot(tSca, pointsSca[ii,:,:])[:3, 3]

    return pointsTLun, pointsTSca
    #tLun = np.empty((totalFrames, 4, 4))
    #tLunBP = np.empty((totalFrames, 3))
    #tSca = np.empty((totalFrames, 4, 4))
    #tScaBP = np.empty((totalFrames, 3))
    #for i in range(totalFrames):
    #    #tLun[i,:,:] = dotAll(global2RadT, transMat['radius'][i], rad2ScaT, transMat['scaphoid'][i], initialRelPosT, transMat['lunate'][i])
    #    tLun[i,:,:] = dotAll(initialRelPosT, transMat['lunate'][i])
    #    tLunBP[i,:] = np.dot(tLun[i,:,:], bpLunate)[:3, 3]
    #    
    #    #tSca[i,:,:]= dotAll(global2RadT, transMat['radius'][i], rad2ScaT, transMat['scaphoid'][i])
    #    tSca[i,:,:]= np.identity(4)
    #    tScaBP[i,:] = np.dot(tSca[i,:,:], bpScaphoid)[:3, 3]

def generateStrains(modelInfo, motionType, useCache = True):
    # Warning! Outputs are:
    #   cut log1
    #   cut log2
    #   normal log1
    #   normal log2
    #   scaffold log1
    #   scaffold log2
    if useCache:
        from slil.cache_results_plot import loadCache, saveCache
        strains = loadCache('strains_' + modelInfo['experimentID'] + '_' + motionType)
        if strains is not None:
            return strains
        print('No strains file found for {} so generating...'.format(modelInfo['experimentID'] + ' ' + motionType))

    from slil.common.math import calcDist
    from slil.common.plotting_functions import generateSLPointKinematics, groupInfoFE, groupInfoUR
    sessions = groupInfoUR['files']
    if 'FE' in motionType:
        sessions = groupInfoFE['files']
    
    from slil.process.functions import getModelCache
    modelCache = getModelCache(modelInfo)
    lunatePoints = [modelCache['SLILpointL']]
    scaphoidPoints = [modelCache['SLILpointS']]

    strains = []
    for session in sessions:
        pointsTLun, pointsTSca = generateSLPointKinematics(modelInfo, scaphoidPoints, lunatePoints, session)
        
        strain = np.empty((pointsTLun.shape[0], pointsTLun.shape[1])) # (n points, n frames)
        for i in range(strain.shape[0]):
            for ii in range(strain.shape[1]):
                strain[i, ii] = calcDist(pointsTLun[i][ii], pointsTSca[i][ii])
        strains.append(strain)
    
    if useCache:
        saveCache(strains, 'strains_' + modelInfo['experimentID'] + '_' + motionType)
    return strains

def getInverseKinematicsTransformationMats(modelInfo, bones, session, coordSys = 'opensim'):
    from slil.common.math import getTransformationMatrix
    from slil.common.plotting_functions import getDataIK
    
    dataOutputDir = modelInfo['dataOutputDir']
    #session = group['files'][0]
    fileExtModelKinematics = r'_model_kinematics.sto'

    # rather read the file again instead of using data otherwise
    # there may be some gimbal lock or something strange
    kinematicsLoaded = getDataIK(dataOutputDir + session + fileExtModelKinematics)

    transMat = {}
    #transMatRP = {}
    for bone in bones:
        tm = getTransformationMatrix(kinematicsLoaded, bone, coordSys)[0]
        transMat[bone] = tm
        #tm, r, p = fm.getTransformationMatrix(kinematicsLoaded, bone, '3-matic')
        #transMat[bone] = tm
        #transMatRP[bone] = {}
        #R = np.empty((tm.shape[0], 4, 4))
        #R[:, :3, :3] = r
        #R[:, :3, 3] = 0
        #R[:, 3, :] = [0,0,0,1]
        #transMatRP[bone]['R'] = R
        #P = np.empty((tm.shape[0], 4, 4))
        #P[:, :3, :3] = np.identity(3)
        #P[:, :3, 3] = p
        #P[:, 3, :] = [0,0,0,1]
        #transMatRP[bone]['P'] = P
    return transMat