# Author: Alastair Quinn 2022

import slil.process.functions as fn
import slil.common.math as fm
import numpy as np

def checkResultsCloseToBounds(experiments, limitPos = 19.0, limitRot = 45, extendedName = ''):
    #limitPos = 19.0 # mm
    #limitRot = 45 # degrees
    print('Searching results close to {} mm and {} degrees'.format(limitPos, limitRot))
    import slil.common.data_configs as dc
    limitRot = np.deg2rad(limitRot)
    for i, experiment in enumerate(experiments):
        modelInfo = dc.load(experiment)
        modelTypes = [ 'normal', 'cut', 'scaffold']
        
        #resAll = fn.saveLoadOptimizationResults(modelInfo)
        #bestFunctionValue = {}
        #for modelType in [ 'normal', 'cut', 'scaffold' ]:
        #    if type(res[modelType]) == list:
        #        x0 = resAll[modelType][0].results['bestFunctionValue']
        #    else:
        #        x0 = resAll[modelType]['t']
        #        if 't' in x0:
        #            x0 = x0['bestFunctionValue']
        #    bestFunctionValue[modelType] = x0

        res = fn.getOptimizationResults(modelInfo, extendedName = extendedName)
        for modelType in modelTypes:
            if not modelType in res:
                continue
            x0p = res[modelType]
            if modelType == 'scaffold':
                modelType = 'scafld'

            angleChange = np.rad2deg(fm.rotMat2AxisAngle(fm.eulerAnglesToRotationMatrix((x0p[3], x0p[4], x0p[5])))[1])

            x0p = list(x0p)
            foundAnyCloseToLimit = False
            for i in range(3):
                if x0p[i] > limitPos or x0p[i] < -1.0 * limitPos:
                    x0p[i] = 'X' + '{:.2f}'.format(x0p[i]) + ''
                    foundAnyCloseToLimit = True
                if x0p[i+3] > limitRot or x0p[i+3] < -1.0 * limitRot:
                    x0p[i+3] = 'X' + '{:.2f}'.format(np.rad2deg(x0p[i+3])) + ''
                    foundAnyCloseToLimit = True
            for i in range(3):
                if type(x0p[i]) == np.float64:
                    x0p[i] = '{:.2f}'.format(x0p[i])
            for i in range(3):
                if type(x0p[i+3]) == np.float64:
                    x0p[i+3] = '{:.2f}'.format(np.rad2deg(x0p[i+3]))
            if foundAnyCloseToLimit:
                print('{} {}\t:X\t{}\t{}\t{}\t{}\t{}\t{} : {:.2f}'.format(
                    experiment, modelType,
                    x0p[0], x0p[1], x0p[2], x0p[3], x0p[4], x0p[5],
                    angleChange))
            else:
                print('{} {}\t:\t{}\t{}\t{}\t{}\t{}\t{} : {:.2f}'.format(
                    experiment, modelType,
                    x0p[0], x0p[1], x0p[2], x0p[3], x0p[4], x0p[5],
                    angleChange))
