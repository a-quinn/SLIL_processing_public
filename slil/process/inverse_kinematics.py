# Author: Alastair Quinn 2022

import slil.common.data_configs as dc
import slil.process.functions as fn
from tqdm import tqdm
import numpy as np
import slil.common.math as fm

def run(experiments, models, overrideOpenSimModel = '', runOnlyOne = False):
    failedIK = []
    for experiment in tqdm(experiments):
        modelInfo = dc.load(experiment)
        
        #mi.open_project(modelInfo['3_matic_file'])

        for model in tqdm(models):

            modelInfo['currentModel'] = model
            trials = modelInfo['trialsRawData_only_'+modelInfo['currentModel']]
            if runOnlyOne: # used for testing
                trials = [trials[0]]
            
            # get transformation matrix of radius markers (relative to world orgin))
            modelCache = fn.getModelCache(modelInfo, return_false_if_not_found=True)
            if modelCache == False:
                print('Skipping exp {} model {}'.format(experiment, model))
            else:

                #markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static'
                ##markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static_after'
                #fn.removeCylindersFromSphereGroup(modelInfo, markerGroupName)
                #fn.convertSphereGroupToPoints(modelInfo, markerGroupName)
                #fn.setMarkersInModel_old(modelInfo)
                #fn.convertPointsToSphereGroup(modelInfo, markerGroupName)
                #fn.addCylindersToSphereGroup(modelInfo, markerGroupName)
                
                #fn.setBonesInModel(modelInfo)

                #modelCache = fn.getModelCache(modelInfo)
                modelInfo['boneplug'] = {}
                modelInfo['boneplug']['lunate'] = modelCache['lunateCOM2bpOS']
                modelInfo['boneplug']['scaphoid'] = modelCache['scaphoidCOM2bpOS']

                if not overrideOpenSimModel == '':
                    openSimModel = overrideOpenSimModel
                else:
                    openSimModel = modelInfo['currentModel']

                for trial in trials:
                    returnCode = fn.runIK(
                        modelInfo,
                        openSimModel,
                        trial + '.c3d',
                        visualise=False,
                        threads = 14, # Change this on your machine
                        timeout_sec = 60)
                    if returnCode != 0 and returnCode != 3221225477: # 3221225477 is the code for a crash but output files are fine
                        print("Failed, adding to retry list.")
                        failedIK.append({
                            'experimentID': modelInfo['experimentID'],
                            'model': model,
                            'trial': trial
                            })
                print('Finished processing trials. {} {}'.format(experiment, model))
                    
        # don't save incase something was mucked
        #mi.save_project()

    if len(failedIK) > 0:
        print("Failed IK: {}".format(len(failedIK)))
        print("Retrying...")

        for retryIK in failedIK:
            modelInfo = dc.load(retryIK['experimentID'])
            
            model = retryIK['model']
            modelInfo['currentModel'] = model

            #mi.open_project(modelInfo['3_matic_file'])

            #fn.generateBoneplugsToBonesDistances(modelInfo)

            modelCache = fn.getModelCache(modelInfo)
            modelInfo['boneplug'] = {}
            modelInfo['boneplug']['lunate'] = modelCache['lunateCOM2bpOS']
            modelInfo['boneplug']['scaphoid'] = modelCache['scaphoidCOM2bpOS']

            if not overrideOpenSimModel == '':
                openSimModel = overrideOpenSimModel
            else:
                openSimModel = modelInfo['currentModel']

            trial = retryIK['trial']
            returnCode = fn.runIK(
                modelInfo,
                openSimModel,
                trial + '.c3d',
                visualise = False,
                threads = 7,
                timeout_sec = 240)
            if returnCode != 0:
                print("Error tried twice running IK on {} {}".format(experiment, trial))
            print('Finished processing trials. {} {}'.format(experiment, model))
            
    print("Check the following files were run: {}".format(failedIK))
    return failedIK


from copy import deepcopy
def findMarkerToPointsAlignmentOnce(pTo, pFrom, x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
    # could be rewritten in a class to be more efficite with multiple frames of data
        
    def calcError(p0, p1):
        return (
                np.power(p1[0, 0] - p0[0, 0], 2) + 
                np.power(p1[0, 1] - p0[0, 1], 2) + 
                np.power(p1[0, 2] - p0[0, 2], 2) +
                np.power(p1[1, 0] - p0[1, 0], 2) + 
                np.power(p1[1, 1] - p0[1, 1], 2) + 
                np.power(p1[1, 2] - p0[1, 2], 2) +
                np.power(p1[2, 0] - p0[2, 0], 2) + 
                np.power(p1[2, 1] - p0[2, 1], 2) + 
                np.power(p1[2, 2] - p0[2, 2], 2))
    def calcError(p0, p1):
        error = 0.0
        for indAx1 in range(p1.shape[0]):
            error += np.power(p1[indAx1, 0] - p0[indAx1, 0], 2) + \
                np.power(p1[indAx1, 1] - p0[indAx1, 1], 2) + \
                np.power(p1[indAx1, 2] - p0[indAx1, 2], 2)
        return error

    def cost(x0, argsExtra):
        x, y, z, rx, ry, rz = x0
        p0, p1Init = argsExtra
        p1 = deepcopy(p1Init)

        t_adjustment = fm.createTransformationMatFromPosAndEuler(x, y, z, rx, ry, rz)
        #p1 = fm.transformPoints_1(p1, t_adjustment)
        pointsTemp = np.ones((p1.shape[0], p1.shape[1] + 1))
        pointsTemp[:, :3] = p1
        for i, vector in enumerate(pointsTemp):
            p1[i] = np.dot(t_adjustment, vector)[:3]

        error = calcError(p0, p1) + 1.0
        return error

    def findT(p0, p1, x0):
        from scipy.optimize import minimize
        result = minimize( \
            fun = cost, \
            x0 = x0, \
            args = ([p0, p1], ), \
            method='L-BFGS-B', \
            options= {
                #'disp': True,
                'maxcor': 40,
                'maxiter': 3000,
                #'ftol': 0.01
                },
            #callback = calllbackSave, \
            #maxiter = 10
            )
        x, y, z, rx, ry, rz = result.x
        return x, y, z, rx, ry, rz
    
    x, y, z, rx, ry, rz = findT(pTo, pFrom, x0)
    return fm.createTransformationMatFromPosAndEuler(x, y, z, rx, ry, rz)
