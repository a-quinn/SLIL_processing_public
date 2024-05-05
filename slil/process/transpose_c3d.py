#%%
# Author: Alastair Quinn 2022
from copy import deepcopy
import slil.process.functions as fn
import slil.common.math as fm
import slil.common.io as fio
import slil.common.data_configs as dc
import numpy as np
import ray
from tqdm import tqdm

def run(experiments, models, runOnlyOne = False, runWithRay = True, clusterAddress = ''):
    missingModelCache = []

    if runWithRay:
        if not ray.is_initialized():
            ray.init(address=clusterAddress, \
                runtime_env={ \
                    "working_dir": ".", \
                    "excludes": [
                        #"/cadaver results 03_08_2021.zip", \
                        "/slil/rayTest/*", \
                        "/slil/3-matic_backup/*", \
                        "/data/*.*", \
                        "/data/data_original/*", \
                        "/data/data_cleaned/*", \
                        "/data/models_3_matic/*", \
                        "*.zip", \
                        "*.PNG", \
                        "*.png", \
                        "*.whl", \
                        "*.pptx", \
                        "*.xlsx", \
                        "*.log", \
                        "*.m", \
                        "**/__pycache__", \
                        #"./*.*",
                        ],
                    }, \
                ignore_reinit_error=True,
                )

    tasksRunning = []
    for cadaverID in tqdm(experiments, desc='Experiment'):
        
        modelInfo = dc.load(cadaverID)
        for i, model in tqdm(enumerate(models)):

            modelInfo['currentModel'] = model
            trials = modelInfo['trialsRawData_only_'+modelInfo['currentModel']]
            if runOnlyOne: # used for testing
                trials = [trials[0]]

            # get transformation matrix of radius markers (relative to world orgin))
            modelCache = fn.getModelCache(modelInfo, return_false_if_not_found=True)
            if modelCache == False:
                print('Skipping exp {} model {}'.format(cadaverID, model))
                missingModelCache.append('exp {} model {}'.format(cadaverID, model))
            else:
                tInitial = modelCache['initialTmatRadiusMarker'][modelInfo['currentModel']]

                for i, trial in tqdm(enumerate(trials), leave=False):
                    fileToImport = modelInfo['dataOutputDir'] + trial + '.c3d'
                    frames = fio.readC3D(fileToImport)
                    if runWithRay:
                        tasksRunning.append(align.remote(modelInfo, modelCache, tInitial, trial, frames))
                    else:
                        [frames, fileToExport] = _align(modelInfo, modelCache, tInitial, trial, frames)
                        fio.writeC3D(frames, markersUsed = 12, outputFile = fileToExport, isPureMarker = True, verbose = False)

    if runWithRay:
        while tasksRunning:
            finished, tasksRunning = ray.wait(tasksRunning, num_returns=1, timeout=None)
            for task in finished:
                [frames, fileToExport] = ray.get(task)
                fio.writeC3D(frames, markersUsed = 12, outputFile = fileToExport, isPureMarker = True, verbose = False)
            print('Tasks remaining:', len(tasksRunning))

    print('Finished tasks.')
    if len(missingModelCache) > 0:
        print('Skipped the following:')
        print(missingModelCache)

@ray.remote
def align(modelInfo0, modelCache0, tInitial, trial, frames):
    return _align(modelInfo0, modelCache0, tInitial, trial, frames)

def _align(modelInfo0, modelCache0, tInitial, trial, frames0):
    modelInfo = deepcopy(modelInfo0)
    modelCache = deepcopy(modelCache0)
    frames = deepcopy(frames0)
    #fileToImport = modelInfo['dataOutputDir'] + trial + '.c3d'
    #frames = fio.readC3D(fileToImport)
    print('Starting {} trial {}'.format(modelInfo['experimentID'], trial))

    #scale = 1.0 # No need to convert, output is in millimeters
    # only need x,y,z
    #for i in range(len(frames)):
    #    frames[i] = frames[i][:,0:3]
    #frames = np.array(frames) # * scale # No need to convert, output is in millimeters

    radiusName = modelInfo['names']['radius']

    # rotate and move all bones through all frames relative to radius in first frame
    #T_WOP1 = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, frames[0])
    

    # actually slower
    #plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
    #plateAssignmentRad = np.array(plateAssignment[radiusName]).astype(int)
    #import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
    #from slil.process.ik import align_1
    #align_1.align(frames, plateAssignmentRad, tInitial)

    # rotate and move all frames relative to static radius from 3-Matic inital guess
    # note: the function getMarkerGroupTranformationMat() to calculate
    #       transform of each marker set is not the same method used
    #       in OpenSim's inverse kinematics
    # this first one move the markers really close but not the same as using
    # OpenSim's IK for marker alignment, however without it the later
    # adjustments fail
    for i in range(0, len(frames)):
        t_WOP1 = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, frames[i])
        frames[i] = fm.transformPoints(frames[i], t_WOP1, tInitial)
    
    runAdjust2 = True
    useCython = False
    showProgress = False
    if runAdjust2:
        modelMarkersPos = modelCache['markers'][modelInfo['currentModel']]
        #mat = fm.rotation_matrix_around_axis(np.pi, np.array([0,1,0]))
        #mat = np.linalg.inv(mat) # invert
        mat = np.eye(3)
        for ind, mark in enumerate(modelMarkersPos):
            modelMarkersPos[ind] = np.dot(mat, modelMarkersPos[ind])

        plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
        p0 = np.array([
            modelMarkersPos[plateAssignment[radiusName][0]],
            modelMarkersPos[plateAssignment[radiusName][1]],
            modelMarkersPos[plateAssignment[radiusName][2]] # middle marker
            ])
        #t_starting = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, newPointsRaw)
        
        from scipy.optimize import fmin, minimize
        if useCython:
            import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})
            from slil.process.ik import fmin_fast, ik
            
        def findT(p0, p1, t_WOP1, x0):
            
            if useCython:
                def cost(x0, argsExtra):
                    x, y, z, rx, ry, rz = x0
                    p0, p1Init, t_WOP1 = argsExtra
                    p1 = deepcopy(p1Init)
                    error = ik.cost_fast(x, y, z, rx, ry, rz, p0, p1, t_WOP1)
                    return error

                result = fmin( \
                    func = cost, \
                    x0 = x0, \
                    args = (argsExtra, ), \
                    disp = False,
                    #callback = callbackF, \
                    #maxiter = 10
                    )
                #result = fmin_fast.nelder_mead(
                #    p0, p1, t_WOP1,
                #    #x0 = x0,
                #    #xmin = np.array([-100.0, -100.0, -100.0, -100.0, -100.0, -100.0]),
                #    #xmax = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0]),
                #    #simplex_scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                #    #xtol = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]),
                #    maxevals=int(3000)
                #)
                #print(result.message)
                #print(result.success)
                #print(result.nit)
                #print(result.nfev)
                #print(result.fun)
                x, y, z, rx, ry, rz = result

            else:
                def cost(x0, argsExtra):
                    x, y, z, rx, ry, rz = x0
                    p0, p1Init, t_WOP1 = argsExtra
                    p1 = deepcopy(p1Init)

                    t_adjustment = fm.createTransformationMatFromPosAndEuler(x, y, z, rx, ry, rz)
                    #t_adjustment = np.dot(t_adjustment, t_WOP1)
                    #p1 = fm.transformPoints(p1, t_WOP1, t_adjustment)
                    # t_adjustment, t_WOP1 * inv(t_WOP1) * vec = t_adjustment, vec
                    p1 = fm.transformPoints_1(p1, t_adjustment)

                    error = (
                            np.power(p1[0, 0] - p0[0, 0], 2) + 
                            np.power(p1[0, 1] - p0[0, 1], 2) + 
                            np.power(p1[0, 2] - p0[0, 2], 2) +
                            np.power(p1[1, 0] - p0[1, 0], 2) + 
                            np.power(p1[1, 1] - p0[1, 1], 2) + 
                            np.power(p1[1, 2] - p0[1, 2], 2) +
                            np.power(p1[2, 0] - p0[2, 0], 2) + 
                            np.power(p1[2, 1] - p0[2, 1], 2) + 
                            np.power(p1[2, 2] - p0[2, 2], 2))
                    error += 1.0
                    return error

                argsExtra = [p0, p1, t_WOP1]

                result = minimize( \
                    fun = cost, \
                    x0 = x0, \
                    args = (argsExtra, ), \
                    method='L-BFGS-B', \
                    options= {
                        #'disp': True,
                        'maxcor': 40,
                        'maxiter': 3000,
                        },
                    #callback = calllbackSave, \
                    #maxiter = 10
                    )
                x, y, z, rx, ry, rz = result.x
            return x, y, z, rx, ry, rz

        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ii = 0
        for i in range(0, len(frames)):
            t_WOP1 = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, frames[i])
            p1 = np.array([
                frames[i][plateAssignment[radiusName][0]],
                frames[i][plateAssignment[radiusName][1]],
                frames[i][plateAssignment[radiusName][2]] # middle marker
                ])
            x, y, z, rx, ry, rz = findT(p0, p1, t_WOP1, x0)
            x0 = np.array([x, y, z, rx, ry, rz]) # faster for next frame
            tAligned = fm.createTransformationMatFromPosAndEuler(x, y, z, rx, ry, rz)
            #tAligned = np.dot(tAligned, t_WOP1)
            #frames[i] = fm.transformPoints(frames[i], t_WOP1, tAligned)
            # t_adjustment, t_WOP1 * inv(t_WOP1) * frames[i] = t_adjustment, frames[i]
            frames[i] = fm.transformPoints_1(frames[i], tAligned)
            if showProgress:
                if ii >= 1000:
                    print('\tframes\t{}/{}'.format(i,len(frames)))
                    ii = 0
                ii += 1

    # rotate and move all bones through all frames relative to radius in first frame
    #plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
    #radiusMidMarkerIndex = plateAssignment[radiusName][2]
    #rotateAbout = np.array(frames[0][radiusMidMarkerIndex])
    #T_WOP1 = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, frames[0])
    #rotFirst = T_WOP1[:3, :3]
    #for i in range(0, len(frames) - 1):
    #    T_WOP1 = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, frames[i])
    #    rot = T_WOP1[:3, :3]
    #    
    #    #rotateAbout = np.array(frames[i][radiusMidMarkerIndex])
    #    #p0 = frames[i][plateAssignment[radiusName][0]]
    #    #p1 = frames[i][plateAssignment[radiusName][1]]
    #    #p2 = frames[i][plateAssignment[radiusName][2]]
    #    #pm = (p0 - p1)/2 + p1 # between p0 and p1
    #    #vec1 = fm.normalizeVector(p2 - pm) # vector where wire is
    #    #vec0 = fm.normalizeVector(fm.calcNormalVec(p0, p1, p2))
    #    #rot = np.array(fm.create3DAxis(vec1, vec0)).T
    #    
    #    r = np.dot(rot, rotAligned.T)
    #    #r = np.dot(rot, rotFirst.T) # if you don't want to static but to radius first frame
    #    for j in range(12):
    #        frames[i][j] = fm.rotateVector(frames[i][j] - rotateAbout, r) + posAligned

#    # load into 3 matic
#    frameN = 1
#    for k, name in enumerate(modelInfo['names']):
#        if (k>3):
#            break
#        markerIndex0 = plateAssignment[modelInfo['names'][name]]
#    
#        cord0 = frames[frameN][markerIndex0[2]]
#        mi.create_point(cord0)
#
#        # create wires
#        p0 = frames[frameN][markerIndex0[0]]
#        p1 = frames[frameN][markerIndex0[1]]
#        p2 = frames[frameN][markerIndex0[2]] # middle
#        pm = (np.array(p0) - p1)/2 + p1 # between p0 and p1
#        vec1 = np.array(p2) - pm # vector where wire is
#        vec1 = fm.normalizeVector(vec1)
#        wireLength = 15.85 + 45
#        m2 = mi.create_line_direction_and_length(pm, vec1, wireLength)

    # 3 matic to opensim
    #if not runAdjust2:
    mat = fm.rotation_matrix_around_axis(np.pi, np.array([0,1,0]))
    for i in range(0, len(frames)):
        for j in range(12):
            frames[i][j] = np.dot(mat, frames[i][j])

    frames = frames * 0.001 # convert mm to m
    fileToExport = modelInfo['dataOutputDir'] + trial + '.c3d'
    #fio.writeC3D(frames, markersUsed = 12, outputFile = fileToExport, isPureMarker = True, verbose = False)
    return frames, fileToExport

#%%
#
#from copy import deepcopy
#import slil.process.functions as fn
#import slil.common.math as fm
#import slil.common.io as fio
#import numpy as np
#import ray
#from tqdm import tqdm
#import slil.common.data_configs as dc
#from copy import deepcopy
#
#import slil.common.opensim as osim
#
#def cost(x0, argsExtra):
#    x, y, z, rx, ry, rz = x0
#    p0, p1Init, t_WOP1 = argsExtra
#    p1 = deepcopy(p1Init)
#    #error = ik.cost_fast(x, y, z, rx, ry, rz, p0, p1, t_WOP1)
#    #return error
#    
#    t_adjustment = fm.createTransformationMatFromPosAndEuler(x, y, z, rx, ry, rz)
#
#    t = np.dot(t_adjustment, t_WOP1)
#    p1 = fm.transformPoints(p1, t_WOP1, t)
#
#    error = (
#            np.power(p1[0, 0] - p0[0, 0], 2) + 
#            np.power(p1[0, 1] - p0[0, 1], 2) + 
#            np.power(p1[0, 2] - p0[0, 2], 2) +
#            np.power(p1[1, 0] - p0[1, 0], 2) + 
#            np.power(p1[1, 1] - p0[1, 1], 2) + 
#            np.power(p1[1, 2] - p0[1, 2], 2) +
#            np.power(p1[2, 0] - p0[2, 0], 2) + 
#            np.power(p1[2, 1] - p0[2, 1], 2) + 
#            np.power(p1[2, 2] - p0[2, 2], 2))
#    error += 1.0
#    return error
#cadaverID = '11534'
#model = 'normal'
#modelInfo = dc.load(cadaverID)
#modelInfo['currentModel'] = model
#trials = modelInfo['trialsRawData_only_'+modelInfo['currentModel']]
#trial = trials[0]
#fileToImport = modelInfo['dataOutputDir'] + trial + '.c3d'
#frames = fio.readC3D(fileToImport)
#
#
#modelCache = fn.getModelCache(modelInfo, breakIfNotFound=True)
#missingModelCache = []
#if modelCache == False:
#    print('Skipping exp {} model {}'.format(cadaverID, model))
#    missingModelCache.append('exp {} model {}'.format(cadaverID, model))
#else:
#    tInitial = modelCache['initialTmatRadiusMarker'][modelInfo['currentModel']]
#
#
#
#radiusName = modelInfo['names']['radius']
#
## need two rotations performed
## rotate and move all frames relative to algined static radius
#for i in range(0, len(frames)):
#    t_WOP1 = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, frames[i])
#    frames[i] = fm.transformPoints(frames[i], t_WOP1, tInitial)
#
#plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
#
#modelMarkersPos = osim.getMarkerPositionsInModelGlobal(modelInfo)
## opensim to 3 matic
#mat = fm.rotation_matrix_around_axis(np.pi, np.array([0,1,0]))
#mat = np.linalg.inv(mat) # invert
#for ind, mark in enumerate(modelMarkersPos):
#    modelMarkersPos[ind] = np.dot(mat, modelMarkersPos[ind]) * 1000.0
#
#t_WOP1_0 = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, frames[0])
#p0 = np.array([
#    modelMarkersPos[plateAssignment[radiusName][0]],
#    modelMarkersPos[plateAssignment[radiusName][1]],
#    modelMarkersPos[plateAssignment[radiusName][2]] # middle marker
#    ])
#
#i = 1
#t_WOP1 = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, frames[i])
#p1 = np.array([
#    frames[i][plateAssignment[radiusName][0]],
#    frames[i][plateAssignment[radiusName][1]],
#    frames[i][plateAssignment[radiusName][2]] # middle marker
#    ])
#
##t_WOP1 = np.eye(4)
#argsExtra = p0, p1, t_WOP1
#x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#cost(x0, argsExtra)
#
#import pyximport; pyximport.install(
#    setup_args={'include_dirs': np.get_include()},
#    reload_support=True)
#from slil.process.ik import ik#, fmin_fast, align_1
#x, y, z, rx, ry, rz, = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#a = ik.cost_fast(x, y, z, rx, ry, rz, p0, p1, t_WOP1)
#
#[
#[a[0][0], a[0][1], a[0][2]],
#[a[1][0], a[1][1], a[1][2]],
#[a[2][0], a[2][1], a[2][2]]]
#[
#[a[0][0], a[0][1], a[0][2], a[0][3]],
#[a[1][0], a[1][1], a[1][2], a[1][3]],
#[a[2][0], a[2][1], a[2][2], a[2][3]],
#[a[3][0], a[3][1], a[3][2], a[3][3]]]
#
#def cost(x, argsExtra):
#    x, y, z, rx, ry, rz = x
#    p0, p1Init, t_WOP1 = argsExtra
#    p1 = deepcopy(p1Init)
#    error = ik.cost_fast(x, y, z, rx, ry, rz, p0, p1, t_WOP1)
#    return error
#
#from scipy.optimize import fmin, minimize
#result = fmin( \
#    func = cost, \
#    x0 = x0, \
#    args = (argsExtra, ), \
#    disp = True,
#    retall = True,
#    #callback = callbackF, \
#    #maxiter = 10
#    )
#result[0]
#len(result[1])
#
#tries = []
#def calllbackSave(xk):
#    tries.append(xk)
#
#result = minimize( \
#    fun = cost, \
#    x0 = x0, \
#    args = (argsExtra, ), \
#    method='L-BFGS-B', \
#    options= {
#        'disp': True,
#        'maxcor': 40,
#        'maxiter': 3000,
#        },
#    callback = calllbackSave, \
#    #maxiter = 10
#    )
#
#x0_s = tries
#x0_s = [result.x]
#x0_s = result[1]
#
#fmins = []
#for r in x0_s:
#    e = cost(r, argsExtra)
#    fmins.append(e)
#
#from pyvistaqt import BackgroundPlotter
#import pyvista as pv
#class MarkerViewer():
#    def __init__(self, x0_s, pInit, argsExtra):
#        p = BackgroundPlotter()
#
#        def showPoints(value):
#            x, y, z, rx, ry, rz = x0_s[int(value)]
#            p0, p1, t_WOP1 = argsExtra
#
#            R = fm.eulerAnglesToRotationMatrix((rx, ry, rz))
#            t_adjustment = np.eye(4)
#            t_adjustment[:3, :3] = R
#            t_adjustment[:3, 3] = np.array((x, y, z)).T
#
#            t =  np.dot(t_adjustment, t_WOP1)
#
#            p1 = fm.transformPoints(p1, t_WOP1, t)
#            #p.add_points(p0, name='target', color='k')
#            #p.add_points(pInit, name='startingPoint', color='r')
#            #p.add_points(p1, name='p1', color='g')
#            
#            p.add_points(np.array([list(p0[0]), list(pInit[0]), list(p1[0])]), name='target', color='k')
#            p.add_points(np.array([list(p0[1]), list(pInit[1]), list(p1[1])]), name='startingPoint', color='r')
#            p.add_points(np.array([list(p0[2]), list(pInit[2]), list(p1[2])]), name='p1', color='g')
#            return
#
#        p.add_slider_widget(showPoints, rng = [0, len(x0_s)-1], value = 0, title='Alignment')
#        p.show()
#mv = MarkerViewer(x0_s, p1, argsExtra)