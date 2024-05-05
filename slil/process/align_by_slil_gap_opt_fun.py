# Author: Alastair Quinn 2022

import slil.process.functions as fn
import slil.common.math as fm
import numpy as np
from copy import deepcopy
import pyvista as pv

import timeit
#start_time = timeit.default_timer()
# run code
#print('timeit: {}'.format(timeit.default_timer() - start_time))

def optFun(x, args):
    x, y, z, rx, ry, rz = x
    #rx, ry, rz = 0.0, 0.0, 0.0
    #x, y, z = x
    #start_time = timeit.default_timer()
    
    modelInfo = args['modelInfo']
    referenceBone = args['referenceBone']
    checkBones = args['checkBones']
    t_World_RadiusMarkers_FromModel = args['M_O2Rad_orig']
    framePointsRaw_Original = args['framesPointsRaw']
    framePointsRaw = deepcopy(framePointsRaw_Original) # stop potential changes for future optimizations

    radiusName = referenceBone['name']
    surfRad_Original = referenceBone['surface']
    t_World_ModelRadCOM_Original = referenceBone['t_WLx']

    R = fm.eulerAnglesToRotationMatrix((rx, ry, rz))
    t_adjustment = np.eye(4)
    t_adjustment[:3, :3] = R
    t_adjustment[:3, 3] = np.array((x, y, z)).T

    # 1. Apply transformation offset to markers
    # 2. Calculate new lunate position
    # 3. Calculate errors

    t_World_RadiusMarkers_Adjusted = np.dot(t_adjustment, deepcopy(t_World_RadiusMarkers_FromModel))

    # move markers for all frames to new position and orientation after adjustment
    for i, frame in enumerate(framePointsRaw): # these markers could be anywhere in space so move them to ensure they're where we think they are
        t_World_RadMarkers_FrameI = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, frame)
        #t_World_RadMarkers_FrameI = np.eye(4)
        framePointsRaw[i] = fm.transformPoints(frame, t_World_RadMarkers_FrameI, t_World_RadiusMarkers_Adjusted)
    frameZero = framePointsRaw[0]

    # t_adjustment * t_World_RadiusMarkers_FromModel * inv(t_World_RadMarkers_FrameI) * frame

    # transformations between markers and lunate
    #m0 = makeMesh(surfRad.points, surfRad.faces.reshape(-1, 4)[:,1:4])
    #print(surfRad)
    #m0 = makeMesh(surfRad.points, np.reshape(surfRad.faces, (-1,4))[:,1:4])

    t_World_RadiusMarkers_Frame0 = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, frameZero) # make sure we have it after transforming frame0
    #t_L1W = fm.inverseTransformationMat(t_WL1)
    t_ModelRadCOM_World = fm.inverseTransformationMat(t_World_ModelRadCOM_Original)
    t_ModelRadCOM_RadMarkers = np.dot(t_ModelRadCOM_World, t_World_RadiusMarkers_Frame0)
    #t_L1P2 = np.dot(t_L1W, t_WP2) # lunate marker at frame 2, relative to lunate COM
    t_RadMarkers_ModelRadCOM = fm.inverseTransformationMat(t_ModelRadCOM_RadMarkers)
    
    outputPairedPoints = {}
    for bone in checkBones:
        if 'pairedPoints' in bone:
            outputPairedPoints[bone['name']] = np.empty((
                len(framePointsRaw), bone['pairedPoints'].shape[0], 3
                ))

    outputSinglePoints = {}
    for bone in checkBones:
        if 'singlePoints' in bone:
            outputSinglePoints[bone['name']] = np.empty((
                len(framePointsRaw), bone['singlePoints'].shape[0], 3
                ))

    outputSurfaces = []
    for bone in checkBones:
        boneName = bone['name']
        t_World_ModelBoneCOM = bone['t_WLx']
        t_World_BoneMarkers_Frame0 = fn.getMarkerGroupTranformationMat(modelInfo, boneName, frameZero)

        #t_L1W = fm.inverseTransformationMat(t_WL1)
        t_ModelBoneCOM_World = fm.inverseTransformationMat(t_World_ModelBoneCOM)
        t_ModelBoneCOM_BoneMarkers = np.dot(t_ModelBoneCOM_World, t_World_BoneMarkers_Frame0)
        #t_L1P2 = np.dot(t_L1W, t_WP2) # lunate marker at frame 2, relative to lunate COM
        t_BoneMarkers_ModelBoneCOM = fm.inverseTransformationMat(t_ModelBoneCOM_BoneMarkers)

        #m1 = makeMesh(boneSurf.points, boneSurf.faces.reshape(-1, 4)[:,1:4])
        
        for i, frame in enumerate(framePointsRaw):
            if True:
            #if i > 0: # skip first frame
                t_World_BoneMarkers_FrameI = fn.getMarkerGroupTranformationMat(modelInfo, boneName, frame)
                t_World_ModelBoneCOM_FrameI = np.dot(t_World_BoneMarkers_FrameI, t_BoneMarkers_ModelBoneCOM)

                if 'pairedPoints' in bone:
                    p1 = fm.transformPoints(bone['pairedPoints'], t_World_ModelBoneCOM, t_World_ModelBoneCOM_FrameI)
                    outputPairedPoints[bone['name']][i] = p1

                if 'singlePoints' in bone:
                    p1 = fm.transformPoints(bone['singlePoints'], t_World_ModelBoneCOM, t_World_ModelBoneCOM_FrameI)
                    outputSinglePoints[bone['name']][i] = p1

                if args['displayViz'] != None:
                    surfBone_2 = deepcopy(bone['surface'])
                    #fm.transformPoints_byRef(surfBone_2.points, t_WL1, t_WL2)
                    fm.transformPoints_byRef(surfBone_2.points, t_World_ModelBoneCOM, t_World_ModelBoneCOM_FrameI)

                    # This should never change if radius never moves throughout all frames, but it may.
                    t_World_RadMarkers_FrameI_CouldHaveChanged = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, frame)
                    t_World_ModelRadCOM_FrameI = np.dot(t_World_RadMarkers_FrameI_CouldHaveChanged, t_RadMarkers_ModelRadCOM)
                    surfRad_2 = deepcopy(surfRad_Original)
                    fm.transformPoints_byRef(surfRad_2.points, t_World_ModelRadCOM_Original, t_World_ModelRadCOM_FrameI)
                    
                    outputSurfaces.append(surfBone_2)
                    outputSurfaces.append(surfRad_2)

    if args['displayViz'] != None:
        outputPairs = []

    errorIntersection = []
    
    for bone in checkBones:
        if 'pairedPointsName' in bone:
            distance = np.empty((len(framePointsRaw), outputPairedPoints[bone['name']].shape[1]))
            for indFrame, pointsInframe in enumerate(outputPairedPoints[bone['name']]):
                for indPoint, pointInFrame in enumerate(pointsInframe):
                    error2distance = fm.calcMag(pointInFrame - outputPairedPoints[bone['pairedPointsName']][indFrame][indPoint])
                    errorIntersection.append(0.0)
                    #distance.append(error2distance)
                    distance[indFrame][indPoint] = error2distance
                    
                    if args['displayViz'] != None:
                        outputPairs.append([pointInFrame, outputPairedPoints[bone['pairedPointsName']][indFrame][indPoint]])
    #print('TimeIt calc error: {}'.format(timeit.default_timer() - start_time))
    
    errorPointsToGlobalOrigin = np.zeros((len(framePointsRaw)))
    for bone in checkBones:
        if 'pairedPoints' in bone:
            for indFrame, pointsInframe in enumerate(outputPairedPoints[bone['name']]):
                distanceToGlobalOrigin_temp = np.zeros((len(pointsInframe)))
                for indPoint, pointInFrame in enumerate(pointsInframe):
                    #distanceToGlobalOrigin[indFrame][indPoint] = fm.calcMag(pointInFrame)
                    distanceToGlobalOrigin_temp[indPoint] = fm.calcMag(pointInFrame)
                errorPointsToGlobalOrigin[indFrame] = np.mean(distanceToGlobalOrigin_temp)

    startingLenths = distance[0, :] # first frame in list is the reference frame!
    
    errorPoint = np.zeros((len(framePointsRaw)))
    errorPointCurve = np.zeros((len(framePointsRaw)))
    axis = 1
    refFrame = 0
    for bone in checkBones:
        if 'singlePoints' in bone:
            # first frame, first point in frame, y-axisz
            startingSinglePoint = outputSinglePoints[bone['name']][0][0, axis] # first frame in list is the reference frame!
            
            for ind in range(len(framePointsRaw)):
                if ind >= len(framePointsRaw)/2:
                    axis = 2 # cahgne to z axis
                    refFrame = int(len(framePointsRaw)/2)
                errorPoint[ind] = abs(
                    outputSinglePoints[bone['name']][ind][0, axis] - outputSinglePoints[bone['name']][refFrame][0, axis]
                    )
                errorPointCurve[ind] = abs(
                        fm.calcMag(
                            outputSinglePoints[bone['name']][ind][0, :]
                        ) -
                        fm.calcMag(
                            outputSinglePoints[bone['name']][refFrame][0, :]
                        )
                    )

    error = 0
    display = args['display']
    gainB = 2.0
    if display:
        print('N = frame number')
        print('error = abs( (individual distances - initial distances) for each point )')
        print('singleError = along y-axis, distance from point in first frame')
        print('N  : avg distance : error = abs(distance) + {:.1f} * singleError + errorPointsToGlobalOrigin # + errorPointCurve'.format(gainB))

    errorDistance = np.zeros((len(framePointsRaw)))
    for ind, errorD in enumerate(distance):
        errorDistance[ind] = np.mean(errorD - startingLenths)

    for ind, errorD in enumerate(errorDistance):
        if display:
            N = ind
            if ind >= len(framePointsRaw)/2:
                N = ind - int(len(framePointsRaw)/2)
                frameN = args['frameNs2'][N]
            else:
                frameN = args['frameNs'][N]

            #print('N {} frame {} : {:.3f} : {:.3f} = abs({:.3f}) + {:.3f} + {:.3f}'.format(
            #    N, frameN,
            #    np.mean(distance[ind, :]), np.abs(errorD) + gainB * errorPoint[ind] + errorPointCurve[ind], errorD, gainB * errorPoint[ind], errorPointCurve[ind]))

            print('N {} frame {} : {:.3f} : {:.3f} = abs({:.3f}) + {:.3f} + {:.3f}'.format(
                N, frameN,
                np.mean(distance[ind, :]), np.abs(errorD) + gainB * errorPoint[ind] + errorPointsToGlobalOrigin[ind], errorD, gainB * errorPoint[ind], errorPointsToGlobalOrigin[ind]))

    for ind, errorD in enumerate(errorDistance):
        errorDistance[ind] = np.abs(errorD)

    for ind, errorD in enumerate(errorDistance):
        error += errorD + gainB * errorPoint[ind] + errorPointsToGlobalOrigin[ind]

    if display:
        print('errorTotal = {}'.format(error))

    if args['displayViz'] != None:
        return error, outputSurfaces, framePointsRaw, outputPairs
        
    return error