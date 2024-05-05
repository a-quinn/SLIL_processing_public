# Author: Alastair Quinn 2022
import slil.process.functions as fn
import slil.common.math as fm
from copy import deepcopy
import numpy as np
from slil.process.inverse_kinematics import findMarkerToPointsAlignmentOnce

def shouldFlip(flip, modelInfo, modelType, boneName):
    if modelInfo['experimentID'] in flip:
        if modelType in flip[modelInfo['experimentID']]:
            if boneName in flip[modelInfo['experimentID']][modelType]:
                return True
    return False

def getMarkerTransformation(flip, modelInfo, modelType, boneName, points):
    doFlip = shouldFlip(flip, modelInfo, modelType, boneName)
    modelInfo['currentModel'] = modelType
    return fn.getMarkerGroupTranformationMat(modelInfo, modelInfo['names'][boneName], points, doFlip)

def excludeMarkersFromPointCloud(modelInfo, cloud, modelType, boneNames):
    inds = []
    for boneName in boneNames:
        inds += modelInfo['plateAssignment_' + modelType][modelInfo['names'][boneName]]
    return np.delete(cloud, inds, axis = 0)

def getPinVector(scene, mapMarkerChanges, inputPoints, boneName, currentModel):
    plateAssignment = scene.modelInfo['plateAssignment_' + currentModel]
    boneModelName = scene.modelInfo['names'][boneName]
    points = np.array([
            inputPoints[plateAssignment[boneModelName][0]],
            inputPoints[plateAssignment[boneModelName][1]],
            inputPoints[plateAssignment[boneModelName][2]] # middle marker
            ])
    pinsSets = generatePinsFromMarkerPoints(points)
    pinsSets = pinsSets[scene.possiblePinSet[scene.modelInfo['experimentID']][currentModel][boneName]]
    pointsCut = np.empty((2, 3))
    if boneName in mapMarkerChanges[scene.modelInfo['experimentID']]:
        mapMarkerChanges = mapMarkerChanges[scene.modelInfo['experimentID']][boneName]

        if mapMarkerChanges == 0:
            pointsCut[:, :] = pinsSets[0, 0:2, :]
        if mapMarkerChanges == 1:
            pointsCut[:, :] = pinsSets[0, 2:4, :]
        if mapMarkerChanges == 2:
            pointsCut[:, :] = pinsSets[1, 0:2, :]
        if mapMarkerChanges == 3:
            pointsCut[:, :] = pinsSets[1, 2:4, :]

    else:
        pointsCut[:, :] = pinsSets[0, 0:2, :]
    return fm.normalizeVector(pointsCut[1, :] - pointsCut[0, :])

def generateEndsOfPins(scene, pointsIn, currentModel, possiblePinSet):
    pointsAtEndsOfPins = {}
    for boneName in fn.boneNames():
        plateAssignment = scene.modelInfo['plateAssignment_' + currentModel]
        boneModelName = scene.modelInfo['names'][boneName]
        points = np.array([
                pointsIn[plateAssignment[boneModelName][0]],
                pointsIn[plateAssignment[boneModelName][1]],
                pointsIn[plateAssignment[boneModelName][2]] # middle marker
                ])
        pinsSets = generatePinsFromMarkerPoints(points)
        pinsSets = pinsSets[possiblePinSet[scene.modelInfo['experimentID']][currentModel][boneName]]
        top = (pinsSets[0, 0, :] + pinsSets[0, 2, :])/2
        bottom = (pinsSets[0, 1, :] + pinsSets[0, 3, :])/2
        pointsAtEndsOfPins[boneName] = np.array([top, bottom])
    return pointsAtEndsOfPins

def generateEndsOfPinsPlanned(scene):
    pins = scene.modelCacheExtra['sensorGuidePins']
    pointsAtEndsOfPins = {}
    for boneName in fn.boneNames():
        pin1 = pins['pin_' + boneName + '1']
        pin2 = pins['pin_' + boneName + '2']
        top = (np.array(pin2['point2']) + np.array(pin1['point2']))/2
        bottom = (np.array(pin2['point1']) + np.array(pin1['point1']))/2
        vecTowardsBone = fm.normalizeVector(np.array(pin1['point1']) - np.array(pin1['point2']))
        depth = 5 # attempted to seat wires 5mm deep into bone
        if boneName in scene.alignmentMethod[scene.modelInfo['experimentID']]['pinDepth_normal']:
            depth = scene.alignmentMethod[scene.modelInfo['experimentID']]['pinDepth_normal'][boneName]
        top = top + vecTowardsBone * depth
        bottom = bottom + vecTowardsBone * depth
        pointsAtEndsOfPins[boneName] = np.array([top, bottom])
    return pointsAtEndsOfPins

def alignUsing(scene, flip, modelInfo, boneNames,
    modelTypeTarget, modelTypeSource,
    pointsTarget, pointsSource):

    if len(boneNames) == 0:
        return np.eye(4)

    if len(boneNames) == 1:
        boneName = boneNames[0]
        # if only using one marker
        t_World_metacarp3_cut = getMarkerTransformation(flip, modelInfo, modelTypeTarget, boneName, pointsTarget)
        t_World_metacarp3_sca = getMarkerTransformation(flip, modelInfo, modelTypeSource, boneName, pointsSource)
        return np.dot(t_World_metacarp3_cut, fm.inverseTransformationMat(t_World_metacarp3_sca))
    
    boneNamesToExclude = fn.boneNames()
    for boneName in boneNames:
        boneNamesToExclude.remove(boneName)

    for boneName in boneNames:
        if shouldFlip(flip, modelInfo, modelTypeTarget, boneName):
            boneModelName = modelInfo['names'][boneName]
            plateAssignment = modelInfo['plateAssignment_' + modelTypeTarget]
            pTemp = deepcopy(pointsTarget[plateAssignment[boneModelName][0]])
            pointsTarget[plateAssignment[boneModelName][0]] = pointsTarget[plateAssignment[boneModelName][1]]
            pointsTarget[plateAssignment[boneModelName][1]] = pTemp
        if shouldFlip(flip, modelInfo, modelTypeSource, boneName):
            boneModelName = modelInfo['names'][boneName]
            plateAssignment = modelInfo['plateAssignment_' + modelTypeSource]
            pTemp = deepcopy(pointsSource[plateAssignment[boneModelName][0]])
            pointsSource[plateAssignment[boneModelName][0]] = pointsSource[plateAssignment[boneModelName][1]]
            pointsSource[plateAssignment[boneModelName][1]] = pTemp

    pointsSource_removed = excludeMarkersFromPointCloud(modelInfo, pointsSource, modelTypeSource, boneNamesToExclude)
    pointsTarget_removed = excludeMarkersFromPointCloud(modelInfo, pointsTarget, modelTypeTarget, boneNamesToExclude)

    # maybe they need to be ordered before alignment
    indToSortSource = np.array([], dtype=int)
    indToSortTarget = np.array([], dtype=int)
    for boneName in boneNames:
        boneModelName = modelInfo['names'][boneName]
        plateAssignment = modelInfo['plateAssignment_' + modelTypeSource]
        indToSortSource = np.append(indToSortSource, plateAssignment[boneModelName])
        
        plateAssignment = modelInfo['plateAssignment_' + modelTypeTarget]
        indToSortTarget = np.append(indToSortTarget, plateAssignment[boneModelName])
    i = 0
    for ind, ii in enumerate(indToSortSource):
        indToSortSource[np.argsort(indToSortSource)[ind]] = i
        i += 1
    i = 0
    for ind, ii in enumerate(indToSortTarget):
        indToSortTarget[np.argsort(indToSortTarget)[ind]] = i
        i += 1
    pointsSource_removed = pointsSource_removed[indToSortSource]
    pointsTarget_removed = pointsTarget_removed[indToSortTarget]

    # show cloud points used for alignment, pairs should have same colors
    #colors = ['red', 'green', 'yellow', 'blue', 'black', 'purple']
    #for ind, color in enumerate(colors):
    #    scene.actors[1]['pointsSource_removed_'+str(ind)] = scene.plotter[0, 1].add_points(
    #        pointsSource_removed[ind,:],
    #        name = 'pointsSource_removed_'+str(ind),
    #        #scalars = colors,
    #        color = color,
    #        render_points_as_spheres=True,
    #        point_size=20.0,
    #        opacity = 0.9
    #    )
    #    scene.actors[1]['pointsTarget_removed_'+str(ind)] = scene.plotter[0, 1].add_points(
    #        pointsTarget_removed[ind,:],
    #        name = 'pointsTarget_removed_'+str(ind),
    #        #scalars = colors,
    #        color = color,
    #        render_points_as_spheres=True,
    #        point_size=20.0,
    #        opacity = 0.9
    #    )

    if len(boneNames) >= 2:
        return findMarkerToPointsAlignmentOnce(pointsTarget_removed, pointsSource_removed)
        
    if len(boneNames) == 2: # not great, and slow...
        # align scaffold marker group to cut marker group using only radius and metacarp3 markers in each group
        # point cloud align with only two marker plates

        import slil.process.alignment_pointcloud as fna
        [result_error, tran, foundErrors, foundTransforms] = fna.cloudPointAlign_v3(
            pointsTarget_removed, pointsSource_removed)

        print('Alignment Error is {} for experiment {}'.format(result_error, modelInfo['experimentID']))
        return fm.inverseTransformationMat(tran)


def moveBones(flip, modelInfo, scene, bonesToMove, modelTypeTarget, modelTypeSource, optimizedFramePoints, alignedPoints):
    for boneName in bonesToMove:
        #t_World_bone_initial = fn.getMarkerGroupTranformationMat(modelInfo, modelInfo['names'][boneName], optimizedFramePoints[modelTypeTarget])
        t_World_bone_initial = getMarkerTransformation(flip, modelInfo, modelTypeTarget, boneName, optimizedFramePoints[modelTypeTarget])
        t_World_bone_aligned = getMarkerTransformation(flip, modelInfo, modelTypeSource, boneName, alignedPoints)

        bonePD = deepcopy(scene.bonePolyData[modelInfo['names'][boneName]])
        fm.transformPoints_byRef(bonePD.points, t_World_bone_initial, t_World_bone_aligned)
        #scene.bonePolyData[modelInfo['names'][boneName]+'_moved'] = bonePD
        scene.bonePolyData[modelInfo['names'][boneName]+'_moved'].shallow_copy(bonePD)

def moveBonesMulti(indPlot, flip, modelInfo, scene, bonesToMove, modelTypeTarget, modelTypeSource, optimizedFramePoints, alignedPoints):
    for boneName in bonesToMove:
        #t_World_bone_initial = fn.getMarkerGroupTranformationMat(modelInfo, modelInfo['names'][boneName], optimizedFramePoints[modelTypeTarget])
        t_World_bone_initial = getMarkerTransformation(flip, modelInfo, modelTypeTarget, boneName, optimizedFramePoints[modelTypeTarget])
        t_World_bone_aligned = getMarkerTransformation(flip, modelInfo, modelTypeSource, boneName, alignedPoints)

        #bonePD = deepcopy(scene.bonePolyData[indPlot][modelInfo['names'][boneName]])
        #fm.transformPoints_byRef(bonePD.points, t_World_bone_initial, t_World_bone_aligned)
        ##scene.bonePolyData[indPlot][modelInfo['names'][boneName]+'_moved'] = bonePD
        #scene.bonePolyData[indPlot][modelInfo['names'][boneName]+'_moved'].shallow_copy(bonePD)
        
        # could be slower
        bonePD = scene.bonePolyData[indPlot][modelInfo['names'][boneName]]
        bonePointsMoved = fm.transformPoints(bonePD.points, t_World_bone_initial, t_World_bone_aligned)
        scene.plotter[0, indPlot].update_coordinates(bonePointsMoved, scene.bonePolyData[indPlot][modelInfo['names'][boneName]+'_moved'])

def getFinalPoints(flip, modelInfo, bonesToMove, modelTypeTarget, modelTypeSource, optimizedFramePoints, alignedAndSwappedPoints, alignedPoints):
    markersFinal = np.array(alignedPoints)
    for boneName in bonesToMove:
        t_World_bone_initial = getMarkerTransformation(flip, modelInfo, modelTypeTarget, boneName, optimizedFramePoints[modelTypeTarget])
        t_World_bone_aligned = getMarkerTransformation(flip, modelInfo, modelTypeSource, boneName, alignedAndSwappedPoints)

        plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
        boneModelName = modelInfo['names'][boneName]
        points = np.array([
                alignedPoints[plateAssignment[boneModelName][0]],
                alignedPoints[plateAssignment[boneModelName][1]],
                alignedPoints[plateAssignment[boneModelName][2]] # middle marker
                ])
        bonePointsMoved = fm.transformPoints(points, t_World_bone_aligned, t_World_bone_initial)

        markersFinal[plateAssignment[boneModelName][0]] = bonePointsMoved[0]
        markersFinal[plateAssignment[boneModelName][1]] = bonePointsMoved[1]
        markersFinal[plateAssignment[boneModelName][2]] = bonePointsMoved[2]
    return markersFinal


def align(scene, indPlot, modelInfo,
    modelTypeTarget, modelTypeSource,
    framePoints,
    transfromAdjustedNormal,
    initTransMarkers,
    alignmentMethod,
    flip, framePointsSwapped):
    optimizedFramePoints = {}

    modelInfo['currentModel'] = modelTypeTarget
    adjustedTransMarkers = np.dot(transfromAdjustedNormal, initTransMarkers[modelTypeTarget])
    t_World_radius_cut = fn.getMarkerGroupTranformationMat(modelInfo, modelInfo['names']['radius'], framePoints[modelTypeTarget])
    optimizedFramePoints[modelTypeTarget] = fm.transformPoints(np.array(framePoints[modelTypeTarget]), t_World_radius_cut, adjustedTransMarkers)
    # this line produces the same as the above three lines, only if the c3d has not changed since model generation
    #newCutPoints = fm.transformPoints(np.array(framePoints[modelTypeTarget]), np.eye(4), transfromAdjustedNormal)
    
    if modelInfo['experimentID'] == '11527':
        tran = np.eye(4)
    else:
        # should be toggleable
        pointsAtEndsOfPins = generateEndsOfPins(scene, optimizedFramePoints[modelTypeTarget], 'normal', scene.possiblePinSet)
        endsOfPinsPlanned = generateEndsOfPinsPlanned(scene)
        pointsExperiment = np.empty((0,3))
        pointsPlanned = np.empty((0,3))
        boneNames = alignmentMethod[modelInfo['experimentID']]['byPins_planned2normal']
        for boneName in boneNames:
            pointsExperiment = np.append(pointsExperiment, pointsAtEndsOfPins[boneName], axis=0)
            pointsPlanned = np.append(pointsPlanned, endsOfPinsPlanned[boneName], axis=0)
        tran = findMarkerToPointsAlignmentOnce(pointsPlanned, pointsExperiment)
    #optimizedFramePoints[modelTypeTarget] = fm.transformPoints(optimizedFramePoints[modelTypeTarget], np.eye(4), tran)
    optimizedFramePoints[modelTypeTarget] = fm.transformPoints(np.array(framePoints[modelTypeTarget]), np.eye(4), tran)
    optimizedFramePoints[modelTypeTarget] = fm.transformPoints(optimizedFramePoints[modelTypeTarget], np.eye(4), transfromAdjustedNormal)
    
    scene.markersFinal[modelTypeTarget] = optimizedFramePoints[modelTypeTarget] # should be 'normal'

    markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static_new'
    scene.addPoints(indPlot, optimizedFramePoints[modelTypeTarget], markerGroupName, color='red')


    #modelInfo['currentModel'] = 'normal'
    #markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static'
    #scene.addPoints(indPlot, np.array(framePoints[modelInfo['currentModel']]), markerGroupName, color='green')


    modelInfo['currentModel'] = modelTypeTarget
    markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static'
    #scene.addPoints(indPlot, np.array(framePoints[modelTypeTarget]), markerGroupName, color='brown')


    #adjustedFramePoints[modelTypeSource] = fm.transformPoints(np.array(framePoints[modelTypeSource]), np.eye(4), transfromAdjustedNormal)
    optimizedFramePoints[modelTypeSource] = np.array(framePoints[modelTypeSource])
    #scene.addPoints(indPlot, adjustedFramePoints[modelTypeSource], 'tttt', color='grey')

    boneNames = alignmentMethod[modelInfo['experimentID']][modelTypeTarget + str(2) + modelTypeSource]
    #boneNames = [ 'metacarp3', 'radius' ]
    #boneNames = [ 'radius' ]
    #boneNames = [ 'metacarp3' ]
    if len(alignmentMethod[modelInfo['experimentID']]['byPins_' + modelTypeTarget + str(2) + modelTypeSource]) > 0:
        pointsAtEndsOfPinsTarget = generateEndsOfPins(scene, optimizedFramePoints[modelTypeTarget], 'normal', scene.possiblePinSet)
        #pointsAtEndsOfPinsSource = generateEndsOfPins(scene, optimizedFramePoints[modelTypeSource], modelTypeSource, scene.possiblePinSet)
        pointsMovedIfChangePinDepth = np.array(deepcopy(optimizedFramePoints[modelTypeSource]))
        if scene.modelInfo['experimentID'] in scene.alignmentMethod:
            for boneName in scene.alignmentMethod[scene.modelInfo['experimentID']]['pinDepth_normal2'+modelTypeSource]:
                vecTowardBone = getPinVector(scene, scene.mapMarkerChangesN2[modelTypeSource], framePointsSwapped, boneName, modelTypeSource)
                depth = scene.alignmentMethod[scene.modelInfo['experimentID']]['pinDepth_normal2'+modelTypeSource][boneName]
                
                plateAssignment = scene.modelInfo['plateAssignment_' + modelTypeSource]
                boneModelName = scene.modelInfo['names'][boneName]
                pointsMovedIfChangePinDepth[plateAssignment[boneModelName][0]] = pointsMovedIfChangePinDepth[plateAssignment[boneModelName][0]] + vecTowardBone * depth
                pointsMovedIfChangePinDepth[plateAssignment[boneModelName][1]] = pointsMovedIfChangePinDepth[plateAssignment[boneModelName][1]] + vecTowardBone * depth
                pointsMovedIfChangePinDepth[plateAssignment[boneModelName][2]] = pointsMovedIfChangePinDepth[plateAssignment[boneModelName][2]] + vecTowardBone * depth

        pointsAtEndsOfPinsSource = generateEndsOfPins(scene, pointsMovedIfChangePinDepth, modelTypeSource, scene.possiblePinSet)

        pointsSource = np.empty((0,3))
        pointsTarget = np.empty((0,3))
        boneNames = alignmentMethod[modelInfo['experimentID']]['byPins_' + modelTypeTarget + str(2) + modelTypeSource]
        for boneName in boneNames:
            pointsSource = np.append(pointsSource, pointsAtEndsOfPinsSource[boneName], axis=0)
            pointsTarget = np.append(pointsTarget, pointsAtEndsOfPinsTarget[boneName], axis=0)
        tran = findMarkerToPointsAlignmentOnce(pointsTarget, pointsSource)
    else:
        tran = alignUsing(scene, flip, modelInfo, boneNames,
            modelTypeTarget = modelTypeTarget,
            modelTypeSource = modelTypeSource,
            pointsTarget = optimizedFramePoints[modelTypeTarget],
            pointsSource = np.array(framePointsSwapped)) # we use swapped points so that visualisation still shows original non-swapped points

    alignedPoints = np.array(deepcopy(optimizedFramePoints[modelTypeSource]))
    alignedPoints = fm.transformPoints(alignedPoints, np.eye(4), tran)
    scene.addPoints(indPlot, alignedPoints, 'alignedPoints', 'blue')


    modelInfo['currentModel'] = 'scaffold'
    markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static'
    #scene.addPoints(indPlot, np.array(framePoints[modelInfo['currentModel']]), markerGroupName, color='blue')


    #t_World_scaphoid_initial = fn.getMarkerGroupTranformationMat(modelInfo, modelInfo['names']['scaphoid'], optimizedFramePoints[modelTypeTarget])
    #t_World_scaphoid_aligned = getMarkerTransformation(flip, modelInfo, modelTypeSource, 'scaphoid', alignedPoints)
    #pdSca = deepcopy(scene.bonePolyData[modelInfo['names']['scaphoid']])
    #fm.transformPoints_byRef(pdSca.points, t_World_scaphoid_initial, t_World_scaphoid_aligned)
    #scene.bonePolyData[modelInfo['names']['scaphoid']+'_moved'] = pdSca
    alignedAndSwappedPoints = fm.transformPoints(np.array(framePointsSwapped), np.eye(4), tran)
    moveBonesMulti(indPlot, flip, modelInfo, scene,
        [ 'lunate', 'scaphoid' ],
        modelTypeTarget, modelTypeSource, optimizedFramePoints, alignedAndSwappedPoints) #alignedPoints

    scene.markersFinal[modelTypeSource] = getFinalPoints(flip, modelInfo,
        [ 'lunate', 'scaphoid', 'radius', 'metacarp3'],
        modelTypeTarget, modelTypeSource,
        optimizedFramePoints, alignedAndSwappedPoints, alignedPoints)

    #scene.addMeshes(indPlot)

    scene.setOpacity(indPlot, modelInfo['names']['radius'], 0.5)
    scene.setOpacity(indPlot, modelInfo['names']['metacarp3'], 0.5)
    capName = [x for x in modelInfo['otherHandBones'] if 'cap' in x][0]
    scene.setOpacity(indPlot, capName, 0.5)

    scene.setOpacity(indPlot, modelInfo['names']['lunate'], 0.2)
    scene.setOpacity(indPlot, modelInfo['names']['scaphoid'], 0.2)

    #scene.plotter[0, indPlot].remove_actor(scene.actors[indPlot]['placedScaffold'])
    #scene.plotter[0, indPlot].remove_actor(scene.actors[modelInfo['names']['lunate']])
    #scene.plotter[0, indPlot].remove_actor(scene.actors[modelInfo['names']['scaphoid']])

    #modelInfo['currentModel'] = 'cut'
    #markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static'
    #framePointsCut = fn.getPointsFromSphereGroup(mi, markerGroupName)
    #scene.addPoints(framePointsCut, markerGroupName)
    #scene.viewScene(indPlot)
    return optimizedFramePoints[modelTypeTarget], alignedPoints

def markerHolderTemplate():
    # Marker A
    markerLef = np.array([-18.56, 43.67, 3.61])
    markerRig = np.array([20.39, 43.67, 3.61])
    markerMid = np.array([0.92, 27.11, 7.11])
    #markerMid = np.array([-27.11, 0.92, -7.11])
    markerPointsTemplate = np.array([markerLef, markerRig, markerMid])

    pin1_top = np.array([0.0, 18.0, 0.0]) # top is closest to markers
    pin1_bottom = np.array([0.0, 18.0 - 45.0, 0.0])
    pinBack = np.array([1.84, 0.0, -1.84])
    pin2_top = pin1_top + pinBack
    pin2_bottom = pin1_bottom + pinBack
    pinsA1 = np.array([
        pin1_top,
        pin1_bottom
    ])
    pinsA2 = np.array([
        pin2_top,
        pin2_bottom
    ])

    markerBetweenLefRig = np.array([(markerLef + markerRig)/2])
    vec1 = fm.normalizeVector(markerMid - markerBetweenLefRig)
    angleToMirror = (np.pi - fm.angleBetweenVectors(vec1, fm.normalizeVector(pin1_top - pin1_bottom))[0])

    # Marker B (mirrored Marker A)
    pin1_top = np.array([0.0 + 1.84, 18.0, 0.0])
    pin1_bottom = np.array([0.0 + 1.84, 18.0 - 45.0, 0.0])
    pinBack = np.array([-1.84, 0.0, -1.84])
    pin2_top = pin1_top + pinBack
    pin2_bottom = pin1_bottom + pinBack
    pinsB1 = np.array([
        pin1_top,
        pin1_bottom
    ])
    pinsB2 = np.array([
        pin2_top,
        pin2_bottom
    ])
    return markerPointsTemplate, angleToMirror, pinsA1, pinsA2, pinsB1, pinsB2

def alignPointsFromMarkerTemplate(markerPoints, allPointToMove):
    # markerPoints : 3 points to align template to
    # allPointToMove : all points to be moved
    
    markerPointsTemplate, angleToMirror, _, _, _, _ = markerHolderTemplate()
    # markerPointsTemplate : 3 points
    # angleToMirror : 1 angle

    tM = findMarkerToPointsAlignmentOnce(pTo=markerPoints, pFrom=markerPointsTemplate)
    
    markerPointsTemplateMoved1 = fm.transformPoints(markerPointsTemplate, np.eye(4), tM)
    r1 = fm.createTransformationMatFromPosAndEuler(0, 0, 0, -angleToMirror*2, np.pi, 0)
    tM2 = np.dot(tM, r1)
    markerPointsTemplateMoved2 = fm.transformPoints(markerPointsTemplate, np.eye(4), tM2)
    a = markerPointsTemplateMoved1[2] - markerPointsTemplateMoved2[2] # middle pin
    r2 = fm.createTransformationMatFromPosAndEuler(a[0], a[1], a[2], 0, 0, 0)

    markerPointsMoved1 = fm.transformPoints(allPointToMove, np.eye(4), tM)
    markerPointsMoved = fm.transformPoints(allPointToMove, np.eye(4), tM2)
    markerPointsMoved = fm.transformPoints(markerPointsMoved, np.eye(4), r2)
    # return two sets of points because markers could be flipped
    return markerPointsMoved1, markerPointsMoved

def generatePinsFromMarkerPoints(markerPoints):
    markerPointsTemplate, angleToMirror, pinsA1, pinsA2, pinsB1, pinsB2 = markerHolderTemplate()
    # 3 marker points
    # 2 pinsA1 points
    # 2 pinsA2 points
    # 2 pinsB1 points
    # 2 pinsB2 points
    markerPointsAndPinsTemplate = np.append(markerPointsTemplate, pinsA1, axis=0)
    markerPointsAndPinsTemplate = np.append(markerPointsAndPinsTemplate, pinsA2, axis=0)
    markerPointsAndPinsTemplate = np.append(markerPointsAndPinsTemplate, pinsB1, axis=0)
    markerPointsAndPinsTemplate = np.append(markerPointsAndPinsTemplate, pinsB2, axis=0)

    m1, m2 = alignPointsFromMarkerTemplate(markerPoints, markerPointsAndPinsTemplate)
    pins = np.empty((4, 4, 3))
    pins[0, :, :] = m1[3:7, :]
    pins[1, :, :] = m1[7:11, :]
    pins[2, :, :] = m2[3:7, :]
    pins[3, :, :] = m2[7:11, :]
    return pins

def generateMarkerPointsFromPins(pins):
    # pins(3x3): 3 points. 2 points from pin1, 1 point from pin2
    
    # The following are for version 2 of marker plates with 2 possible configurations
    # Each pair of pins should have four possible marker orientations

    markerPointsTemplate, angleToMirror, pinsA1, pinsA2, pinsB1, pinsB2 = markerHolderTemplate()
    markerSets = np.empty((4, 3, 3))
    # generate all points from marker pins
    pp = np.empty((3, 3))

    pp[:2, :] = pinsA1
    pp[2, :] = pinsA2[0, :]
    tM = findMarkerToPointsAlignmentOnce(pTo=pins, pFrom=pp)
    markerSets[0, :, :] = fm.transformPoints(markerPointsTemplate, np.eye(4), tM)

    pp[:2, :] = pinsA2
    pp[2, :] = pinsA1[0, :]
    tM = findMarkerToPointsAlignmentOnce(pTo=pins, pFrom=pp)
    markerSets[1, :, :] = fm.transformPoints(markerPointsTemplate, np.eye(4), tM)

    pp[:2, :] = pinsB1
    pp[2, :] = pinsB2[0, :]
    tM = findMarkerToPointsAlignmentOnce(pTo=pins, pFrom=pp)
    markerSets[2, :, :] = fm.transformPoints(markerPointsTemplate, np.eye(4), tM)

    pp[:2, :] = pinsB2
    pp[2, :] = pinsB1[0, :]
    tM = findMarkerToPointsAlignmentOnce(pTo=pins, pFrom=pp)
    markerSets[3, :, :] = fm.transformPoints(markerPointsTemplate, np.eye(4), tM)
    return markerSets