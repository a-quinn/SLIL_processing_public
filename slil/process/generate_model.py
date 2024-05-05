# Author: Alastair Quinn 2022
from statistics import mode
import slil.process.functions as fn
import slil.process.alignment_pointcloud as fna
import slil.common.data_configs as dc
import slil.common.math as fm
import slil.common.data_configs as dc
import slil.common.io as fio
import numpy as np
from slil.mesh.interface import MeshInterface

def findPossibleAligment(modelInfo, rawPoints, pins, coms):

    possibleMarkers, boneWithMarkerSets = fna.generatePossibleMarkerLocationsFromPins(modelInfo, pins, coms)
    
    numCombs = np.power(4, 4)
    numCombs += (np.power(4, 3) * 4)
    #for i in possibleMarkers:
    #    for ii in i:
    #        for iii in ii:
    #            mi.create_point(iii)

    # For all bones
    # possible marker configurations for each bone
    #boneIndexs = range(0, 12)
    ##p1 = np.empty(shape=(len(boneIndexs),3))
    ##for ind, i in enumerate(possibleMarkers[0]):
    ##    p1[ind*3:(ind*3+3)] = possibleMarkers[ind][0]
    ## points imported from static
    #
    #p2s = fna.getPoints(mi, modelInfo)
    #p2 = np.empty(shape=(len(boneIndexs),3))
    #for ind, i in enumerate(boneIndexs):
    #    p2[ind] = p2s[i]
    
    p2 = fna.getPointsOrdered(rawPoints, modelInfo['plateAssignment_' + modelInfo['currentModel']])

    possibleMarkers2, combinations = fna.generateCombinations(modelInfo, numCombs, possibleMarkers)

    # add combinatiosn without marker sets (4 possible combinatiosn if 4 sets of markers)

    # TODO: add combinations without plates
    # TODO: need to check inlier_rmse is best mesaure for closeness of two point clouds
    # TODO: maybe not all marker plate configuratiosn should be searched as some point in opposite direction to camera direction

    # find transform with minimum error
    alignmentResults = np.empty(shape=(numCombs), dtype=float)
    alignmentResultsFitness = np.empty(shape=(numCombs), dtype=float)
    alignmentMat = np.empty(shape=(numCombs, 4, 4), dtype=float)
    for ind, markerCombo in enumerate(possibleMarkers2):
        # check for comb where nan
        p2Temp = p2
        markerComboTemp = markerCombo
        if (np.any(np.isnan(markerCombo[:, 0]))):
            markerComboTemp = np.delete(markerCombo, np.where(np.isnan(markerCombo[:, 0]))[0], axis=0)
            # should be 0, 3, 6, or 9
            ind2 = np.where(np.isnan(markerCombo[:, 0]))[0][0]
            ind2 = int(ind2 / 3)
            # has to be in order of boneWithMarkerSets
            #boneWithMarkerSets[ind2]
            ind3 = modelInfo['plateAssignment_' + modelInfo['currentModel']][modelInfo['names'][boneWithMarkerSets[3]]]
            p2Temp = np.delete(p2, ind3, axis=0)

        [result_ransac, p1Out, final, tran] = fna.cloudPointAlign_v2(markerComboTemp, p2Temp, False)
        alignmentResults[ind] = result_ransac.inlier_rmse
        alignmentResultsFitness[ind] = result_ransac.fitness
        alignmentMat[ind] = tran

    if (len(np.where(alignmentResults == 0.0)[0]) > 0):
        print("Warning! Some ({}) marker plate configurations failed to align!".format(len(np.where(alignmentResults == 0.0)[0])))
    alignmentResults2 = np.delete(alignmentResults, np.where(alignmentResults == 0.0))

    #markerCombo = possibleMarkers2[np.where(alignmentResults == alignmentResults2.min())][0]
    #[result_ransac, p1Out, final, tran] = fna.cloudPointAlign_v2(markerCombo, p2, True)

    #alignmentResults3 = np.delete(alignmentResults, np.where(alignmentResults > 10.0))
    #alignmentResults3 = np.delete(alignmentResults3, np.where(alignmentResults3 == alignmentResults3.min()))
    #markerCombo = possibleMarkers2[np.where(alignmentResults == alignmentResults3.min())][0]
    #[result_ransac, p1Out, final, tran] = fna.cloudPointAlign_v1(markerCombo, p2, True)


    nthSmallest = 0 # smallest value in alignmentResults
    index = int(np.where(alignmentResults == np.partition(alignmentResults, nthSmallest)[nthSmallest])[0])
    # I thinks all fitness's are 1 unless alginment didn't work...
    print("Smallest Inlier RMSE: {}mm, Fitness: {}".format(alignmentResults[index], alignmentResultsFitness[index] ))
    comb = combinations[index]
    print("Combo used: {}".format(comb))
    markerCombo = possibleMarkers2[index]
    tran = alignmentMat[index]
    #show static new position
    #a0 = np.empty(shape=(len(p2),3))
    #for ind, i in enumerate(p2):
    #    a0[ind] = fm.rotateVector(i, tran[:3, :3]) + (tran[:3, 3].T)
    #    mi.create_point(a0[ind])
    # show marker plate combination
    #for i in markerCombo:
    #    mi.create_point(i)

    adjustRot = np.eye(3)
    # correction to align FE wrist motion to be in line with an axis plane
    if False:
        from slil.common.cache import loadCache
        dataRotations = loadCache('dataRotations')
        name = modelInfo['experimentID'] + '_' + modelInfo['currentModel']
        exisitngIndex = [i for i, x in enumerate(dataRotations) if name in x.keys()]
        if exisitngIndex:
            dictContents = dataRotations[exisitngIndex[0]][name]
            adjustRot = dictContents['rot1']
            adjustRot = adjustRot * [[1., -1., 0.], [-1., 1., 0.], [0., 0., 1.]]
            m4 = fm.rotationMatrixToEulerAngles(adjustRot)
            m4 = [m4[2], -m4[0], -m4[1]] # Coordinate systems OpenSim to 3-Matic
            adjustRot = fm.eulerAnglesToRotationMatrix(m4)
        else:
            print("error no adjustment rotation matrix found")
    
    return tran

def generate_STLs(mi: MeshInterface, experiments):
    for experiment in experiments:
        modelInfo = dc.load(experiment)
        mi.open_project(modelInfo['3_matic_file'])
        fn.exportBones(modelInfo)

def generate_model(mi: MeshInterface, experiments):
    showMarkersIn3Matic = False # also saves any changes to file
    for experiment in experiments:
    #if True:
        #experiment = '11538'
        modelInfo = dc.load(experiment)
        modelTypes = [ 'normal', 'cut', 'scaffold']
        #modelTypes = [ 'cut' ]
        
        mi.open_project(modelInfo['3_matic_file'])
        
        # if first time and no boneplug points generated from raw scaffold STL, run this part manually
        # Needs two parts
        # 1. line named 'scaffold_edge_l_to_s'
        # 2. bottom surface of lunate bone-plug named 'lunate_boneplug_bottom'
        # It think left hands require True, second argument
        #fn.generateBoneplugPoints(modelInfo, True)

        modelCache = fn.getModelCache(modelInfo) # creates cache
        modelCacheExtra = fn.getModelCacheExtra(modelInfo)
        pins = modelCacheExtra['sensorGuidePins']
        
        coms = {
            modelInfo['names']['lunate']: modelCache['lunateCOM'],
            modelInfo['names']['scaphoid']: modelCache['scaphoidCOM'],
            modelInfo['names']['radius']: modelCache['radiusCOM'],
            modelInfo['names']['metacarp3']: modelCache['metacarp3COM'],
        }
        #def getCOMs2(names):
        #    coms = {}
        #    for name in names:
        #        geometry = fn.getGeometry(modelInfo, name)
        #        com = fm.calcCOM(np.array(geometry[1]), geometry[0], 'mesh')
        #        coms[name] = {
        #            'COM': com,
        #        }
        #    return coms
        #coms = getCOMs2(fn.boneModelNames(modelInfo))
        
        for modelType in modelTypes:
            modelInfo['currentModel'] = modelType
            fn.setBonesInModel(modelInfo, coms)

        newMarkerPositions = {}
        for modelType in modelTypes:
            modelInfo['currentModel'] = modelType
    
            #trial = '\\' + modelInfo['currentModel'] + '_static\log1'
            trial = '\\' + modelInfo['currentModel'] + '_fe_40_40\log1'
            #trial = '\\' + modelInfo['currentModel'] + '_static_after\log1'
            markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static'
            #markerGroupName = 'tempMarkerSpheres_scaffold_static_after'

            isTrialSet = False
            for i in modelInfo['trialsRawData_only_static']:
                if modelInfo['currentModel'] in i:
                    print("Found static trial: {}".format(i))
                    if not isTrialSet:
                        isTrialSet = True
                        trial = i

            #trial = r'\normal_fe_40_40\log1'
            #trial = r'\normal_ur_30_30\log1'
            #trial = r'\normal_static_after\log'

            if isTrialSet:
                print("Using static trial: {}".format(trial))

            fileToImport = modelInfo['dataOutputDir'] + trial + '.c3d'
            frames = fio.readC3D(fileToImport)

            frameN = 1
            rawPoints = frames[frameN,:,0:3]

            if showMarkersIn3Matic:
                latestPoints = []
                for point in rawPoints:
                    #point[1] = point[1]-1000 # just to move them closer to the center
                    latestPoints.append(mi.create_point(point))
                fn.namePoints(latestPoints)
                fn.colourPoints(latestPoints, modelInfo)
                if not fn.checkMarkersValid(latestPoints):
                    print("Warning! Invalid static marker points!")

            if len(pins) == 0: # if model has no marker pins
                if modelInfo['experimentID'] == '11527':
                    # get from 3-Matic environment
                    points = fn.getPointsFromSphereGroup(mi, markerGroupName)
                    t_WOP1 = fn.getMarkerGroupTranformationMat(modelInfo, modelInfo['names']['radius'], points)
                    t_WOP1_raw = fn.getMarkerGroupTranformationMat(modelInfo, modelInfo['names']['radius'], rawPoints)
                    t_WOP1_raw = fm.inverseTransformationMat(t_WOP1_raw)
                    transformation = np.dot(t_WOP1, t_WOP1_raw)
                else:
                    #then find transformation to origin
                    [result_ransac, p1Out, final, tran] = fna.cloudPointAlign_v2([[0.0, 0.0, 0.0]], rawPoints, False)
                    transformation = tran
            else:
                transformation = findPossibleAligment(modelInfo, rawPoints, pins, coms)

            newMarkerPoints0 = []
            for point in rawPoints:
                coord = fm.rotateVector(point, transformation[:3, :3]) + (transformation[:3, 3].T)
                newMarkerPoints0.append(np.array([coord[0], coord[1], coord[2]]))

            if showMarkersIn3Matic:
                for ind, i in enumerate(latestPoints):
                    coord = fm.rotateVector(i.coordinates, transformation[:3, :3]) + (transformation[:3, 3].T)
                    
                    #pointsTemp = np.array([coord[0], coord[1], coord[2], 1.0])
                    #coord = np.dot(adjustRot, pointsTemp)[:3]

                    #coord = np.dot(adjustRot, coord)
                    nameTemp = i.name
                    colourTemp = i.color
                    mi.delete(i)
                    latestPoints[ind] = mi.create_point(coord)
                    latestPoints[ind].name = nameTemp
                    latestPoints[ind].color = colourTemp
                fn.colourPoints(latestPoints, modelInfo)
            #else:
            #    latestPoints = []
            #    for point in rawPoints:
            #        coord = fm.rotateVector(point, tran[:3, :3]) + (tran[:3, 3].T)
            #        pointsTemp = np.array([coord[0], coord[1], coord[2], 1.0])
            #        coord = np.dot(adjustRot, pointsTemp)[:3]
            #        latestPoints.append(mi.create_point(coord))
            #    fn.namePoints(latestPoints)
            #    fn.colourPoints(latestPoints, modelInfo)


            #tran = alignmentMat[np.where(alignmentResults == alignmentResults2.min())][0]
            #a0 = np.empty(shape=(len(p2),3))
            #for ind, i in enumerate(p2):
            #    a0[ind] = fm.rotateVector(i, tran[:3, :3]) + (tran[:3, 3].T)
            #    mi.create_point(a0[ind])
            #
            #for i in final:
            #    mi.create_point(i)

            if showMarkersIn3Matic:
                fn.deleteSphereGroupToPoints(markerGroupName) # if already exists
                fn.convertPointsToSphereGroup(modelInfo, markerGroupName)

                mi.save_project(modelInfo['3_matic_file'])
            
            t_WOP1 = fn.getMarkerGroupTranformationMat(modelInfo, modelInfo['names']['radius'], newMarkerPoints0)
            fn.modifyModelCache_initTmatRadiusMarker(modelInfo, t_WOP1)

            newMarkerPositions[modelInfo['currentModel']] = newMarkerPoints0
        
        fn.modifyModelCache_markerPositions(modelInfo, newMarkerPositions)

    
        for modelType in modelTypes:
            modelInfo['currentModel'] = modelType
            newMarkerPoints0 = newMarkerPositions[modelInfo['currentModel']]

            fn.setMarkersInModel(modelInfo, coms, newMarkerPoints0)

    print("Done generating models.")

def setOptimizedMarkers(experiments, modelTypes):
    for experiment in experiments:
        modelInfo = dc.load(experiment)
        modelCache = fn.getModelCache(modelInfo, return_false_if_not_found=True)
        optResults = fn.getOptimizationResultsAsTransMat(modelInfo, '_SLILGap')
        for modelType in modelTypes:
            modelInfo['currentModel'] = modelType

            adjustRot = optResults[modelInfo['currentModel']]

            newMarkerPoints0 = modelCache['markers'][modelInfo['currentModel']]

            newMarkerPoints = []
            for point in newMarkerPoints0:
                pointsTemp = np.array([point[0], point[1], point[2], 1.0])
                coord = np.dot(adjustRot, pointsTemp)[:3]
                newMarkerPoints.append(coord)

            coms = {
                modelInfo['names']['lunate']: modelCache['lunateCOM'],
                modelInfo['names']['scaphoid']: modelCache['scaphoidCOM'],
                modelInfo['names']['radius']: modelCache['radiusCOM'],
                modelInfo['names']['metacarp3']: modelCache['metacarp3COM'],
            }

            fn.setMarkersInModel(modelInfo, coms, newMarkerPoints)


