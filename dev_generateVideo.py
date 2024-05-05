# Author: Alastair Quinn 2022

def generateCut2ScaffoldVideo(experiment):
    import numpy as np
    import slil.process.functions as fn
    import slil.common.data_configs as dc
    import slil.common.math as fm
    from copy import deepcopy
    from slil.process.qt_plotter import Scene
    from slil.common.data_configs import outputFolders

    modelInfo = dc.load(experiment)
    # only needed if missing some geometry (e.g. capitate)
    #projectFile = modelInfo['3_matic_file']
    #mi.open_project(projectFile)

    scene = Scene(modelInfo, useBackgroundPlotter = False)
    scene.loadGeometry()
    scene.addMeshes()

    modelCache = fn.getModelCache(modelInfo)
    framePoints = modelCache['markers']
    initTransMarkers = modelCache['initialTmatRadiusMarker']

    adjustTransMarkers = fn.getOptimizationResultsAsTransMat(modelInfo)

    modelInfo['currentModel'] = 'cut'
    adjustedTransMarkers = np.dot(adjustTransMarkers['cut'], initTransMarkers['cut'])
    t_World_radius_cut = fn.getMarkerGroupTranformationMat(modelInfo, modelInfo['names']['radius'], framePoints['cut'])
    newCutPoints = fm.transformPoints(np.array(framePoints['cut']), t_World_radius_cut, adjustedTransMarkers)
    # this line produces the same as the above three lines, only if the c3d has not changed since model generation
    #newCutPoints = fm.transformPoints(np.array(framePoints['cut']), np.eye(4), adjustTransMarkers['cut'])
    markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static_new'
    scene.addPoints(newCutPoints, markerGroupName, color='red')

    #modelInfo['currentModel'] = 'normal'
    #markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static'
    #scene.addPoints(np.array(framePoints[modelInfo['currentModel']]), markerGroupName, color='green')

    modelInfo['currentModel'] = 'cut'
    markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static'
    #scene.addPoints(np.array(framePoints['cut']), markerGroupName, color='red')
    t_World_lunate_cut = fn.getMarkerGroupTranformationMat(modelInfo, modelInfo['names']['lunate'], newCutPoints)
    t_World_scaphoid_cut = fn.getMarkerGroupTranformationMat(modelInfo, modelInfo['names']['scaphoid'], newCutPoints)


    
    def excludeMarkersFromPointCloud(cloud, modelType, boneNames):
        inds = []
        for boneName in boneNames:
            inds += modelInfo['plateAssignment_' + modelType][modelInfo['names'][boneName]]
        return np.delete(cloud, inds, axis = 0)

    # align scaffold marker group to cut marker group using only radius and metacarp3 markers in each group
    # point cloud align with only two marker plates
    boneNames = ['lunate', 'scaphoid']
    framePointsScaffold_removed = excludeMarkersFromPointCloud(framePoints['scaffold'], 'scaffold', boneNames)
    boneNames = ['lunate', 'scaphoid']
    framePointsCut_removed = excludeMarkersFromPointCloud(newCutPoints, 'cut', boneNames)

    # show two cloud point groups
    #scene.addPoints(framePointsScaffold_removed, 'framePointsScaffold_removed', 'yellow')
    #scene.addPoints(framePointsCut_removed, 'framePointsCut_removed', 'purple')

    import slil.process.alignment_pointcloud as fna
    [result_error, tran] = fna.cloudPointAlign_v3(
        framePointsCut_removed, framePointsScaffold_removed)
    tran = fm.inverseTransformationMat(tran)

    print('Alignment Error is {} for experiment {}'.format(result_error, modelInfo['experimentID'])) 
    #if result_error > 5.0: # bad then try using only radius to align
    #    print('Alignment Error is greater than 5.0 ({}) for experiment {}, using only radius.'.format(result_error, modelInfo['experimentID']))
    #    boneNames = ['lunate', 'scaphoid', 'metacarp3']
    #    framePointsScaffold_removed = excludeMarkersFromPointCloud(framePoints['scaffold'], 'scaffold', boneNames)
    #    boneNames = ['lunate', 'scaphoid', 'metacarp3']
    #    framePointsCut_removed = excludeMarkersFromPointCloud(newCutPoints, 'cut', boneNames)
    #    
    #    [result_error, tran] = fna.cloudPointAlign_v3(
    #        framePointsCut_removed, framePointsScaffold_removed)


    newScaffoldPoints = np.array(deepcopy(framePoints['scaffold']))
    newScaffoldPoints = fm.transformPoints(newScaffoldPoints, np.eye(4), tran)
    scene.addPoints(newScaffoldPoints, 'newScaffoldPoints', 'blue')



    modelInfo['currentModel'] = 'scaffold'
    markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static'
    #scene.addPoints(np.array(framePoints[modelInfo['currentModel']]), markerGroupName, color='blue')
    t_World_lunate_sca = fn.getMarkerGroupTranformationMat(modelInfo, modelInfo['names']['lunate'], newScaffoldPoints)
    t_World_scaphoid_sca = fn.getMarkerGroupTranformationMat(modelInfo, modelInfo['names']['scaphoid'], newScaffoldPoints)



    pdLun = deepcopy(scene.bonePolyData[modelInfo['names']['lunate']])
    fm.transformPoints_byRef(pdLun.points, t_World_lunate_cut, t_World_lunate_sca)
    scene.bonePolyData[modelInfo['names']['lunate']+'_moved'] = pdLun

    pdSca = deepcopy(scene.bonePolyData[modelInfo['names']['scaphoid']])
    fm.transformPoints_byRef(pdSca.points, t_World_scaphoid_cut, t_World_scaphoid_sca)
    scene.bonePolyData[modelInfo['names']['scaphoid']+'_moved'] = pdSca

    scene.addMeshes()

    scene.setOpacity(modelInfo['names']['radius'], 0.5)
    scene.setOpacity(modelInfo['names']['metacarp3'], 0.5)
    capName = [x for x in modelInfo['otherHandBones'] if 'cap' in x][0]
    scene.setOpacity(capName, 0.5)


    scene.setOpacity(modelInfo['names']['lunate'], 0.2)
    scene.setOpacity(modelInfo['names']['scaphoid'], 0.2)

    #scene.plotter.remove_actor(scene.actors['placedScaffold']) # Redacted
    #scene.plotter.remove_actor(scene.actors[modelInfo['names']['lunate']])
    #scene.plotter.remove_actor(scene.actors[modelInfo['names']['scaphoid']])

    filename = outputFolders()['graphics'] + '\\videos\\' + modelInfo['experimentID'] + '_cut2sacffold.mp4'
    scene.plotter.open_movie(filename)
    # Update scalars on each frame
    r = 360 + 180
    scene.plotter.camera_position = 'xz'
    scene.plotter.camera.roll = -90
    scene.plotter.camera.azimuth = -180
    scene.plotter.write_frame()
    for i in range(r):
        scene.plotter.camera.azimuth += 1
        scene.plotter.update()
        #scene.plotter.show()
        #scene.plotter.view_vector += (0.0, 0.1, 0.0)
        #scene.plotter.view_vector([-1 * np.pi, -1 * np.pi + (2 * np.pi*i/r), 1 * np.pi - (2 * np.pi*i/r)])
        #scene.plotter.add_text(f"Iteration: {i}", name='time-label')
        scene.plotter.write_frame()

    scene.plotter.close() # Be sure to close the scene.plotter when finished
