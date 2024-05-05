import numpy as np
import slil.common.math as fm

def visualizeMarkerPins(mi, modelInfo, framePointsRaw, specificBone = ''):
    # need to have markers outside of a marker sphere group
    import slil.process.functions as fn
    # realtive to markers, where are all the possible pins?

    # TODO: Measure the pins in the lab last used.
    pinLength = 45 # 5mm of 45mm is in the bone

    markerPinDepth = 18

    # location of LED emission
    # relative to center of front pin hole at base/bottom
    markerLef = [-18.56, 43.67, 3.61]
    markerRig = [20.39, 43.67, 3.61]
    markerMid = [0.92, 27.11, 7.11]
    markerMid = np.array([-27.11, 0.92, -7.11])

    # back pin relative to front pin
    pinBack = [-1.84, 0.0, 1.84]
    pinBack = np.array([0.0, -1.84, -1.84])

    # should take average of a couple frame to get a more accurate position
    # of all LEDs. They will likely not be the same (relative to eachother)
    # as those define above

    def addPins(TMrad, markerMid, pinBackOffset):
        m1 = np.dot(TMrad[:3, :3], markerMid* -1.0) + TMrad[:3, 3].T
        #mi.create_point(m1)
        v1 = TMrad[:3, 0]
        pinEnd1 = m1 + v1 * -1.0 * markerPinDepth
        pinEnd2 = pinEnd1 + v1 * pinLength
        mi.create_line(pinEnd1, pinEnd2)

        m1 = np.dot(TMrad[:3, :3], (markerMid + pinBackOffset)* -1.0) + TMrad[:3, 3].T
        #mi.create_point(m1)
        v1 = TMrad[:3, 0]
        pinEnd1 = m1 + v1 * -1.0 * markerPinDepth
        pinEnd2 = pinEnd1 + v1 * pinLength
        mi.create_line(pinEnd1, pinEnd2)
    configs = [
        [1,1,1],
        [1,-1,1],
        [1,1,-1],
        [1,-1,-1]
        ]

    #[lunateName, scaphoidName, radiusName, metacarp3Name] = fn.boneModelNames(modelInfo)
    #TMrad = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, framePointsRaw)
    if specificBone == '':
        for boneName in fn.boneModelNames(modelInfo):
            TMrad = fn.getMarkerGroupTranformationMat(modelInfo, boneName, framePointsRaw)
            for i in configs:
                a = markerMid * i
                b = pinBack * i
                addPins(TMrad, a, b)
    else:
        TMrad = fn.getMarkerGroupTranformationMat(modelInfo, specificBone, framePointsRaw)
        for i in configs:
            a = markerMid * i
            b = pinBack * i
            addPins(TMrad, a, b)

# Generate all possible marker locations given pins.

# find pins with names like pin_lunate1, pin_lunate2, and pin_scaphoid1
def genMarkersPos_old(modelInfo, pins, coms, boneName, isMirroredPlate, isOtherPinOrder):
    # The following are for version 2 of marker plates with 2 possible configurations
    # Each pair of pins should have four possible marker orientations
    # All measures in millimeters

    # TODO: Measure the pins in the lab last used.
    pinLength = 45 # 5mm of 45mm is in the bone

    markerPinDepth = 18

    # location of LED emission
    # relative to center of front pin hole at base/bottom
    markerLef = [-18.56, 43.67, 3.61]
    markerRig = [20.39, 43.67, 3.61]
    markerMid = [0.92, 27.11, 7.11]

    # back pin relative to front pin
    pinBack = [-1.84, 0.0, 1.84]
    # angle relative to SolidWords coordinate system
    # between blue and black line (see SolidWorks screenshots)
    angleBetween = np.arctan(pinBack[2]/pinBack[0])

    pinOrder = [1, 2]

    if (isMirroredPlate):
        markerLef[0] = markerLef[0] * -1.0
        markerRig[0] = markerRig[0] * -1.0
        markerMid[0] = markerMid[0] * -1.0
    if (isOtherPinOrder):
        pinOrder = [2, 1]

    pin1 = pins['pin_' + boneName + str(pinOrder[0])]
    #pin1 = mi.find_line('pin_' + boneName + str(pinOrder[0]))
    boneCOM = coms[modelInfo['names'][boneName]]
    #boneCOM = mi.compute_center_of_gravity(
    #    mi.find_part(modelInfo['names'][boneName])
    #    , method ='Based on mesh')
    # should always be point2 but check anyway
    if (fm.calcDist(boneCOM, pin1['point1']) > fm.calcDist(boneCOM, pin1['point2'])):
        pinFrontTop = np.array(pin1['point1'])
        pinFrontVec = np.array(pin1['direction'])
    else:
        pinFrontTop = np.array(pin1['point2'])
        pinFrontVec = np.array(pin1['direction']) * -1.0

    markerPlateCentreOfFrontHole = pinFrontTop + (pinFrontVec * markerPinDepth)

    pin2 = pins['pin_' + boneName + str(pinOrder[1])]
    #pin2 = mi.find_line('pin_' + boneName + str(pinOrder[1]))
    pinBackTop = np.array(pin2['point2'])
    pinBackVec = pinFrontVec
    pinBackVec = np.array(pin2['direction']) * -1.0

    # find difference along x axis (in frame of reference of front Vec).
    r3 = fm.rotation_matrix_from_vectors2((1.0,0.0,0.0), -1.0 *pinFrontVec)
    a0 = fm.rotateVector(pinBackTop - pinFrontTop,r3)[0]

    # find point along pinBackTop which is same x distance as pinFrontTop (in frame of reference of front Vec)
    aa = pinBackTop+(-1.0 *pinFrontVec*a0)
    #mi.create_point(aa)
    a00 = pinFrontTop-aa
    a00=fm.normalizeVector(a00)
    #mi.create_line(pinFrontTop, aa)

    [v1, v2, v3] = fm.create3DAxis(pinFrontVec, a00)
    v1 = v1 * -1.0
    v2 = v2 * -1.0
    [vX, vY, vZ] = [v3, v1, v2]

    #vecP1toP2 = fm.normalizeVector(pinBackTop - pinFrontTop)
    #[v1, v2, v3] = fm.create3DAxis(pinFrontVec, vecP1toP2)
    #
    ## make sure they're in the correct direction
    ## one axis should be pointing to the other pin
    #v1 = v1 * -1.0
    #v3 = v3 * -1.0 # TODO: check with other configs
    ##if (fm.angleBetweenVectors(vecP1toP2, v2) > np.pi/2):
    ##    v3 = v3 * -1.0
    ##    v2 = v2 * -1.0 # TODO: find a way to check right handed axis
    #[vX, vY, vZ] = [v3, v1, v2]

    rot = np.array([vX, vY, vZ]).T
    #import slil.process.functions as fn
    #fn.makeAxis(pinFrontTop, vX, vY, vZ)
    
    # 'angleBetween' should be used here, but the angle is 45 degrees
    if (isMirroredPlate):
        rotXtoRotatedZ = fm.rotation_matrix_from_vectors((1.0,0.0,0.0), fm.normalizeVector((-0.5,0.0,0.5)))
    else:
        rotXtoRotatedZ = fm.rotation_matrix_from_vectors((1.0,0.0,0.0), fm.normalizeVector((0.5,0.0,0.5)))
    rotXtoNewV3 = np.dot(rot, rotXtoRotatedZ)

    pinFrontMid = pinFrontTop + (markerPinDepth*pinFrontVec)
    #fn.makeAxisR(pinFrontMid, rotXtoNewV3)
    m1 = pinFrontMid + fm.rotateVector(markerLef, rotXtoNewV3)
    m2 = pinFrontMid + fm.rotateVector(markerRig, rotXtoNewV3)
    m3 = pinFrontMid + fm.rotateVector(markerMid, rotXtoNewV3)
    #mi.create_point(m1)
    #mi.create_point(m2)
    #mi.create_point(m3)

    # close enough
    #fm.calcDist(m1, pinFrontMid) # = 47.58754669028443
    #fm.calcMag(markerLef) # = 47.58754669028442
    
    if (isMirroredPlate):
        return [m2, m1, m3]
    return [m1, m2, m3]

# find pins with names like pin_lunate1, pin_lunate2, and pin_scaphoid1
def genMarkersPos(modelInfo, pins, coms, boneName):
    # The following are for version 2 of marker plates with 2 possible configurations
    # Each pair of pins should have four possible marker orientations
    # All measures in millimeters

    import slil.process.align_main as pam

    boneCOM = coms[modelInfo['names'][boneName]]
    #boneCOM = mi.compute_center_of_gravity(
    #    mi.find_part(modelInfo['names'][boneName])
    #    , method ='Based on mesh')
    pin1 = pins['pin_' + boneName + str(1)]
    pin2 = pins['pin_' + boneName + str(2)]
    # should always be second case, but check anyway
    if (fm.calcDist(boneCOM, pin1['point1']) > fm.calcDist(boneCOM, pin1['point2'])):
        pin1Top = np.array(pin1['point1'])
        pin1Bottom = np.array(pin1['point2'])
        pin2Top = np.array(pin2['point1'])
        pin2Bottom = np.array(pin2['point2'])
    else:
        pin1Top = np.array(pin1['point2'])
        pin1Bottom = np.array(pin1['point1'])
        pin2Top = np.array(pin2['point2'])
        pin2Bottom = np.array(pin2['point1'])

    pp = np.empty((3, 3))
    pp[0, :] = pin1Top
    pp[1, :] = pin1Bottom
    pp[2, :] = pin2Top
    return pam.generateMarkerPointsFromPins(pp)

def generatePossibleMarkerLocationsFromPins(modelInfo, pins, coms):
    # order: left, right, middle
    #markersPos= [
    #    [0.0, 0.0, 0.0],
    #    [0.0, 0.0, 0.0],
    #    [0.0, 0.0, 0.0]]

    possibleMarkers = []
    # only ['lunate', 'scaphoid', 'radius', 'metacarp3']
    boneWithMarkerSets = list(modelInfo['names'].keys())[:4]
    for boneName in boneWithMarkerSets:
        a = genMarkersPos(modelInfo, pins, coms, boneName).tolist()
        a.append([np.nan, np.nan, np.nan])
        #a = [
        #    genMarkersPos_old(modelInfo, pins, coms, boneName,
        #        isMirroredPlate = False,
        #        isOtherPinOrder = False),
        #    genMarkersPos_old(modelInfo, pins, coms, boneName,
        #        isMirroredPlate = False,
        #        isOtherPinOrder = True),
        #    genMarkersPos_old(modelInfo, pins, coms, boneName,
        #        isMirroredPlate = True,
        #        isOtherPinOrder = True),
        #    genMarkersPos_old(modelInfo, pins, coms, boneName,
        #        isMirroredPlate = True,
        #        isOtherPinOrder = False),
        #    [np.nan, np.nan, np.nan], # for no marker
        #]
        possibleMarkers.append(a)
    return possibleMarkers, boneWithMarkerSets

def generateCombinations(modelInfo, numCombs, possibleMarkers):
    # all combinations of marker plate configurations
    combinations = np.empty(shape=(numCombs,4), dtype=int)
    c = 0 
    # this would need to be changed if more marker sets are used
    for i in range(0,4):
        for ii in range(0,4):
            for iii in range(0,4):
                for iiii in range(0,4):
                    comb = [int(i), int(ii), int(iii), int(iiii)]
                    combinations[c,:] = comb
                    c = c + 1
    indOfNaN = 4
    for ii in range(0,4):
        for iii in range(0,4):
            for iiii in range(0,4):
                comb = [indOfNaN, int(ii), int(iii), int(iiii)]
                combinations[c,:] = comb
                c = c + 1
    for ii in range(0,4):
        for iii in range(0,4):
            for iiii in range(0,4):
                comb = [int(ii), indOfNaN, int(iii), int(iiii)]
                combinations[c,:] = comb
                c = c + 1
    for ii in range(0,4):
        for iii in range(0,4):
            for iiii in range(0,4):
                comb = [int(ii), int(iii), indOfNaN, int(iiii)]
                combinations[c,:] = comb
                c = c + 1
    for ii in range(0,4):
        for iii in range(0,4):
            for iiii in range(0,4):
                comb = [int(ii), int(iii), int(iiii), indOfNaN]
                combinations[c,:] = comb
                c = c + 1

    possibleMarkers2 = np.empty(shape=(numCombs,12,3))
    possibleMarkersTemp = np.empty(shape=(12,3))
    for ind, comb in enumerate(combinations):
        possibleMarkersTemp[0:3,:] = possibleMarkers[0][comb[0]]
        possibleMarkersTemp[3:6,:] = possibleMarkers[1][comb[1]]
        possibleMarkersTemp[6:9,:] = possibleMarkers[2][comb[2]]
        possibleMarkersTemp[9:12,:] = possibleMarkers[3][comb[3]]
        possibleMarkers2[ind,:,:] = possibleMarkersTemp
    return possibleMarkers2, combinations


def getPoints(mi, modelInfo):
    if (not mi.find_point('marker0')):
        return
    points = []
    #numPoints = 0
    for i, p in enumerate(modelInfo['plateAssignment_' + modelInfo['currentModel']]):
        for j in range(len(modelInfo['plateAssignment_' + modelInfo['currentModel']][p])):
            #numPoints += 1
            id = modelInfo['plateAssignment_' + modelInfo['currentModel']][p][j]
            points.append(np.array(mi.find_point('marker' + str(id)).coordinates))
    #for i in range(numPoints):
    #    points.append(np.array(mi.find_point('marker' + str(i)).coordinates))

    return points

def getPointsOrdered(pointsUnordered, orderedBy):
    points = []
    for i, p in enumerate(orderedBy):
        for j in range(len(orderedBy[p])):
            ind = orderedBy[p][j]
            points.append(pointsUnordered[ind])
    return points

def draw_registration_result(source, target, transformation):
    import open3d as o3d
    import copy
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0]) # yellow
    target_temp.paint_uniform_color([0, 0.651, 0.929]) # blue
    source_temp.transform(transformation)
    #o3d.visualization.draw_geometries([source_temp, target_temp],
    #                                zoom=0.4459,
    #                                front=[0.9288, -0.2951, -0.2242],
    #                                lookat=[1.6784, 2.0612, 1.4451],
    #                                up=[-0.3402, -0.9189, -0.1996])
    o3d.visualization.draw_geometries([source_temp, target_temp])

def cloudPointAlign_v1(p1, p2, showInfo=False):
    import open3d as o3d
    import copy
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(
        copy.deepcopy(p2))
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(
        copy.deepcopy(p1))
    #print("Initial alignment")
    #evaluation = o3d.pipelines.registration.evaluate_registration(
    #    source, target, threshold, trans_init)
    #print(evaluation)

    #draw_registration_result(source, target, fm.createMatrixI())

    def preprocess_point_cloud(pcd, voxel_size):
        #print(":: Downsample with a voxel size %.3f." % voxel_size)
        #pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down = pcd
        radius_normal = voxel_size * 2
        if (showInfo):
            print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        if (showInfo):
            print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh
    def prepare_dataset(voxel_size, source, target):
        #print(":: Load two point clouds and disturb initial pose.")
        #demo_icp_pcds = o3d.data.DemoICPPointClouds()
        #source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
        #target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
        trans_init = np.identity(4)
        source.transform(trans_init)
        if (showInfo):
            draw_registration_result(source, target, trans_init)

        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh
    voxel_size = 100.0  # for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        voxel_size, source, target)
    def execute_global_registration(source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        if (showInfo):
            print(":: RANSAC registration on downsampled point clouds.")
            print("   Since the downsampling voxel size is %.3f," % voxel_size)
            print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            #], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    i = 0
    while (result_ransac.inlier_rmse == 0.0):
        i = i + 1
        result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
        print("Inlier RMSE: {}".format(result_ransac.inlier_rmse))
        if (i > 100):
            print("Solution not found!")
            break
    #result_ransac = execute_global_registration(source, target,
    #                                            source_fpfh, target_fpfh,
    #                                            voxel_size)
    if (showInfo):
            print(result_ransac)
    #draw_registration_result(source_down, target_down, result_ransac.transformation)
    if (showInfo):
            draw_registration_result(source_down, target_down, result_ransac.transformation)
    final = source.transform(result_ransac.transformation)
    return [result_ransac, np.asarray(target.points), np.asarray(final.points), result_ransac.transformation]

def cloudPointAlign_v2(p1, p2, showInfo=False):
    import open3d as o3d
    import copy
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(
        copy.deepcopy(p1))
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(
        copy.deepcopy(p2))

    if (showInfo):
        draw_registration_result(source, target, np.identity(4))

    #result = calcRigidRegistration(p1, p2)
    #trans = np.identity(4)
    #trans[:3,:3] = result[0]
    #trans[:3, 3] = result[1].T[0]

    #if (showInfo):
    #    draw_registration_result(source, target, trans)
    lenPoints = len(p1)
    corr = np.zeros((lenPoints, 2))
    corr[:, 0] = range(0,lenPoints)
    corr[:, 1] = range(0,lenPoints)
    # estimate rough transformation using correspondences
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))
    if (showInfo):
        draw_registration_result(source, target, trans_init)

    #trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
    #                         [-0.139, 0.967, -0.215, 0.7],
    #                         [0.487, 0.255, 0.835, -1.4],
    #                         [0.0, 0.0, 0.0, 1.0]])
    #trans_init= result_ransac.transformation
    if (showInfo):
        print("Input transformation is:")
        print(trans_init)
        print("Apply point-to-point ICP")
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria()
    criteria.relative_rmse = 0.001
    threshold = 0.000000002
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())#,
    #    criteria)
    if (showInfo):
        print(reg_p2p)
        print("Inlier RMSE: {}".format(reg_p2p.inlier_rmse))
        print("Transformation is:")
        print(reg_p2p.transformation)
        draw_registration_result(source, target, reg_p2p.transformation)
    
    reg = o3d.pipelines.registration.evaluate_registration(
        source, target, 1000, transformation=reg_p2p.transformation)
    if (showInfo):
        print(reg)
    final = source.transform(reg_p2p.transformation)
    return [reg, np.asarray(target.points), np.asarray(final.points), reg_p2p.transformation]

def cloudPointAlign_v3(p1, p2):
    from copy import deepcopy

    target = deepcopy(p1)
    source = deepcopy(p2)
    l1 = source.shape[0]
    l2 = source.shape[1]

    def cost(x0, argsExtra):
        x, y, z, rx, ry, rz = x0
        p0, p1Init = argsExtra
        p1 = deepcopy(p1Init)

        t_adjustment = fm.createTransformationMatFromPosAndEuler(x, y, z, rx, ry, rz)
        p1 = fm.transformPoints_1(p1, t_adjustment)

        error = 0.0
        for i in range(l1):
            for ii in range(l2):
                error += np.power(p1[i, ii] - p0[i, ii], 2)

        #error = (
        #        np.power(p1[0, 0] - p0[0, 0], 2) + 
        #        np.power(p1[0, 1] - p0[0, 1], 2) + 
        #        np.power(p1[0, 2] - p0[0, 2], 2) +
        #        np.power(p1[1, 0] - p0[1, 0], 2) + 
        #        np.power(p1[1, 1] - p0[1, 1], 2) + 
        #        np.power(p1[1, 2] - p0[1, 2], 2) +
        #        np.power(p1[2, 0] - p0[2, 0], 2) + 
        #        np.power(p1[2, 1] - p0[2, 1], 2) + 
        #        np.power(p1[2, 2] - p0[2, 2], 2))

        error += 1.0
        return error

    #source = np.array([
    #    [0.0, 2.0, 3.0],
    #    [1.0, 2.0, 3.0],
    #    [2.0, 2.0, 3.0],
    #    [4.0, 4.0, 3.0],
    #    [5.0, 4.0, 3.0],
    #    [6.0, 4.0, 3.0],
    #    ])
    #l1 = source.shape[0]
    #l2 = source.shape[0]

    numMarkers = int(l1 / 3)

    markCombMap = np.array([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [2, 0, 1],
        ])
    eachMarkerCombs = markCombMap.shape[0] # 3^2 = 6

    ms1_all = np.empty((numMarkers, eachMarkerCombs, 3, source.shape[1]))
    
    for i in range(numMarkers):
        ind1 = i * 3
        for ii in range(eachMarkerCombs):
            ms1_all[i, ii, 0, :] = source[ind1 + markCombMap[ii, 0], :]
            ms1_all[i, ii, 1, :] = source[ind1 + markCombMap[ii, 1], :]
            ms1_all[i, ii, 2, :] = source[ind1 + markCombMap[ii, 2], :]
    
    #ms1_all2 = np.empty((np.power(eachMarkerCombs, numMarkers), ms1.shape[0], ms1.shape[1]))
    #for i0 in range(numMarkers):
    #    for ii0 in range(eachMarkerCombs):
    #            for i in range(eachMarkerCombs):
    #                for iix in range(numMarkers):
    #                    ms1_all2[i0 * ii0, iix*3:iix*3+3, :] = ms1_all[iix, i, :3, :]
    #                #ms1_all2[i0 * ii0, ii*3:ii*3+3, :] = ms1_all[ii, i, 0:3, :]
    #                #ms1_all2[i + ii, 3:6, :] = ms1_all[1, i, :3, :]

    if numMarkers == 1:
        sourceCombinations = ms1_all[0, :, :, :]
    else:
        if numMarkers == 2:
            l = []
            #for i0 in range(numMarkers):
            for ii0 in range(eachMarkerCombs):
                for i in range(numMarkers-1):
                    for ii in range(eachMarkerCombs):
                        l.append(list(ms1_all[0, ii0, :3, :]) + list(ms1_all[i+1, ii, :3, :]))
            sourceCombinations = np.array(l)
        else:
            print('Function has no way to use more than 2 markers!')
            return [0.0, np.eye(4)]

    foundTransforms = []
    foundErrors = []
    from scipy.optimize import minimize
    for i in range(sourceCombinations.shape[0]):
        source = sourceCombinations[i]

        argsExtra = [target, source]
        
        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = minimize( \
            fun = cost, \
            x0 = x0, \
            args = (argsExtra, ), \
            method='L-BFGS-B', \
            options= {
                #'disp': True,
                'maxcor': 400,
                'maxiter': 30000,
                },
            #callback = calllbackSave, \
            #maxiter = 10
            )
        error = cost(result.x, argsExtra) - 1.0
        transformation = fm.createTransformationMatFromPosAndEuler(result.x[0], result.x[1], result.x[2], result.x[3], result.x[4], result.x[5])
        foundTransforms.append(transformation)
        foundErrors.append(error)
    minErrorInd = np.argmin(np.array(foundErrors))
    error = foundErrors[minErrorInd]
    transformation = fm.inverseTransformationMat(foundTransforms[minErrorInd])

    return [error, transformation, foundErrors, foundTransforms]
