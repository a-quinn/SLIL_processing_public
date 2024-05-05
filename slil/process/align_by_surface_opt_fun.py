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

# vtk doesn't work on hpc linux, I think because of no xvfb (frame buffer)
#from vtk import vtkIntersectionPolyDataFilter as IPDF
#from vtk import vtkPolyDataIntersectionFilter as PDIF
#from vtk import vtkBooleanOperationPolyDataFilter as BOPDF
#from vtk import vtkMassProperties as MP
#from vtkmodules.vtkFiltersGeneral import vtkBooleanOperationPolyDataFilter as BOPDF
#from vtkmodules.vtkFiltersCore import vtkMassProperties as MP

import trimesh
import fcl

def makeMesh(verts, tris):
    m = fcl.BVHModel()
    m.beginModel(len(verts), len(tris))
    m.addSubModel(verts, tris)
    m.endModel()
    return m

def intersectionError(display, surf1, surf2):
    ## could do with less creation of new objects...
    #mesh1 = trimesh.base.Trimesh(surf1.points, surf1.faces.reshape(-1, 4)[:,1:4])
    #mesh2 = trimesh.base.Trimesh(surf2.points, surf2.faces.reshape(-1, 4)[:,1:4])
    #coll = trimesh.collision.CollisionManager()
    #coll.add_object('bone1', mesh1)
    #coll.add_object('bone2', mesh2)
    ##is_collision, names, contacts = coll.in_collision_single(mesh1)
    #distance = coll.min_distance_single(mesh1)
    #nIntersections = distance
    #print(nIntersections)
    ##collD = trimesh.collision.ContactData(names, contacts)

    start_time = timeit.default_timer()
    m1 = makeMesh(surf1.points, surf1.faces.reshape(-1, 4)[:,1:4])
    print('collide1: {}'.format(timeit.default_timer() - start_time))
    start_time = timeit.default_timer()
    m2 = makeMesh(surf2.points, surf2.faces.reshape(-1, 4)[:,1:4])
    print('collide2: {}'.format(timeit.default_timer() - start_time))
    start_time = timeit.default_timer()

    t1 = fcl.Transform()
    o1 = fcl.CollisionObject(m1, t1)

    t2 = fcl.Transform()
    o2 = fcl.CollisionObject(m2, t2)

    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()
    
    nIntersections = 0
    if fcl.collide(o1, o2, request, result) == 1:
        print('Number of contacts = {}'.format(len(result.contacts)))
        nIntersections = 10.0 * np.abs(result.contacts[0].penetration_depth)
        for i in result.contacts:
            print('Depth = {}'.format(i.penetration_depth))
    print('collide6: {}'.format(timeit.default_timer() - start_time))

    #request = fcl.DistanceRequest()
    #request.enable_nearest_points = True
    #result = fcl.DistanceResult()
    #ret = fcl.distance(o1, o2, request, result)
    #result.nearest_points[0] # one from each surface
    #result.nearest_points[1]

    # another option: https://stackoverflow.com/questions/56211310/how-can-i-detect-a-intersection-between-a-pointcloud-and-a-triangle-mesh
    
    #intfilter = IPDF()
    ##intfilter = PDIF() # only in newer vtk versions
    #intfilter.SetInputDataObject(0, surf1)
    #intfilter.SetInputDataObject(1, surf2)
    #intfilter.SetComputeIntersectionPointArray(True)
    #intfilter.SetCheckInput(False)
    #intfilter.SetSplitFirstOutput(False) #
    #intfilter.SetSplitSecondOutput(False) #
    #intfilter.Update()
    #nIntersections = intfilter.GetNumberOfIntersectionPoints()

    # vtkBooleanOperationPolyDataFilter uses vtkIntersectionPolyDataFilter internally anyway
    #intBoolOp = BOPDF()
    #intBoolOp.DebugOn()
    #intBoolOp.SetOperationToIntersection()
    #intBoolOp.SetInputDataObject(0, surf1)
    #intBoolOp.SetInputDataObject(1, surf2)
    #intBoolOp.Update()
    #massIntersection = MP()
    #massIntersection.SetInputData(intBoolOp.GetOutput())
    #nIntersections = massIntersection.GetVolume()

    #intersection, _, _ = surf1.intersection(surf2, split_first=False, split_second=False)
    #nIntersections = intersection.n_cells
    if nIntersections == 0:
        if display:
            print('No intersection.')
        error1intersection = 0
    else:
        if display:
            print('Intersection = {}'.format(nIntersections))
        error1intersection = nIntersections
    return error1intersection


def intersectionError2(display, m1, m2, t):

    #start_time = timeit.default_timer()
    t1 = fcl.Transform()
    o1 = fcl.CollisionObject(m1, t1)

    t2 = fcl.Transform(t[:3, :3], t[:3, 3].T)
    o2 = fcl.CollisionObject(m2, t2)

    request = fcl.CollisionRequest(enable_contact=True)
    result = fcl.CollisionResult()
    
    nIntersections = 0
    if fcl.collide(o1, o2, request, result) == 1:
        nIntersections = 100.0 * np.abs(result.contacts[0].penetration_depth)
        if display:
            print('Number of contacts = {}'.format(len(result.contacts)))
            for i in result.contacts:
                print('Intersection Depth = {}'.format(i.penetration_depth))
    else:
        if display:
            print('No intersection.')
    #print('collide6: {}'.format(timeit.default_timer() - start_time))

    #request = fcl.DistanceRequest()
    #request.enable_nearest_points = True
    #result = fcl.DistanceResult()
    #ret = fcl.distance(o1, o2, request, result)
    #result.nearest_points[0] # one from each surface
    #result.nearest_points[1]

    return nIntersections

#from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
def distanceError(surf1, surf2, maxDist):
    dists = cdist(surf2.points, surf1.points)
    # for greater number of points
    #dists = pairwise_distances(surfRad.points, surfSca.points, metric = 'euclidean', n_jobs = -1)
    pointsBetweenMag1 = dists[np.where(dists < maxDist)]

    if len(pointsBetweenMag1) != 0:
        error = np.sqrt(np.mean((pointsBetweenMag1)**2)) # rmse
        error = error + 10000/len(pointsBetweenMag1)
        return error
        #return np.nanmean(pointsBetweenMag1)
    else:
        return 100.0

def transformBone(modelInfo, radiusName, boneName, framePointsRaw, t_WL1, t_O2Rad, surfBone, t_P1L1):
    t_O2RadFrame2 = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, framePointsRaw)
    framePointsRaw = fm.transformPoints(framePointsRaw, t_O2RadFrame2, t_O2Rad)
    t_WP2 = fn.getMarkerGroupTranformationMat(modelInfo, boneName, framePointsRaw)
    t_WL2 = np.dot(t_WP2, t_P1L1)
    surfBone_2 = deepcopy(surfBone)
    fm.transformPoints_byRef(surfBone_2.points, t_WL1, t_WL2)
    return surfBone_2

def optFun(x, args):
    x, y, z, rx, ry, rz = x
    #rx, ry, rz = 0.0, 0.0, 0.0
    #x, y, z = x
    #start_time = timeit.default_timer()
    #modelInfo, referenceBone, checkBones, t_WL1, M_O2Rad_orig, framePointsRaw, display, displayViz = args
    modelInfo = args['modelInfo']
    referenceBone = args['referenceBone']
    checkBones = args['checkBones']
    t_World_RadiusMarkers_FromModel = args['M_O2Rad_orig']
    framePointsRaw_Original = args['framesPointsRaw']
    framePointsRaw = deepcopy(framePointsRaw_Original) # stop potential changes for future optimizations
    display = args['display']
    displayViz = args['displayViz']

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
        
    outputSurfaces = []
    errorIntersection = []
    errorDistance = []
    for bone in checkBones:
        boneSurf = bone['surface']
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
                #surfBone_2 = transformBone(modelInfo, radiusName, boneName, frame, \
                #    t_WL1, t_O2Rad, boneSurf, t_P1L1)
                #
                t_World_BoneMarkers_FrameI = fn.getMarkerGroupTranformationMat(modelInfo, boneName, frame)
                t_World_ModelBoneCOM_FrameI = np.dot(t_World_BoneMarkers_FrameI, t_BoneMarkers_ModelBoneCOM)
                surfBone_2 = deepcopy(boneSurf)
                #fm.transformPoints_byRef(surfBone_2.points, t_WL1, t_WL2)
                fm.transformPoints_byRef(surfBone_2.points, t_World_ModelBoneCOM, t_World_ModelBoneCOM_FrameI)

                # This should never change if radius never moves throughout all frames, but it may.
                t_World_RadMarkers_FrameI_CouldHaveChanged = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, frame)
                t_World_ModelRadCOM_FrameI = np.dot(t_World_RadMarkers_FrameI_CouldHaveChanged, t_RadMarkers_ModelRadCOM)
                surfRad_2 = deepcopy(surfRad_Original)
                fm.transformPoints_byRef(surfRad_2.points, t_World_ModelRadCOM_Original, t_World_ModelRadCOM_FrameI)
                #
                #M2 = fm.inverseTransformationMat(t_WL1)
                #t_m = np.dot(t_WL2, M2)
                #m0 = makeMesh(surfRad.points, surfRad.faces.reshape(-1, 4)[:,1:4])
                #m1 = makeMesh(surfBone_2.points, surfBone_2.faces.reshape(-1, 4)[:,1:4])
                m0 = makeMesh(surfRad_2.points, np.reshape(surfRad_2.faces, (-1,4))[:,1:4])
                m1 = makeMesh(surfBone_2.points, np.reshape(surfBone_2.faces, (-1,4))[:,1:4])
                t_m = np.eye(4)
                error2intersection = intersectionError2(display, m0, m1, t_m)
                #coord = np.dot(t_m, vector)


                # Calculate error
                #error2intersection = intersectionError(display, surfBone_2, surfRad)
                #error2intersection = 0.0
                maxDist = 5.0 # mm
                error2distance = distanceError(surfBone_2, surfRad_Original, maxDist)
                
                errorIntersection.append(error2intersection)
                errorDistance.append(error2distance)
                if displayViz != None:
                    outputSurfaces.append(surfBone_2)
                    outputSurfaces.append(surfRad_2)

    #print('TimeIt calc error: {}'.format(timeit.default_timer() - start_time))

    error = 0
    if display:
        print('N  :    e = errorIntersection + errorDistance')
    #error += error1intersection + error1distance
    for i, errorD in enumerate(errorDistance):
        e = errorIntersection[i] + errorD
        if display:
            print('N{} : {:.3f} = {:.3f} + {:.3f}'.format(i, e, errorIntersection[i], errorD))
        error += e
    if display:
        print('errorTotal = {}'.format(error))

    if displayViz != None:
        surfs = []
        surfs.append(surfRad_Original)
        for bone in checkBones: # zero frame surfaces
            boneSurf = bone['surface']
            surfs.append(boneSurf)
        for surf in outputSurfaces:
            surfs.append(surf)

        if displayViz == True: # but not setup yet
            from pyvistaqt import BackgroundPlotter
            displayViz = BackgroundPlotter(window_size=(1000, 1000))
        for ind, surf in enumerate(surfs):
            displayViz.add_mesh(
                surf,
                name = 'surf_' + str(ind),
                scalars=np.arange(surf.n_faces),
                color='silver',
                specular=1.0,
                specular_power=10,
                opacity=1.0
            )
        for ind, frame in enumerate(framePointsRaw):
            displayViz.add_points(
                points = np.array(frame),
                name = 'frame_' + str(ind),
                render_points_as_spheres=True,
                point_size=20.0
                )

        displayViz.show_axes_all()
        displayViz.view_vector([0, 1, 0])
        displayViz.set_viewup([0, 0, -1])

        origin = (0.0, 0.0, 0.0) # for axies lines at origin
        displayViz.add_mesh(pv.Line(origin, (1.0, 0.0, 0.0)), color='red')
        displayViz.add_mesh(pv.Line(origin, (0.0, 1.0, 0.0)), color='green')
        displayViz.add_mesh(pv.Line(origin, (0.0, 0.0, 1.0)), color='blue')

        #displayViz.show()
        #displayViz.screenshot('screenshot_{:.4f}_.png'.format(error))

    return error