#%% Script for manipulating markers in 3-Matic
# Author: Alastair Quinn 2021
# These scripts work with the 3-Matic API Trimatic and Python Jupyter
import os
import sys

#from cv2 import CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE

#from numpy.lib.twodim_base import tri
import slil.mesh.interface as smi
mi = smi.MeshInterface(1)

import numpy as np

import slil.process.functions as fn
import slil.process.alignment_pointcloud as fna
import slil.common.data_configs as dc
import slil.common.math as fm
import slil.common.io as fio

#py -m pip install pyvista trimesh optimparallel pythreejs

#%% To allow re-importing modules without restarting jupyter
#%load_ext autoreload
#%autoreload 2
#%matplotlib qt
#%% Change settings here
import slil.common.data_configs as dc

experiments = [
    #'11525',
    #'11526',
    ##'11527', # bad, do not run
    #'11534',
    #'11535',
    '11536',
    #'11537',
    #'11538',
    #'11539'
    ] 

from slil.process import generate_model
#generate_model.generate(mi, experiments)


#%%


experiments = [
    #'11525',
    #'11526',
    #'11527', # bad, do not run
    #'11534',
    #'11535',
    '11536',
    #'11537',
    #'11538',
    #'11539'
    ]
modelInfo = dc.load(experiments[0])
modelTypes = [ 'normal', 'cut', 'scaffold']
modelInfo['currentModel'] = modelTypes[0]

trial = r'\normal_fe_40_40\log1'
print("Using trial: {}".format(trial))
fileToImport = modelInfo['dataOutputDir'] + trial + '.c3d'
frames = np.array(fio.readC3D(fileToImport))

from copy import deepcopy
def getPointsAtFrame(frames, N):
    return np.array(deepcopy(frames[N,:,0:3]))

frame = frames[1200,:]
frame = frames[5200,:]
frame = frames[1,:]
#frame = frames[23500]
newPointsRaw = []
newPointsRaw.extend(frame[:,0:3])
#del frames
def insertPoints(points):
    latestPoints = []
    for point in points:
        #point[1] = point[1]-1000 # just to move them closer to the center
        latestPoints.append(mi.create_point(point))
    fn.namePoints(latestPoints)
    fn.colourPoints(latestPoints, modelInfo)
    if not fn.checkMarkersValid(latestPoints):
        print("Warning! Invalid static marker points!")
    return latestPoints

#latestPoints = insertPoints(newPointsRaw)


def transformPoints_show(points, tmStart, tmEnd):
    M2 = fm.inverseTransformationMat(tmStart)
    for ind, i in enumerate(points):
        #pm = np.array(i.coordinates)
        #coord = np.dot(M2[:3, :3], pm + (M2[:3, 3].T)) # first move to origin
        #coord = np.dot(tmEnd[:3, :3], coord) + tmEnd[:3, 3].T # then move to new pos
        # now using 4-vectors (appending a 1 to the array)
        pm = np.array((i.coordinates[0], i.coordinates[1], i.coordinates[2], 1))
        coord = np.dot(M2, pm) # first move to origin
        coord = np.dot(tmEnd, coord)[:3] # then move to new pos
        nameTemp = i.name
        colourTemp = i.color
        mi.delete(i)
        points[ind] = mi.create_point(coord)
        points[ind].name = nameTemp
        points[ind].color = colourTemp
    fn.colourPoints(points, modelInfo)





#fn.deleteMarkers()
#fn.deleteMarkers()

modelCache = fn.getModelCache(modelInfo)
M_O2Rad_orig = modelCache['initialTmatRadiusMarker']['normal']

[lunateName, scaphoidName, radiusName, metacarp3Name] = fn.boneModelNames(modelInfo)

import pyvista as pv
# pip install pythreejs
import numpy as np
pv.set_jupyter_backend('pythreejs')

def getGeometry(modelInfo, boneName):
    geometry = fn.getGeometry(modelInfo, boneName)

    faces0 = geometry[0]
    vertices = geometry[1]
    vertices = np.array(vertices)
    facesTemp = np.array(faces0)
    faces = np.empty((facesTemp.shape[0], facesTemp.shape[1]+1), dtype=int)
    faces[:, 1:] = facesTemp
    faces[:, 0] = 3
    return pv.PolyData(vertices, faces)

surfLun = getGeometry(modelInfo, lunateName)
surfSca = getGeometry(modelInfo, scaphoidName)
surfRad_original = getGeometry(modelInfo, radiusName)


lunCOM = fm.calcCOM(surfLun.points, surfLun.faces.reshape(-1, 4), method = 'mesh')



#transformPoints_show(latestPoints, M_O2RadFrame, M_O2Rad)
#
#lunateName = modelInfo['names']['lunate']
#fna.visualizeMarkerPins(modelInfo, lunateName)


# find transform of lunate from one frame to another
lunateName = modelInfo['names']['lunate']

frame1PointsRaw = getPointsAtFrame(frames, 2)
frame2PointsRaw = getPointsAtFrame(frames, 5000)
frame3PointsRaw = getPointsAtFrame(frames, 10000)

# transform all so radius is in line with the 
M_O2Rad = deepcopy(M_O2Rad_orig)
M_O2RadFrame = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, frame1PointsRaw)
frame1PointsRaw = fm.transformPoints(frame1PointsRaw, M_O2RadFrame, M_O2Rad)
M_O2RadFrame = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, frame2PointsRaw)
frame2PointsRaw = fm.transformPoints(frame2PointsRaw, M_O2RadFrame, M_O2Rad)

#insertPoints(frame1PointsRaw)
#insertPoints(frame2PointsRaw)

# use homogeneous transformation matrices
# note: The multiplication of transformation matrices is associative,
# but not generally commutative
T_WP1 = fn.getMarkerGroupTranformationMat(modelInfo, lunateName, frame1PointsRaw)
T_WP2 = fn.getMarkerGroupTranformationMat(modelInfo, lunateName, frame2PointsRaw)

# M_1 lunate marker at frame 1, relative to world origin
# M_2 lunate marker at frame 2, relative to world origin

# make realteive to lunate COM
#partLun = mi.find_part(lunateName)
#partRad = mi.find_part(radiusName)
#lunCOM = np.array(mi.compute_center_of_gravity(partLun, method ='Based on mesh'))
# check if intersecting any bones
#inters = mi.create_intersection_curve(partLun, partRad, intersection_curve_in=2)
#if len(inters) == 0:
#    print('No intersection.')
#else:
#    print('Intersection')
#    mi.delete(inters) # actually leaves


T_WL1 = np.eye(4)
T_WL1[:3, 3] = lunCOM.T

T_L1W = fm.inverseTransformationMat(T_WL1)

T_L1P1 = np.dot(T_L1W, T_WP1)

T_L1P2 = np.dot(T_L1W, T_WP2) # lunate marker at frame 2, relative to lunate COM

def movePart(part, rotAbout, T):
    rot2 = fm.rotationMatrixToEulerAngles(T[:3, :3])
    mi.rotate(part, float(np.rad2deg(rot2[0])), rotAbout, (1,0,0))
    mi.rotate(part, float(np.rad2deg(rot2[1])), rotAbout, (0,1,0))
    mi.rotate(part, float(np.rad2deg(rot2[2])), rotAbout, (0,0,1))
    #rot2 = (float(np.rad2deg(rot2[0])), float(np.rad2deg(rot2[1])), float(np.rad2deg(rot2[2])))
    #mi.rotate_around_axes(part, rot2, rotAbout)
    t2 = T[:3, 3].T
    mi.translate(part, t2)


T_P1L1 = fm.inverseTransformationMat(T_L1P1)
#T_WL1 = np.dot(T_WP1, T_P1L1)
T_WL2 = np.dot(T_WP2, T_P1L1)


#movePart(partLun, lunCOM, T_L1W) # to world origin
#movePart(partLun, lunCOM, T_WL2)
#movePart(partLun, lunCOM, T_WL1)


def findPointsBetweenTwoBones(verticesA, verticesB, maxDist = 3.0):
    from scipy.spatial.distance import cdist
    dists = cdist(verticesA, verticesB)

    # should remove duplicates, one point of each should only have one other point
    #dists

    d0 = np.where(dists < maxDist)

    i0 = 0
    pointsBetween = []
    pointsBetweenMag = []
    for i in range(len(d0[0])):
        #i0 += 1
        if i0 == 0:
            p1 = verticesA[d0[0][i]]
            p2 = verticesB[d0[1][i]]
            #mi.create_line(p1, p2)
            v1 = np.array(p1) - np.array(p2)
            mag = fm.calcMag(v1)
            v1 = fm.normalizeVector(v1)
            p3 = p2 + v1 * mag/2
            pointsBetween.append(p3)
            pointsBetweenMag.append(mag)
            #mi.create_point(p3)
        if i0 > 200:
            i0 = -1
    return pointsBetween, pointsBetweenMag


surfs = []
plane = pv.Plane()
plane.scale((100,100,0))
plane.rotate_y(90)
plane.translate((6,0,0))
plane = plane.triangulate()
surfs.append(plane)

# maybe split the lunate vectors to only have part of the surface we need.
# same could be done for the radius
surfRad = surfRad_original.boolean_difference(plane)

#facesRad, verticesRad = getFacesAndVertices(radiusName)
#facesLun0, verticesLun = getFacesAndVertices(lunateName)
#pointsBetween, pointsBetweenMag = findPointsBetweenTwoBones(verticesRad, verticesLun, maxDist = 3.0)
#pset = pv.PolyData(pointsBetween)
#pset.plot(point_size=10)
#surfs.append(pset)

#surfs.append(surfLun)
#surfs.append(surfSca)
#surfs.append(surfRad)
#
##surfLun_test = deepcopy(surfLun)
##fm.transformPoints_byRef(surfLun_test.points, T_WL1, T_WL2)
##surfs.append(surfLun_test)
#
#p = pv.Plotter(window_size=[1000, 1000])
#for ind, surf in enumerate(surfs):
#    # only use smooth shading for the teapot
#    p.add_mesh(
#        surf,
#    scalars=np.arange(surf.n_faces), color='silver', specular=1.0, specular_power=10
#    )
#p.show_axes_all()
#p.view_vector([0, 1, 0])
#p.set_viewup([0, 0, -1])
#p.show()

def rayIntersectsTriangle(rayOrigin, rayVector, triangle):
    # from https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    EPSILON = 0.0000001
    rayOrigin = np.array(rayOrigin)
    rayVector = np.array(rayVector)

    vertex0 = triangle[0]
    vertex1 = triangle[1]
    vertex2 = triangle[2]
    #edge1, edge2, h, s, q = np.array((0.0, 0.0, 0.0))
    a,f,u,v = 0.0, 0.0, 0.0, 0.0
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    h = rayVector.cross(edge2)
    a = edge1.dot(h)
    if (a > -EPSILON and a < EPSILON):
        return False # This ray is parallel to this triangle.
    f = 1.0/a
    s = rayOrigin - vertex0
    u = f * s.dot(h)
    if (u < 0.0 or u > 1.0):
        return False
    q = s.cross(edge1)
    v = f * rayVector.dot(q)
    if (v < 0.0 or u + v > 1.0):
        return False
    # At this stage we can compute t to find out where the intersection point is on the line.
    t = f * edge2.dot(q)
    if (t > EPSILON): # ray intersection
        outIntersectionPoint = rayOrigin + rayVector * t
        return True, outIntersectionPoint
    else: # This means that there is a line intersection but not a ray intersection.
        return False, np.array([0.0, 0.0, 0.0])

def checkIntersection(surf1, surf2):
    com = fm.calcCOM(surf1.points, surf1.faces.reshape(-1, 4), method = 'points')
    #intersected, point = rayIntersectsTriangle(com, rayVector, triangle)

framesPointsRaw = [
    getPointsAtFrame(frames, 2),
    getPointsAtFrame(frames, 2083),
    getPointsAtFrame(frames, 2083 * 3),
]

from slil.process.align_by_surface_opt_fun import optFun

Nfeval = 1
def callbackF(Xi):
    global Nfeval
    #print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], f(Xi, argsExtra)))
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}   {7: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5], optFun(Xi, argsExtra)))
    Nfeval += 1
#print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)'))

import timeit

Nfeval = 1
start_time_global = timeit.default_timer()
def callbackFde(Xi, convergence):
    global Nfeval
    global start_time_global
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}'.format(Nfeval, convergence, Xi[0], Xi[1], Xi[2], optFun(Xi, argsExtra)))
    Nfeval += 1
    #if (timeit.default_timer() - start_time_global) > 30.0:
    #    return True # stops optimizer
    return False

startingT = (5,0,5,0,0,0)
bounds = [(-20,20), (-20,20), (-20,20), \
    (-1.0 * np.deg2rad(50), np.deg2rad(50)), \
    (-1.0 * np.deg2rad(50), np.deg2rad(50)), \
    (-1.0 * np.deg2rad(50), np.deg2rad(50))
    ]

#startingT = (5,0,5)
#bounds = [(-20,20), (-20,20), (-20,20)]

displayViz = False
display = False
referenceBone = {
    'name': radiusName,
    'surface': surfRad
    }

surfMeta = getGeometry(modelInfo, metacarp3Name)
checkBones = [
    { 'name': lunateName,'surface': surfLun },
    { 'name': scaphoidName,'surface': surfSca },
    #{ 'name': metacarp3Name,'surface': surfMeta },
    ]
#framesPointsRaw = [frame1PointsRaw, frame2PointsRaw, frame3PointsRaw]
argsExtra = (modelInfo, referenceBone, checkBones, T_WL1, M_O2Rad_orig, framesPointsRaw, display, displayViz)

#%%
optFun(startingT, argsExtra) # test function works

#%%
from scipy import optimize
result = optimize.fmin( \
    func=optFun, \
    x0=startingT, \
    args=(argsExtra, ), \
    callback=callbackF, \
    maxiter = 10)
#%%
result = optimize.fmin_powell( \
    func=optFun, \
    x0=startingT, \
    args=(argsExtra, ), \
    callback=callbackF, \
    maxiter = 20000)
#%%
result = optimize.differential_evolution( \
    func = optFun, \
    bounds=bounds, \
    x0 =startingT, \
    args=(argsExtra, ), \
    callback=callbackFde, \
    disp=True, \
    workers=-1, \
    maxiter = 1000)
#%%
result = optimize.minimize( \
    fun = optFun, \
    bounds=bounds, \
    x0 =startingT, \
    args=(argsExtra, ), \
    method='L-BFGS-B', \
    callback=callbackF, \
    options = { 'disp': False }
    )

from optimparallel import minimize_parallel
result = minimize_parallel( \
    fun = optFun, \
    bounds=bounds, \
    x0 =startingT, \
    #jac=None, \
    args=(argsExtra, ), \
    callback=callbackF, \
    parallel={
        'verbose': True, \
        'loginfo': True, 
        #'forward': False \
        }, \
    )

print(result)

displayViz = True
display = True

startingT = (4.83780404,  0.28850772,  5.01573, -3.14159265,  3.14159246,  3.14159265)
startingT = ( 5.323302,   0.000289,   3.836420,   0.000095,  -0.000164,   0.000581)

startingT = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

argsExtra = (modelInfo, referenceBone, checkBones, T_WL1, M_O2Rad_orig, framesPointsRaw, display, displayViz)
optFun(startingT, argsExtra) # test function works

#%%
import ray
ray.init(address='ray://localhost:10002', \
    runtime_env={ \
        "working_dir": ".", \
        "excludes": [
            "./cadaver results 03_08_2021.zip", \
            "./slil/rayTest/*", \
            "./slil/3-matic_backup/*", \
            "*.zip", \
            "*.PNG", \
            "*.png", \
            "*.whl",
            ],
        }, \
    ignore_reinit_error=True,
    )

#ray.shutdown()

@ray.remote
def generateIndividualGraphics(optFun, bounds, startingT, argsExtra, callbackF):
    #from pyvista.utilities import xvfb
    #xvfb.start_xvfb() # for linux only!
    #from slil.process.align_by_surface import optFun
    #result = optimize.differential_evolution( \
    #    func = optFun, \
    #    bounds=bounds, \
    #    x0 =startingT, \
    #    args=(argsExtra, ), \
    #    callback=callbackF, \
    #    disp=True, \
    #    workers=-1, \
    #    maxiter = 10)
    #print(result)
    #result.x
    #result.fun
    #result.nfev

    
    #result = optimize.minimize( \
    #    fun = optFun, \
    #    bounds=bounds, \
    #    x0 =startingT, \
    #    args=(argsExtra, ), \
    #    method='L-BFGS-B', \
    #    callback=callbackF, \
    #    options = { 'disp': False }
    #    )
    from optimparallel import minimize_parallel
    result = minimize_parallel( \
        fun = optFun, \
        bounds=bounds, \
        x0 =startingT, \
        #jac=None, \
        args=(argsExtra, ), \
        callback=callbackF, \
        parallel={
            'verbose': True, \
            #'forward': False \
            }, \
        )
    print(result)

tasksRunning = []
for ind in range(1):
    tasksRunning.append(generateIndividualGraphics.remote(optFun, bounds, startingT, argsExtra, callbackF))

while tasksRunning:
    finished, tasksRunning = ray.wait(tasksRunning, num_returns=1, timeout=None)
    for task in finished:
        result = ray.get(task)
        #print('result:', result)
    print('Tasks remaining:', len(tasksRunning))
print('Finished tasks.')

##%%
#s = startingT
#s = (4.83780404,  0.28850772,  5.01573, -3.14159265,  3.14159246,  3.14159265)
#t_optimized = np.eye(4)
#t_optimized[:3, :3] = fm.eulerAnglesToRotationMatrix((s[3], s[4], s[5]))
#t_optimized[:3, 3] = [s[0], s[1], s[2]]
#fn.modifyModelCache_adjTmatRadiusMarker(modelInfo, t_optimized)


#%%

experiments = [
    #'11525',
    #'11526',
    #'11527', # bad, do not run
    #'11534',
    #'11535',
    '11536',
    #'11537',
    #'11538',
    #'11539'
    ]
for experiment in experiments:
    #if True:
        #experiment = '11538'
        modelInfo = dc.load(experiment)
        modelTypes = [ 'normal', 'cut', 'scaffold']
        modelTypes = [ 'normal' ]
        for modelType in modelTypes:
        #if True:
            #modelInfo['currentModel'] = 'normal'
            #modelInfo['currentModel'] = 'cut'
            #modelInfo['currentModel'] = 'scaffold'
            modelInfo['currentModel'] = modelType

            #mi.open_project(modelInfo['3_matic_file'])

            numPoints = 12
            latestPoints = mi.get_points()[int(-1*numPoints):]


            if (modelInfo['currentModel'] == 'normal'):
                markerGroupName = 'tempMarkerSpheres_normal_static'
                #markerGroupName = 'tempMarkerSpheres_normal_static_after'
            if (modelInfo['currentModel'] == 'cut'):
                markerGroupName = 'tempMarkerSpheres_cut_static'
                #markerGroupName = 'tempMarkerSpheres_cut_static_after'
            if (modelInfo['currentModel'] == 'scaffold'):
                markerGroupName = 'tempMarkerSpheres_scaffold_static'
                #markerGroupName = 'tempMarkerSpheres_scaffold_static_after'



#%%
fn.convertPointsToSphereGroup(modelInfo, markerGroupName)
[lunateName, scaphoidName, radiusName, metacarp3Name] = fn.boneModelNames(modelInfo)
fn.moveToCloseToPart(markerGroupName, lunateName)
fn.convertSphereGroupToPoints(modelInfo, markerGroupName)
fn.generateMarkerPinsFromMarkerPoints(modelInfo)
fn.convertLinesToCylinders(modelInfo)
fn.convertPointsToSphereGroup(modelInfo, markerGroupName)
fn.addCylindersToSphereGroup(modelInfo, markerGroupName)


#%%

import slil.process.functions as fn
import slil.common.data_configs as dc
modelInfo = dc.load('11524')

# Using the marker tool select a triangle on the inside of a marker guide pin hole.
fn.generatePin(modelInfo)

#%%
fn.colourPoints(latestPoints, modelInfo)

#%% Flip them
mi.rotate(latestPoints,90,(0,0,0),(0,0,-1))
#%%
mi.rotate(latestPoints,90,(0,0,0),(1,0,0))
mi.rotate(latestPoints,180,(0,0,0),(0,1,0))
mi.translate(latestPoints,(0,0,-80))


#%%
fn.convertPointsToSphereGroup(modelInfo, markerGroupName)
#%%
fn.removeCylindersFromSphereGroup(modelInfo, markerGroupName)
#%%
fn.convertSphereGroupToPoints(modelInfo, markerGroupName)

#%%
fn.addCylindersToSphereGroup(modelInfo, markerGroupName)
# %%
fn.removeCylindersFromSphereGroup(modelInfo, markerGroupName)


#%% When there is a marker model in the scene
fn.generatePointOnMarkerPlates(modelInfo, 'marker_v2_low_resolution_lunate', lunateName)
fn.generatePointOnMarkerPlates(modelInfo, 'marker_v2_low_resolution_scaphoid', scaphoidName)
fn.generatePointOnMarkerPlates(modelInfo, 'marker_v2_low_resolution_radius', radiusName)
fn.generatePointOnMarkerPlates(modelInfo, 'marker_v2_low_resolution_3meta', metacarp3Name)

#%%
fn.printDistancesFromMarkers(modelInfo)

#%%
fn.calcBodyDistancesFromGround(modelInfo)

#%%
# Needs two parts
# 1. line named 'scaffold_edge_l_to_s'
# 2. bottom surface of lunate bone-plug named 'lunate_boneplug_bottom'
# It think left hands require True, second argument
fn.generateBoneplugPoints(modelInfo, True)

#%%
fn.printBoneplugsToBonesDistances(modelInfo)
#%%
fn.generateBoneplugsToBonesDistances(modelInfo)

#%%
fn.printDistancesFromMarkersForSphereGroup(modelInfo, 'tempMarkerShperes2_cut_fe_log')

fn.printDistancesFromMarkersForSphereGroup(modelInfo, 'tempMarkerSpheres_v4_static_try2')

fn.printDistancesFromMarkersForSphereGroup(modelInfo, 'tempMarkerSpheres_static_try1')

fn.printDistancesFromMarkersForSphereGroup(modelInfo, 'tempMarkerSpheres_static scaffold before_duplicate')
#%%
fn.printDistancesFromMarkersForSphereGroup(modelInfo, 'tempMarkerShperes')
#%% For resetting marker order
import slil.common.data_configs as dc
modelInfo = dc.load(experiment)
fn.removeCylindersFromSphereGroup(modelInfo, markerGroupName)
fn.convertSphereGroupToPoints(modelInfo, markerGroupName)
fn.convertPointsToSphereGroup(modelInfo, markerGroupName)
fn.addCylindersToSphereGroup(modelInfo, markerGroupName)

#%%
fn.removeCylindersFromSphereGroup(modelInfo, markerGroupName)
fn.convertSphereGroupToPoints(modelInfo, markerGroupName)
fn.setMarkersInModel_old(modelInfo)
fn.convertPointsToSphereGroup(modelInfo, markerGroupName)
fn.addCylindersToSphereGroup(modelInfo, markerGroupName)
#%% OpenSim put markers into model
fn.setMarkersInModel_old(modelInfo)

#%%
fn.exportBones(modelInfo)
#%%

fn.setBonesInModel(modelInfo)

# %%
trial = "\\normal_fe_40_40\\log1.c3d"
fn.generateBoneplugsToBonesDistances(modelInfo)
fn.runIK(modelInfo, trial, visualise=False)

#%%
#fn.convertSphereGroupToPoints(modelInfo, markerGroupName)
fn.setMarkersInModel_old(modelInfo)
fn.convertPointsToSphereGroup(modelInfo, markerGroupName)

# %%
fn.generateBoneplugsToBonesDistances(modelInfo)

trials = modelInfo['trialsRawData']

if (modelInfo['currentModel'] == 'normal'):
    trials = modelInfo['trialsRawData_only_normal']
if (modelInfo['currentModel'] == 'cut'):
    trials = modelInfo['trialsRawData_only_cut']
if (modelInfo['currentModel'] == 'scaffold'):
    trials = modelInfo['trialsRawData_only_scaffold']

for i, trial in enumerate(trials):
    fn.runIK(modelInfo, trial + '.c3d', visualise=False, threads = 3)
print('Finished processing trials.')

#%%

markerGroupName = 'tempMarkerSpheres_RefTest1-001'
plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
[lunateName, scaphoidName, radiusName, metacarp3Name] = fn.boneModelNames(modelInfo)
lunateCOM = mi.compute_center_of_gravity(mi.find_part(lunateName))
scaphoidCOM = mi.compute_center_of_gravity(mi.find_part(scaphoidName))
radiusCOM = mi.compute_center_of_gravity(mi.find_part(radiusName))
metacarp3COM = mi.compute_center_of_gravity(mi.find_part(metacarp3Name))

#fn.convertSphereGroupToPoints(modelInfo, markerGroupName)

#points = [lunateCOM, scaphoidCOM, radiusCOM, metacarp3COM]
#for i in points:
#    mi.create_point()
p0 = mi.find_point('marker' + str(plateAssignment[lunateName][2])).coordinates
p1 = mi.find_point('marker' + str(plateAssignment[scaphoidName][2])).coordinates
p2 = mi.find_point('marker' + str(plateAssignment[radiusName][2])).coordinates
p3 = mi.find_point('marker' + str(plateAssignment[metacarp3Name][2])).coordinates

#fn.convertPointsToSphereGroup(modelInfo, markerGroupName)
#
#mi.n_points_registration(
#    [lunateCOM, scaphoidCOM, radiusCOM, metacarp3COM],
#    [p0, p1, p2, p3],
#    mi.find_part(markerGroupName)
#)


#%% Move and rotate the markers to origin roughly aligned with bones


from math import atan
#origin/reference
p2[0]
vec1 = [
    p3[0]-p2[0],
    p3[1]-p2[1],
    p3[2]-p2[2]
    ]
vec1 = np.array(p3)
#vec1 = fm.normalizeVector(vec1)

#newpoint = mi.create_point(vec1)
vec2 = np.array([-1, 0, 0 ])


mat = fm.rotation_matrix_from_vectors(vec1, vec2)
newpos1 = mat.dot(vec1)
#newpos1 = np.allclose(fm.normalizeVector(vec1_rot), fm.normalizeVector(vec2))

#newpos1 = mat.dot(p0)
newpoint = mi.create_point(newpos1)
newpoint = mi.create_point(mat.dot(p0))
newpoint = mi.create_point(mat.dot(p1))
newpoint = mi.create_point(mat.dot(p2))

#%%
ang = (180,0,0)

ang = fm.degToRad(ang)
rotMat = fm.eulerAnglesToRotationMatrix(ang)
points = [p0]

newpos = np.dot(points, mat)

newpos = newpos[0]
newpoint = mi.create_point(newpos)

#%%
modelInfo = dc.load(experiment)
modelInfo['currentModel'] = 'normal'
#modelInfo['currentModel'] = 'cut'
#modelInfo['currentModel'] = 'scaffold'

trial = r'\normal_fe_40_40\log1'
#trial = r'\normal_ur_30_30\log1'

fileToImport = modelInfo['dataOutputDir'] + trial + '.c3d'
frames = fio.readC3D(fileToImport)

frame1 = frames[0]
frame2 = frames[23500]

plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]

radiusName = modelInfo['names']['radius']
markerIndex1 = plateAssignment[radiusName][2]
metacarp3Name = modelInfo['names']['metacarp3']
markerIndex2 = plateAssignment[metacarp3Name][2]

cord0 = frame1[markerIndex1,0:3]
cord1 = frame1[markerIndex2,0:3]
cord2 = frame2[markerIndex2,0:3]

posOffset = cord0
#rotOffset = 

#%% Calculating arc of scaphoid
modelInfo = dc.load(experiment)
modelInfo['currentModel'] = 'normal'
#modelInfo['currentModel'] = 'cut'
#modelInfo['currentModel'] = 'scaffold'

trial = r'\normal_fe_40_40\log1'
#trial = r'\normal_ur_30_30\log1'

fileToImport = modelInfo['dataOutputDir'] + trial + '.c3d'

frames = fio.readC3D(fileToImport)
trial = r'\normal_ur_30_30\log1'
#fileToImport = modelInfo['dataOutputDir'] + trial + '.c3d'
frames2 = fio.readC3D(fileToImport)

frameNums = [
    3700+8333,
    4700+8333,
    6000 + 8333
]
frameNums = range(0, 8000, 300)
plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]

radiusName = modelInfo['names']['radius']
markerIndex1 = plateAssignment[radiusName][2]
scaphoidName = modelInfo['names']['scaphoid']
metacarp3Name = modelInfo['names']['metacarp3']
markerIndex2 = plateAssignment[scaphoidName][2]

#%%


#%%
frames0 = fio.readC3D(fileToImport)
markerName = modelInfo['names']['radius']
markerIndex0 = plateAssignment[markerName]

# only need x,y,z
for i in range(len(frames)):
    frames[i] = frames0[i][:,0:3]

# find average of all points at ends of wires
frameNum = 0
pointsInMiddle = []
pointsAtEndOfWires = []
for k, name in enumerate(modelInfo['names']):
    if (k>3):
        break
    markerIndex0 = plateAssignment[modelInfo['names'][name]]

    p0 = frames[frameNum][markerIndex0[0]]
    p1 = frames[frameNum][markerIndex0[1]]
    p2 = frames[frameNum][markerIndex0[2]] # middle
    pm = (np.array(p0) - p1)/2 + p1 # between p0 and p1
    vec1 = np.array(p2) - pm # vector where wire is
    vec1 = (fm.normalizeVector(vec1)).reshape(3)
    wireLength = 15.85 + 45
    p4 = (vec1 * wireLength) + pm
    pointsAtEndOfWires.append(p4)
    pointsInMiddle.append(pm)
    #mi.create_point(p4)
avg1 = np.array([0,0,0])
avg2 = np.array([0,0,0])
for i in range(len(pointsAtEndOfWires)):
    avg1 = avg1 + pointsInMiddle[i]
    avg2 = avg2 + pointsAtEndOfWires[i]
avg1 = avg1 / len(pointsAtEndOfWires)
avg2 = avg2 / len(pointsAtEndOfWires)

vec3 = avg1 - avg2
vec3 = (fm.normalizeVector(vec3)).reshape(3)
#m2 = mi.create_line_direction_and_length([0,0,0], vec3, 20)

# move all frames to origin from average point
vec2 = np.array([0, -1, 0 ])
vec2 = (fm.normalizeVector(vec2)).reshape(3)
for i in range(len(frames)):
    offset = (np.array(avg2) * -1)
    rotateAbout = np.array(avg2)
    mat = fm.rotation_matrix_from_vectors(vec3, vec2)  
    for j in range(12):
        frames[i][j] = np.dot(mat, frames[i][j] - rotateAbout) + rotateAbout + offset



# rotate and move all frames relative to first radius frame
offset1 = np.array(frames[0][markerIndex1])
rotateAbout = np.array(frames[0][markerIndex1])
p0 = frames[0][plateAssignment[radiusName][0]]
p1 = frames[0][plateAssignment[radiusName][1]]
p2 = frames[0][plateAssignment[radiusName][2]] # middle
pm = (np.array(p0) - p1)/2 + p1 # between p0 and p1
vec2 = np.array(p2) - pm # vector where wire is
vec2 = (fm.normalizeVector(vec2)).reshape(3)
for i in range(1, len(frames)): # skip first frame
    offset = (np.array(frames[i][markerIndex1]) * -1.0)
    rotateAbout = np.array(frames[i][markerIndex1])
    p0 = frames[i][plateAssignment[radiusName][0]] + offset
    p1 = frames[i][plateAssignment[radiusName][1]] + offset
    p2 = frames[i][plateAssignment[radiusName][2]] + offset # middle
    pm = (np.array(p0) - p1)/2 + p1 # between p0 and p1
    vec1 = np.array(p2) - pm # vector where wire is
    vec1 = (fm.normalizeVector(vec1)).reshape(3)
    mat = fm.rotation_matrix_from_vectors(vec1, vec2)  
    for j in range(12):
        frames[i][j] = np.dot(mat, frames[i][j] - rotateAbout) + rotateAbout + offset + offset1

#vec2 = np.array([-0.5, 1, 0 ])
#vec2 = (fm.normalizeVector(vec2)).reshape(3)
#for i in range(len(frames)):
#    wireLength0 = 45
#    offset = (np.array(frames[i][markerIndex1]) * -1.0) + np.array([0,-1*wireLength0,0])
#    rotateAbout = np.array(frames[i][markerIndex1])
#    p0 = frames[i][plateAssignment[radiusName][0]] + offset
#    p1 = frames[i][plateAssignment[radiusName][1]] + offset
#    p2 = frames[i][plateAssignment[radiusName][2]] + offset # middle
#    pm = (np.array(p0) - p1)/2 + p1 # between p0 and p1
#    vec1 = np.array(p2) - pm # vector where wire is
#    vec1 = (fm.normalizeVector(vec1)).reshape(3)
#    mat = fm.rotation_matrix_from_vectors(vec1, vec2)  
#    for j in range(12):
#        frames[i][j] = np.dot(mat, frames[i][j] - rotateAbout) + rotateAbout + offset


# Rotate around radius marker
#offset = np.array([0,0,0])
#rotateAbout = np.array(frames[0][markerIndex1])
#p0 = frames[0][plateAssignment[radiusName][0]]
#p1 = frames[0][plateAssignment[radiusName][1]]
#p2 = frames[0][plateAssignment[radiusName][2]] # middle
#pm = (np.array(p0) - p1)/2 + p1 # between p0 and p1
#vec1 = np.array(p2) - pm # vector where wire is
#vec1 = (fm.normalizeVector(vec1)).reshape(3)
#mat = fm.rotation_matrix_around_axis(3.14, vec1)
#for i in range(len(frames)):
#    for j in range(12):
#        frames[i][j] = np.dot(mat, frames[i][j] - rotateAbout) + rotateAbout + offset

#%% Average scaphoid wire end poitns to create arc
markerIndex0 = plateAssignment[modelInfo['names']['scaphoid']]
wireLength = 15.85 + 45
# end points
pointsAtEndOfWires = []
for i in frameNums:
    p0 = frames[i][markerIndex0[0]]
    p1 = frames[i][markerIndex0[1]]
    p2 = frames[i][markerIndex0[2]] # middle
    pm = (np.array(p0) - p1)/2 + p1 # between p0 and p1
    vec1 = np.array(p2) - pm # vector where wire is
    vec1 = (fm.normalizeVector(vec1)).reshape(3)
    p4 = (vec1 * wireLength) + pm
    pointsAtEndOfWires.append(p4)

groupsOf = 7
numGroups = int(len(frameNums)/groupsOf)
avg = np.zeros((numGroups,3), dtype=float)
for i in range(0,numGroups):
    for j in range(groupsOf):
        avg[i] = avg[i] + pointsAtEndOfWires[(i*groupsOf)+j]
    avg[i] = avg[i] / groupsOf

#for i in range(len(avg)):
#    mi.create_point(avg[i])

arc1 = mi.create_arc_3_points(avg[0], avg[1], avg[2])
arc1_coord = arc1.center
arc1_dir = arc1.direction
mi.delete(arc1)

# move all frames so scaphoid rotation is negative z vector
vec2 = np.array([0, 0, -1 ])
vec2 = (fm.normalizeVector(vec2)).reshape(3)
for i in range(len(frames)):
    offset = (np.array(arc1_coord) * -1)
    rotateAbout = np.array(arc1_coord)
    mat = fm.rotation_matrix_from_vectors(arc1_dir, vec2)  
    for j in range(12):
        frames[i][j] = np.dot(mat, frames[i][j] - rotateAbout) + rotateAbout + offset

#%% # meaure angle to rotate markers about z axis so radius marker is closer to landmark

markerIndex0 = plateAssignment[modelInfo['names']['radius']]
p0 = frames[0][markerIndex0[0]]
p1 = frames[0][markerIndex0[1]]
p2 = frames[0][markerIndex0[2]] # middle
pm = (np.array(p0) - p1)/2 + p1 # between p0 and p1
vec1 = np.array(p2) - pm # vector where wire is
vec1 = (fm.normalizeVector(vec1)).reshape(3)
p4 = (vec1 * wireLength) + pm
vec1 = (fm.normalizeVector(p4)).reshape(3)

x = 20
y = -10
ang1 = np.arctan(y/x)

ang2 = np.arctan(vec1[1]/vec1[0])
ang = ang2 - ang1

offset = np.array([0,0,0])
rotateAbout = np.array([0,0,0])
vec1 = np.array([0,0,-1])
mat = fm.rotation_matrix_around_axis(ang, vec1)
for i in range(len(frames)):
    for j in range(12):
        frames[i][j] = np.dot(mat, frames[i][j] - rotateAbout) + rotateAbout + offset



#%% Average scaphoid wire end poitns to create arc
markerIndex0 = plateAssignment[modelInfo['names']['scaphoid']]
wireLength = 15.85 + 45
# end points
pointsAtEndOfWires = []
for i in frameNums:
    p0 = frames[i][markerIndex0[0]]
    p1 = frames[i][markerIndex0[1]]
    p2 = frames[i][markerIndex0[2]] # middle
    pm = (np.array(p0) - p1)/2 + p1 # between p0 and p1
    vec1 = np.array(p2) - pm # vector where wire is
    vec1 = (fm.normalizeVector(vec1)).reshape(3)
    p4 = (vec1 * wireLength) + pm
    pointsAtEndOfWires.append(p4)

groupsOf = 7
numGroups = int(len(frameNums)/groupsOf)
avg = np.zeros((numGroups,3), dtype=float)
for i in range(0,numGroups):
    for j in range(groupsOf):
        avg[i] = avg[i] + pointsAtEndOfWires[(i*groupsOf)+j]
    avg[i] = avg[i] / groupsOf

for i in range(len(avg)):
    mi.create_point(avg[i])

arc1 = mi.create_arc_3_points(avg[0], avg[1], avg[2])
#%%
for k, name in enumerate(modelInfo['names']):
    if (k>3):
        break
    markerIndex0 = plateAssignment[modelInfo['names'][name]]

    newPoints=[]
    for i in frameNums:
        cord0 = frames[i][markerIndex0[2]]
        newPoints.append(mi.create_point(cord0))

    # create wires
    for i in frameNums:
        p0 = frames[i][markerIndex0[0]]
        p1 = frames[i][markerIndex0[1]]
        p2 = frames[i][markerIndex0[2]] # middle
        pm = (np.array(p0) - p1)/2 + p1 # between p0 and p1
        vec1 = np.array(p2) - pm # vector where wire is
        vec1 = (fm.normalizeVector(vec1)).reshape(3)
        wireLength = 15.85 + 45
        m2 = mi.create_line_direction_and_length(pm, vec1, wireLength)
        
#%% on generate first frame
for k, name in enumerate(modelInfo['names']):
    if (k>3):
        break
    markerIndex0 = plateAssignment[modelInfo['names'][name]]

    cord0 = frames[0][markerIndex0[2]]
    mi.create_point(cord0)

    # create wires
    p0 = frames[0][markerIndex0[0]]
    p1 = frames[0][markerIndex0[1]]
    p2 = frames[0][markerIndex0[2]] # middle
    pm = (np.array(p0) - p1)/2 + p1 # between p0 and p1
    vec1 = np.array(p2) - pm # vector where wire is
    vec1 = (fm.normalizeVector(vec1)).reshape(3)
    wireLength = 15.85 + 45
    m2 = mi.create_line_direction_and_length(pm, vec1, wireLength)
#%%
markerIndex0 = plateAssignment[modelInfo['names']['lunate']]

newPoints=[]
for i in frameNums:
    cord0 = frames[i][markerIndex0[2]]
    newPoints.append(mi.create_point(cord0))
#arc1 = mi.create_arc_3_points(newPoints[0], newPoints[1], newPoints[2])
#arc1_coord = arc1.center
#arc1_dir = arc1.direction
#mi.create_line_direction_and_length(arc1_coord, arc1_dir, 20)

# create wires
wireLength = 15.85 + 45
for i in frameNums:
    p0 = frames[i][markerIndex0[0]]
    p1 = frames[i][markerIndex0[1]]
    p2 = frames[i][markerIndex0[2]] # middle
    pm = (np.array(p0) - p1)/2 + p1 # between p0 and p1
    vec1 = np.array(p2) - pm # vector where wire is
    vec1 = (fm.normalizeVector(vec1)).reshape(3)
    m2 = mi.create_line_direction_and_length(pm, vec1, wireLength)



#%%
newPoints=[]
for i in frameNums:
    cord0 = frames[i][markerIndex0[2]]
    newPoints.append(mi.create_point(cord0))
#arc1 = mi.create_arc_3_points(newPoints[0], newPoints[1], newPoints[2])
#arc1_coord = arc1.center
#arc1_dir = arc1.direction
#mi.create_line_direction_and_length(arc1_coord, arc1_dir, 20)

# create wires
for i in frameNums:
    p0 = frames[i][markerIndex0[0]]
    p1 = frames[i][markerIndex0[1]]
    p2 = frames[i][markerIndex0[2]] # middle
    pm = (np.array(p0) - p1)/2 + p1 # between p0 and p1
    vec1 = np.array(p2) - pm # vector where wire is
    vec1 = (fm.normalizeVector(vec1)).reshape(3)
    wireLength = 15.85 + 45
    m2 = mi.create_line_direction_and_length(pm, vec1, wireLength)

        
#%%
# make relative to radius center marker
for i in range(len(frames2)):
    frames2[i] = frames2[i][:,0:3] - frames2[i][markerIndex1,0:3]

# move to origin
cordOriginOffset = frames2[0][markerIndex0[2],0:3]
for i in range(len(frames2)):
    frames2[i] = frames2[i][:,0:3] - cordOriginOffset

for i in frameNums:
    cord0 = frames2[i][markerIndex0[2],0:3]
    newPoints.append(mi.create_point(cord0))
arc2 = mi.create_arc_3_points(newPoints[3], newPoints[4], newPoints[5])
arc2_coord = arc2.center
arc2_dir = arc2.direction
mi.create_line_direction_and_length(arc2_coord, arc2_dir, 20)

# create wires
for i in frameNums:
    p0 = frames2[i][markerIndex0[0],0:3]
    p1 = frames2[i][markerIndex0[1],0:3]
    p2 = frames2[i][markerIndex0[2],0:3] # middle
    pm = (np.array(p0) - p1)/2 + p1 # between p0 and p1
    vec1 = np.array(p2) - pm # vector where wire is
    vec1 = (fm.normalizeVector(vec1)).reshape(3)
    wireLength = 15.85 + 45
    m2 = mi.create_line_direction_and_length(pm, vec1, wireLength)

#mi.create_plane_3_points(newPoints[0], newPoints[1], newPoints[2])
#%%
offset = np.array([0,0,0])
rotateAbout = np.array([0,0,0])
p1 = np.array(newPoints[0])
p2 = np.array(newPoints[1])
p3 = np.array(newPoints[2])
vec1 = [
    p2[0]-p1[0],
    p2[1]-p1[1],
    p2[2]-p1[2]
    ]
vec1 = np.array(vec1)
vec1 = np.array(arc1_dir)
rotateAbout = np.array(arc1_coord)
offset = np.array(arc1_coord) * -1.0
vec2 = np.array([0, 1, 0 ])
mat = fm.rotation_matrix_from_vectors(vec1, vec2)

pointsToRotate = [p1, p2, p3]
pointsToRotate = newPoints
#rotateAbout = p1
newpoint = []
for i in pointsToRotate:
    newpoint.append(mi.create_point(mat.dot(i - rotateAbout) + rotateAbout + offset))
arc1_temp = mi.create_arc_3_points(newpoint[0], newpoint[1], newpoint[2])
arc1_temp_coord = arc1_temp.center
arc1_temp_dir = arc1_temp.direction
mi.create_line_direction_and_length(arc1_temp_coord, arc1_temp_dir, 20)


arc2 = mi.create_arc_3_points(newpoint[3], newpoint[4], newpoint[5])
arc2_coord = arc2.center
arc2_dir = arc2.direction
mi.create_line_direction_and_length(arc2_coord, arc2_dir, 20)


vec1 = np.array(arc2_dir)
rotateAbout = np.array(arc2_coord)
offset = np.array(arc2_coord) * -1.0
vec2 = np.array([0, 0, 1 ])
mat = fm.rotation_matrix_from_vectors(vec1, vec2)

pointsToRotate = newPoints
#rotateAbout = p1
newpoint = []
for i in pointsToRotate:
    newpoint.append(mi.create_point(mat.dot(i - rotateAbout) + rotateAbout + offset))

arc1_temp = mi.create_arc_3_points(newpoint[0], newpoint[1], newpoint[2])
arc1_temp_coord = arc1_temp.center
arc1_temp_dir = arc1_temp.direction
mi.create_line_direction_and_length(arc1_temp_coord, arc1_temp_dir, 20)

arc2_temp = mi.create_arc_3_points(newpoint[3], newpoint[4], newpoint[5])
arc2_temp_coord = arc2_temp.center
arc2_temp_dir = arc2_temp.direction
mi.create_line_direction_and_length(arc2_temp_coord, arc2_temp_dir, 20)


posOffset = cord0
# %%

# Generate a circle around the top of a hole
import itertools
a = mi.get_selection()
part = a[0].get_parent()
b = a[0].get_triangles()
partPoints = b[0]
partPointIndexes = b[1]

points = []
[[points.append(xx) for xx in x] for x in partPointIndexes]

holeSize = 0.7 # mm
holeTolerance = 0.01 # mm

rS = []
for pointComb in list(itertools.combinations(points, 3)):
    r = fm.calcRadius(partPoints[pointComb[0]], partPoints[pointComb[1]], partPoints[pointComb[2]])
    rS.append(r)
    #print(r)
    if ( holeSize - holeTolerance < r < holeSize + holeTolerance):
        print("Found: ", r)
        arc = mi.create_arc_3_points(partPoints[pointComb[0]], partPoints[pointComb[1]], partPoints[pointComb[2]])
vectorOrigin = arc.center

#%%

def calcRigidRegistration(p1_t, p2_t):
    #From: https://stackoverflow.com/questions/66923224/rigid-registration-of-two-point-clouds-with-known-correspondence
    ##Based on Arun et al., 1987
    #Writing points with rows as the coordinates

    #Take transpose as columns should be the points
    p1 = p1_t.transpose()
    p2 = p2_t.transpose()

    #Calculate centroids
    p1_c = np.mean(p1, axis = 1).reshape((-1,1)) #If you don't put reshape then the outcome is 1D with no rows/colums and is interpeted as rowvector in next minus operation, while it should be a column vector
    p2_c = np.mean(p2, axis = 1).reshape((-1,1))

    #Subtract centroids
    q1 = p1-p1_c
    q2 = p2-p2_c

    #Calculate covariance matrix
    H=np.matmul(q1,q2.transpose())

    #Calculate singular value decomposition (SVD)
    U, X, V_t = np.linalg.svd(H) #the SVD of linalg gives you Vt

    #Calculate rotation matrix
    R = np.matmul(V_t.transpose(),U.transpose())

    assert np.allclose(np.linalg.det(R), 1.0), "Rotation matrix of N-point registration not 1, see paper Arun et al."

    #Calculate translation matrix
    T = p2_c - np.matmul(R,p1_c)

    #Check result
    result = T + np.matmul(R,p1)
    if np.allclose(result,p2):
        print("transformation is correct!")
    else:
        print("transformation is wrong...")
    return [R,T]
#p1_t = np.array([[0,0,0], [1,0,0],[0,1,0]])
#p2_t = np.array([[0,0,1], [1,0,1],[0,0,2]]) #Approx transformation is 90 degree rot over x-axis and +1 in Z axis
#result = calcRigidRegistration(p1_t, p2_t)
#result = calcRigidRegistration(p1, p2)
#rot = result[0]
#tran = result[1].T[0]
#
#a0 = np.empty(shape=(len(p2),3))
#for ind, i in enumerate(p2):
#    a0[ind] = fm.rotateVector(i-tran, rot)
#
#for i in a0:
#    mi.create_point(i)