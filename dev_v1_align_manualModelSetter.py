# Author: Alastair Quinn 2022
# These scripts work with the 3-Matic API Trimatic and Python Jupyter

#%% To allow re-importing modules without restarting jupyter
%load_ext autoreload
%autoreload 2
#%matplotlib qt
#%%
from slil.common.cache import loadCache, saveCache
import slil.mesh.interface as smi
mi = smi.MeshInterface(1)

import numpy as np

import slil.process.functions as fn
import slil.common.data_configs as dc
import slil.common.math as fm

import pyvista as pv
#pv.set_jupyter_backend('pythreejs')

#py -m pip install pyvista trimesh optimparallel pythreejs


#%%
modelInfo = dc.load('11534')
mi.open_project(modelInfo['3_matic_file'])

experiments = [
    '11525',
    '11526',
    '11527', # bad, do not run
    '11534',
    '11535',
    '11536',
    '11537',
    '11538',
    '11539'
    ] 

# '11525' : 15.162742918740157,
# '11526' : 30.66243080203471,
# '11527' : 1.4464904122754896,
# '11534' : 0.793292277166685,
# '11535' : 1.7340741614066015,
# '11536' : 28.968188988417076,
# '11537' : 3.325593732440061,
# '11538' : 27.109040767744716,
# '11539' : 1.202595228938231

import slil.process.alignment_pointcloud as fna

results = {}
for i, experiment in enumerate(experiments):
    modelInfo = dc.load(experiment)

    mi.open_project(modelInfo['3_matic_file'])

    #modelInfo['currentModel'] = 'cut'
    #markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static'
    #framePointsCut = fn.getPointsFromSphereGroup(mi, markerGroupName)
    #lunateName = modelInfo['names']['lunate']
    #fna.visualizeMarkerPins(mi, modelInfo, framePointsCut, specificBone = lunateName)
    #fna.visualizeMarkerPins(mi, modelInfo, framePointsCut, specificBone = '')


    modelInfo['currentModel'] = 'cut'
    markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static'
    framePointsCut = fn.getPointsFromSphereGroup(mi, markerGroupName)

    modelInfo['currentModel'] = 'scaffold'
    markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static'
    framePointsScaffold = fn.getPointsFromSphereGroup(mi, markerGroupName)

    [lunateName, scaphoidName, radiusName, metacarp3Name] = fn.boneModelNames(modelInfo)

    # point cloud align with only two marker plates
    ind = modelInfo['plateAssignment_' + 'scaffold'][lunateName] + \
        modelInfo['plateAssignment_' + 'scaffold'][scaphoidName]
    framePointsScaffold_removed = (np.delete(framePointsScaffold, ind, axis = 0))
    ind = modelInfo['plateAssignment_' + 'cut'][lunateName] + \
        modelInfo['plateAssignment_' + 'cut'][scaphoidName]
    framePointsCut_removed = (np.delete(framePointsCut, ind, axis = 0))

    [result_ransac, p1Out, final, tran] = fna.cloudPointAlign_v2(
        framePointsCut_removed, framePointsScaffold_removed, False)
    results[experiment] = [result_ransac, p1Out, final, tran]

for r in results:
    results[r][0] = results[r][0].inlier_rmse

saveCache(results, 'pointCloudAlign_cut2scaffold_radiusAndMetacarp3')
results = loadCache('pointCloudAlign_cut2scaffold_radiusAndMetacarp3')

errors = [results[x][0] for x in results]

final = results['11534'][2]
for point in final:
    mi.create_point(point)
tran = results['11534'][3]

modelInfo['currentModel'] = 'scaffold'
markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static'
framePointsScaffold = fn.getPointsFromSphereGroup(mi, markerGroupName)

framePointsScaffold6 = fm.transformPoints(framePointsScaffold,np.eye(4), tran)
for point in framePointsScaffold6:
    #coord = fm.rotateVector(point, tran[:3, :3]) + (tran[:3, 3].T)
    #mi.create_point(coord)
    mi.create_point(point)

modelInfo['currentModel'] = 'scaffold'
modelCache = fn.getModelCache(modelInfo) # creates cache
coms = {
    modelInfo['names']['lunate']: modelCache['lunateCOM'],
    modelInfo['names']['scaphoid']: modelCache['scaphoidCOM'],
    modelInfo['names']['radius']: modelCache['radiusCOM'],
    modelInfo['names']['metacarp3']: modelCache['metacarp3COM'],
}

fn.setMarkersInModel(modelInfo, coms, framePointsScaffold6)

t_O2RadCut = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, framePointsCut)
t_O2MetCut = fn.getMarkerGroupTranformationMat(modelInfo, metacarp3Name, framePointsCut)


t_O2RadScaffold = fn.getMarkerGroupTranformationMat(modelInfo, radiusName, framePointsScaffold)
t_O2MetScaffold = fn.getMarkerGroupTranformationMat(modelInfo, metacarp3Name, framePointsScaffold)

framePointsScaffold2 = fm.transformPoints(framePointsScaffold, t_O2RadScaffold, t_O2RadCut)


framePointsScaffold3 = fm.transformPoints(framePointsScaffold, t_O2MetScaffold, t_O2MetCut)

framePointsScaffold4 = fm.transformPoints(framePointsScaffold, np.eye(4), tran)
for point in framePointsScaffold4:
    mi.create_point(point)



for point in framePointsCut_removed:
    mi.create_point(point)

for point in framePointsScaffold2:
    mi.create_point(point)
for point in framePointsScaffold3:
    mi.create_point(point)

fna.visualizeMarkerPins(mi, modelInfo, framePointsRaw = framePointsCut, specificBone = '')
fna.visualizeMarkerPins(mi, modelInfo, framePointsRaw = framePointsScaffold3, specificBone = '')
fna.visualizeMarkerPins(mi, modelInfo, framePointsRaw = framePointsScaffold2, specificBone = '')


#%% # This is the actual code that was run for the publication data. All other cells are for testing and development
import slil.common.data_configs as dc
from slil.process.align_viewer_settings import settings

#Groups
# D: Couldn't align any normal to cut
# C: Couldn't align scaphoid and/or lunate normal to cut
# B: Couldn't align scaphoid and/or lunate noraml to implanted
# A: All found reasonable alignment

# CAD#  Group
# 11524 C # But scaffold to normal looks fine...
          # Could be A with right initial aligment.. tried many times
# 11525 D # scaffold to normal looks almost fine...
# 11526 B # scaffold: scaphoid pin looks moved... just
# 11527 C # cut and scaffold look very similar... Could try a different static file or timestamp
# 11534 A # lunate is twisted, could be real and not marker error...
# 11535 B # scaffold: lunate looks moved
# 11536 A
# 11537 A
# 11538 B # scaffold: lunate pin looks moved
# 11539 B # scaffold: lunate and/or scaphoid pins look moved

modelInfo = dc.load('11534')
# only needed if missing some geometry (e.g. capitate)
#mi.open_project(modelInfo['3_matic_file'])

from slil.process.align_viewer import AlignViewer
scene = AlignViewer(modelInfo)
scene.loadScene()
flip, possiblePinSet, alignmentMethod, mapMarkerChangesN2C, mapMarkerChangesN2S = settings()

scene.setConfigurations(flip, possiblePinSet, alignmentMethod,
    mapMarkerChangesN2C, mapMarkerChangesN2S)
scene.align()

# uncomment and call this manually
#scene.setMarkersInModels(['normal', 'cut', 'scaffold'])



#%%

import numpy as np
import slil.process.functions as fn
import slil.common.math as fm
import slil.process.align_main as fnam
from slil.process.inverse_kinematics import findMarkerToPointsAlignmentOnce


alignedPoints = fm.transformPoints(scene.normalPoints, np.eye(4), tran)
scene.addPoints(0, alignedPoints, 'alignedPoints2', 'brown')




pins = fn.getModelCacheExtra(scene.modelInfo)['sensorGuidePins']

pp = np.empty((3, 3))
boneName = 'metacarp3'#'scaphoid'
pp[0, :] = pins['pin_' + boneName + '1']['point2']
pp[1, :] = pins['pin_' + boneName + '1']['point1']
pp[2, :] = pins['pin_' + boneName + '2']['point2']
markerSets = fnam.generateMarkerPointsFromPins(pp)

def calcError(p0, p1):
        error = 0.0
        for indAx1 in range(p1.shape[0]):
            error += np.power(p1[indAx1, 0] - p0[indAx1, 0], 2) + \
                np.power(p1[indAx1, 1] - p0[indAx1, 1], 2) + \
                np.power(p1[indAx1, 2] - p0[indAx1, 2], 2)
        return error

currentModel = 'normal'
errors = []
for indSet, markerPoints in enumerate(markerSets):
    plateAssignment = modelInfo['plateAssignment_' + currentModel]
    boneModelName = modelInfo['names'][boneName]
    points = np.array([
            scene.normalPoints[plateAssignment[boneModelName][0]],
            scene.normalPoints[plateAssignment[boneModelName][1]],
            scene.normalPoints[plateAssignment[boneModelName][2]] # middle marker
            ])
    errors.append(calcError(points, markerPoints))
np.argmin(errors)
scene.plotter[0, 0].add_points(
    markerSets[1],
    name = 'points_close_'+str(indSet),
    color = 'yellow',
    render_points_as_spheres=True,
    point_size=15.0,
    opacity = 0.9
)

for indSet, markerPoints in enumerate(markerSets):
    scene.plotter[0, 0].add_points(
        markerPoints,
        name = 'points_'+str(indSet),
        color = 'black',
        render_points_as_spheres=True,
        point_size=15.0,
        opacity = 0.9
    )





from copy import deepcopy

def getMarkersRelativeToBonesNormal(flip, modelInfo, modelTypeTarget, modelTypeSource, normalPoints, alignedPoints):
    alignedPointsReferencedInTarget = deepcopy(alignedPoints)
    for boneName in fn.boneNames():
        t_World_bone_initial = fnam.getMarkerTransformation(flip, modelInfo, modelTypeTarget, boneName, normalPoints)
        t_World_bone_aligned = fnam.getMarkerTransformation(flip, modelInfo, modelTypeSource, boneName, alignedPoints)

        boneModelName = modelInfo['names'][boneName]
        plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
        p0 = np.array([
            alignedPoints[plateAssignment[boneModelName][0]],
            alignedPoints[plateAssignment[boneModelName][1]],
            alignedPoints[plateAssignment[boneModelName][2]]
            ])
        newPoints = fm.transformPoints(p0, t_World_bone_aligned, t_World_bone_initial)

        plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
        alignedPointsReferencedInTarget[plateAssignment[boneModelName][0]] = newPoints[0]
        alignedPointsReferencedInTarget[plateAssignment[boneModelName][1]] = newPoints[1]
        alignedPointsReferencedInTarget[plateAssignment[boneModelName][2]] = newPoints[2]
    return alignedPointsReferencedInTarget
    
alignedPointsReferencedInTarget = getMarkersRelativeToBonesNormal(flip, modelInfo, 'normal', 'scaffold', normalPoints, alignedPoints1)
scene.plotter[0, 1].add_points(
    alignedPointsReferencedInTarget,
    name = 'alignedPointsReferencedInTarget',
    color = 'black',
    render_points_as_spheres=True,
    point_size=20.0,
    opacity = 0.9
)


scene.plotter[0, 0].add_points(
    scene.normalPoints,
    name = 'normalP',
    color = 'black',
    render_points_as_spheres=True,
    point_size=20.0,
    opacity = 0.9
)

modelInfo['currentModel'] = 'normal'
coms = {
    modelInfo['names']['lunate']: modelCache['lunateCOM'],
    modelInfo['names']['scaphoid']: modelCache['scaphoidCOM'],
    modelInfo['names']['radius']: modelCache['radiusCOM'],
    modelInfo['names']['metacarp3']: modelCache['metacarp3COM'],
}
fn.setMarkersInModel(modelInfo, coms, normalPoints)
#%%

import slil.process.functions as fn
import numpy as np
from pyvistaqt import MultiPlotter
import slil.common.data_configs as dc
import slil.common.math as fm
import slil.process.align_main as fnam

def getMarkerTran_temp(newPointsRaw, plateAssignment):
    p0 = np.array(newPointsRaw[plateAssignment[0]])
    p1 = np.array(newPointsRaw[plateAssignment[1]])
    p2 = np.array(newPointsRaw[plateAssignment[2]]) # middle marker
    #p0 = np.array(mi.find_point('marker' + str(plateAssignment[boneName][0])).coordinates)
    #p1 = np.array(mi.find_point('marker' + str(plateAssignment[boneName][1])).coordinates)
    #p2 = np.array(mi.find_point('marker' + str(plateAssignment[boneName][2])).coordinates) # middle marker
    pm = (p0 - p1)/2 + p1 # between p0 and p1
    vecAligned = fm.normalizeVector(p2 - pm) # vector where wire is
    posAligned = p2
    vecAlignedNorm = fm.normalizeVector(fm.calcNormalVec(p0, p1, p2))
    #if flip:
    #    vecAlignedNorm = vecAlignedNorm * -1.0
    rotAligned = np.array(fm.create3DAxis(vecAligned, vecAlignedNorm)).T
    MTrans = np.eye(4)
    MTrans[:3, :3] = rotAligned
    MTrans[:3, 3] = posAligned.T
    return MTrans

class Scene():
    def __init__(self):
        self.plotter = MultiPlotter(window_size=(1500, 1000), nrows=1, ncols=2, show=True)
scene = Scene()

scene.plotter[0, 1].enable_parallel_projection()
scene.plotter[0, 1].show_axes_all()

file = 'marker_offset_test.c3d'
from slil.common.io import readC3D
points = readC3D(file)
points = points[3000:3010, 0:6, :]
# 2 4 3 # marker 1
# 0 1 5 # marker 2
framePointsRaw = points[10, :, :]
scene.plotter[0, 1].add_points(
            framePointsRaw,
            name = 'marker_offest_test',
            color = 'black',
            render_points_as_spheres=True,
            point_size=20.0,
            opacity = 0.9
        )


markerMap = [2, 4, 3]
for indFrame, frame in enumerate(points):
    TMrad = getMarkerTran_temp(frame, markerMap)
    points[indFrame, :, :] = fm.transformPoints(points[indFrame, :, :], TMrad, np.eye(4))
for indFrame, frame in enumerate(points):
    framePointsRaw = frame[:, :]
    scene.plotter[0, 1].add_points(
                framePointsRaw,
                name = 'marker_offest_test_' + str(indFrame),
                color = 'black',
                render_points_as_spheres=True,
                point_size=10.0,
                opacity = 0.9
            )

markerPointsTemplate, angleToMirror, pinsA1, pinsA2, pinsB1, pinsB2 = fnam.markerHolderTemplate()


scene.plotter[0, 1].add_points(
            markerPointsTemplate,
            name = 'marker_template',
            color = 'black',
            render_points_as_spheres=True,
            point_size=10.0,
            opacity = 0.9
        )


scene.plotter[0, 1].add_lines(
        lines = pinsA1,
        color = 'black',
        width = 2.0,
        name = 'marker_template_pin_1'
    )
scene.plotter[0, 1].add_lines(
        lines = pinsA2,
        color = 'black',
        width = 2.0,
        name = 'marker_template_pin_2'
    )


markerMap = [2, 4, 3]
#markerMap = [1, 0, 5]
pinsSets = fnam.generatePinsFromMarkerPoints(framePointsRaw[markerMap])

for indSet, pins in enumerate(pinsSets):
    scene.plotter[0, 1].add_points(
                pins,
                name = 'pins_'+str(indSet),
                color = 'green',
                render_points_as_spheres=True,
                point_size=15.0,
                opacity = 0.9
            )


markerSets = fnam.generateMarkerPointsFromPins(pinsSets[0, :3])

for indSet, markerPoints in enumerate(markerSets):
    scene.plotter[0, 1].add_points(
                markerPoints,
                name = 'points_'+str(indSet),
                color = 'blue',
                render_points_as_spheres=True,
                point_size=15.0,
                opacity = 0.9
            )


def loadMarkerHolderMesh(isMirroredVersion):
    import pyvista as pv
    if not isMirroredVersion:
        fileMarkerPlate = r'data\marker_plate_version3.STL'
    else:
        fileMarkerPlate = r'data\marker_plate_version3_mirrored.STL'
    mesh = pv.read(fileMarkerPlate)

    dx = -np.min(mesh.points, axis=0)[0] # offset from overall edge of stl
    dx -= (np.max(mesh.points, axis=0)[0] - np.min(mesh.points, axis=0)[0]) / 2 # half way of model
    if not isMirroredVersion:
        dx += 1.84/2 # to pin1 hold center
    else:
        dx -= 1.84/2
    dy = -np.min(mesh.points, axis=0)[1] # offset from overall edge of stl

    dz = -np.min(mesh.points, axis=0)[2] # offset from overall edge of stl
    dz -= 0.7291 # distance from flat back face to tip of back upper marker pin holder
    dz -= 2.0 # outer diameter
    dz -= 1.84 # pin2 to pin1, hold centers

    mesh.translate(np.array([dx, dy, dz]))

    scene.plotter[0, 1].add_mesh(
        mesh = mesh,
        name = 'plate',
        #scalars = np.arange(surf.n_faces),
        show_scalar_bar = False,
        color='blanchedalmond',
        specular=1.0,
        specular_power=10
    )
loadMarkerHolderMesh(False)

scene.plotter[0, 1].add_points(
            np.array([0,0,0]),
            name = 'origin',
            color = 'red',
            render_points_as_spheres=True,
            point_size=15.0,
            opacity = 0.9
        )




#%%

#modelType = 'cut'
#for experiment in experiments:
#    modelInfo = dc.load(experiment)
#    modelCache = fn.getModelCache(modelInfo, breakIfNotFound=True)
#
#    coms = {
#        modelInfo['names']['lunate']: modelCache['lunateCOM'],
#        modelInfo['names']['scaphoid']: modelCache['scaphoidCOM'],
#        modelInfo['names']['radius']: modelCache['radiusCOM'],
#        modelInfo['names']['metacarp3']: modelCache['metacarp3COM'],
#    }
#
#    modelInfo['currentModel'] = modelType
#    fn.setMarkersInModel(modelInfo, coms, alignedPoints)
