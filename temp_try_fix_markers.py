# Author: Alastair Quinn 2022
# do not use as it will make results worse!
import slil.common.math as fm
import numpy as np
import slil.common.data_configs as dc
import slil.common.plotting_functions as pf
from copy import deepcopy
import temp_try_fix_markers_functions as tf
from slil.common.cache import loadCache, saveCache

modelInfo = dc.load('11534')
modelInfo['currentModel'] = 'normal'

# Ideal Marker configuration (e.i., from CAD model)
pR = np.array([20.39, 43.67, 3.61])
pL = np.array([-18.56, 43.67, 3.61])
pM = np.array([0.92, 27.11, 7.11])
rd1, rd2, rd3, r1, d2C1, d2C2, r2, r3 = tf.markerCharacteristics(pR, pL, pM)


filePath = modelInfo['dataInputDir'] + '\\' + modelInfo['currentModel'] + '_fe_40_40\\log1'
filePath = modelInfo['rootFolder'] + r'\data_original' + \
    '\\' + modelInfo['experimentID'] + r'\vel 10 acc 10 dec 10 jer 3.6' + '\\' + modelInfo['currentModel'] + '_fe_40_40\\log1'
expData = pf.convertToNumpyArrays(pf.loadTest2(filePath))

scale = 1000.0
framesPointsRaw = np.empty((expData['marker/1_1/x'].shape[0],12,3))
for indMarker in range(1,13):
    framesPointsRaw[:,indMarker-1,0] = expData['marker/1_' + str(indMarker) + '/x'] * scale
    framesPointsRaw[:,indMarker-1,1] = expData['marker/1_' + str(indMarker) + '/y'] * scale
    framesPointsRaw[:,indMarker-1,2] = expData['marker/1_' + str(indMarker) + '/z'] * scale

from pyvistaqt import BackgroundPlotter
plotter = BackgroundPlotter(window_size=(1000, 1000))
import pyvista as pv

plotter.show_grid()
plotter.enable_parallel_projection()
#plotter.disable_parallel_projection()

boneName = 'scaphoid'
boneFileName = modelInfo['names'][boneName]
plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
        
errorTolerance = 15 # mm
indErrorFrames = []
indGoodFrames = []
indErrorFramesRealBad = []
for indFrame in range(len(framesPointsRaw)):
    newPointsRaw = framesPointsRaw[indFrame]
    p0 = np.array(newPointsRaw[plateAssignment[boneFileName][0]])
    p1 = np.array(newPointsRaw[plateAssignment[boneFileName][1]])
    p2 = np.array(newPointsRaw[plateAssignment[boneFileName][2]]) # middle marker
    
    dE1 = fm.calcDist(p0, p2) - rd1
    dE2 = fm.calcDist(p1, p2) - rd2
    dE3 = fm.calcDist(p1, p0) - rd3
    errors = np.array([dE1, dE2, dE3])
    if np.any(np.abs(errors) > errorTolerance): # any have error
        numBad = (np.where(np.abs(errors) > errorTolerance))[0].shape[0]
        if numBad == 1 or numBad == 3:
            indErrorFramesRealBad.append(indFrame)
        else:
            indErrorFrames.append(indFrame)
    else:
        indGoodFrames.append(indFrame)
print('Frames | Good: {} Bad: {} RealBad: {}'.format(len(indGoodFrames), len(indErrorFrames), len(indErrorFramesRealBad)))

#originalMarkerPoints = np.array([pR, pL, pM])
#pR, pL, pM, pAround = tf.getTrialMarkerConfig(modelInfo, originalMarkerPoints, boneName, framesPointsRaw, indGoodFrames)
#plotter.add_points(points = np.array(pAround), name = 'pAround', color='green')
#newMarkerConfig = [pR, pL, pM]
#saveCache(newMarkerConfig, 'newMarkerConfig')

#newMarkerConfig = loadCache('newMarkerConfig')
#[pR, pL, pM] = newMarkerConfig
#rd1, rd2, rd3, r1, d2C1, d2C2, r2, r3 = tf.markerCharacteristics(pR, pL, pM)

pAround = []
for indFrame in range(len(framesPointsRaw)):
    newPointsRaw = framesPointsRaw[indFrame]
    for i in range(12):
        pAround.append(np.array(newPointsRaw[i]))
plotter.add_points(points = np.array(pAround),
    name = 'p_original',
    #color = 'black')
    scalars = range(0, len(pAround)))

pAround = []
for indFrame in range(len(framesPointsRaw)):
    newPointsRaw = framesPointsRaw[indFrame]
    p0 = np.array(newPointsRaw[plateAssignment[boneFileName][0]])
    p1 = np.array(newPointsRaw[plateAssignment[boneFileName][1]])
    p2 = np.array(newPointsRaw[plateAssignment[boneFileName][2]]) # middle marker
    
    pAround.append(p0)
    pAround.append(p1)
    pAround.append(p2)
plotter.add_points(points = np.array(pAround),
    name = 'p_original',
    #color = 'black')
    scalars = range(0, len(pAround)))

originalMarkerPoints = np.array([pR, pL, pM])
framesPointsFixed = deepcopy(framesPointsRaw)


#for indErrorFrame in indErrorFramesRealBad:
#    framesPointsFixed[indErrorFrame, plateAssignment[boneFileName][0], :] = np.zeros((3))
#    framesPointsFixed[indErrorFrame, plateAssignment[boneFileName][1], :] = np.zeros((3))
#    framesPointsFixed[indErrorFrame, plateAssignment[boneFileName][2], :] = np.zeros((3))

pAround = []
for indErrorFrame in indErrorFramesRealBad:
    frame = framesPointsFixed[indErrorFrame, :, :]
    p0 = np.array(frame[plateAssignment[boneFileName][0]])
    p1 = np.array(frame[plateAssignment[boneFileName][1]])
    p2 = np.array(frame[plateAssignment[boneFileName][2]]) # middle marker
    
    pAround.append(p0)
    pAround.append(p1)
    pAround.append(p2)
if len(pAround) > 0:
    plotter.add_points(points = np.array(pAround),
        name = 'pAround_224',
        color = 'yellow')


firstRun = True
show = False
for ind, indErrorFrame in enumerate(indErrorFrames):

    #indFrame2 = 587
    indFrame2 = indErrorFrame
    # get 10 frames either side
    indGoodFrames2 = np.array(indGoodFrames)
    in1 = np.where(indGoodFrames2 <= indFrame2)[0]
    in2 = np.where(indGoodFrames2 >= indFrame2)[0]
    indGoodFramesClose = np.append(indGoodFrames2[in1[-10:]], indGoodFrames2[in2[:10]])
    if not firstRun and not np.all(np.isclose(indGoodFramesClosePrevious, indGoodFramesClose)):
        if show:
            print('same frames for reference')
        pR, pL, pM, pAround = tf.getTrialMarkerConfig(
            modelInfo, originalMarkerPoints, boneName,
            framesPointsFixed, indGoodFramesClose)
        if show:
            plotter.add_points(points = np.array(pAround), name = 'pAround', color='green')
        rd1, rd2, rd3, r1, d2C1, d2C2, r2, r3 = tf.markerCharacteristics(pR, pL, pM)
    firstRun = False
    indGoodFramesClosePrevious = indGoodFramesClose

    # display frames around
    pAround = []
    for indFrame in range(indFrame2 - 10, indFrame2 + 10):
        newPointsRaw = framesPointsRaw[indFrame]
        for name in [boneName]:#, 'scaphoid', 'metacarp3', 'radius']:
            boneFileName = modelInfo['names'][name]
            plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
            p0 = np.array(newPointsRaw[plateAssignment[boneFileName][0]])
            p1 = np.array(newPointsRaw[plateAssignment[boneFileName][1]])
            p2 = np.array(newPointsRaw[plateAssignment[boneFileName][2]]) # middle marker
            
            pAround.append(p0)
            pAround.append(p1)
            pAround.append(p2)
    plotter.add_points(points = np.array(pAround),
        name = 'pAround',
        scalars = range(0,len(pAround)))

    newPointsRaw = framesPointsFixed[indFrame2]
    #pStr = '{}'.format(indFrame)
    p0 = np.array(newPointsRaw[plateAssignment[boneFileName][0]])
    p1 = np.array(newPointsRaw[plateAssignment[boneFileName][1]])
    p2 = np.array(newPointsRaw[plateAssignment[boneFileName][2]]) # middle marker

    dE1 = fm.calcDist(p0, p2) - rd1
    dE2 = fm.calcDist(p1, p2) - rd2
    dE3 = fm.calcDist(p1, p0) - rd3
    #    pStr += ' ( {:.3f} {:.3f} {:.3f} )'.format(dE1, dE2, dE3)
    #print(pStr)

    errorTolerance = 2 # mm
    errors = np.array([dE1, dE2, dE3])
    if np.any(np.abs(errors) > errorTolerance): # any have error

        # two will be large error, meaning the common marker is bad
        indBadPoint = np.argmin(errors) 
        indBadPoint = [1, 0, 2][indBadPoint]

        badPoint = np.array([p0, p1, p2])[indBadPoint]
        if show:
            plotter.add_points(points = np.array([badPoint]), name = 'p2', color='red')

        if indBadPoint == 2:
            pm = (p0 - p1)/2 + p1 # between p0 and p1

            if show:
                pOnR = tf.createCircle(r1, pm, np.pi/10, p0, p1)
                plotter.add_points(points = np.array([p0, p1]), name = 'pGood')
                plotter.add_points(points = np.array(pOnR), name = 'pOnRadius')
            pLikely = tf.findLikely(badPoint, p0, p1, pm, r2)

        if indBadPoint == 0:
            pm = p1 + fm.normalizeVector(p2-p1) * d2C1
            
            if show:
                pOnR = tf.createCircle(r2, pm, np.pi/10, p1, p2)
                plotter.add_points(points = np.array([p1, p2]), name = 'pGood')
                plotter.add_points(points = np.array(pOnR), name = 'pOnRadius')

            pLikely = tf.findLikely(badPoint, p1, p2, pm, r2)
    
        if indBadPoint == 1:
            pm = p0 + fm.normalizeVector(p2-p0) * d2C2
            
            if show:
                pOnR = tf.createCircle(r3, pm, np.pi/10, p0, p2)
                plotter.add_points(points = np.array([p0, p2]), name = 'pGood')
                plotter.add_points(points = np.array(pOnR), name = 'pOnRadius')

            pLikely = tf.findLikely(badPoint, p0, p2, pm, r3)
            #pLikely = np.zeros((3))
        if show:
            plotter.add_points(points = np.array(pLikely), name = 'p_onPlane2'+str(0), color = 'purple')
        framesPointsFixed[indFrame2, plateAssignment[boneFileName][indBadPoint], :] = pLikely[:]
    print('{}/{}'.format(ind, len(indErrorFrames)))


pAround = []
for indGoodFrame in indGoodFrames:
    frame = framesPointsFixed[indGoodFrame, :, :]
    p0 = np.array(frame[plateAssignment[boneFileName][0]])
    p1 = np.array(frame[plateAssignment[boneFileName][1]])
    p2 = np.array(frame[plateAssignment[boneFileName][2]]) # middle marker
    
    pAround.append(p0)
    pAround.append(p1)
    pAround.append(p2)
#plotter.add_points(points = np.array(pAround),
#    name = 'pAround_2',
#    scalars = range(len(pAround) + 0,len(pAround) + len(pAround)))
plotter.add_points(points = np.array(pAround),
    name = 'pAround_22',
    color = 'green')

pAround2 = []
#for frame in framesPointsFixed:
for indErrorFrame in indErrorFrames:
    frame = framesPointsFixed[indErrorFrame, :, :]
    p0 = np.array(frame[plateAssignment[boneFileName][0]])
    p1 = np.array(frame[plateAssignment[boneFileName][1]])
    p2 = np.array(frame[plateAssignment[boneFileName][2]]) # middle marker
    
    pAround2.append(p0)
    pAround2.append(p1)
    pAround2.append(p2)
plotter.add_points(points = np.array(pAround2),
    name = 'pAround_2',
    color = 'red')

