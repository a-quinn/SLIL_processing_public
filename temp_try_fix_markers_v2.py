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


velocityRaw = deepcopy(framesPointsRaw[:-1, :, :])
for indFrame in range(framesPointsRaw.shape[0]-1):
    velocityRaw[indFrame] = framesPointsRaw[indFrame, :, :] - framesPointsRaw[indFrame + 1, :, :]

velocityRawX = deepcopy(framesPointsRaw[:-1, :, :])
for indFrame in range(framesPointsRaw.shape[0]-1):
    velocityRawX[indFrame, :, 0] = framesPointsRaw[indFrame, :, 0] - framesPointsRaw[indFrame + 1, :, 0]

velocityRawY = deepcopy(framesPointsRaw[:-1, :, :])
for indFrame in range(framesPointsRaw.shape[0]-1):
    velocityRawY[indFrame, :, 1] = framesPointsRaw[indFrame, :, 1] - framesPointsRaw[indFrame + 1, :, 1]

velocityRawZ = deepcopy(framesPointsRaw[:-1, :, :])
for indFrame in range(framesPointsRaw.shape[0]-1):
    velocityRawZ[indFrame, :, 2] = framesPointsRaw[indFrame, :, 2] - framesPointsRaw[indFrame + 1, :, 2]

def findPlane(points):
    # from: https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    # do fit
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)

    # Manual solution
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)

    #print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    #print("errors: \n", errors)
    #print("residual:", residual)
    return fit

def group():
    return



points = []
for indFrame in range(len(velocityRawY)):
    newPointsRaw = velocityRawY[indFrame]
    for i in range(12):
        points.append(np.array(newPointsRaw[i]))
points = np.array(points)



import ckwrap
results = ckwrap.ckmeans(points[:, 1], (2, 9))
print('{}, {}, {}, {}, {}, {}, {}'.format(
    results.betweenss,
    results.centers,
    results.k,
    results.labels,
    results.sizes,
    results.totss,
    results.withinss
))
labels = results.labels

for indLabel in range(max(labels) + 1):
    groupedPoints = points[np.where(labels == indLabel), 1][0]
    rangeP = abs(groupedPoints.max() - groupedPoints.min())
    if rangeP > 50.0:
        results = ckwrap.ckmeans(groupedPoints, 2)
        print('{}, {}, {}, {}, {}, {}, {}'.format(
            results.betweenss,
            results.centers,
            results.k,
            results.labels,
            results.sizes,
            results.totss,
            results.withinss
        ))
        labels1 = results.labels
        labels[np.where(labels == indLabel)[0][np.where(labels1 == 1)]] = max(labels) + 1

colours = ['white', 'red', 'blue', 'green', 'brown', 'black', 'orange', 'yellow', 'purple']
for indGroup in range(max(labels) + 1):
    #ys = points[np.where(labels == indGroup), 1]
    #p = np.zeros((ys.shape[1], 3))
    #p[:, 1] = ys
    p = points[np.where(labels == indGroup)]
    plotter.add_points(points = p,
        name = 'p_clusters_'+str(indGroup),
        color = colours[indGroup])
    #scalars = labels)


groupedPoints = points[np.where(labels == 2)]
fit = findPlane(groupedPoints)

def affine_fit(points):
    
    p = np.mean(points, 0)
    
    # The samples are reduced:
    R = points - p
    # Computation of the principal directions if the samples cloud
    V, _ = np.linalg.eig(R*R)
    # Extract the output from the eigenvectors
    n = V[:, 1]
    V = V[:, 2:]
    return n, V, p

def findPlane2(points):
    # ripped from my other project and converted from MATLAB
    res_xy = 0.5 # mm resolution
    [n_1, V_1, p_1] = affine_fit(points)
    # Find distance to furthest point of interest from center of plane
    from scipy.spatial.distance import cdist
    D_2 = cdist(points, p_1)
    #D_2 = minDist(points, p_1, np.nan)
    indMin = D_2.min()
    #[_, index_p_1_to_lun_insert] = sort(D_2, 'descend')
    #indMin = index_p_1_to_lun_insert[1]
    a = np.sqrt(sum((p_1 - points[indMin,:])^2))
    # find number of points rqeuired for given resolution and minimum area
    a = np.ceil((a*2)/res_xy) # has to be an odd number
    if a % 2 == 0: # if even
        numPoints = a+1
    else:
        numPoints = a

    #[S1,S2] = meshgrid(linspace(min(z)-p_1(3), max(z)-p_1(3), numPoints)); % distances to min and max Z points from mid point
    [S1,S2] = np.meshgrid(np.arange(-(a/2*res_xy), a/2*res_xy, numPoints))
    #[S1,S2] = meshgrid([-1 0 1]);
    #generate the pont coordinates
    X = p_1[1]+[S1[:], S2[:]]*V_1[1,:].T
    Y = p_1[2]+[S1[:], S2[:]]*V_1[2,:].T
    Z = p_1[3]+[S1[:], S2[:]]*V_1[3,:].T
    return X, Y, Z
X, Y, Z = findPlane2(groupedPoints)


#plot
xlim = [np.min(groupedPoints[:, 0]), np.max(groupedPoints[:, 0])]
ylim = [np.min(groupedPoints[:, 1]), np.max(groupedPoints[:, 1])]
X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                  np.arange(ylim[0], ylim[1]))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]

pAround = []
for ind0 in range(X.shape[0]):
    for ind1 in range(X.shape[1]):
        pAround.append(np.array([X[ind0, ind1], Y[ind0, ind1], Z[ind0, ind1]]))
plotter.add_points(points = np.array(pAround),
    name = 'p_plane',
    color = 'red')
    #scalars = range(0, len(pAround)))


pAround = []
for indFrame in range(len(velocityRawY)):
    newPointsRaw = velocityRawY[indFrame]
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

