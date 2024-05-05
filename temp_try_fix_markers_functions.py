# Author: Alastair Quinn 2022
import slil.common.math as fm
import numpy as np
import slil.common.data_configs as dc
import slil.common.plotting_functions as pf
from copy import deepcopy

def createCircle(radius, centrePoint, intervals, p0, p1):
    # (Parametric) If v1 and v2 are orthogonal unit vectors in R3, p is an arbitrary point, and r>0 is real, then the set of points of the form
    # p + r (cos(t)) v1 + r ( sin(t)) v2, t is real
    # is the circle with center p and radius r lying in the plane parallel to v1 and v2. (The same technique describes an arbitrary circle in Rn for n≥2.))
    pOnR =[]
    for t in np.arange(0, np.pi*2, intervals):
        #v1 and v2 just have to l
        v0 = fm.normalizeVector(p0 - p1)
        v1 = fm.orthogVector(v0) # vector where wire is
        v_temp = fm.create3DAxis(v0, v1)
        v1 = v_temp[1]
        v2 = v_temp[2]
        pOnR.append(centrePoint + radius * (np.cos(t)) * v1 + radius * ( np.sin(t)) * v2)
    return pOnR

def findLikely(pe, pOther1, pOther2, pMidOfCirlce, circleRadius):
    # checks intersection to plane from all possible axies.
    # If one axis is has bad data then can correct for it.
    v0 = fm.normalizeVector(pOther1 - pOther2)

    combs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    onPlane = [] # should be at least 2 points (likely three as third is almost never perfectly parallel to plane)
    for ind, vi in enumerate(combs):
        found, pLikely = fm.findLinePlaneIntersectPoint(pe, vi, v0*-1.0, pMidOfCirlce)
        if found:
            onPlane.append(pLikely)
    # there really should be a tolerance check here for distance error...
    ind = np.argmin([fm.calcDist(x, pMidOfCirlce) for x in np.array(onPlane)]) # most likely closest to centre point
    pLikely = pMidOfCirlce + fm.normalizeVector(onPlane[ind] - pMidOfCirlce) * circleRadius
    #plotter.add_points(points = np.array(pLikely), name = 'p_onPlane2'+str(ind), color = 'purple')
    return pLikely

def markerCharacteristics(pR, pL, pM):
    rd1 = fm.calcDist(pR, pM)
    rd2 = fm.calcDist(pL, pM)
    rd3 = fm.calcDist(pR, pL)
    pm = (pR - pL)/2 + pL # between p0 and p1
    r1 = fm.calcDist(pM, pm)
    a1 = (np.pi - np.pi/2 - fm.angleBetweenVectors((pR-pM), (pR-pL)))
    d2C1 = rd3 * np.sin(a1) # distance to center, should be greater than rd1 and rd2
    a2 = (np.pi - np.pi/2 - fm.angleBetweenVectors((pL-pM), (pL-pR)))
    d2C2 = rd3 * np.sin(a2) # distance to center, should be greater than rd1 and rd2
    r2 = rd3 * np.cos(a1)
    r3 = rd3 * np.cos(a2)
    #print(' ( {} {:.3f} {:.3f} {:.3f})'.format('Marker', rd1, rd2, rd3))
    return rd1, rd2, rd3, r1, d2C1, d2C2, r2, r3

def getTrialMarkerConfig(modelInfo, originalMarkerPoints, boneName, framesPointsRaw, indGoodFrames):
    # generate marker points based on trial

    def calcError(p0, p1):
        return (
                np.power(p1[0, 0] - p0[0, 0], 2) + 
                np.power(p1[0, 1] - p0[0, 1], 2) + 
                np.power(p1[0, 2] - p0[0, 2], 2) +
                np.power(p1[1, 0] - p0[1, 0], 2) + 
                np.power(p1[1, 1] - p0[1, 1], 2) + 
                np.power(p1[1, 2] - p0[1, 2], 2) +
                np.power(p1[2, 0] - p0[2, 0], 2) + 
                np.power(p1[2, 1] - p0[2, 1], 2) + 
                np.power(p1[2, 2] - p0[2, 2], 2))

    from scipy.optimize import minimize
    def cost(x0, argsExtra):
        x, y, z, rx, ry, rz = x0
        p0, p1Init = argsExtra
        p1 = deepcopy(p1Init)

        t_adjustment = fm.createTransformationMatFromPosAndEuler(x, y, z, rx, ry, rz)
        #p1 = fm.transformPoints_1(p1, t_adjustment)
        pointsTemp = np.ones((p1.shape[0], p1.shape[1] + 1))
        pointsTemp[:, :3] = p1
        for i, vector in enumerate(pointsTemp):
            p1[i] = np.dot(t_adjustment, vector)[:3]

        error = calcError(p0, p1) + 1.0
        return error
    def findT(p0, p1, x0):
        result = minimize( \
            fun = cost, \
            x0 = x0, \
            args = ([p0, p1], ), \
            method='L-BFGS-B', \
            options= {
                #'disp': True,
                'maxcor': 40,
                'maxiter': 3000,
                #'ftol': 0.01
                },
            #callback = calllbackSave, \
            #maxiter = 10
            )
        x, y, z, rx, ry, rz = result.x
        return x, y, z, rx, ry, rz

    boneFileName = modelInfo['names'][boneName]
    plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]

    everyNth = 1
    pointsTransposed = np.empty((int(len(indGoodFrames)/everyNth), 3, 3))
    for ind in np.arange(0, int(len(indGoodFrames)/everyNth)): # take every 3rd frame
    #for ind, indFrame in enumerate(indGoodFrames):
        frame = framesPointsRaw[indGoodFrames[ind*everyNth]]
        pointsTransposed[ind, :, :] = np.array([
            frame[plateAssignment[boneFileName][0]],
            frame[plateAssignment[boneFileName][1]],
            frame[plateAssignment[boneFileName][2]] # middle marker
            ])

    # move them so they're all exactly on top of the middle marker
    # helps optimizer find solutions fater
    for ind in range(pointsTransposed.shape[0]):
        pFrom = np.array([
            pointsTransposed[ind, 0, :],
            pointsTransposed[ind, 1, :],
            pointsTransposed[ind, 2, :] # middle marker
            ])
        pointsTransposed[ind, :, :] = pFrom - pointsTransposed[ind, 2, :] # middle marker

    pTo = originalMarkerPoints - originalMarkerPoints[2] # middle marker
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tAligned = fm.createTransformationMatFromPosAndEuler(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ii = 0

    #for ind, pFrom in enumerate(pointsTransposed):
    #    pFrom2 = fm.transformPoints_1(pFrom, tAligned)
    #    if calcError(pTo, pFrom2) > 1.5:
    #        x, y, z, rx, ry, rz = findT(pTo, pFrom, x0)
    #        x0 = np.array([x, y, z, rx, ry, rz]) # faster for next frame
    #        tAligned = fm.createTransformationMatFromPosAndEuler(x, y, z, rx, ry, rz)
    #        pointsTransposed[ind, :, :] = fm.transformPoints_1(pFrom, tAligned)
    #    else:
    #        pointsTransposed[ind, :, :] = pFrom
    #    if True:
    #        if ii >= 1000:
    #            print('\tframes\t{}/{}'.format(ind, pointsTransposed.shape[0]))
    #            ii = 0
    #        ii += 1

    # run again as some are wrong
    ii = 0
    #for ind in range(pointsTransposed.shape[0]):
    #    pFrom = np.array([
    #        pointsTransposed[ind, 0, :],
    #        pointsTransposed[ind, 1, :],
    #        pointsTransposed[ind, 2, :] # middle marker
    #        ])
    for ind, pFrom in enumerate(pointsTransposed):
        x, y, z, rx, ry, rz = findT(pTo, pFrom, x0)
        x0 = np.array([x, y, z, rx, ry, rz]) # faster for next frame
        tAligned = fm.createTransformationMatFromPosAndEuler(x, y, z, rx, ry, rz)
        pointsTransposed[ind, :, :] = fm.transformPoints_1(pFrom, tAligned)
        if True:
            if ii >= 1000:
                print('\tframes\t{}/{}'.format(ind, pointsTransposed.shape[0]))
                ii = 0
            ii += 1

    # calculate error relateive to middle marker
    pAround = []
    #errorOffset = np.empty((2, pointsTransposed.shape[0], 3))
    for ind in range(pointsTransposed.shape[0]):
        #for ind in range(20):
        p0 = pointsTransposed[ind][0, :]
        p1 = pointsTransposed[ind][1, :]
        p2 = pointsTransposed[ind][2, :] # middle marker

    #    errorOffset[0, ind, :] = p0 - p2
    #    errorOffset[1, ind, :] = p1 - p2

        pAround.append(p0)
        pAround.append(p1)
        pAround.append(p2)
    
    pR = np.mean(pointsTransposed[:, 0, :], axis=0)
    pL = np.mean(pointsTransposed[:, 1, :], axis=0)
    pM = np.mean(pointsTransposed[:, 2, :], axis=0)
    return pR, pL, pM, pAround
