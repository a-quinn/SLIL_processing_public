# Author: Alastair Quinn 2021
# Used for data from cadaver RE9 and onward

import numpy as np
import slil.common.filtering as ft
import copy
from itertools import groupby

import numpy as np
class CriticallyDampedFilter():
    def __init__(self, sampFreq: float, cutoffFreq: float):
        self.sampling_frequency = sampFreq # Hz
        self.cutoff_frequency = cutoffFreq # Hz, low pass
        self.filter_passes = 2 # dual pass
        self.verbose = False
        self.createFilter()
    
    def createFilter(self):
        c = 1 / np.sqrt(np.power(2,(1/(2*self.filter_passes)))-1)
        #c_critical = 1 / (((2^(1/(2*filter_passes)))-1)^(1/2))
        f_adjusted = self.cutoff_frequency * c # 10 Hz becomes 22.9896 Hz

        w_adjusted = np.tan(np.pi*f_adjusted/self.sampling_frequency)

        k1 = 2 * w_adjusted
        k2 = np.power(w_adjusted,2)

        self.a0 = k2 / (1 + k1 + k2)
        self.a2 = self.a0
        self.a1 = 2 * self.a0

        self.b1 = 2 * self.a0 * (1/k2 - 1)
        self.b2 = 1 - (self.a0 + self.a1 + self.a2 + self.b1)
        if (self.verbose):
            print(self.a0 + self.a1 + self.a2 + self.b1 + self.b2)
            print(self.a0, self.a1, self.a2, self.b1, self.b2)


    def runFilter(self, data_raw_array):
        # Use this instead for a step response.
        #data_raw_array = [ zeros(1,500) ones(1,500) ]
        data_raw_copy_array = data_raw_array

        # Filter the data.
        data_array = np.ones(np.size(data_raw_array))

        # First pass in the forward direction.
        for x in range(0, np.size(data_raw_array)):
            if (x >= 2):

                data_array[x] = self.a0*data_raw_array[x] + self.a1*data_raw_array[x-1] + self.a2*data_raw_array[x-2] + self.b1*data_array[x-1] + self.b2*data_array[x-2]
            else:
                if (x == 1):
                    data_array[x] = self.a0*data_raw_array[x] + self.a1*data_raw_array[x-1] + self.b1*data_array[x-1]
                else:
                    data_array[x] = self.a0*data_raw_array[x]

        # Flip the data for the second pass.
        #data_raw_array = data_array(end:-1:1) # this is now the once filtered data
        data_raw_array = np.flip(data_array) # this is now the once filtered data
        data_array = np.ones(np.size(data_raw_array)) * 0.0

        # Second pass.
        for x in range(0, np.size(data_raw_array)):
            if (x >= 2):
                data_array[x] = self.a0*data_raw_array[x] + self.a1*data_raw_array[x-1] + self.a2*data_raw_array[x-2] + self.b1*data_array[x-1] + self.b2*data_array[x-2]
            else:
                if (x == 1):
                    data_array[x] = self.a0*data_raw_array[x] + self.a1*data_raw_array[x-1] + self.b1*data_array[x-1]
                else:
                    data_array[x] = self.a0*data_raw_array[x]

        # Flip the data back.
        data_array = np.flip(data_array)

        # Restore the raw array.
        data_raw_array = data_raw_copy_array

        # Plot the results.
        #plot(data_raw_array, ‘r’)
        #plot(data_butter_array, ‘g’)
        #plot(data_array, ‘b’)
        return data_array

def fun2(y):
    # find data which repeats for more then 20 points
    yIgnore = np.full((len(y)), False)
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(y)]
    cursor = 0
    for k, l in groups:
        if l >= 20:
            #yIgnore[cursor : cursor + l] = np.inf
            yIgnore[cursor : cursor + l] = True
        cursor += l

    yRemoveIgnore = np.array(y, dtype=float)
    if len(np.where(yIgnore)[0]) > 0:
        yRemoveIgnore[np.where(yIgnore)[0]] = np.nan
        yRemoveIgnore = ft.interpolate_np(yRemoveIgnore)
    return copy.deepcopy(yRemoveIgnore)

def fun3(y):
    yRemoveIgnore = np.array(y)

    # find data which repeats for more then 20 points
    yIgnore = np.full((len(y)), False)
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(y)]
    cursor = 0
    for k, l in groups:
        if l >= 20:
            #yIgnore[cursor : cursor + l] = np.inf
            yIgnore[cursor : cursor + l] = True
            yRemoveIgnore[cursor : cursor + l] = yRemoveIgnore[cursor-1]
        cursor += l

    return copy.deepcopy(yRemoveIgnore)

def getNumRepeating(y, ifRepeatFor = 20):
    # find data which repeats for more then X points
    yIgnore = np.full((len(y)), False)
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(y)]
    cursor = 0
    for k, l in groups:
        if l >= ifRepeatFor:
            yIgnore[cursor : cursor + l] = True
        cursor += l
    return yIgnore

def fun1(y, time, velocityLimit, verbose = False):
    # get derivative
    yDot = np.array(np.zeros(len(y)-1), dtype=float) # unit: meters/second
    dt = time[1] - time[0]
    for i in range(0, len(y) - 1):
        yDot[i] = (y[i+1] - y[i])/dt

    velocityLimitNeg = velocityLimit * -1.0

    badDataPoints = np.full((len(y)), False, dtype=bool)
    badDataPointsPos = np.full((len(y)), False, dtype=bool)
    badDataPointsNeg = np.full((len(y)), False, dtype=bool)
    badDataPointsX = np.empty([0], dtype=float) # unit: second

    for i, a in enumerate(yDot[:-1]):
        if abs(a) > velocityLimit:
            badDataPoints[i] = True
            badDataPointsX = np.append(badDataPointsX, time[i])
        if a > velocityLimit:
            badDataPointsPos[i] = True
        if a < velocityLimitNeg:
            badDataPointsNeg[i] = True
    badDataPointsY = np.full((len(badDataPointsX)), 0.0, dtype=float) # unit: none
    
    if (verbose):
        print("Bad points total: {} (positive: {} negative: {})".format(
            len(np.where(badDataPoints)[0]),
            len(np.where(badDataPointsPos)[0]),
            len(np.where(badDataPointsNeg)[0])
            ))
    yOutput = np.array(y)
    if (len(np.where(badDataPointsPos)[0]) > 0) and (len(np.where(badDataPointsNeg)[0]) > 0):

        # find data which repeats for more then 20 points
        yIgnore = np.full((len(y)), False)
        groups = [(k, sum(1 for _ in g)) for k, g in groupby(y)]
        cursor = 0
        for k, l in groups:
            if l >= 20:
                #yIgnore[cursor : cursor + l] = np.inf
                yIgnore[cursor : cursor + l] = True
            cursor += l

        #yRemoveIgnore = np.array(y)
        #yRemoveIgnore[np.where(yIgnore)[0]] = np.nan
        #yRemoveIgnore = ft.interpolate_np(yRemoveIgnore)

        doubleJump = []
        # find pairs (move away paired to when signal moves back to 'correct')
        pairs = []
        foundFirst = False
        foundFirstInd = 0
        # Is first jump negative or positive?
        if (np.where(badDataPointsPos)[0][0] < np.where(badDataPointsNeg)[0][0]):
            s = badDataPointsPos
            s2 = badDataPointsNeg
        else:
            s = badDataPointsNeg
            s2 = badDataPointsPos
        for ind, i in enumerate(s):

            # This doesn't really do the job...
            if yIgnore[ind] and not( i or s2[ind] ):
                continue

            if (foundFirst and (i or s2[ind])):
                pairs.append([foundFirstInd, ind])
                foundFirst = False
                continue
            if not foundFirst and ( i or s2[ind] ):
                if (foundFirst):
                    print("double jump? {}".format(ind))
                    doubleJump.append(ind)
                foundFirst = True
                foundFirstInd = ind
        if (verbose):
            print("Pairs found: {}".format(len(pairs)))
            if (not len(pairs) == len(np.where(badDataPointsPos)[0])) or (not (len(pairs) == len(np.where(badDataPointsNeg)[0]))):
                print("Warning! Pairs found is not the same as positive or negative points!".format(len(pairs)))
        

        # seperately find average change for positive and negative jump
        avgPos = 0
        avgPosI = 0
        avgNeg = 0
        avgNegI = 0
        for ind, i in enumerate(pairs):
            if (badDataPointsPos[i[0]]):
                avgPos = avgPos + (yOutput[i[0]] - yOutput[i[0]+1])
                avgPosI = avgPosI + 1
            if (badDataPointsNeg[i[0]]):
                avgNeg = avgNeg + (yOutput[i[0]] - yOutput[i[0]+1])
                avgNegI = avgNegI + 1
        if (avgPosI > 0):
            avgPos = avgPos / avgPosI
            for ind, i in enumerate(pairs):
                if (badDataPointsPos[i[0]]):
                    yOutput[(i[0]+1):i[1]+1] = yOutput[(i[0]+1):i[1]+1] + avgPos
                    #print("{} {} times: {} to {}".format(ind, i, time[i[0]], time[i[1]]))
        if (avgNegI > 0):
            avgNeg = avgNeg / avgNegI
            for ind, i in enumerate(pairs):
                if (badDataPointsNeg[i[0]]):
                    yOutput[(i[0]+1):i[1]+1] = yOutput[(i[0]+1):i[1]+1] + avgNeg
        #print(avgPos)
        #print(avgNeg)
    return [
        yOutput,
        yDot,
        badDataPoints,
        badDataPointsPos,
        badDataPointsNeg,
        badDataPointsX,
        badDataPointsY]