
from scipy.interpolate import interp1d#, InterpolatedUnivariateSpline
import numpy as np
from scipy import signal
import pandas as pd
import copy
from itertools import groupby
#from sklearn.decomposition import FastICA, PCA

def interpolateMoCap(data, outlierLimit = 0.5):
    count = len(data)
    #find first non-zero value
    num = 0
    for i in range(count -1):
        if (data[i]!= 0):
            num = i
            break
    #average = np.average(data[i:i+20]) # average first 20 samples, incase simple techinique deosn't work
    for i in range(num,count - 2):
        sCurrent = data[i]
        sNext = data[i + 1]
        if (sNext - outlierLimit > sCurrent or sNext + outlierLimit < sCurrent):
            y = [data[i - 1], data[i],data[i+1], data[i+2]]
            x = np.arange(0,4)
            f2 = interp1d(x, y, kind='cubic')
            data[i+1] = f2(x)[1]

def interpolateMoCap_dataframe(data, columnName, outlierLimit = 0.5):
    count = len(data[columnName])

    #find first non-zero value
    num = 0
    for i in range(count -1):
        if (data.loc[i, columnName]!= 0):
            num = i
            break

    # find average of first 250 frames instead of first zero
    average = sum(data.loc[num:250, columnName])/(250-num)
    for i in range(num, 250):
        if (data.loc[i, columnName] - outlierLimit < average and data.loc[i, columnName] + outlierLimit > average):
            num = i
            break

    #average = np.average(data[columnName][i:i+20]) # average first 20 samples, incase simple techinique deosn't work
    x = np.arange(0,4)
    for i in range(num, count - 2):
        sCurrent = data.loc[i, columnName]
        sNext = data.loc[i + 1, columnName]
        # only interpolate if the data point after the bad one is within limits
        if ((sNext - outlierLimit > sCurrent or sNext + outlierLimit < sCurrent) and (data.loc[i+2, columnName] - outlierLimit < sCurrent and data.loc[i+2, columnName] + outlierLimit > sCurrent)):
        #if (sNext - outlierLimit > sCurrent or sNext + outlierLimit < sCurrent):
            y = [data.loc[i-1, columnName], sCurrent, sNext, data.loc[i+2, columnName]]
            #y = [data.loc[i-1, columnName], data.loc[i, columnName], data.loc[i+1, columnName], data.loc[i+2, columnName]]
            f2 = interp1d(x, y, kind='cubic')
            data.loc[i+1, columnName] = f2(x)[1]

def findBadData(data, columnName, outlierLimit = 0.5):
    data_temp = data[columnName].to_numpy()
    x, badDataPoints = findBadData_np(data_temp, outlierLimit)
    data[columnName] = x
    return badDataPoints

def findBadData_np(data, outlierLimit = 0.5):
    data_temp = data
    data_temp2 = copy.deepcopy(data_temp)
    count = len(data_temp)
    badDataPoints = np.array(np.zeros(count), dtype=bool)
    badDataDefinition = np.nan


    #data_temp3 = np.vstack([data_temp2,range(0,data_temp2.shape[0])])
    ## Compute ICA
    #ica = FastICA(n_components=3)
    #S_ = ica.fit_transform(data_temp3)  # Reconstruct signals
    #A_ = ica.mixing_  # Get estimated mixing matrix

    # find data which repeats for more then 20 points
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(data_temp2)]
    cursor = 0
    for k, l in groups:
        if l >= 20:
            data_temp2[cursor : cursor + l] = badDataDefinition
            badDataPoints[cursor : cursor + l] = 1
        cursor += l

    # find data which repeats for less then 20 points but more than 4 points, and
    # greater than the offset form surrounding points
    #cursor = 0
    #outlierLimit2 = 0.01
    #for k, l in groups:
    #    if (cursor >= 20 and l >= 4 and l <= 20):
    #        lenGoodData = len(np.where(np.isfinite(data_temp2[cursor - 20 : cursor])==True)[0])
    #        avgBefore = sum(data_temp2[cursor - 20 : cursor])/lenGoodData
    #        if (np.isfinite(avgBefore)):
    #            avg = avgBefore
    #            if (k - outlierLimit2 > avg):
    #                data_temp2[cursor : cursor + l] = badDataDefinition
    #                badDataPoints[cursor : cursor + l] = 1
    #            else:
    #                if (k + outlierLimit2 < avg):
    #                    data_temp2[cursor : cursor + l] = badDataDefinition
    #                    badDataPoints[cursor : cursor + l] = 1
    #    cursor += l

    #find first non-zero value
    num = 0
    for i in range(count -1):
        if (data_temp[i]!= 0):
            num = i
            break

    # find average of first N frames instead of first zero
    numAvr = 400
    # beginnings of some data can be wildly off so give some leeway
    outlierLimitBias = 0.02
    average = sum(data_temp[num:numAvr])/(numAvr-num)
    for i in range(num, numAvr):
        if (data_temp[i] - (outlierLimit + outlierLimitBias) < average and data_temp[i] + (outlierLimit + outlierLimitBias) > average):
            num = i
            break
    
    data_temp2[0:num] = badDataDefinition
    badDataPoints[0:num] = 1

    # use gradient to check if the following point is within that limit.
    rate = 250 #Hz
    rateLimit = 0.120 # mm/s
    stepLimit = rateLimit/rate # mm/step
    outlierLimit2 = 0.002
    i = num
    while i < count - 1:
        nextPoint = data_temp2[i + 1]
        point = data_temp2[i]
        
        if (np.isfinite(data_temp2[i])):
            # this section could be re-written...
            if (nextPoint > point + outlierLimit2):
                data_temp2[i+1] = badDataDefinition
                badDataPoints[i+1] = 1
                for ii in range(1, count - i - 1): # find the next point within limits
                    followingPoint = data_temp2[i + 1 + ii]
                    if (point - (stepLimit * (ii - 1)) - outlierLimit2 < followingPoint < point + (stepLimit * (ii - 1)) + outlierLimit2):
                    #if (not (followingPoint - outlierLimit2 > point - (stepLimit * (ii - 1) ) > followingPoint + outlierLimit2 ) \
                    #    or not (followingPoint - outlierLimit2 > point + (stepLimit * (ii - 1) ) > followingPoint + outlierLimit2 ) ):
                        if (not i == 1): # if there is only one bad point
                            data_temp2[i + 2 : i + 1 + ii] = badDataDefinition
                            badDataPoints[i + 2 : i + 1 + ii] = 1
                        i += ii
                        break
            else:
                if (nextPoint < point - outlierLimit2):
                    data_temp2[i+1] = badDataDefinition
                    badDataPoints[i+1] = 1
                    for ii in range(1, count - i - 1):
                        followingPoint = data_temp2[i + 1 + ii]
                        if (point - (stepLimit * (ii - 1)) - outlierLimit2 < followingPoint < point + (stepLimit * (ii - 1)) + outlierLimit2):
                        #if ( not (followingPoint + outlierLimit2 < point - (stepLimit * (ii - 1)) < followingPoint - outlierLimit2 ) \
                        #    or not (followingPoint + outlierLimit2 < point + (stepLimit * (ii - 1)) < followingPoint - outlierLimit2 ) ):
                            if (not i == 1):
                                data_temp2[i + 2 : i + 1 + ii] = badDataDefinition
                                badDataPoints[i + 2 : i + 1 + ii] = 1
                            i += ii
                            break
        i += 1

    # get valid values either side of a group of points, check they are inside range an average surrounding range
    #for checks in range(2): # just runs twice
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(badDataPoints)]
    cursor = 0
    for k, l in groups:
        if (l <= 65 and k == False):
            numAvgPoints = l
            indexOfLastGoodPoints = np.where(badDataPoints[:cursor]==False)[0][-numAvgPoints:]
            avgBefore = sum(data_temp2[indexOfLastGoodPoints])/numAvgPoints
            
            indexOfFutureGoodPoints = np.where(badDataPoints[cursor + l:]==False)[0][:numAvgPoints] + (cursor + l)
            avgAfter = sum(data_temp2[indexOfFutureGoodPoints])/numAvgPoints

            pointsAvg = sum(data_temp2[cursor : cursor + l])/l
            if ( (not avgBefore + outlierLimit2 <= pointsAvg <= avgAfter - outlierLimit2) and (not avgBefore - outlierLimit2 >= pointsAvg >= avgAfter + outlierLimit2) ):
                if (pointsAvg > avgBefore + outlierLimit2):
                    data_temp2[cursor : cursor + l] = badDataDefinition
                    badDataPoints[cursor : cursor + l] = 1
                else:
                    if (pointsAvg < avgBefore - outlierLimit2):
                        data_temp2[cursor : cursor + l] = badDataDefinition
                        badDataPoints[cursor : cursor + l] = 1
        cursor += l
    
    badDataPoints = np.flip(badDataPoints)
    data_temp2 = np.flip(data_temp2)
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(badDataPoints)]
    cursor = 0
    for k, l in groups:
        if (l <= 65 and k == False):
            numAvgPoints = l
            indexOfLastGoodPoints = np.where(badDataPoints[:cursor]==False)[0][-numAvgPoints:]
            avgBefore = sum(data_temp2[indexOfLastGoodPoints])/numAvgPoints
            
            indexOfFutureGoodPoints = np.where(badDataPoints[cursor + l:]==False)[0][:numAvgPoints] + (cursor + l)
            avgAfter = sum(data_temp2[indexOfFutureGoodPoints])/numAvgPoints

            pointsAvg = sum(data_temp2[cursor : cursor + l])/l
            if ( (not avgBefore + outlierLimit2 <= pointsAvg <= avgAfter - outlierLimit2) and (not avgBefore - outlierLimit2 >= pointsAvg >= avgAfter + outlierLimit2) ):
                if (pointsAvg > avgBefore + outlierLimit2):
                    data_temp2[cursor : cursor + l] = badDataDefinition
                    badDataPoints[cursor : cursor + l] = 1
                else:
                    if (pointsAvg < avgBefore - outlierLimit2):
                        data_temp2[cursor : cursor + l] = badDataDefinition
                        badDataPoints[cursor : cursor + l] = 1
        cursor += l
    badDataPoints = np.flip(badDataPoints)
    data_temp2 = np.flip(data_temp2)

    #outlierLimitFine = 0.0015
    #cursor = 0
    #for k, l in groups:
    #    if (3 <= l <= 20):
    #        numAvgPoints = l
    #        indexOfLastGoodPoints = np.where(badDataPoints[:cursor]==False)[0][-numAvgPoints:]
    #        avgBefore = sum(data_temp2[indexOfLastGoodPoints])/numAvgPoints
    #        
    #        indexOfFutureGoodPoints = np.where(badDataPoints[cursor + l:]==False)[0][:numAvgPoints] + (cursor + l)
    #        avgAfter = sum(data_temp2[indexOfFutureGoodPoints])/numAvgPoints
#
    #        pointsAvg = sum(data_temp2[cursor : cursor + l])/l
    #        if ( (not avgBefore <= pointsAvg <= avgAfter) and (not avgBefore >= pointsAvg >= avgAfter) ):
    #            if (pointsAvg - outlierLimitFine > avgBefore):
    #                data_temp2[cursor : cursor + l] = badDataDefinition
    #                badDataPoints[cursor : cursor + l] = 1
    #            else:
    #                if (pointsAvg + outlierLimitFine < avgBefore):
    #                    data_temp2[cursor : cursor + l] = badDataDefinition
    #                    badDataPoints[cursor : cursor + l] = 1
    #    cursor += l






    #sGood = data_temp[i]
    #dataSide = 0
    #for i in range(num, count-1):
    #    sCurrent = data_temp[i]
    #    sNext = data_temp[i + 1]
#
    #    if (dataSide == 0):
    #        sGood = sCurrent
    #    # only interpolate if the data point after the bad one is within limits
    #    #if ((sNext - outlierLimit > sCurrent or sNext + outlierLimit < sCurrent) and (data_temp[i+2] - outlierLimit < sCurrent and data_temp[i+2] + outlierLimit > sCurrent)):
    #    if (sNext - outlierLimit > sGood):
    #        dataSide = 1
    #        data_temp2[i+1] = badDataDefinition
    #        badDataPoints[i+1] = 1
    #    else:
    #        if (sNext + outlierLimit < sGood):
    #            dataSide = 1
    #            data_temp2[i+1] = badDataDefinition
    #            badDataPoints[i+1] = 1
    #        else:
    #            dataSide = 0
    
    data = data_temp2
    return (data_temp2, badDataPoints)

def interpolate_np(data):
    count = len(data)
    goodPoints = np.isfinite(data)
    if (max(goodPoints) > 0):
        x = np.array(range(count))[goodPoints]
        f2 = interp1d(x, data[goodPoints], kind='slinear', fill_value="extrapolate")
        #f2 = interp1d(x, data[goodPoints], fill_value="extrapolate")
        #f2 = InterpolatedUnivariateSpline(x, data[goodPoints])
        return f2(np.arange(0,count,1))
    return data

def interpolate(data, columnName):
    data_temp = data[columnName].to_numpy()
    data_temp = interpolate_np(data_temp)
    data[columnName] = pd.DataFrame(data=data_temp)


def filterLowpass_old(dataIn, cutoffFreq, sampleFreq):
    NyquistFreq = sampleFreq/2
    b, a = signal.butter(6, cutoffFreq/NyquistFreq, 'lowpass', fs = sampleFreq)
    dataIn = signal.filtfilt(b, a, dataIn)

def filterLowpass(dataIn, cutoffFreq, sampleFreq):
    NyquistFreq = sampleFreq/2
    b, a = signal.butter(2, cutoffFreq/NyquistFreq, 'lowpass', fs = 1)
    return signal.filtfilt(b, a, dataIn)

def filterLowpass_dataframe(dataIn, columnName, cutoffFreq, sampleFreq):
    dataIn2 = dataIn[columnName].to_numpy()
    NyquistFreq = sampleFreq/2
    b, a = signal.butter(6, cutoffFreq/NyquistFreq, 'lowpass', fs = sampleFreq)
    y2 = signal.filtfilt(b, a, dataIn2)
    dataIn[columnName] = pd.DataFrame(data=y2)