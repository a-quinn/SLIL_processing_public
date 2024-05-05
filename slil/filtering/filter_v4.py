# Author: Alastair Quinn 2021
# Used for data from cadaver RE9 and onward
#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib.collections as collections
import slil.common.filtering as ft
from copy import deepcopy
import slil.common.data_configs as dc
import slil.common.plotting_functions as pf
import slil.filtering.filter_v4_functions as frm4f
import slil.common.io as fc
from scipy import signal

from tqdm import tqdm
#from tqdm.notebook import tqdm
### auto-reload modules when editing functions
#%load_ext autoreload
#%autoreload 2
#
## for popout graphs
#%matplotlib qt
#%%

def findBadDataSet(data, markerNum, cutoff):
    badData = {}
    for i in ['x', 'y', 'z']:
        data['marker/1_'+str(markerNum)+'/' + i], badDataPoints = ft.findBadData_np(data['marker/1_'+str(markerNum)+'/' + i], cutoff)
        badData['bdp/1_'+str(markerNum)+'/' + i] = badDataPoints
    return badData
    
def interpolateSet(data, markerNum):
    for i in ['x', 'y', 'z']:
        data['marker/1_'+str(markerNum)+'/' + i] = ft.interpolate_np(data['marker/1_'+str(markerNum)+'/' + i])

def filterSet(data, markerNum, lowpassCutoffFreq, samplingFreq):
    for i in ['x', 'y', 'z']:
        ft.filterLowpass(data['marker/1_'+str(markerNum)+'/' + i], lowpassCutoffFreq, samplingFreq)

samplingFreq = 250 #Hz
lowpassCutoffFreq = 80 #Hz

#print('Plotting.')
def findCycles(data):
    if (type(data['desired/rot/x']) == np.ndarray):
        rot = data['desired/rot/x']
    else:
        rot = data['desired/rot/x'].to_numpy()

    if ((max(rot)-min(rot))==0):
        if (type(data['desired/rot/y']) == np.ndarray):
            rot = data['desired/rot/y']
        else:
            rot = data['desired/rot/y'].to_numpy()
    #rot = rot - ((max(rot)-min(rot))/2)
    cycles = np.argwhere(np.diff(np.sign(rot))).flatten()
    # every second interseciont of 0
    cycles = cycles[range(0,len(cycles),2)]
    def chunks(lst, n):
        for i in range(0, len(lst), 1):
            yield lst[i:i + n]
    cycles = [x for x in list(chunks(cycles,2)) if len(x)==2]
    return cycles
    #indexThirds = int((len(data_1['time'])-(len(data_1['time'])%3.5))/3.5)


def plotSingleAxis(dataIn, badPoints, markers, useCycles, showBadMarkerPoints = False, axis = 'x'):
    plt.figure(figsize=[8,8])
    ax1 = plt.subplot(1,1,1)
    for data in dataIn:
        idx = findCycles(data)

        if (idx == [] or not useCycles):
            for markerNum in markers:
                key = str(markerNum)+'/' + axis

                ax1.plot(data['time'], data['marker/1_'+key], linewidth=0.9, alpha=0.9, label=axis)
                    
                if (showBadMarkerPoints):
                    overlapTime = data['time']
                    collection = collections.BrokenBarHCollection.span_where(
                        overlapTime,
                        ymin=ax1.get_ylim()[0],
                        ymax=ax1.get_ylim()[1],
                        where=badPoints['bdp/1_'+key],
                        facecolor='green',
                        alpha=0.7)
                    ax1.add_collection(collection)
        else:
            for markerNum in markers:
                key = str(markerNum)+'/' + axis

                for i in (range(0,len(idx))):
                    ax1.plot(data['time'][idx[0][0]:idx[0][1]],data['marker/1_'+key][idx[i][0]:idx[i][1]], linewidth=0.9, alpha=0.9, label=axis + "_"+str(i))
                    
                if (showBadMarkerPoints):
                    overlapTime = data['time']
                    alphaDiv = (1.0/(len(badPoints['bdp/1_'+key]) * len(idx)* len(idx))/2)
                    #for i in (range(0,len(badPoints['bdp/1_'+key]))):
                    for ii in (range(0,len(idx))):
                        collection = collections.BrokenBarHCollection.span_where(
                            overlapTime[idx[0][0]:idx[0][1]],
                            ymin=ax1.get_ylim()[0],
                            ymax=ax1.get_ylim()[1],
                            where=badPoints['bdp/1_'+key][idx[ii][0]:idx[ii][1]],
                            facecolor='green',
                            alpha=0.7)
                        ax1.add_collection(collection)

    ax1.spines['right'].set_visible(False) # Hide the right and top spines
    ax1.spines['top'].set_visible(False)
    #ax1.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
    #ax1.legend(edgecolor='None')
    plt.title('1_'+str(markerNum))
    plt.xlabel('Time (sec)')
    plt.ylabel('Position (mm)', fontsize=10, position = (-0.8,0.5))

def plotOverlappingRaw(dataIn, badPoints, markers, showBadMarkerPoints = False):
    plt.figure(figsize=[8,8])
    ax1 = plt.subplot(3,1,1)
    ax2 = plt.subplot(3,1,2)
    ax3 = plt.subplot(3,1,3)
    for data in dataIn:
        idx = findCycles(data)

        for markerNum in markers:
            for i in (range(0,len(idx))):
                ax1.plot(data['time'][idx[0][0]:idx[0][1]],data['marker/1_'+str(markerNum+1)+'/x'][idx[i][0]:idx[i][1]], linewidth=0.9, alpha=0.9, label="d1_x_"+str(i))
                ax2.plot(data['time'][idx[0][0]:idx[0][1]],data['marker/1_'+str(markerNum+1)+'/y'][idx[i][0]:idx[i][1]], linewidth=0.9, alpha=0.9, label="d1_y_"+str(i))
                ax3.plot(data['time'][idx[0][0]:idx[0][1]],data['marker/1_'+str(markerNum+1)+'/z'][idx[i][0]:idx[i][1]], linewidth=0.9, alpha=0.9, label="d1_z_"+str(i))
                #ax1.plot(data2['time'].iloc[idx2[0][0]:idx2[0][1]],data2['marker/1_'+str(markerNum+1)+'/x'].iloc[idx2[i][0]:idx2[i][1]], linewidth=0.9, alpha=0.9, label="d2_x_"+str(i))
                #ax1.plot(data['time'].iloc[idx[0][0]:idx[0][1]],data_1_original['marker/1_'+str(markerNum+1)+'/x'].iloc[idx[i][0]:idx[i][1]], linewidth=0.7, alpha=0.9, label="d1_original_x_"+str(i))
                #ax1.plot(data['time'].iloc[idx[0][0]:idx[0][1]],data_1_original['marker/1_'+str(markerNum+1)+'/y'].iloc[idx[i][0]:idx[i][1]], linewidth=0.7, alpha=0.9, label="d1_original_y_"+str(i))
                #ax1.plot(data['time'].iloc[idx[0][0]:idx[0][1]],data_1_original['marker/1_'+str(markerNum+1)+'/z'].iloc[idx[i][0]:idx[i][1]], linewidth=0.7, alpha=0.9, label="d1_original_z_"+str(i))

            if (showBadMarkerPoints):
                overlapTime = data['time']
                alphaDiv = (1.0/(len(badPoints) * len(idx)* len(idx))/2)
                for i in (range(0,len(badPoints[markerNum]))):
                    for ii in (range(0,len(idx))):
                        collection = collections.BrokenBarHCollection.span_where(
                            overlapTime[idx[0][0]:idx[0][1]], ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], where=badPoints[markerNum][i][idx[ii][0]:idx[ii][1]], facecolor='green', alpha=alphaDiv)
                        ax1.add_collection(collection)

    ax1.spines['right'].set_visible(False) # Hide the right and top spines
    ax1.spines['top'].set_visible(False)
    #ax1.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
    #ax1.legend(edgecolor='None')
    plt.title('1_'+str(markerNum+1))
    plt.xlabel('Time (sec)')
    plt.ylabel('Position (mm)', fontsize=10, position = (-0.8,0.5))

# Below code looks at indiviual signals for filtering

#cadaverID = 11537
#modelInfo = dc.load(cadaverID)
#dataFiles = modelInfo['trialsRawData']
##dataFiles = modelInfo['trialsRawData_only_scaffold']
##dataFiles = modelInfo['trialsRawData_only_static']
##dataFiles = modelInfo['trialsRawData_only_normal']
#dataFiles = [ '\\normal_fe_40_40\\log1' ]
#dataFiles = [ '\\normal_ur_30_30\\log1' ]
#numMarkers = modelInfo['numMarkers']
#
#dataOutputDir = modelInfo['dataOutputDir']
#dataInputDir = modelInfo['dataInputDir']

##%%
#rint('Interpolating & filtering data')
# = 0
#or dataInStr in dataFiles:
#   n += 1
#   print('Loading (' + str(n) + '/' + str(len(dataFiles)) + ') ' + str(dataInStr))
#   dataIn = pf.convertToNumpyArrays(pf.loadTest2(dataInputDir + dataInStr))
#   print('Processing...')
#   badPoints = {}
#
#   markerNum = 6
#   axis = 'y'
#   
#   #cutAt = findCycles(dataIn)[0][1] + 20
#   #cutAt1 = 0
#   #cutAt2 = 30000
#   #dataIn['desired/rot/x'] = dataIn['desired/rot/x'][cutAt1:cutAt2]
#   #dataIn['desired/rot/y'] = dataIn['desired/rot/y'][cutAt1:cutAt2]
#   #dataIn['time'] = dataIn['time'][cutAt1:cutAt2]
#   #for i in (range(1,numMarkers+1)):
#   #    dataIn['marker/1_'+str(i)+'/x'] = dataIn['marker/1_'+str(i)+'/x'][cutAt1:cutAt2]
#   #    dataIn['marker/1_'+str(i)+'/y'] = dataIn['marker/1_'+str(i)+'/y'][cutAt1:cutAt2]
#   #    dataIn['marker/1_'+str(i)+'/z'] = dataIn['marker/1_'+str(i)+'/z'][cutAt1:cutAt2]
#
#   if (True):
#       print(str(markerNum))
#       #plotSingleAxis([dataIn], badPoints, [markerNum], False, axis)
#       #plotOverlappingRaw([dataIn], badPoints, [markerNum], False)
#       name = 'marker/1_'+str(markerNum)+'/' + axis
#
#       
#def removeBadFrames (y, limit = 0.0002, withinReturn = 0.0005):
#       limit = 0.0005
#       withinReturn = 0.0005
#       y = deepcopy( dataIn[name]) # unit: meters
#       
#       upperLimit = limit
#       lowerLimit = -1 * upperLimit
#
#       within = 0.000001
#
#       badDataPoints = np.array(np.zeros(len(y)), dtype=bool)
#       indexOfSame = 0
#       
#       for i in tqdm(range(0,len(y)), desc='Lookign for bad frames'):
#           if (i >= len(y) - 2 or indexOfSame != 0):
#               if (indexOfSame > 0):
#                   indexOfSame -= 1
#               continue
#           #if (y[i+1] == y[i+2]):
#           #if (y[i+1] == y[i+2] or y[i+1] < y[i+2] + within or y[i+1] > y[i+2] - within):      
#
#           p = y[i+1]
#           count = 0
#           # if outside bounds
#           if (y[i] + upperLimit < p or y[i] + lowerLimit > p):
#               count += 1
#               #print('Found Bad Frame ({}): {}'.format(count, i))
#               # find all consecutive values the same
#               indexOfSame = 1
#               for ii in range(1, min(10000, len(y) - i - 3)):
#                   # if inside bounds
#                   p1 = y[i + ii]
#                   p2 = y[i + ii + 1]
#                   if (p1 == p2 or 
#                   (p1 + within > p2 and p1 - within < p2) or
#                   not (y[i] + withinReturn > p2 and y[i] - withinReturn < p2)):
#                   #if (p1 == p2):
#                       indexOfSame += 1
#                   else:
#                       break
#
#               avg = (y[i] + y[i + indexOfSame + 1]) /2
#               for ii in range(indexOfSame + 1):
#                   y[i+ii + 1] = avg
#                   badDataPoints[i + ii + 1] = True
#       #return y
#           #dataIn[name] = y
#           dataTemp = deepcopy(dataIn)
#           badPoints['bdp/1_'+str(markerNum)+'/' + axis] = badDataPoints
#           dataTemp[name] = y
#
#        diff = {}
#        for ii in range(len(dataIn[name])-1):
#            y = dataIn[name]
#            if (y[ii] > y[ii+1] + 0.01):
#                diff[ii+1] = y[ii] - y[ii+1]
#
#                print('bad frame: {} = {} sec\n\ty[i]: {} y[i+1]: {}'.format(
#                    ii+1,
#                    (ii+1)*0.004,
#                    y[ii],
#                    y[ii+1]
#                    )
#                    )
#        
#        offset = next(iter(diff.values()))
#        for ii in diff:
#            dataIn[name][ii] = dataIn[name][ii] + offset
#        print('Plotting...')

#%%
# fresh attempt 2022 04 22
import ray

def run(experiments, models, runOnlyOne = False, clusterAddress = ''):

    if not ray.is_initialized():
        ray.init(
            #_node_ip_address=clusterAddress, \
            address=clusterAddress, \
            runtime_env={ \
                "working_dir": ".", \
                "excludes": [
                    #"/cadaver results 03_08_2021.zip", \
                    "/slil/rayTest/*", \
                    "/slil/3-matic_backup/*", \
                    "/data/*", \
                    "*.zip", \
                    "*.PNG", \
                    "*.png", \
                    "*.whl", \
                    "*.pptx", \
                    "*.xlsx", \
                    "*.log", \
                    "*.m", \
                    "**/__pycache__", \
                    #"./*.*",
                    ],
                }, \
            ignore_reinit_error=True,
            )

    tasksRunning = []
    for cadaverID in tqdm(experiments, desc='Experiment'):
        modelInfo = dc.load(cadaverID)
        dataFiles = modelInfo['trialsRawData']
        #dataFiles = modelInfo['trialsRawData_only_static']
        dataFiles = []
        for model in models:
            dataFiles += modelInfo['trialsRawData_only_' + model]
        if runOnlyOne: # used for testing
            dataFiles = [dataFiles[0]]
        #dataFiles = [ '\\normal_fe_40_40\\log1' ]
        #dataFiles = [ '\\normal_ur_30_30\\log2' ]
        #dataFiles = [ '\\scaffold_ur_30_30\\log2' ]
        #dataFiles = [ '\\scaffold_fe_40_40\\log2' ]
        #dataFiles = [ '\\cut_static\\log_fromFE' ]
        numMarkers = modelInfo['numMarkers']

        dataOutputDir = modelInfo['dataOutputDir']
        dataInputDir = modelInfo['dataInputDir']

        des = "Trials started: {}".format(cadaverID)
        for n, dataInStr in tqdm(enumerate(dataFiles), desc=des, leave=False):
            dataInRaw = pf.loadTest2(dataInputDir + dataInStr)
            tasksRunning.append(filterIt.remote(dataInRaw, dataInStr, dataOutputDir, numMarkers, disableProgressBar = True))

    while tasksRunning:
        finished, tasksRunning = ray.wait(tasksRunning, num_returns=1, timeout=None)
        for task in finished:
            dataIn, dataOutputDir, dataInStr = ray.get(task)

            fc.writeC3D(
                dataIn,
                markersUsed = 12,
                outputFile = dataOutputDir + dataInStr + '.c3d',
                isPureMarker = False,
                verbose = False)

            #print('result:', result)
        print('Tasks remaining:', len(tasksRunning))
    print('Finished tasks.')

@ray.remote
def filterIt(dataInRaw, dataInStr, dataOutputDir, numMarkers, disableProgressBar):
    if True:
        if True:            
            #tqdm.write('Loading (' + str(n) + '/' + str(len(dataFiles)) + ') ' + str(dataInStr))
            #dataIn = pf.convertToNumpyArrays(pf.loadTest2(dataInputDir + dataInStr))
            dataIn = pf.convertToNumpyArrays(dataInRaw)
            #print('Processing...')
            badPoints = {}

            markerNum = 11
            des = "Trial: {}".format(dataInStr)
            for markerNum in tqdm(range(1, numMarkers + 1), desc=des, leave=False, disable=disableProgressBar):
            #for markerNum in range(1, numMarkers+1):
            #if True:
                axis = 'z'
                for axis in tqdm(['x', 'y', 'z'], desc='Axis', disable=True):
                #if True:
                    
                    #print(str(markerNum))
                    #plotSingleAxis([dataIn], badPoints, [markerNum], False, axis)
                    #plotOverlappingRaw([dataIn], badPoints, [markerNum], False)
                    name = 'marker/1_'+str(markerNum)+'/' + axis

                    #find bad points
                    limit = 0.0005
                    withinReturn = 0.0005
                    y = deepcopy( dataIn[name]) # unit: meters
                    yOriginal = deepcopy(y)
                    
                    upperLimit = limit
                    lowerLimit = -1 * upperLimit

                    within = 0.000001

                    indexOfSame = 0

                    velocityLimit = 5 # meters/sec
                    
                    yAdjustRepeating = frm4f.fun3(y)
                    y = yAdjustRepeating

                    velocityLimit1 = 40 # meters/sec
                    [y1,
                    yDot,
                    badDataPoints,
                    badDataPointsPos,
                    badDataPointsNeg,
                    badDataPointsX,
                    badDataPointsY] = frm4f.fun1(y, dataIn['time'], velocityLimit = velocityLimit1)
                    maxYDot = max(abs(yDot))
                    if (max(abs(yDot)) > 5.0):
                        #print("Max yDot: {}".format(max(abs(yDot))))

                        y1 = y
                        for i in range(0,2):
                            valLimitLargest = max(abs(yDot.max()) , abs(yDot.min()))
                            valLimitLargest = valLimitLargest * 0.99
                            [y1,
                            yDot,
                            badDataPoints,
                            badDataPointsPos,
                            badDataPointsNeg,
                            badDataPointsX,
                            badDataPointsY] = frm4f.fun1(y1, dataIn['time'], velocityLimit = valLimitLargest)


                        velocityLimit2 = 5 # meters/sec
                        [y2,
                        yDot2,
                        badDataPoints2,
                        badDataPoints2Pos,
                        badDataPoints2Neg,
                        badDataPoints2X,
                        badDataPoints2Y] = frm4f.fun1(y1, dataIn['time'], velocityLimit = velocityLimit2)

                        #yRemoveIgnore = fun2(y2)
                        #y2 = yRemoveIgnore
                        
                        velocityLimit3 = 5 # meters/sec
                        [y3,
                        yDot3,
                        badDataPoints3,
                        badDataPoints3Pos,
                        badDataPoints3Neg,
                        badDataPoints3X,
                        badDataPoints3Y] = frm4f.fun1(y2, dataIn['time'], velocityLimit = velocityLimit3)

                        yRemoveBadData = np.array(y1, dtype=float)
                        offset = 0.0
                        offsetPrev = 0.0
                        for ind, a in enumerate(y1):
                            if ind >= len(y1)-1:
                                break
                            if badDataPoints[ind]:
                                offset = offset - (yRemoveBadData[ind+1] - yRemoveBadData[ind])
                            yRemoveBadData[ind] = yRemoveBadData[ind] + offsetPrev
                            offsetPrev = offset
                        #yRemoveBadData = ft.interpolate_np(yRemoveBadData)
                    else:
                        y3 = y
                        yDot2 = yDot
                        yDot3 = yDot
                        badDataPoints2 = badDataPoints
                        badDataPoints2X = badDataPointsX
                        badDataPoints2Y = badDataPointsY
                        badDataPoints3 = badDataPoints
                        badDataPoints3X = badDataPointsX
                        badDataPoints3Y = badDataPointsY

                    y3 = frm4f.fun2(y3)

                    samplingFreq = 250 #Hz
                    lowpassCutoffFreq = 0.5 #Hz
                    #yfilt = ft.filterLowpass(y, lowpassCutoffFreq, samplingFreq)

                    #filt = frm4f.CriticallyDampedFilter(sampFreq = samplingFreq, cutoffFreq=lowpassCutoffFreq)
                    #y3In = deepcopy(y3)
                    #y3In = np.insert(y3In, 0, [y3In[0] for i in range(10)], axis=0)
                    #y3In = np.append(y3In, [y3In[-1] for i in range(10)], axis=0)
                    #yfilt = filt.runFilter(y3In)
                    #yfilt = yfilt[10:-10]
                    
                    NyquistFreq = (samplingFreq)/2
                    b, a = signal.butter(2, lowpassCutoffFreq/NyquistFreq, 'lowpass', fs = 1.0)
                    y3In = deepcopy(y3)
                    y3In = np.insert(y3In, 0, [y3In[0] for i in range(20)], axis=0)
                    y3In = np.append(y3In, [y3In[-1] for i in range(20)], axis=0)
                    yfilt = signal.filtfilt(b, a, y3In)
                    yfilt = yfilt[20:-20]

                    #dt = dataIn['time'][1] - dataIn['time'][0]
                    #yDot3 = np.array(np.zeros(len(y2)-1), dtype=float) # unit: meters/second
                    #for i in range(0, len(y2) - 1):
                    #    yDot3[i] = (y2[i+1] - y2[i])/dt
                

                    if False:
                    #if True:
                        #print("Plotting...")

                        # For visualisation, BrokenBarHCollection needs at least two points
                        badDataPointsVis = np.array(badDataPoints)
                        for i, a in reversed(list(enumerate(badDataPointsVis))):
                            if a == True:
                                badDataPointsVis[i+1] = True

                        badDataPoints2Vis = np.array(badDataPoints2)
                        for i, a in reversed(list(enumerate(badDataPoints2Vis))):
                            if a == True:
                                badDataPoints2Vis[i+1] = True

                        rep = frm4f.getNumRepeating(yOriginal)
                        repInv = np.logical_not(rep)
                        mp = len(np.where(np.isclose(yOriginal[repInv], yfilt[repInv], 0.001, 0.0001))[0])
                        #if len(badDataPoints3Y) > 0:
                        r = len(np.where(rep)[0])
                        if (r == len(yOriginal)):
                            r = len(yOriginal) - 1
                        #print("Repeating points: {}".format(r))
                        #print("Plotting: {}".format(
                        #        dataInStr + ' /1_' + str(markerNum) + '/' + axis +
                        #        ' Matching points: ' + str(mp) + ' of ' + str(len(y))) +
                        #        ' with missing ' + str((mp / (len(yOriginal) - r))*100)+ '%')
                        
                        # if less than 95% of matching points found (excluding repeating points)
                        if ((mp / (len(yOriginal) - r)) * 100.0) < 95.0:
                        #if True:
                            #print("Matching points: {}".format(mp))
                            #print("Plotting: {}".format(
                            #    dataInStr + ' /1_' + str(markerNum) + '/' + axis +
                            #    ' Matching points: ' + str(mp) + ' of ' + str(len(y))) +
                            #    ' with missing ' + str((mp / (len(yOriginal) - r))*100)+ '%')
                            
                            if True:
                                plt.figure(figsize=[8,8])
                                ax1 = plt.subplot(1,1,1)
                                
                                plt.title(cadaverID + ' ' + dataInStr + ' /1_'+str(markerNum) + '/' + axis + ' MP: ' + str(mp) + ' is ' + str((mp / (len(yOriginal) - r))*100)+ '%')

                                x = np.array(range(len(dataIn['time'])), dtype=float)
                                ax1.plot(x, yOriginal, alpha=0.9, label='y Original')
                                ax1.plot(x, y3, alpha=0.9, label='y Final')
                                ax1.plot(x, yfilt, alpha=0.5, label='y Final Filtered')
                                ax1.plot(x[np.where(badDataPoints)[0]], y3[np.where(badDataPoints)[0]], 'o', alpha=0.2, label='Bad points')
                                ax1.set_ylim( bottom = min(y3), top = max(y3) )
                                #ax1.legend(loc='right')
                                ax1.set(ylabel='Position (m)')

                                plt.xlabel('Time (sec)')
                                plt.show()
                            else:
                                plt.figure(figsize=[8,8])
                                ax1 = plt.subplot(7,1,1)
                                ax2 = plt.subplot(7,1,2)
                                ax3 = plt.subplot(7,1,3)
                                ax4 = plt.subplot(7,1,4)
                                ax5 = plt.subplot(7,1,5)
                                ax6 = plt.subplot(7,1,6)
                                ax7 = plt.subplot(7,1,7)
                                
                                x = dataIn['time']
                                x = np.array(range(len(dataIn['time'])), dtype=float)
                                ax1.plot(x, yOriginal, linewidth=0.9, alpha=0.9, label='Raw')
                                #collection = collections.BrokenBarHCollection.span_where(
                                #    dataIn['time'],
                                #    ymin=ax1.get_ylim()[0],
                                #    ymax=ax1.get_ylim()[1],
                                #    where=yIgnore,
                                #    facecolor='red',
                                #    alpha=0.7)
                                #ax1.add_collection( collection )
                                #ax1.plot(dataIn['time'], yRemoveIgnore, alpha=0.9, label='y Remove Ignore')
                                ax1.legend(loc='right')
                                ax1.set(ylabel='Position (m)')
                                plt.title(dataInStr + ' /1_'+str(markerNum) + '/' + axis + ' Matching points: ' + str(mp))

                                x = dataIn['time']
                                x = np.array(range(len(dataIn['time'])), dtype=float)
                                ax2.plot(x[:-1], yDot, linewidth=0.9, alpha=0.9, label='Raw')
                                collection = collections.BrokenBarHCollection.span_where(
                                    x,
                                    ymin=ax2.get_ylim()[0],
                                    ymax=ax2.get_ylim()[1],
                                    where=badDataPointsVis,
                                    facecolor='green',
                                    alpha=0.7)
                                ax2.add_collection(collection)
                                ax2.plot(badDataPointsX, badDataPointsY, 'o', alpha=0.9, label='Raw bad points')
                                ax2.hlines([velocityLimit1, -1.0* velocityLimit1], 0, 1, transform=ax2.get_yaxis_transform(), colors='r')
                                ax2.legend(loc='right')
                                ax2.set(ylabel='Velocity (m/sec)')

                                #ax3.plot(dataIn['time'], yRemoveIgnore, alpha=0.9, label='y Remove Ignore')
                                collection = collections.BrokenBarHCollection.span_where(
                                    dataIn['time'],
                                    ymin=ax3.get_ylim()[0],
                                    ymax=ax3.get_ylim()[1],
                                    where=badDataPointsVis,
                                    facecolor='green',
                                    alpha=0.7)
                                ax3.add_collection( collection )
                                x = dataIn['time']
                                x = np.array(range(len(dataIn['time'])), dtype=float)
                                #ax3.plot(dataIn['time'], yfilt, linewidth=0.9, alpha=0.9, label='Filtered')
                                ax3.plot(x, y1, linewidth=0.9, alpha=0.9, label='y Offset Jumps')
                                #ax3.plot(dataIn['time'], yRemoveBadData, linewidth=0.9, alpha=0.9, label='Bad points offset')
                                #ax3.plot(dataIn['time'][doubleJump], yfilt[doubleJump], 'o', alpha=0.9, label='Double jumps')
                                ax3.plot(x[np.where(badDataPoints)[0]], y1[np.where(badDataPoints)[0]], 'o', alpha=0.2, label='Bad points')
                                #ax3.plot(dataIn['time'][np.where(badDataPoints)[0]], yfilt[np.where(badDataPoints)[0]], 'o', alpha=0.2, label='Bad points')
                                ax3.legend(loc='right')
                                ax3.set(ylabel='Position (m)')

                                x = dataIn['time']
                                x = np.array(range(len(dataIn['time'])), dtype=float)
                                ax4.plot(x[:-1], yDot2, linewidth=0.9, alpha=0.9, label='Raw')
                                collection = collections.BrokenBarHCollection.span_where(
                                    dataIn['time'],
                                    ymin=ax4.get_ylim()[0],
                                    ymax=ax4.get_ylim()[1],
                                    where=badDataPoints2Vis,
                                    facecolor='green',
                                    alpha=0.7)
                                ax4.add_collection(collection)
                                ax4.plot(badDataPoints2X, badDataPoints2Y, 'o', alpha=0.9, label='Raw bad points')
                                ax4.hlines([velocityLimit2, -1.0* velocityLimit2], 0, 1, transform=ax4.get_yaxis_transform(), colors='r')
                                ax4.legend(loc='right')
                                ax4.set(ylabel='Velocity (m/sec)')

                                x = dataIn['time']
                                x = np.array(range(len(dataIn['time'])), dtype=float)
                                ax5.plot(x, y1, alpha=0.9, label='y Offset Jumps')
                                ax5.plot(x, y2, alpha=0.9, label='y Offset Jumps 2')
                                #collection = collections.BrokenBarHCollection.span_where(
                                #    x,
                                #    ymin=ax5.get_ylim()[0],
                                #    ymax=ax5.get_ylim()[1],
                                #    where=badDataPointsVis,
                                #    facecolor='green',
                                #    alpha=0.7)
                                #ax5.add_collection( collection )
                                ax5.plot(x[np.where(badDataPoints)[0]], y2[np.where(badDataPoints)[0]], 'o', alpha=0.2, label='Bad points')
                                ax5.legend(loc='right')
                                ax5.set(ylabel='Position (m)')

                                ax6.plot(dataIn['time'][:-1], yDot3, linewidth=0.9, alpha=0.9, label='Raw')
                                #collection = collections.BrokenBarHCollection.span_where(
                                #    dataIn['time'],
                                #    ymin=ax6.get_ylim()[0],
                                #    ymax=ax6.get_ylim()[1],
                                #    where=badDataPoints2Vis,
                                #    facecolor='green',
                                #    alpha=0.7)
                                #ax6.add_collection(collection)
                                ax6.plot(badDataPoints3X, badDataPoints3Y, 'o', alpha=0.9, label='Raw bad points')
                                ax6.hlines([velocityLimit3, -1.0* velocityLimit3], 0, 1, transform=ax6.get_yaxis_transform(), colors='r')
                                ax6.legend(loc='right')
                                ax6.set(ylabel='Velocity (m/sec)')

                                x = np.array(range(len(dataIn['time'])), dtype=float)
                                ax7.plot(x, yOriginal, alpha=0.9, label='y Original')
                                ax7.plot(x, y3, alpha=0.9, label='y Final')
                                ax7.plot(x, yfilt, alpha=0.5, label='y Final Filtered')
                                ax7.plot(x[np.where(badDataPoints)[0]], y2[np.where(badDataPoints)[0]], 'o', alpha=0.2, label='Bad points')
                                ax7.set_ylim( bottom = min(y3), top = max(y3) )
                                ax7.legend(loc='right')
                                ax7.set(ylabel='Position (m)')

                                plt.xlabel('Time (sec)')
                                plt.show()
                    else:
                        dataIn[name] = deepcopy(yfilt)
            
            #fc.writeC3D(
            #    dataIn,
            #    markersUsed = 12,
            #    outputFile = dataOutputDir + dataInStr + '.c3d',
            #    isPureMarker = False,
            #    verbose = False)
            
            return dataIn, dataOutputDir, dataInStr


                #plt.figure(figsize=[8,8])
                #plt.title('1_'+str(markerNum))
                #ax1 = plt.subplot(2,1,1)
                #ax2 = plt.subplot(2,1,2)
                #ax1.plot(dataIn['time'], yfilt, linewidth=0.9, alpha=0.9, label='Filtered')
                #ax1.legend(loc='best')
                #ax1.set(ylabel='Position (m)')
                #
                #ax2.plot(dataIn['time'][:-1], yDot3, linewidth=0.9, alpha=0.9, label='Raw')
                #ax2.hlines([velocityLimit, -1.0* velocityLimit], 0, 1, transform=ax2.get_yaxis_transform(), colors='r')
                #ax2.legend(loc='best')
                #ax2.set(ylabel='Velocity (m/sec)')
                #plt.xlabel('Time (sec)')
                #plt.show()
        


#%%
#plotSingleAxis([dataIn, dataTemp], badPoints, [markerNum], False, True, axis)

        #badPoints.update(findBadDataSet(dataIn, i, 0.06))
        #plotSingleAxis([dataIn], badPoints, [markerNum], True, axis)
#        #plotOverlappingRaw([dataIn], badPoints, [markerNum], False)
#
#        interpolateSet(dataIn, i)
#        plotSingleAxis([dataIn], badPoints, [markerNum], False, axis)
#        #plotOverlappingRaw([dataIn], badPoints, [markerNum], False)
#
#        filterSet(dataIn, i, lowpassCutoffFreq, samplingFreq)
#        plotSingleAxis([dataIn], badPoints, [markerNum], False, axis)
#        #plotOverlappingRaw([dataIn], badPoints, [markerNum], False)
#
#        #badPoints.extend(findBadDataSet(data_2, i, 0.03))
#        #interpolateSet(data_2, i)
#        #filterSet(data_2, i, lowpassCutoffFreq, samplingFreq)
#        dataIn['marker/1_'+str(i)+'/x'] = dataIn['marker/1_'+str(i)+'/x'] * 100
#        dataIn['marker/1_'+str(i)+'/y'] = dataIn['marker/1_'+str(i)+'/y'] * 100
#        dataIn['marker/1_'+str(i)+'/z'] = dataIn['marker/1_'+str(i)+'/z'] * 100


    #plotOverlappingRaw([dataIn], badPoints, [markerNum], False)
    #dataIn.to_csv( dataOutputDir + dataInStr + '.csv', index=False)
    #pf.checkAndCreateFolderExists(dataOutputDir + dataInStr)
    #writeC3D(dataIn, markersUsed = 12, outputFile = dataOutputDir + dataInStr + '_raw.c3d')
    #writeC3D(dataIn, markersUsed = 12, outputFile = dataOutputDir + dataInStr + '_rm_bad.c3d')
    #writeC3D(dataIn, markersUsed = 12, outputFile = dataOutputDir + dataInStr + '_interp.c3d')
    #writeC3D(dataIn, markersUsed = 12, outputFile = dataOutputDir + dataInStr + '.c3d')
    
#print('Finished processing trails, .log files to .c3d files.')

if __name__ == "__main__":
    experiments = [
    '11525',
    '11526',
    '11527',
    '11534',
    '11535',
    '11536',
    '11537',
    '11538',
    '11539'
    ]
    models = [ 'normal', 'cut', 'scaffold'] #
    run(experiments, models)
# %%

#experiments = [
#'11526',
#]
#models = [ 'normal', 'cut', 'scaffold']
#run(experiments, models)
# %%
