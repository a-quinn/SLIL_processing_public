# Author: Alastair Quinn 2021
# Used for data from cadaver RE9 and onward
#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib.collections as collections
import slil.common.filtering as ft
#from scipy import signal
import slil.common.c3d_modified as c3d
import sys
import copy
from itertools import groupby
import slil.common.data_configs as dc
import slil.common.plotting_functions as pf

#%%

# 
# from 3-matic to opensim
# rot: -1.57 0 -1.57
# trans: -y(3-matic) z(3-matic) -x(3-matic)
# 
# Markers
# Radius: 5, 8, 9
# 3rd MetCarp: 2, 3, 6
# Lunate: 0, 1, 4
# Scaphoid: 7, 10, 11

#print('Interpolating data.')

#ft.interpolateMoCap(data_1, 'marker/1_2/x', 0.02)
def findBadDataSet(data, markerNum, cutoff):
    badData = []
    badData.append(ft.findBadData(data,'marker/1_'+str(markerNum)+'/x', cutoff))
    badData.append(ft.findBadData(data,'marker/1_'+str(markerNum)+'/y', cutoff))
    badData.append(ft.findBadData(data,'marker/1_'+str(markerNum)+'/z', cutoff))
    return badData
    
def interpolateSet(data, markerNum):
    ft.interpolate(data,'marker/1_'+str(markerNum)+'/x')
    ft.interpolate(data,'marker/1_'+str(markerNum)+'/y')
    ft.interpolate(data,'marker/1_'+str(markerNum)+'/z')

def removeBadDataSet(data, markerNum, badpoints):
    axes = ['x', 'y', 'z']
    for axis in range(len(axes)):
        # find data which repeats for more then 20 points
        groups = [(k, sum(1 for _ in g)) for k, g in groupby(badpoints[axis])]
        cursor = 0
        for k, l in groups:
            if k == True and l <= 20:
                badpoints[axis][cursor : cursor + l] = False
            cursor += l

        dataIn = data['marker/1_'+str(markerNum)+'/'+str(axes[axis])].to_numpy()
        dataIn[badpoints[axis]]=np.nan
        data['marker/1_'+str(markerNum)+'/'+str(axes[axis])] = pd.DataFrame(data=dataIn)
        #dataIn = data['marker/1_'+str(markerNum)+'/y'].to_numpy()
        #dataIn[badpoints[1]]=np.nan
        #data['marker/1_'+str(markerNum)+'/y'] = pd.DataFrame(data=dataIn)
        #dataIn = data['marker/1_'+str(markerNum)+'/z'].to_numpy()
        #dataIn[badpoints[2]]=np.nan
        #data['marker/1_'+str(markerNum)+'/z'] = pd.DataFrame(data=dataIn)

# Filtering
#print('Filtering data.')

#data_1['marker/1_2/x'] = ft.filterLowpass(data_1['marker/1_2/x'], lowpassCutoffFreq, samplingFreq)
def filterSet(data, markerNum, lowpassCutoffFreq, samplingFreq):
    ft.filterLowpass_dataframe(data,'marker/1_'+str(markerNum)+'/x', lowpassCutoffFreq, samplingFreq)
    ft.filterLowpass_dataframe(data,'marker/1_'+str(markerNum)+'/y', lowpassCutoffFreq, samplingFreq)
    ft.filterLowpass_dataframe(data,'marker/1_'+str(markerNum)+'/z', lowpassCutoffFreq, samplingFreq)

#filterSet( 1, lowpassCutoffFreq, samplingFreq)
#filterSet( 2, lowpassCutoffFreq, samplingFreq)
#filterSet( 3, lowpassCutoffFreq, samplingFreq)
def writeC3D(data_1, markersUsed = 12, outputFile = 'random_data.c3d'):
    writer = c3d.Writer(point_scale=-0.1, point_rate=250)
    pointNum = len(data_1['marker/1_2/x'])
    print('Writing file: ' + str(outputFile))
    frames = []
    lables = []
    x = np.empty(( markersUsed, pointNum))
    y = np.empty(( markersUsed, pointNum))
    z = np.empty(( markersUsed, pointNum))
    for ii in range( markersUsed ):
        x[ii] = data_1['marker/1_'+str(ii+1)+'/x'].to_numpy()
        y[ii] = data_1['marker/1_'+str(ii+1)+'/y'].to_numpy()
        z[ii] = data_1['marker/1_'+str(ii+1)+'/z'].to_numpy()

    for i in range(pointNum):
        points = np.empty((markersUsed, 5))
        
        for ii in range( markersUsed ):
            #points[ii] = [x[ii][i], -1.0 * z[ii][i], y[ii][i], ii+1, 0]
            #points[ii] = [ y[ii][i], -1.0 * z[ii][i], -1.0 * (x[ii][i]-100), ii+1, 0] # works with opensim for RE9
            points[ii] = [ y[ii][i], -1.0 * z[ii][i], x[ii][i], ii+1, 0] # RE10
            #points[ii] = [ y[ii][i], -1.0 * z[ii][i], (x[ii][i]-100), ii+1, 0]
            #points[ii] = [x[ii][i], y[ii][i], z[ii][i], ii+1, 0]
        frames.append(( points, np.array([[],[]]) ))
    writer.add_frames(frames)
    
    for i in range( markersUsed ):
        lables.append('marker'+str(i))

    with open(outputFile, 'wb') as h:
        writer.write(h, lables)

#writeC3D(data_1, markersUsed = 12, outputFile = 'random_data.c3d')

dataToConvert = [
    r'\cut_fe_40_40\log',
    r'\cut_fe_40_40\log2',
    r'\cut_ur_30_30\log',
    r'\cut_ur_30_30\log2',
    r'\normal_fe_40_40\log',
    r'\normal_fe_40_40\log2',
    r'\normal_ur_30_30\log',
    r'\normal_ur_30_30\log2',
    #r'\removed_fe_40_40\log',
    #r'\removed_ur_30_30\log',
    r'\scaffold_fe_40_40\log',
    r'\scaffold_fe_40_40\log2',
    r'\scaffold_ur_30_30\log',
    r'\scaffold_ur_30_30\log2',
    r'\scaffold_ur_30_30\log3',
    r'\scaffold_ur_30_30\log4',
]
#for i in range(len(dataToConvert)):
#    print('Writing data ' + str(i) + '.')
#    data_1 = loadTest(dataInputDir + dataToConvert[i] + '.csv', 0)
#    writeC3D(data_1, markersUsed = 12, outputFile = dataOutputDir + dataToConvert[i] + '.c3d')

#  data_1 = loadTest(dataInputDir + '\cut_fe_40_40\log.csv', 0)
#  data_2 = loadTest(dataInputDir + '\cut_fe_40_40\log2.csv', 0)
#  data_1_original = copy.deepcopy(data_1)
#writeC3D(data_1, markersUsed = 12, outputFile = dataOutputDir + '\cut_ur_30_30\log_test.c3d')

#print('Interpolating data 1')
#interpolateSet(data_1, 1, 0.02)
#interpolateSet(data_1, 2, 0.02)
#interpolateSet(data_1, 3, 0.02)

#print('Interpolating data 2')
#interpolateSet(data_2, 1, 0.02)
#interpolateSet(data_2, 2, 0.02)
#interpolateSet(data_2, 3, 0.02)

samplingFreq = 250 #Hz
lowpassCutoffFreq = 80 #Hz
#print('Filtering data 1.')
#filterSet(data_1, 1, lowpassCutoffFreq, samplingFreq)
#filterSet(data_1, 2, lowpassCutoffFreq, samplingFreq)
#filterSet(data_1, 3, lowpassCutoffFreq, samplingFreq)
#print('Filtering data 2.')
#filterSet(data_2, 1, lowpassCutoffFreq, samplingFreq)
#filterSet(data_2, 2, lowpassCutoffFreq, samplingFreq)
#filterSet(data_2, 3, lowpassCutoffFreq, samplingFreq)


#print('Exiting.')
#sys.exit()

print('Plotting.')
def findCycles(data):
    rot = data['desired/rot/x'].to_numpy()
    if ((max(rot)-min(rot))==0):
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

def plotOverlappingRaw(dataIn, badPoints, numMarkers, showBadMarkerPoints = False):
    fig, ax1 = plt.subplots()
    for data in dataIn:
        idx = findCycles(data)

        for markerNum in range(numMarkers):
            for i in (range(0,len(idx))):
                ax1.plot(data['time'].iloc[idx[0][0]:idx[0][1]],data['marker/1_'+str(markerNum+1)+'/x'].iloc[idx[i][0]:idx[i][1]], linewidth=0.9, alpha=0.9, label="d1_x_"+str(i))
                ax1.plot(data['time'].iloc[idx[0][0]:idx[0][1]],data['marker/1_'+str(markerNum+1)+'/y'].iloc[idx[i][0]:idx[i][1]], linewidth=0.9, alpha=0.9, label="d1_y_"+str(i))
                ax1.plot(data['time'].iloc[idx[0][0]:idx[0][1]],data['marker/1_'+str(markerNum+1)+'/z'].iloc[idx[i][0]:idx[i][1]], linewidth=0.9, alpha=0.9, label="d1_z_"+str(i))
                #ax1.plot(data2['time'].iloc[idx2[0][0]:idx2[0][1]],data2['marker/1_'+str(markerNum+1)+'/x'].iloc[idx2[i][0]:idx2[i][1]], linewidth=0.9, alpha=0.9, label="d2_x_"+str(i))
                #ax1.plot(data['time'].iloc[idx[0][0]:idx[0][1]],data_1_original['marker/1_'+str(markerNum+1)+'/x'].iloc[idx[i][0]:idx[i][1]], linewidth=0.7, alpha=0.9, label="d1_original_x_"+str(i))
                #ax1.plot(data['time'].iloc[idx[0][0]:idx[0][1]],data_1_original['marker/1_'+str(markerNum+1)+'/y'].iloc[idx[i][0]:idx[i][1]], linewidth=0.7, alpha=0.9, label="d1_original_y_"+str(i))
                #ax1.plot(data['time'].iloc[idx[0][0]:idx[0][1]],data_1_original['marker/1_'+str(markerNum+1)+'/z'].iloc[idx[i][0]:idx[i][1]], linewidth=0.7, alpha=0.9, label="d1_original_z_"+str(i))

            if (showBadMarkerPoints):
                overlapTime = data['time'].to_numpy()
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

#%%
cadaverID = 11526
modelInfo = dc.load(cadaverID)
dataFiles = modelInfo['trialsRawData']
#dataFiles = modelInfo['trialsRawData_only_scaffold']
#dataFiles = modelInfo['trialsRawData_only_static']
#dataFiles = modelInfo['trialsRawData_only_normal']
numMarkers = modelInfo['numMarkers']

dataOutputDir = modelInfo['dataOutputDir']
dataInputDir = modelInfo['dataInputDir']

print('Interpolating & filtering data')
n = 0
for dataInStr in dataFiles:
    n += 1
    print('Processing (' + str(n) + '/' + str(len(dataFiles)) + ') ' + str(dataInStr))
    dataIn = pf.loadTest2(dataInputDir + dataInStr)
    badPoints = []
    for i in (range(1,numMarkers+1)):
        print(str(i))
        badPoints.append(findBadDataSet(dataIn, i, 0.06))
        interpolateSet(dataIn, i)
        filterSet(dataIn, i, lowpassCutoffFreq, samplingFreq)
        #badPoints.extend(findBadDataSet(data_2, i, 0.03))
        #interpolateSet(data_2, i)
        #filterSet(data_2, i, lowpassCutoffFreq, samplingFreq)
        dataIn['marker/1_'+str(i)+'/x'] = dataIn['marker/1_'+str(i)+'/x'] * 100
        dataIn['marker/1_'+str(i)+'/y'] = dataIn['marker/1_'+str(i)+'/y'] * 100
        dataIn['marker/1_'+str(i)+'/z'] = dataIn['marker/1_'+str(i)+'/z'] * 100
    plotOverlappingRaw([dataIn], badPoints, numMarkers)
    #dataIn.to_csv( dataOutputDir + dataInStr + '.csv', index=False)
#    pf.checkAndCreateFolderExists(dataOutputDir + dataInStr)
    #writeC3D(dataIn, markersUsed = 12, outputFile = dataOutputDir + dataInStr + '_raw.c3d')
    #writeC3D(dataIn, markersUsed = 12, outputFile = dataOutputDir + dataInStr + '_rm_bad.c3d')
    #writeC3D(dataIn, markersUsed = 12, outputFile = dataOutputDir + dataInStr + '_interp.c3d')
#    writeC3D(dataIn, markersUsed = 12, outputFile = dataOutputDir + dataInStr + '.c3d')
    
print('Finished processing trails, .log files to .c3d files.')
#%%
print('Exiting.')
sys.exit()

singleFile = r'\cut_fe_40_40\log'
dataIn = pf.loadTest2(singleFile)

data_1_original = copy.deepcopy(dataIn)
badPoints = []
for i in (range(1,numMarkers+1)):
    print(str(i))
    badPoints.append(findBadDataSet(dataIn, i, 0.03))
    interpolateSet(dataIn, i)
    filterSet(dataIn, i, lowpassCutoffFreq, samplingFreq)
    #removeBadDataSet(dataIn, i, badPoints[i-1])
    dataIn['marker/1_'+str(i)+'/x'] = dataIn['marker/1_'+str(i)+'/x'] * 100
    dataIn['marker/1_'+str(i)+'/y'] = dataIn['marker/1_'+str(i)+'/y'] * 100
    dataIn['marker/1_'+str(i)+'/z'] = dataIn['marker/1_'+str(i)+'/z'] * 100
#plotOverlappingRaw([dataIn], badPoints, numMarkers)
#dataIn.to_csv( dataOutputDir + singleFile + '_test2.csv', index=False,  na_rep='NaN')
writeC3D(dataIn, markersUsed = 12, outputFile = dataOutputDir + singleFile + '_test2.c3d')

plt.show()

print('Exiting.')
sys.exit()

fig, ax1 = plt.subplots()
ax1.plot(data_1['time'],data_1['marker/1_1/x'], linewidth=0.8, alpha=0.9, label="x")
ax1.plot(data_1['time'],data_1['marker/1_1/y'], linewidth=0.8, alpha=0.9, label="y")
ax1.plot(data_1['time'],data_1['marker/1_1/z'], linewidth=0.8, alpha=0.9, label="z")
ax1.plot(data_1['time'],data_1_original['marker/1_1/x'], linewidth=0.6, alpha=0.9, label="x")
ax1.plot(data_1['time'],data_1_original['marker/1_1/y'], linewidth=0.6, alpha=0.9, label="y")
ax1.plot(data_1['time'],data_1_original['marker/1_1/z'], linewidth=0.6, alpha=0.9, label="z")
ax1.spines['right'].set_visible(False) # Hide the right and top spines
ax1.spines['top'].set_visible(False)
#ax1.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
ax1.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
ax1.legend(edgecolor='None')
plt.title('1_1')
plt.xlabel('Time (sec)')
plt.ylabel('Position (mm)', fontsize=10, position = (-0.8,0.5))

fig, ax1 = plt.subplots()
ax1.plot(data_1['time'],data_1['marker/1_3/x'], linewidth=0.8, alpha=0.9, label="x")
ax1.plot(data_1['time'],data_1['marker/1_3/y'], linewidth=0.8, alpha=0.9, label="y")
ax1.plot(data_1['time'],data_1['marker/1_3/z'], linewidth=0.8, alpha=0.9, label="z")
ax1.plot(data_1['time'],data_1_original['marker/1_3/x'], linewidth=0.6, alpha=0.9, label="x")
ax1.plot(data_1['time'],data_1_original['marker/1_3/y'], linewidth=0.6, alpha=0.9, label="y")
ax1.plot(data_1['time'],data_1_original['marker/1_3/z'], linewidth=0.6, alpha=0.9, label="z")
ax1.spines['right'].set_visible(False) # Hide the right and top spines
ax1.spines['top'].set_visible(False)
#ax1.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
ax1.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
ax1.legend(edgecolor='None')
plt.title('1_3')
plt.xlabel('Time (sec)')
plt.ylabel('Position (mm)', fontsize=10, position = (-0.8,0.5))

fig, ax1 = plt.subplots()
#g = sns.lineplot(x="Displacement",y="Force",sort=True,ci=False,data=test)
ax1.plot(data_1['time'],data_1['marker/1_2/x'], linewidth=0.8, alpha=0.9, label="x")
ax1.plot(data_1['time'],data_1['marker/1_2/y'], linewidth=0.8, alpha=0.9, label="y")
ax1.plot(data_1['time'],data_1['marker/1_2/z'], linewidth=0.8, alpha=0.9, label="z")
ax1.plot(data_1['time'],data_1_original['marker/1_2/x'], linewidth=0.6, alpha=0.9, label="x")
ax1.plot(data_1['time'],data_1_original['marker/1_2/y'], linewidth=0.6, alpha=0.9, label="y")
ax1.plot(data_1['time'],data_1_original['marker/1_2/z'], linewidth=0.6, alpha=0.9, label="z")
#ax1.plot(test2['Displacement'],test2['Force'], linewidth=0.8, alpha=0.9, label="Test 2")
#ax1.plot(test3['Displacement'],test3['Force'], linewidth=0.8, alpha=0.9, label="Test 3")
#ax1.plot(test4['Displacement'],test4['Force'], linewidth=0.8, alpha=0.9, label="Test 4")
#ax1.plot(test5['Displacement'],test5['Force'], linewidth=0.8, alpha=0.9, label="Test 5 10%/s to 10% strain")
#ax1.plot(test6['Displacement'],test6['Force'], linewidth=0.8, alpha=0.9, label="Test 6 6%/s to 6% strain")
#ax1.plot(test7['Displacement'],test7['Force'], linewidth=0.8, alpha=0.9, label="Test 7 1%/s to 1% strain")
#ax1.plot(test8['Displacement'],test8['Force'], linewidth=0.8, alpha=0.9, label="Test 8 1%/s to 6% strain")
#ax1.plot(test9['Displacement'],test9['Force'], linewidth=0.8, alpha=0.9, label="Test 9 4%/s to 4% strain")
ax1.spines['right'].set_visible(False) # Hide the right and top spines
ax1.spines['top'].set_visible(False)
#ax1.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
ax1.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
ax1.legend(edgecolor='None')
plt.title('1_2')
plt.xlabel('Time (sec)')
plt.ylabel('Position (mm)', fontsize=10, position = (-0.8,0.5))
plt.show()
#plt.savefig(r'C:\Users\s2952109\OD\MTP - SLIL project\Tensile loading\CYCLIC V2\cyclev2.is_tcyclic_Exports\figure2.png',dpi=2000)

# basic plot of data

