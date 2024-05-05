#%% imports
# Author: Alastair Quin 2021
from os import path
import numpy as np
import matplotlib.pyplot as plt
import slil.common.filtering as ft
import slil.common.io as fio

dirRootOD = path.join(path.expandvars("%userprofile%"),"OneDrive - Griffith University")
dataOutPutFolderPath = dirRootOD + r'\Projects\MTP - SLIL project\cadaver experiements\Data processed\RE9_newBonePlugLoc'
rawMoCapData = r'C:\Users\Griffith\Desktop\Motion Capture data\Cadaver RE6'

#%% Functions

def plot(data):
    fig, ax1 = plt.subplots()

    ax1.plot(range(len(data)),data, linewidth=0.9, alpha=0.9, label="data")
    #ax1.set_ylim([60,100])

    ax1.spines['right'].set_visible(False) # Hide the right and top spines
    ax1.spines['top'].set_visible(False)
    #ax1.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
    #ax1.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
    ax1.legend(edgecolor='None')
    plt.title('Plot test')
    plt.xlabel('Points')
    plt.ylabel('Position (mm)', fontsize=10, position = (-0.8,0.5))



#%% processes
#static = fio.readC3D(dataOutPutFolderPath + r'\crop2.c3d')
static = fio.readC3D(rawMoCapData + r'\full.c3d')
markerToPlot = 0
axisToPlot = 0
plot(static[:,markerToPlot,axisToPlot])

numMarkers = 12 # static.shape[1]
numFrames = static.shape[0]

cutoff = 0.03
cutoff = 4 # for data in mm

for marker in range(numMarkers):
    static[:,marker,0], x = ft.findBadData_np(static[:,marker,0], cutoff)
    static[:,marker,1], x = ft.findBadData_np(static[:,marker,1], cutoff)
    static[:,marker,2], x = ft.findBadData_np(static[:,marker,2], cutoff)

fio.writeC3D(static, 12, rawMoCapData + r'\full_1_bad_frames_removed.c3d')
#%%
plot(static[:,markerToPlot,axisToPlot])
#%% Fix for missing data
def setBadFirstFrame(data, marker):
    firstGoodFrame = np.argmax(np.isfinite(data[:,marker,0]))
    data[0,marker,0] = data[firstGoodFrame,marker,0]
    firstGoodFrame = np.argmax(np.isfinite(data[:,marker,1]))
    data[0,marker,1] = data[firstGoodFrame,marker,1]
    firstGoodFrame = np.argmax(np.isfinite(data[:,marker,2]))
    data[0,marker,2] = data[firstGoodFrame,marker,2]
    
def setBadLastFrame(data, marker):
    firstGoodFrame = np.argmax(np.isfinite(data[:,marker,0])[::-1])
    data[-1,marker,0] = data[-firstGoodFrame-1,marker,0]
    firstGoodFrame = np.argmax(np.isfinite(data[:,marker,1])[::-1])
    data[-1,marker,1] = data[-firstGoodFrame-1,marker,1]
    firstGoodFrame = np.argmax(np.isfinite(data[:,marker,2])[::-1])
    data[-1,marker,2] = data[-firstGoodFrame-1,marker,2]

setBadFirstFrame(static,0)
setBadLastFrame(static,0)
#%% Interpolate
for marker in range(numMarkers):
    static[:,marker,0] = ft.interpolate_np(static[:,marker,0])
    static[:,marker,1] = ft.interpolate_np(static[:,marker,1])
    static[:,marker,2] = ft.interpolate_np(static[:,marker,2])

fio.writeC3D(static, 12, rawMoCapData + r'\full_2_interpolated.c3d')
plot(static[:,markerToPlot,axisToPlot])
#%% Filtering
samplingFreq = 250 #Hz
lowpassCutoffFreq = 80 #Hz
for marker in range(numMarkers):
    static[:,marker,0] = ft.filterLowpass(static[:,marker,0], lowpassCutoffFreq, samplingFreq)
    static[:,marker,1] = ft.filterLowpass(static[:,marker,1], lowpassCutoffFreq, samplingFreq)
    static[:,marker,2] = ft.filterLowpass(static[:,marker,2], lowpassCutoffFreq, samplingFreq)

fio.writeC3D(static, 12, rawMoCapData + r'\full_3_filtered.c3d')
plot(static[:,markerToPlot,axisToPlot])
#%%

fio.writeC3D(static, 12, dataOutPutFolderPath + r'\static_processed.c3d')
# %%
