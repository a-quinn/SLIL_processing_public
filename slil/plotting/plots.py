
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import matplotlib.collections as collections
#from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from slil.common.plotting_functions import figColours, figNames, cropData3, cropDataRemoveBeginning
from matplotlib.cbook import get_sample_data
from copy import deepcopy
import slil.common.opensim as fo
import slil.common.math as fm
from slil.common.data_configs import outputFolders
from slil.common.math import nan_helper
from slil.common.cache import deleteCache

def deleteAllPreviousCache(experiment):
    cacheToDelete = [
        ['kinematics_BreakDown_', '_FE'],
        ['kinematics_BreakDown_', '_UR'],
        ['kinematics_BreakDownSD_crop_', '_FE'],
        ['kinematics_BreakDownSD_crop_', '_UR'],
        ['kinematics_cropped_2083_4165_', '_FE'],
        ['kinematics_cropped_2083_4165_', '_UR'],
        ['kinematics_Rel2Met_', '_FE'],
        ['kinematics_Rel2Met_', '_UR'],
        ['strains_', '_FE'],
        ['strains_', '_UR'],
        ]
    for cache in cacheToDelete:
        deleteCache(cache[0] + experiment + cache[1])

def addIcon(modelInfo, fig, img, coord, zorder = -1):
    im = plt.imread(get_sample_data(modelInfo['graphicsInputDir'] + "\\" + img))
    newax = fig.add_axes(coord, anchor='NE', zorder=zorder)
    newax.imshow(im)
    newax.axis('off')

def plotBPtoBPvTime(modelInfo, dataIn):
    fig = plt.figure()
    fig.set_size_inches(14.5, 6.5) # for Jupyter viewing
    fig.set_size_inches(19.5, 8.5) # for exporting
    gs0 = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3,1])
    ax1 = fig.add_subplot(gs0[0])
    ax2 = fig.add_subplot(gs0[1])

    ax1.hlines([modelInfo['scaffoldBPtoBPlength']], 0, 1, transform=ax1.get_yaxis_transform(), colors='r')
    motionType = dataIn[0]['type']
    
    maxTime = 0
    minTime = 9999
    for data in dataIn:
        time = data['time']
        label1 = figNames[data['title']]
        col = figColours[data['title']]
        
        ax1.plot(time,data['difference'], linewidth=0.9, alpha=0.7, label=label1, color=col)
        maxTime = max(time[-1],maxTime)
        minTime = min(time[0],minTime)

    ax1.spines['right'].set_visible(False) # Hide the right and top spines
    ax1.spines['top'].set_visible(False)
    #ax1.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
    #ax1.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
    ax1.set_xlim([minTime,maxTime])
    ax1.legend(edgecolor='None')
    ax1.set(ylabel='BP to BP Distance (mm)', title='Bone-Plug to Bone-Plug')


    time = deepcopy(dataIn[0]['time']) if len(dataIn) == 1 else deepcopy(dataIn[2]['time'])
    # raw rot data is in positive direction for both FE and UR
    rot = deepcopy(dataIn[0]['time']) if len(dataIn) == 1 else deepcopy(dataIn[2]['rot'])
    if 'FE' in motionType:
        rot = rot * -1.0
        ax2.set_yticks(np.arange(round(min(rot),-1),max(rot),20), minor=False)
        ax2.set_ylim([round(max(rot),-1)+7,round(min(rot),-1)-5])
    else:
        # if RE9
        #time = dataIn[0]['time'] if len(dataIn) == 1 else dataIn[4]['time']
        #rot = dataIn[0]['time'] if len(dataIn) == 1 else dataIn[4]['rot']
        time = deepcopy(dataIn[0]['time']) if len(dataIn) == 1 else deepcopy(dataIn[3]['time'])
        rot = deepcopy(dataIn[0]['time']) if len(dataIn) == 1 else deepcopy(dataIn[3]['rot'])

        # UR rotation was clock wise first
        if ('UR' in motionType) and not modelInfo['isLeftHand']:
            rot = rot * -1.0

        ax2.set_yticks(np.arange(round(min(rot),-1), max(rot), 15), minor=False)
        ax2.set_ylim([min(rot)-5,max(rot)+5])

    ax2.plot(time,rot, linewidth=0.9, alpha=0.9, label="d")
    ax2.spines['right'].set_visible(False) # Hide the right and top spines
    ax2.spines['top'].set_visible(False)
    #ax2.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
    ax2.set_xlim([minTime,maxTime])
    ax2.grid(True)
    #ax2.legend(edgecolor='None')

    if 'FE' in motionType:
        ax2.set(xlabel='Time (sec)', ylabel='    Wrist             Wrist      \nFlexion(°)    Extension(°)')
        addIcon(modelInfo, fig, "wrist_extension.png", [0.069, 0.30, 0.05, 0.05])
        addIcon(modelInfo, fig, "wrist_flexion.png", [0.069, 0.07, 0.05, 0.05])
    else:
        ax2.set(xlabel='Time (sec)', ylabel='     Ulnar             Radial      \nDeviation(°)    Deviation(°)')
        addIcon(modelInfo, fig, "wrist_radial_deviation.png", [0.069, 0.30, 0.05, 0.05])
        addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.069, 0.07, 0.05, 0.05])
    
    fig.patch.set_facecolor('white')
    plt.savefig(
        fname = outputFolders()['graphics'] + '\\' + modelInfo['experimentID'] + '_BPtoBP_vs_time_' + dataIn[0]['type'],
        dpi = 600,
        facecolor=fig.get_facecolor(),
        transparent=False)
    plt.close() # prevent Jupyter from showing plot

    
def plotSLILGapvTime(modelInfo, dataIn):
    fig = plt.figure()
    fig.set_size_inches(14.5, 6.5) # for Jupyter viewing
    fig.set_size_inches(19.5, 8.5) # for exporting
    gs0 = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[3,1,1])
    ax1 = fig.add_subplot(gs0[0])
    ax2 = fig.add_subplot(gs0[1])
    ax3 = fig.add_subplot(gs0[2])

    #ax1.hlines([modelInfo['scaffoldBPtoBPlength']], 0, 1, transform=ax1.get_yaxis_transform(), colors='r')
    motionType = dataIn[0]['type']
    from slil.common.plotting_functions import generateStrains
    strains = generateStrains(modelInfo, motionType)

    maxY = 0
    maxTime = 0
    minTime = 9999
    lineHandles = []
    prevLable = ''
    labels = []
    
    # same order as what generateStrains() returns
    from slil.common.plotting_functions import groupInfoUR, groupInfoFE
    dataInOrdered = []
    d = {data['file']: data for data in dataIn}
    if 'FE' in motionType:
        for fileName in groupInfoFE['files']:
            dataInOrdered.append(d[fileName])
    else:
        for fileName in groupInfoUR['files']:
          dataInOrdered.append(d[fileName])

    for i, data in enumerate(dataIn):
        time = data['time']
        label1 = figNames[data['title']]
        col = figColours[data['title']]
        
        lineTemp, = ax1.plot(time, strains[i][0], linewidth=0.9, alpha=0.7, label=label1, color=col)
        maxY = max(max(strains[i][0]), maxY)
        maxTime = max(time[-1],maxTime)
        minTime = min(time[0],minTime)
        if not label1 in prevLable:
            prevLable = label1
            lineHandles.append(lineTemp)
            labels.append(figNames[data['title']])
    ax1.legend(handles=lineHandles, labels = labels, edgecolor='None') 

    ax1.spines['right'].set_visible(False) # Hide the right and top spines
    ax1.spines['top'].set_visible(False)
    #ax1.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
    #ax1.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
    ax1.set_xlim([minTime,maxTime])
    ax1.set_ylim([0.0, maxY])
    ax1.grid(True)
    ax1.set(ylabel='SLIL Gap (mm)', title='SLIL Gap')

    arrN, arrC, arrS = plotKinematics_Generate(modelInfo, dataIn, True, "BreakDown_")
    
    for cat in ['normal', 'cut', 'scaffold']:
        label1 = figNames[cat]
        lineColour = figColours[cat]

        if ( cat =='cut'):
            arr = arrC
        if ( cat =='normal' ):
            arr = arrN
        if ( cat =='scaffold' ):
            arr = arrS


        if ('FE' in motionType):
            y = arr[6]
        else:
            y = arr[7]
        for ar in y:
            #if 'FE' in motionType:
            #    ar = ar * -1.0
            ax2.plot(time, ar, color=lineColour, linewidth=0.9, alpha=0.9, label=label1)
            
    if 'FE' in motionType:
        ax2.set_ylim([45, -45])
    else:
        ax2.set_ylim([-35, 35])
    ax2.spines['right'].set_visible(False) # Hide the right and top spines
    ax2.spines['top'].set_visible(False)
    ax2.set_yticks(np.arange(-40, 40, 20), minor=False)
    ax2.grid(True)
    ax2.set_xlim([minTime,maxTime])

    time = deepcopy(dataIn[0]['time']) if len(dataIn) == 1 else deepcopy(dataIn[2]['time'])
    # raw rot data is in positive direction for both FE and UR
    rot = deepcopy(dataIn[0]['time']) if len(dataIn) == 1 else deepcopy(dataIn[2]['rot'])
    if 'FE' in motionType:
        rot = rot * -1.0
        ax3.set_yticks(np.arange(round(min(rot),-1),max(rot),20), minor=False)
        ax3.set_ylim([round(max(rot),-1)+7,round(min(rot),-1)-6])
    else:
        # if RE9
        #time = dataIn[0]['time'] if len(dataIn) == 1 else dataIn[4]['time']
        #rot = dataIn[0]['time'] if len(dataIn) == 1 else dataIn[4]['rot']
        time = deepcopy(dataIn[0]['time']) if len(dataIn) == 1 else deepcopy(dataIn[3]['time'])
        rot = deepcopy(dataIn[0]['time']) if len(dataIn) == 1 else deepcopy(dataIn[3]['rot'])

        # UR rotation was clock wise first
        if dataIn[0]['type'] == 'UR' and not modelInfo['isLeftHand']:
            rot = rot * -1.0

        ax3.set_yticks(np.arange(round(min(rot),-1), max(rot), 15), minor=False)
        ax3.set_ylim([min(rot)-5,max(rot)+5])

    ax3.plot(time, rot, linewidth=0.9, alpha=0.9, label="d")
    ax3.spines['right'].set_visible(False) # Hide the right and top spines
    ax3.spines['top'].set_visible(False)
    #ax3.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
    ax3.set_xlim([minTime,maxTime])
    ax3.grid(True)
    #ax2.legend(edgecolor='None')
    ax3.set(ylabel='Robot\nangle\n(degrees)')
    ax3.set(xlabel='Time (sec)')

    s = 0.05 # figure size
    b1 = 0.44
    b2 = 0.24
    l = 0.069
    if 'FE' in motionType:
        ax2.set(ylabel='  Flexion         Extension\n(degrees)      (degrees)')
        addIcon(modelInfo, fig, "wrist_extension.png", [l, b1, s, s])
        addIcon(modelInfo, fig, "wrist_flexion.png", [l, b2, s, s])
    else:
        ax2.set(ylabel=' Ulnar             Radial \nDeviation        Deviation\n(degrees)        (degrees)')
        addIcon(modelInfo, fig, "wrist_radial_deviation.png", [l, b1, s, s])
        addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [l, b2, s, s])
    
    fig.patch.set_facecolor('white')
    plt.savefig(
        fname = outputFolders()['graphics'] + '\\' + modelInfo['experimentID'] + '_SLILGap_vs_time_' + dataIn[0]['type'],
        dpi = 400,
        facecolor=fig.get_facecolor(),
        transparent=False)
    plt.close() # prevent Jupyter from showing plot

def plotSLILGapvWristAngle(modelInfo, dataIn, isMeanSD = False):
    # Relative to Metacarpal rotation, not robot.
    for_publication = True
    
    fig = plt.figure()
    fig.set_size_inches(7, 7)
    gs0 = gridspec.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs0[0])

    #ax1.hlines([modelInfo['scaffoldBPtoBPlength']], 0, 1, transform=ax1.get_yaxis_transform(), colors='r')
    motionType = dataIn[0]['type']
    from slil.common.plotting_functions import generateStrains
    allStrains = generateStrains(modelInfo, motionType)

    def removeBeginning(dataIn, removedInds = 4165):
        tmpCrop = []
        for data in dataIn:
            tmp = deepcopy(data)
            tmp = tmp[removedInds:]
            tmpCrop.append(tmp)
        return tmpCrop

    # I think this is only needed when comparing samples
    #if ('UR' in motionType) and (modelInfo['isLeftHand']):
    #    for indStrain, strain in enumerate(allStrains):
    #        allStrains[indStrain] = removeBeginning(strain)
    

    def cropDataStrains(dataIn, offset = 2083, cycleLength = 8333):
        from math import floor
        cl = cycleLength
        of = offset
        tmpCrop = []
        len1 = len(dataIn[0])
        for data in dataIn:
            for i in range(floor((len1-of)/cl)):
                j = i+1
                cl1 = of + (cl*i)
                cl2 = of + (cl*j)
                tmp = deepcopy(data)
                tmp = tmp[cl1:cl2]
                tmpCrop.append(tmp)
        return tmpCrop

    # output of generateStrains() is: 1 cut, 2 normal, 3 scaffold
    strainsCropped = [
        cropDataStrains(allStrains[2], 2083, 4165) + cropDataStrains(allStrains[3], 2083, 4165),
        cropDataStrains(allStrains[0], 2083, 4165) + cropDataStrains(allStrains[1], 2083, 4165),
        cropDataStrains(allStrains[4], 2083, 4165) + cropDataStrains(allStrains[5], 2083, 4165)
    ]
    strainsCropped = np.array(strainsCropped)

    motionType = dataIn[0]['type']

    arrN, arrC, arrS = plotKinematics_Generate_cropped(modelInfo, dataIn)

    def flipChronologicalOrder(arrN, arrC, arrS):
        # this is a fix, otherwise they are the wrong way around
        for ind, arr in enumerate(arrN):
            arrN[ind] = np.flip(arr, axis=0)
        for ind, arr in enumerate(arrC):
            arrC[ind] = np.flip(arr, axis=0)
        for ind, arr in enumerate(arrS):
            arrS[ind] = np.flip(arr, axis=0)
        return arrN, arrC, arrS
    arrN, arrC, arrS = flipChronologicalOrder(arrN, arrC, arrS)

    allArrs = np.array([arrN, arrC, arrS])
    
    allStrains = {'normal': [], 'cut': [], 'scaffold': []}

    allX = {'normal': {}, 'cut': {}, 'scaffold': {}}
    for expType in allX:
        tempArr = []
        for i in range(allArrs.shape[2]):
            tempArr.append([])
        allX[expType] = deepcopy(tempArr) # what a bad way to create this...

    # make relative to 3rd metacarpal range of motion
    for indType, typeName in enumerate(['normal', 'cut', 'scaffold']):
        for indCycle in range(allArrs.shape[2]):
            if ('FE' in motionType):
                arrX = allArrs[indType, 6, indCycle, :]
            else:
                arrX = allArrs[indType, 7, indCycle, :]
            
            inc = 0.1 # split data into groups on this sized angle
            minX = round(float(min(arrX)) / inc) * inc
            maxX = round(float(max(arrX)) / inc) * inc

            xnew = np.array(np.arange(minX, maxX, inc))
            allX[typeName][indCycle] = xnew

            strains = strainsCropped[indType, indCycle, :]
            
            strainsNew = np.empty((len(xnew)), dtype=float)
            strainsNew[:] = np.nan
            for ind4, c in enumerate(xnew):
                indexGrouped = np.where((arrX > c) & (arrX < inc + c))[0]
                if len(indexGrouped) > 0:
                    strainsNew[ind4] = np.mean(strains[indexGrouped])

            allStrains[typeName].append(strainsNew)

    maxY = 0
    maxX = 0
    minX = 9999
    lineHandles = []
    if isMeanSD:
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        allXLargestRange = []
        allStrainsMeans = []
        for typeName in ['normal', 'cut', 'scaffold']:
            maxX = 0
            minX = 0
            for indCycle in range(allArrs.shape[2]):
                maxX = max(allX[typeName][indCycle].max(), maxX)
                minX = min(allX[typeName][indCycle].min(), minX)
            inc = 0.1
            xnew = np.array(np.arange(minX - inc, maxX + inc, inc))
            ynew = np.empty((len(allStrains[typeName]), xnew.shape[0]))
            ynew[:] = np.nan
            for indCycle in range(allArrs.shape[2]):
                indStart = find_nearest(xnew, allX[typeName][indCycle].min())
                indEnd = find_nearest(xnew, allX[typeName][indCycle].max())+1
                ynew[indCycle, indStart:indEnd] = allStrains[typeName][indCycle][:]
            allXLargestRange.append(xnew)
            allStrainsMeans.append(ynew)

            
        for indType, typeName in enumerate(['normal', 'cut', 'scaffold']):
            mean = allStrainsMeans[indType].mean(axis=0)
            lineTemp, = ax.plot(
                allXLargestRange[indType], mean,
                linewidth=0.9, alpha=0.7, label=figNames[typeName], color=figColours[typeName])
            ax.fill_between(allXLargestRange[indType],
                mean - 1*allStrainsMeans[indType].std(axis=0),
                mean + 1*allStrainsMeans[indType].std(axis=0),
                color=figColours[typeName], alpha=0.2)
            maxX = max(allXLargestRange[indType].max(), maxX)
            minX = min(allXLargestRange[indType].min(), minX)
            for indCycle in range(allArrs.shape[2]):
                maxY = max(max(allStrains[typeName][indCycle]), maxY)
            lineHandles.append(lineTemp)
        if not for_publication:
            ax.legend(handles=lineHandles, labels = list(figNames.values())) 
    else:
        for typeName in ['normal', 'cut', 'scaffold']:
            for indCycle in range(allArrs.shape[2]):
                lineTemp, = ax.plot(allX[typeName][indCycle], allStrains[typeName][indCycle],
                    linewidth=0.9, alpha=0.7, label=figNames[typeName], color=figColours[typeName])
                maxY = max(max(allStrains[typeName][indCycle]), maxY)
                maxX = max(allX[typeName][indCycle].max(), maxX)
                minX = min(allX[typeName][indCycle].min(), minX)
            lineHandles.append(lineTemp)
        if not for_publication:
            ax.legend(handles=lineHandles, labels = list(figNames.values()), edgecolor='None')   
    
    ax.spines['right'].set_visible(False) # Hide the right and top spines
    ax.spines['top'].set_visible(False)
    ax.grid(True)
    ax.set(ylabel='Gap (mm)', xlabel="Wrist Angle (degrees)", title='SLIL Gap Estimate')
    ax.set_ylim([0.0, maxY + 0.5])
    ax.set_xlim([minX - 5.0, maxX + 5.0])
    if for_publication:
        ax.set_ylim([0.0, 11.0])
        ax.set_xlim([-45.0, 40.0])
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.title.set_size(18)
    ax.xaxis.set_tick_params(labelsize=13)
    ax.yaxis.set_tick_params(labelsize=13)

    s = 0.1 # figure size
    b = 0.020
    d = 0.273
    if not for_publication:
        if 'FE' in motionType:
            addIcon(modelInfo, fig, "wrist_extension.png", [0.07, b, s, s])
            addIcon(modelInfo, fig, "wrist_flexion.png",   [0.290 + d + d, b - 0.025, s, s])
        else:
            b = b - 0.02
            d = 0.273
            addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.065, b, s, s])
            addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.290 + d + d, b, s, s])
        
    fig.patch.set_facecolor('white')
    fileName = outputFolders()['graphics'] + '\\' + modelInfo['experimentID'] + '_SLILGap_vs_WristAngle_'
    if isMeanSD:
        fileName += 'wSD_'
    fileName +=  dataIn[0]['type']
    plt.savefig(
        fname = fileName,
        dpi = 400,
        facecolor=fig.get_facecolor(),
        transparent=False)
    plt.close() # prevent Jupyter from showing plot

def keepInBounds(ls):
    # This should not be needed, if it is then it's bad IK data!
    return ls
    #for i in range(len(ls)):
    #    if (ls[i] < 0.0):
    #        ls[i] = ls[i] + 360.0
    #ls = ls + 2000.0
    for i in range(len(ls)):
        #if (ls[i]/150.0 > 1.0):
        #    ls[i] = ls[i] - (180.0 * floor(abs(ls[i]/180.0)))
        if (ls[i]/150.0 < -1.0 or ls[i]/150.0 > 1.0):
            ls[i] = ls[i] + (-180.0 * floor(abs(ls[i]/180.0)) * copysign(1,ls[i]))
    for i in range(len(ls)):
        if (ls[i]/150.0 < -1.0 or ls[i]/150.0 > 1.0):
            ls[i] = ls[i] + (-180.0 * floor(abs(ls[i]/180.0)) * copysign(1,ls[i]))

    #for i in range(len(ls)):
    #    if (ls[i] > 0.0):
    #        ls[i] = ls[i] - 360.0
    #ls = ls - 2000.0
    return ls


def plotKinematics_Generate_cropped(modelInfo, dataIn):
    from slil.cache_results_plot import loadCache, saveCache
    appendedName = 'cropped_2083_4165_'
    try:
        arr = loadCache('kinematics_' + appendedName + modelInfo['experimentID'] + '_' + dataIn[0]['type'])
        return arr[0], arr[1], arr[2]
    except:
        print('No kinematics file found for {} so generating...'.format(modelInfo['experimentID'] + ' ' + dataIn[0]['type']))

    #if ('UR' in dataIn[0]['type']) and (modelInfo['isLeftHand']):
    #    dataIn = cropDataRemoveBeginning(dataIn)
    dataIn = cropData3(dataIn, 2083, 4165)

    arrN, arrC, arrS = plotKinematics_Generate(modelInfo, dataIn, False)
    
    saveCache([arrN, arrC, arrS], 'kinematics_' + appendedName + modelInfo['experimentID'] + '_' + dataIn[0]['type'])
    return arrN, arrC, arrS

def plotKinematics_Generate(modelInfo, dataIn, useCache = True, appendedName = ""):
    '''
    Returns grouped rotations for each axis for each bone.
    If dataIn is already chopped per cycle then will return the same.
    '''
    if useCache:
        from slil.cache_results_plot import loadCache, saveCache
        try:
            arr = loadCache('kinematics_' + appendedName + modelInfo['experimentID'] + '_' + dataIn[0]['type'])
            return arr[0], arr[1], arr[2]
        except:
            print('No kinematics file found for {} so generating...'.format(modelInfo['experimentID'] + ' ' + dataIn[0]['type']))

    l1 = dataIn[0]['kinematics']['lunate_flexion'].shape[0]
    if (l1 != dataIn[1]['kinematics']['lunate_flexion'].shape[0]):
        print("This function is make for only plotting with full cycles")
        return
    
    l2c, l2n, l2s = 0, 0, 0
    for data in dataIn:
        if ('cut' in data['title']):
            l2c += 1
        if ('normal' in data['title']):
            l2n += 1
        if ('scaffold' in data['title']):
            l2s += 1

    def toDegrees(data, xyz):
        # flexion extension
        r1 = data['kinematics'][xyz[0]].to_numpy()*(180.0/np.pi)
        # ulnar radial deviation
        r2 = data['kinematics'][xyz[1]].to_numpy()*(180.0/np.pi)
        # pronation supination
        r3 = data['kinematics'][xyz[2]].to_numpy()*(180.0/np.pi)
        return [r1, r2, r3]

    arrN = np.zeros((9, l2n, l1), dtype=float)
    arrC = np.zeros((9, l2c, l1), dtype=float)
    arrS = np.zeros((9, l2s, l1), dtype=float)
    for data in dataIn:
        [r1_1, r1_2, r1_3] = toDegrees(data, ['lunate_flexion', 'lunate_deviation', 'lunate_rotation'])
        [r2_1, r2_2, r2_3] = toDegrees(data, ['sca_xrot', 'sca_yrot', 'sca_zrot'])
        [r3_1, r3_2, r3_3] = toDegrees(data, ['hand_flexion', 'hand_deviation', 'hand_rotation'])
        if (not modelInfo['isLeftHand']):
            r1_2 *= -1.0
            r1_3 *= -1.0
            r2_2 *= -1.0
            r2_3 *= -1.0
            r3_2 *= -1.0
            r3_3 *= -1.0
        
        if ('cut' in data['title']):
            l2c -= 1
            arr = arrC
            i = l2c
        if ('normal' in data['title']):
            l2n -= 1
            arr = arrN
            i = l2n
        if ('scaffold' in data['title']):
            l2s -= 1
            arr = arrS
            i = l2s
        arr[0:3, i, :] = [r1_1, r1_2, r1_3]
        arr[3:6, i, :] = [r2_1, r2_2, r2_3]
        arr[6:9, i, :] = [r3_1, r3_2, r3_3]
        
    if useCache:
        saveCache([arrN, arrC, arrS], 'kinematics_' + appendedName + modelInfo['experimentID'] + '_' + dataIn[0]['type'])
    return arrN, arrC, arrS

def plotKinematicsBreakDown(modelInfo, dataIn):

    # raw rot data is in positive direction for both FE and UR
    # UR rotation was clock wise first
    rot = deepcopy(dataIn[0]['rot'])
    if (dataIn[0]['type'] == 'FE'):
        rot *= -1.0
    # graphs are represented as righ hand so flip left hand data
    if ((dataIn[0]['type'] == 'UR') and (not modelInfo['isLeftHand'])):
        rot *= -1.0
    #rot = rot[:4165]
    #time = dataIn[0]['kinematics']['time'].to_numpy()

    arrN, arrC, arrS = plotKinematics_Generate(modelInfo, dataIn, True, "BreakDown_")
    
    fig = plt.figure()
    fig.set_size_inches(14.5, 14.5)
    gs0 = gridspec.GridSpec(4, 3, figure=fig)#, height_ratios=[3,3,1])
    ax = []
    for i in range(0, 9):
        ax.append(fig.add_subplot(gs0[i]))

    remapGraphs = [
        0, 3, 6,
        1, 4, 7,
        2, 5, 8
    ]
    for cat in ['normal', 'cut', 'scaffold']:
        label1 = figNames[cat]
        lineColour = figColours[cat]

        if ( cat =='cut'):
            arr = arrC
        if ( cat =='normal' ):
            arr = arrN
        if ( cat =='scaffold' ):
            arr = arrS

        for ind, ax0 in enumerate(ax):
            for ar in arr[remapGraphs[ind]]:
                ax0.plot(rot, ar, color=lineColour, linewidth=0.9, alpha=0.9, label=label1)

    ylimMin = -50
    ylimMax = 50
    #ylimMin = min(rot)
    #ylimMax = max(rot)
    def axAdjust(ax, ylabel = "", xlabel = "", title = ""):
        ax.spines['right'].set_visible(False) # Hide the right and top spines
        ax.spines['top'].set_visible(False)
        #ax.legend(edgecolor='None')
        ax.grid(True)
        ax.set(ylabel=ylabel, title=title, xlabel=xlabel)
        ax.set_ylim([ylimMin,ylimMax])
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.title.set_size(18)
        ax.xaxis.set_tick_params(labelsize=13)
        ax.yaxis.set_tick_params(labelsize=13)
    axAdjust(ax[0], ylabel = "Flexion-Extension\n(degrees)", title = "Lunate")
    axAdjust(ax[1], title = "Scaphoid")
    axAdjust(ax[2], title = "$3^{rd}$ Metacarpal")
    axAdjust(ax[3], ylabel = "Radial-Ulnar Deviation\n(degrees)")
    axAdjust(ax[4])
    axAdjust(ax[5])
    axAdjust(ax[6], ylabel = "Pro-Supination\n(degrees)", xlabel = "Wrist Angle (degrees)")
    axAdjust(ax[7], xlabel = "Wrist Angle (degrees)")
    axAdjust(ax[8], xlabel = "Wrist Angle (degrees)")
    #ax1_0.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
    #ax1_0.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
    figTitle = "Wrist Motion: Flexion-Extension"
    if 'UR' in dataIn[0]['type']:
        figTitle = "Wrist Motion: Radial-Ulnar Deviation"
    ax[1].text(-0.3, 1.3, figTitle,
        verticalalignment='center',
        rotation='horizontal', fontsize=20, transform=ax[1].transAxes)

    #ax[0].text(-0.35, 0.5, 'Lunate',
    #    verticalalignment='center',
    #    rotation='vertical', fontsize=18, transform=ax[0].transAxes)
    #ax[3].text(-0.35, 0.5, 'Scaphoid',
    #    verticalalignment='center',
    #    rotation='vertical', fontsize=18, transform=ax[3].transAxes)
    #ax[6].text(-0.35, 0.5, '$3^{rd}$ Metacarpal',
    #    verticalalignment='center',
    #    rotation='vertical', fontsize=18, transform=ax[6].transAxes)
        
    s = 0.05 # figure size
    b = 0.26
    d = 0.273
    if (dataIn[0]['type'] == 'FE'):
        addIcon(modelInfo, fig, "wrist_extension.png", [0.100, b, s, s])
        addIcon(modelInfo, fig, "wrist_extension.png", [0.100 + d, b, s, s])
        addIcon(modelInfo, fig, "wrist_extension.png", [0.100 + d + d, b, s, s])
        addIcon(modelInfo, fig, "wrist_flexion.png",   [0.320, b - 0.03, s, s])
        addIcon(modelInfo, fig, "wrist_flexion.png",   [0.320 + d, b - 0.03, s, s])
        addIcon(modelInfo, fig, "wrist_flexion.png",   [0.320 + d + d, b - 0.03, s, s])
    else:
        b = 0.25
        d = 0.273
        addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.095, b, s, s])
        addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.095 + d, b, s, s])
        addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.095 + d + d, b, s, s])
        addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.320, b, s, s])
        addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.320 + d, b,  s, s])
        addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.320 + d + d, b, s, s])
    
    #fig.legend((, l2), ('Line 1', 'Line 2'), 'upper left')
    fig.patch.set_facecolor('white')
    fileName = '\\' + modelInfo['experimentID'] + '_bone_kinematics_' + dataIn[0]['type']
    
    plt.savefig(
        fname = outputFolders()['graphics'] + fileName,
        dpi=600,
        facecolor=fig.get_facecolor(),
        transparent=False)
    plt.close() # prevent Jupyter from showing plot

def plotKinematicsBreakDownSD(modelInfo, dataIn):

    arrN, arrC, arrS = plotKinematics_Generate(modelInfo, dataIn, True, "BreakDownSD_crop_")

    #arrN2 = np.zeros((9, l2n, l1/2), dtype=float)
    #arrC2 = np.zeros((9, l2c, l1/2), dtype=float)
    #arrS2 = np.zeros((9, l2s, l1/2), dtype=float)
    
    useTime = True
    if useTime:
        rot = deepcopy(dataIn[0]['kinematics']['time'].to_numpy())
        # make sure time starts at zero. May not due to data cropping
        if (rot[0] != 0.0):
            rot = rot - rot[0]
    else:
        # raw rot data is in positive direction for both FE and UR
        # UR rotation was clock wise first
        rot = deepcopy(dataIn[0]['rot'])
        if (dataIn[0]['type'] == 'FE'):
            rot *= -1.0
        if (dataIn[0]['type'] == 'UR' and not modelInfo['isLeftHand']):
            rot *= -1.0

    fig = plt.figure()
    fig.set_size_inches(14.5, 14.5)
    gs0 = gridspec.GridSpec(4, 3, figure=fig)#, height_ratios=[3,3,1])
    ax = []
    for i in range(0, 9):
        ax.append(fig.add_subplot(gs0[i]))

    remapGraphs = [
        0, 3, 6,
        1, 4, 7,
        2, 5, 8
    ]
    for cat in ['normal', 'cut', 'scaffold']:
        label1 = figNames[cat]
        lineColour = figColours[cat]

        if ( cat =='cut'):
            arr = arrC
        if ( cat =='normal' ):
            arr = arrN
        if ( cat =='scaffold' ):
            arr = arrS

        for ind, ax0 in enumerate(ax):
            mean = arr[remapGraphs[ind]].mean(axis=0)
            ax0.plot(rot, mean, color=lineColour, linewidth=0.9, alpha=0.9, label=label1)
            ax0.fill_between(rot,
                mean - 1*arr[i].std(axis=0),
                mean + 1*arr[i].std(axis=0),
                color=lineColour, alpha=0.2)
        #    for ar in arr[ind]:
        #        ax0.plot(rot, ar, color=lineColour, linewidth=0.9, alpha=0.9, label=label1)
            

    for ind, ax0 in enumerate(ax):
        meanN = arrN[remapGraphs[ind]].mean(axis=0)
        meanC = arrC[remapGraphs[ind]].mean(axis=0)
        meanS = arrS[remapGraphs[ind]].mean(axis=0)
        rmseNC = np.sqrt(np.mean((meanN-meanC)**2))
        rmseNS = np.sqrt(np.mean((meanN-meanS)**2))
        ax0.text(0.84, 0.15, "$R_1$: {:.2f}".format(rmseNC),
        verticalalignment='center',
        rotation='horizontal', fontsize=14, transform=ax0.transAxes)
        ax0.text(0.84, 0.05, "$R_2$: {:.2f}".format(rmseNS),
        verticalalignment='center',
        rotation='horizontal', fontsize=14, transform=ax0.transAxes)

    ylimMin = -50
    ylimMax = 50
    #ylimMin = min(rot)
    #ylimMax = max(rot)
    def axAdjust(ax, ylabel = "", xlabel = "", title = ""):
        ax.spines['right'].set_visible(False) # Hide the right and top spines
        ax.spines['top'].set_visible(False)
        #ax.legend(edgecolor='None')
        ax.grid(True)
        ax.set(ylabel=ylabel, title=title, xlabel=xlabel)
        ax.set_ylim([ylimMin,ylimMax])
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.title.set_size(18)
        ax.xaxis.set_tick_params(labelsize=13)
        ax.yaxis.set_tick_params(labelsize=13)
    #axAdjust(ax[0], ylabel = "Rotation (degrees)", title = "Flexion-Extension")
    #axAdjust(ax[1], title = "Radial-Ulnar Deviation")
    #axAdjust(ax[2], title = "Pro-Supination")
    #axAdjust(ax[3], ylabel = "Rotation (degrees)")
    #axAdjust(ax[4])
    #axAdjust(ax[5])
    #axAdjust(ax[6], ylabel = "Rotation (degrees)", xlabel = "Time (seconds)")
    #axAdjust(ax[7], xlabel = "Time (seconds)")
    #axAdjust(ax[8], xlabel = "Time (seconds)")
    axAdjust(ax[0], ylabel = "Flexion-Extension\n(degrees)", title = "Lunate")
    axAdjust(ax[1], title = "Scaphoid")
    axAdjust(ax[2], title = "$3^{rd}$ Metacarpal")
    axAdjust(ax[3], ylabel = "Radial-Ulnar Deviation\n(degrees)")
    axAdjust(ax[4])
    axAdjust(ax[5])
    axAdjust(ax[6], ylabel = "Pro-Supination\n(degrees)", xlabel = "Time (seconds)")
    axAdjust(ax[7], xlabel = "Time (seconds)")
    axAdjust(ax[8], xlabel = "Time (seconds)")
    #ax1_0.set_xlim([minTime,maxTime])
    #ax1_0.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
    #ax1_0.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
    figTitle = "Wrist Motion: Flexion-Extension"
    if 'UR' in dataIn[0]['type']:
        figTitle = "Wrist Motion: Radial-Ulnar Deviation"
    ax[1].text(-0.3, 1.3, figTitle,
        verticalalignment='center',
        rotation='horizontal', fontsize=20, transform=ax[1].transAxes)

    #ax[0].text(-0.35, 0.5, 'Lunate',
    #    verticalalignment='center',
    #    rotation='vertical', fontsize=18, transform=ax[0].transAxes)
    #ax[3].text(-0.35, 0.5, 'Scaphoid',
    #    verticalalignment='center',
    #    rotation='vertical', fontsize=18, transform=ax[3].transAxes)
    #ax[6].text(-0.35, 0.5, '$3^{rd}$ Metacarpal',
    #    verticalalignment='center',
    #    rotation='vertical', fontsize=18, transform=ax[6].transAxes)

    note = "$R_1$: Root-mean-squared-error between {} and {}\n$R_2$: Root-mean-squared-error between {} and {}".format(figNames['normal'], figNames['cut'], figNames['normal'], figNames['scaffold'])
    ax[6].text(-0.3, -0.8, note,
        verticalalignment='center',
        rotation='horizontal', fontsize=18, transform=ax[6].transAxes)

    ax[2].legend(bbox_to_anchor=(0.75, 1.05))
    
    s = 0.05 # figure size
    if useTime:
        t = 8.33 # seconds from start to max
        t2 = t * 3
        for a in ax:
            a.axvline(x = t, color='grey', linestyle="--")
            a.axvline(x = t2, color='grey', linestyle="--")
        b = 0.235
        d = 0.273
        if (dataIn[0]['type'] == 'FE'):
            addIcon(modelInfo, fig, "wrist_extension.png", [0.160, b, s, s])
            addIcon(modelInfo, fig, "wrist_extension.png", [0.160 + d, b, s, s])
            addIcon(modelInfo, fig, "wrist_extension.png", [0.160 + d + d, b, s, s])
            addIcon(modelInfo, fig, "wrist_flexion.png",   [0.269, b - 0.025, s, s])
            addIcon(modelInfo, fig, "wrist_flexion.png",   [0.269 + d, b - 0.025, s, s])
            addIcon(modelInfo, fig, "wrist_flexion.png",   [0.269 + d + d, b - 0.025, s, s])
        else:
            b = 0.23
            d = 0.273
            addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.155, b, s, s])
            addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.155 + d, b, s, s])
            addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.155 + d + d, b, s, s])
            addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.269, b, s, s])
            addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.269 + d, b,  s, s])
            addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.269 + d + d, b, s, s])
    else:
        b = 0.255
        d = 0.273
        if (dataIn[0]['type'] == 'FE'):
            addIcon(modelInfo, fig, "wrist_extension.png", [0.100, b, s, s])
            addIcon(modelInfo, fig, "wrist_extension.png", [0.100 + d, b, s, s])
            addIcon(modelInfo, fig, "wrist_extension.png", [0.100 + d + d, b, s, s])
            addIcon(modelInfo, fig, "wrist_flexion.png",   [0.320, b - 0.025, s, s])
            addIcon(modelInfo, fig, "wrist_flexion.png",   [0.320 + d, b - 0.025, s, s])
            addIcon(modelInfo, fig, "wrist_flexion.png",   [0.320 + d + d, b - 0.025, s, s])
        else:
            b = 0.25
            d = 0.273
            addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.095, b, s, s])
            addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.095 + d, b, s, s])
            addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.095 + d + d, b, s, s])
            addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.320, b, s, s])
            addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.320 + d, b,  s, s])
            addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.320 + d + d, b, s, s])
    
    
    fig.patch.set_facecolor('white')
    fileName = '\\' + modelInfo['experimentID'] + '_bone_kinematics_wSD_' + dataIn[0]['type']
    plt.savefig(
        fname = outputFolders()['graphics'] + fileName,
        dpi=600,
        facecolor=fig.get_facecolor(),
        transparent=False)
    plt.close() # prevent Jupyter from showing plot

def plotKinematicsBreakDownSD_RelativeToMet_Generate(dataIn, modelInfo, useCache = True):
    '''
    Find the min and max that the metacarpal got to during each cycle. And resamples all other
    bones rotations relative to the new x.

    Assumes robot cycles rought align with specimen manipulation cycles. Meaning
    there was no compliance/play between the specimen and robot end effector.

    Returns:
        allX - min to max x data at intervals of 0.2 (unit should be degrees)
        allArrs2 - resampled y data
        allArrs - same as plotKinematics_Generate()
    '''
    if useCache:
        from slil.cache_results_plot import loadCache, saveCache
        try:
            arr = loadCache('kinematics_Rel2Met_' + modelInfo['experimentID'] + '_' + dataIn[0]['type'])
            return arr[0], arr[1], arr[2]
        except:
            print('No kinematics file found for {} so generating...'.format(modelInfo['experimentID'] + ' ' + dataIn[0]['type']))

    motionType = dataIn[0]['type']

    arrN, arrC, arrS = plotKinematics_Generate_cropped(modelInfo, dataIn)

    allArrs = np.array([arrN, arrC, arrS])
    
    rotationNames = [
        'lunate_flexion', 'lunate_deviation', 'lunate_rotation',
        'sca_xrot', 'sca_yrot', 'sca_zrot',
        'hand_flexion', 'hand_deviation', 'hand_rotation']
    allArrs2 = {'normal': {}, 'cut': {}, 'scaffold': {}}
    for ind, expType in enumerate(allArrs2):
        for rotationName in rotationNames:
            tempArr = []
            for i in range(allArrs.shape[2]):
                tempArr.append([])
            allArrs2[expType][rotationName] = deepcopy(tempArr) # what a bad way to create this...

    allX = {'normal': {}, 'cut': {}, 'scaffold': {}}
    for ind, expType in enumerate(allX):
        tempArr = []
        for i in range(allArrs.shape[2]):
            tempArr.append([])
        allX[expType] = deepcopy(tempArr) # what a bad way to create this...

    # make relative to 3rd metacarpal range of motion
    for indType, typeName in enumerate(allArrs2):
        for indCycle in range(allArrs.shape[2]):
            if ('FE' in motionType):
                arrX = allArrs[indType, 6, indCycle, :]
            else:
                arrX = allArrs[indType, 7, indCycle, :]
            
            inc = 0.1 # split data into groups on this sized angle
            minX = round(float(min(arrX)) / inc) * inc
            maxX = round(float(max(arrX)) / inc) * inc

            xnew = np.array(np.arange(minX, maxX, inc))
            allX[typeName][indCycle] = xnew

            for indBone in range(allArrs.shape[1]):
                #if (data['type'] == 'FE'):
                #    rot *= -1.0
                #if ((data['type'] == 'UR') and (not model['isLeftHand'])):
                #    rot *= -1.0
                y = deepcopy(allArrs[indType, indBone, indCycle, :])
                #if (not modelInfo['isLeftHand']):
                #    y *= -1.0
                
                ynew = np.empty((len(xnew)), dtype=float)
                ynew[:] = np.nan
                #ynew[:] = 0.0
                for ind4, c in enumerate(xnew):
                    indexGrouped = np.where((arrX > c) & (arrX < inc + c))[0]
                    if len(indexGrouped) > 0:
                        ynew[ind4] = np.mean(y[indexGrouped])

                allArrs2[typeName][rotationNames[indBone]][indCycle] = ynew

    if useCache:
        saveCache([allX, allArrs2, allArrs], 'kinematics_Rel2Met_' + modelInfo['experimentID'] + '_' + dataIn[0]['type'])
    return allX, allArrs2, allArrs

def plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataIn, ScaRel2Lun = True, isMeanSD = False):#, dataKinematics):
    # This function is bad, do not use. It is incorrect to average points at a given angle, later stats would be incorrect.

    modelType = dataIn[0]['type']

    allX, allArrs2, allArrs = plotKinematicsBreakDownSD_RelativeToMet_Generate(dataIn, modelInfo)
    rotationNames = [
        'lunate_flexion', 'lunate_deviation', 'lunate_rotation',
        'sca_xrot', 'sca_yrot', 'sca_zrot',
        'hand_flexion', 'hand_deviation', 'hand_rotation']

    useTime = False

    if ScaRel2Lun:
        # use hand rotations arrays for scaphoid realtieve to lunate
        for indType, typeName in enumerate(allArrs2):
            for indCycle in range(allArrs.shape[2]):
                allArrs2[typeName]['hand_flexion'][indCycle] = \
                    allArrs2[typeName]['sca_xrot'][indCycle] - \
                    allArrs2[typeName]['lunate_flexion'][indCycle]
                allArrs2[typeName]['hand_deviation'][indCycle] = \
                    allArrs2[typeName]['sca_yrot'][indCycle] - \
                    allArrs2[typeName]['lunate_deviation'][indCycle]
                allArrs2[typeName]['hand_rotation'][indCycle] = \
                    allArrs2[typeName]['sca_zrot'][indCycle] - \
                    allArrs2[typeName]['lunate_rotation'][indCycle]
                

    fig = plt.figure()
    fig.set_size_inches(14.5, 14.5)
    gs0 = gridspec.GridSpec(3, 3, figure=fig)#, height_ratios=[3,3,1])
    ax = []
    for i in range(0, 9):
        ax.append(fig.add_subplot(gs0[i]))

    remapGraphs = [
        0, 3, 6,
        1, 4, 7,
        2, 5, 8
    ]
    lineHandles = []
    if isMeanSD:
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
        
        allXLargestRange = []
        allRotsMeans = []
        for typeName in ['normal', 'cut', 'scaffold']:
            maxX = 0
            minX = 0
            for indCycle in range(allArrs.shape[2]):
                maxX = max(allX[typeName][indCycle].max(), maxX)
                minX = min(allX[typeName][indCycle].min(), minX)
            inc = 0.1
            xnew = np.array(np.arange(minX - inc, maxX + inc, inc))
            ynew = np.empty((len(rotationNames), len(allArrs2[typeName][rotationNames[0]]), xnew.shape[0]))
            ynew[:] = np.nan
            for indCycle in range(allArrs.shape[2]):
                indStart = find_nearest(xnew, allX[typeName][indCycle].min())
                indEnd = find_nearest(xnew, allX[typeName][indCycle].max())+1
                for indRotName, rotationName in enumerate(rotationNames):
                    ynew[indRotName, indCycle, indStart:indEnd] = allArrs2[typeName][rotationName][indCycle][:]
            allXLargestRange.append(xnew)
            allRotsMeans.append(ynew)

            
        for indType, typeName in enumerate(['normal', 'cut', 'scaffold']):
            for ind, ax0 in enumerate(ax):
                mean = allRotsMeans[indType][remapGraphs[ind], :, :].mean(axis=0)
                lineTemp, = ax0.plot(
                    allXLargestRange[indType], mean,
                    linewidth=0.9, alpha=0.7, label=figNames[typeName], color=figColours[typeName])
                ax0.fill_between(allXLargestRange[indType],
                    mean - 1*allRotsMeans[indType][remapGraphs[ind], :, :].std(axis=0),
                    mean + 1*allRotsMeans[indType][remapGraphs[ind], :, :].std(axis=0),
                    color=figColours[typeName], alpha=0.2)
                if ind == 2:
                    lineHandles.append(lineTemp)
        ax[2].legend(handles=lineHandles, labels = list(figNames.values()))   
    else:
        for typeName in ['normal', 'cut', 'scaffold']:

            #    for ar in arr[ind]:
            #        ax0.plot(rot, ar, color=figColours[typeName], linewidth=0.9, alpha=0.9, label=figNames[typeName])
            for ind, ax0 in enumerate(ax):
                for indCycle in range(allArrs.shape[2]):
                    #mean = np.mean(allArrs2[typeName][rotationNames[ind]][:])
                    #std = np.std(allArrs2[typeName][rotationNames[ind]][:])
                    lineTemp, = ax0.plot(
                        allX[typeName][indCycle],
                        allArrs2[typeName][rotationNames[remapGraphs[ind]]][indCycle],
                        color=figColours[typeName], linewidth=0.9, alpha=0.9, label=figNames[typeName])
                
                    #ax0.fill_between(allX[typeName][indCycle],
                    #    mean - 1*std,
                    #    mean + 1*std,
                    #    color=figColours[typeName], alpha=0.2)
                    if ind == 2 and indCycle == 0:
                        lineHandles.append(lineTemp)
        ax[2].legend(handles=lineHandles, labels = list(figNames.values()))         

    #for ind, ax0 in enumerate(ax):
    #    meanN = arrN[ind].mean(axis=0)
    #    meanC = arrC[ind].mean(axis=0)
    #    meanS = arrS[ind].mean(axis=0)
    #    rmseNC = np.sqrt(np.mean((meanN-meanC)**2))
    #    rmseNS = np.sqrt(np.mean((meanN-meanS)**2))
    #    ax0.text(0.84, 0.15, "$R_1$: {:.2f}".format(rmseNC),
    #    verticalalignment='center',
    #    rotation='horizontal', fontsize=14, transform=ax0.transAxes)
    #    ax0.text(0.84, 0.05, "$R_2$: {:.2f}".format(rmseNS),
    #    verticalalignment='center',
    #    rotation='horizontal', fontsize=14, transform=ax0.transAxes)

    ylimMin = -50
    ylimMax = 50
    #ylimMin = min(rot)
    #ylimMax = max(rot)
    def axAdjust(ax, ylabel = "", xlabel = "", title = ""):
        ax.spines['right'].set_visible(False) # Hide the right and top spines
        ax.spines['top'].set_visible(False)
        #ax.legend(edgecolor='None')
        ax.grid(True)
        ax.set(ylabel=ylabel, title=title, xlabel=xlabel)
        ax.set_ylim([ylimMin,ylimMax])
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.title.set_size(18)
        ax.xaxis.set_tick_params(labelsize=13)
        ax.yaxis.set_tick_params(labelsize=13)
    axAdjust(ax[0], ylabel = "Flexion-Extension\n(degrees)", title = "Lunate")
    axAdjust(ax[1], title = "Scaphoid")
    if ScaRel2Lun:
        axAdjust(ax[2], title = "Scaphoid relative\nto Lunate")
    else:
        axAdjust(ax[2], title = "$3^{rd}$ Metacarpal")
    axAdjust(ax[3], ylabel = "Radial-Ulnar Deviation\n(degrees)")
    axAdjust(ax[4])
    axAdjust(ax[5])
    axAdjust(ax[6], ylabel = "Pro-Supination\n(degrees)", xlabel = "Wrist Angle (degrees)")
    axAdjust(ax[7], xlabel = "Wrist Angle (degrees)")
    axAdjust(ax[8], xlabel = "Wrist Angle (degrees)")
    #ax1_0.set_xlim([minTime,maxTime])
    #ax1_0.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
    #ax1_0.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
    figTitle = "Wrist Motion: Flexion-Extension"
    if 'UR' in modelType:
        figTitle = "Wrist Motion: Radial-Ulnar Deviation"
    ax[1].text(-0.3, 1.3, figTitle,
        verticalalignment='center',
        rotation='horizontal', fontsize=20, transform=ax[1].transAxes)

    #ax[0].text(-0.35, 0.5, 'Lunate',
    #    verticalalignment='center',
    #    rotation='vertical', fontsize=18, transform=ax[0].transAxes)
    #ax[3].text(-0.35, 0.5, 'Scaphoid',
    #    verticalalignment='center',
    #    rotation='vertical', fontsize=18, transform=ax[3].transAxes)
    #ax[6].text(-0.35, 0.5, '$3^{rd}$ Metacarpal',
    #    verticalalignment='center',
    #    rotation='vertical', fontsize=18, transform=ax[6].transAxes)

    #note = "$R_1$: Root-mean-squared-error between {} and {}\n$R_2$: Root-mean-squared-error between {} and {}".format(figNames['normal'], figNames['cut'], figNames['normal'], figNames['scaffold'])
    #ax[6].text(-0.3, -0.8, note,
    #    verticalalignment='center',
    #    rotation='horizontal', fontsize=18, transform=ax[6].transAxes)

    #ax[2].legend(bbox_to_anchor=(0.75, 1.05))
    
    s = 0.05 # figure size
    b1FE = 0.18 # use if no bottom text
    b1UR = 0.2 # use if no bottom text
    if useTime:
        t = 8.33 # seconds from start to max
        t2 = t * 3
        for a in ax:
            a.axvline(x = t, color='grey', linestyle="--")
            a.axvline(x = t2, color='grey', linestyle="--")
        b = 0.235 - b1FE
        d = 0.273
        if (modelType == 'FE'):
            addIcon(modelInfo, fig, "wrist_extension.png", [0.160, b, s, s])
            addIcon(modelInfo, fig, "wrist_extension.png", [0.160 + d, b, s, s])
            addIcon(modelInfo, fig, "wrist_extension.png", [0.160 + d + d, b, s, s])
            addIcon(modelInfo, fig, "wrist_flexion.png",   [0.269, b - 0.025, s, s])
            addIcon(modelInfo, fig, "wrist_flexion.png",   [0.269 + d, b - 0.025, s, s])
            addIcon(modelInfo, fig, "wrist_flexion.png",   [0.269 + d + d, b - 0.025, s, s])
        else:
            b = 0.23 - b1UR
            d = 0.273
            addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.155, b, s, s])
            addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.155 + d, b, s, s])
            addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.155 + d + d, b, s, s])
            addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.269, b, s, s])
            addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.269 + d, b,  s, s])
            addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.269 + d + d, b, s, s])
    else:
        b = 0.255 - b1FE
        d = 0.273
        if (modelType == 'FE'):
            addIcon(modelInfo, fig, "wrist_extension.png", [0.100, b, s, s])
            addIcon(modelInfo, fig, "wrist_extension.png", [0.100 + d, b, s, s])
            addIcon(modelInfo, fig, "wrist_extension.png", [0.100 + d + d, b, s, s])
            addIcon(modelInfo, fig, "wrist_flexion.png",   [0.320, b - 0.025, s, s])
            addIcon(modelInfo, fig, "wrist_flexion.png",   [0.320 + d, b - 0.025, s, s])
            addIcon(modelInfo, fig, "wrist_flexion.png",   [0.320 + d + d, b - 0.025, s, s])
        else:
            b = 0.25 - b1UR
            d = 0.273
            addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.095, b, s, s])
            addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.095 + d, b, s, s])
            addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.095 + d + d, b, s, s])
            addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.320, b, s, s])
            addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.320 + d, b,  s, s])
            addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.320 + d + d, b, s, s])
    
    fig.patch.set_facecolor('white')
    fileName = '\\' + modelInfo['experimentID']
    if ScaRel2Lun:
        fileName += '_bone_kinematics3_'
    else:
        fileName += '_bone_kinematics2_'
    
    if isMeanSD:
        fileName += 'wSD_'
    
    fileName += modelType
    plt.savefig(
        fname = outputFolders()['graphics'] + fileName,
        dpi=600,
        facecolor=fig.get_facecolor(),
        transparent=False)
    plt.close() # prevent Jupyter from showing plot

def plotKinematicsBreakDownMean(modelInfo, dataIn):
    dataIn = cropData3(dataIn, 0, 2083)
    
    # find min max closes to 0.05 of a degree
    # find indecies in groups of 0.05 degrees
    # group kinematics by index groups
    inc = 0.1 # split data into groups on this sized angle
    if ('UR' in dataIn[0]['type']):
        minX = round(float(-30.1) / inc) * inc
        maxX = round(float(30.1) / inc) * inc
    else:
        minX = round(float(-40.1) / inc) * inc
        maxX = round(float(40.1) / inc) * inc
    xnew = np.array(np.arange(minX, maxX, inc))

    # 3 types: noraml, cut, sacffold
    # x angles
    # 14 sections * 2 trials of each type
    kinematicAngles = [
        'lunate_flexion', 'lunate_deviation', 'lunate_rotation',
        'sca_xrot', 'sca_yrot', 'sca_zrot',
        'hand_flexion', 'hand_deviation', 'hand_rotation'
    ]
    # flip these if model is right handed
    kinematicAnglesFlip = [
        'lunate_deviation', 'lunate_rotation',
        'sca_yrot', 'sca_zrot',
        'hand_deviation', 'hand_rotation'
    ]
    l = 14 * 2 * 1 # one model
    y2new = np.empty((3, len(kinematicAngles), l, len(xnew)), dtype=float)
    for ind2, kinematicAngle in enumerate(kinematicAngles):
        l2c, l2n, l2s = 0, 0, 0
        for ind1, data in enumerate(dataIn):
            # raw rot data is in positive direction for both FE and UR
            # UR rotation was clock wise first
            rot = deepcopy(data['rot'])
            if (data['type'] == 'FE'):
                rot *= -1.0
            if ((data['type'] == 'UR') and (not modelInfo['isLeftHand'])):
                rot *= -1.0
            y = deepcopy(data['kinematics'][kinematicAngle].to_numpy()*(180.0/np.pi))
            if (kinematicAngle in kinematicAnglesFlip) and (not modelInfo['isLeftHand']):
                y *= -1.0
            
            ynew = np.empty((len(xnew)), dtype=float)
            ynew[:] = np.nan
            #ynew[:] = 0.0
            for ind3, c in enumerate(xnew):
                indexGrouped = np.where((rot > c) & (rot < inc + c))[0]
                if len(indexGrouped) > 0:
                    ynew[ind3] = y[indexGrouped].mean()

            #nans, x1= nan_helper(ynew)
            #ynew[nans]= np.interp(x1(nans), x1(~nans), ynew[~nans])
            if 'normal' in data['title']:
                y2new[0, ind2, l2n, :] = ynew
                l2n += 1
            if 'cut' in data['title']:
                y2new[1, ind2, l2c, :] = ynew
                l2c += 1
            if 'scaffold' in data['title']:
                y2new[2, ind2, l2s, :] = ynew
                l2s += 1

    fig = plt.figure()
    fig.set_size_inches(14.5, 14.5)
    gs0 = gridspec.GridSpec(4, 3, figure=fig)#, height_ratios=[3,3,1])
    ax = []
    for i in range(0, 9):
        ax.append(fig.add_subplot(gs0[i]))
    
    xnew = np.arange(minX, maxX, inc)

    remapGraphs = [
        0, 3, 6,
        1, 4, 7,
        2, 5, 8
    ]
    for indFig, cat in enumerate(['normal', 'cut', 'scaffold']):
        for indAx, ax0 in enumerate(ax):
            mean = np.nanmean(y2new[indFig,remapGraphs[indAx],:,:], axis=0)
            ax0.plot(xnew, mean, color=figColours[cat], linewidth=0.9, alpha=0.9, label=figNames[cat])
            # + - one standard deviation
            ax0.fill_between(xnew,
                mean - 1*np.nanstd(y2new[indFig,remapGraphs[indAx],:,:], axis=0),
                mean + 1*np.nanstd(y2new[indFig,remapGraphs[indAx],:,:], axis=0),
                color=figColours[cat], alpha=0.2)

    for indAx, ax0 in enumerate(ax):
        meanN = np.nanmean(y2new[0,remapGraphs[indAx],:,:], axis=0)
        nans, x1= nan_helper(meanN)
        meanN[nans]= 0.0
        meanC = np.nanmean(y2new[1,remapGraphs[indAx],:,:], axis=0)
        nans, x1= nan_helper(meanC)
        meanC[nans]= 0.0
        meanS = np.nanmean(y2new[2,remapGraphs[indAx],:,:], axis=0)
        nans, x1= nan_helper(meanS)
        meanS[nans]= 0.0
        rmseNC = np.sqrt(np.mean((meanN-meanC)**2))
        rmseNS = np.sqrt(np.mean((meanN-meanS)**2))
        ax0.text(0.84, 0.15, "$R_1$: {:.2f}".format(rmseNC),
        verticalalignment='center',
        rotation='horizontal', fontsize=14, transform=ax0.transAxes)
        ax0.text(0.84, 0.05, "$R_2$: {:.2f}".format(rmseNS),
        verticalalignment='center',
        rotation='horizontal', fontsize=14, transform=ax0.transAxes)

    ylimMin = -35
    ylimMax = 35
    #ylimMin = min(rot)
    #ylimMax = max(rot)
    def axAdjust(ax, ylabel = "", xlabel = "", title = ""):
        ax.spines['right'].set_visible(False) # Hide the right and top spines
        ax.spines['top'].set_visible(False)
        #ax.legend(edgecolor='None')
        ax.grid(True)
        ax.set(ylabel=ylabel, title=title, xlabel=xlabel)
        ax.set_ylim([minX - 5.0, maxX + 5.0])
        ax.set_xlim([minX - 5.0, maxX + 5.0])
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.title.set_size(18)
        ax.xaxis.set_tick_params(labelsize=13)
        ax.yaxis.set_tick_params(labelsize=13)
    #axAdjust(ax[0], ylabel = "Rotation (degrees)", title = "Flexion-Extension")
    #axAdjust(ax[1], title = "Radial-Ulnar Deviation")
    #axAdjust(ax[2], title = "Pro-Supination")
    #axAdjust(ax[3], ylabel = "Rotation (degrees)")
    #axAdjust(ax[4])
    #axAdjust(ax[5])
    #axAdjust(ax[6], ylabel = "Rotation (degrees)", xlabel = "Wrist Angle (degrees)")
    #axAdjust(ax[7], xlabel = "Wrist Angle (degrees)")
    #axAdjust(ax[8], xlabel = "Wrist Angle (degrees)")
    axAdjust(ax[0], ylabel = "Flexion-Extension\n(degrees)", title = "Lunate")
    axAdjust(ax[1], title = "Scaphoid")
    axAdjust(ax[2], title = "$3^{rd}$ Metacarpal")
    axAdjust(ax[3], ylabel = "Radial-Ulnar Deviation\n(degrees)")
    axAdjust(ax[4])
    axAdjust(ax[5])
    axAdjust(ax[6], ylabel = "Pro-Supination\n(degrees)", xlabel = "Wrist Angle (degrees)")
    axAdjust(ax[7], xlabel = "Wrist Angle (degrees)")
    axAdjust(ax[8], xlabel = "Wrist Angle (degrees)")
    #ax1_0.set_xlim([minTime,maxTime])
    #ax1_0.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
    #ax1_0.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
    figTitle = "Wrist Motion: Flexion-Extension"
    if 'UR' in dataIn[0]['type']:
        figTitle = "Wrist Motion: Radial-Ulnar Deviation"
    ax[1].text(-0.3, 1.3, figTitle,
        verticalalignment='center',
        rotation='horizontal', fontsize=20, transform=ax[1].transAxes)

    #ax[0].text(-0.35, 0.5, 'Lunate',
    #    verticalalignment='center',
    #    rotation='vertical', fontsize=18, transform=ax[0].transAxes)
    #ax[3].text(-0.35, 0.5, 'Scaphoid',
    #    verticalalignment='center',
    #    rotation='vertical', fontsize=18, transform=ax[3].transAxes)
    #ax[6].text(-0.35, 0.5, '$3^{rd}$ Metacarpal',
    #    verticalalignment='center',
    #    rotation='vertical', fontsize=18, transform=ax[6].transAxes)

    note = "$R_1$: Root-mean-squared-error between {} and {}\n$R_2$: Root-mean-squared-error between {} and {}".format(figNames['normal'], figNames['cut'], figNames['normal'], figNames['scaffold'])
    ax[6].text(-0.3, -0.8, note,
        verticalalignment='center',
        rotation='horizontal', fontsize=18, transform=ax[6].transAxes)

    ax[2].legend(bbox_to_anchor=(0.75, 1.05))
        
    s = 0.05 # figure size
    b = 0.255
    d = 0.273
    if (dataIn[0]['type'] == 'FE'):
        addIcon(modelInfo, fig, "wrist_extension.png", [0.100, b, s, s])
        addIcon(modelInfo, fig, "wrist_extension.png", [0.100 + d, b, s, s])
        addIcon(modelInfo, fig, "wrist_extension.png", [0.100 + d + d, b, s, s])
        addIcon(modelInfo, fig, "wrist_flexion.png",   [0.320, b - 0.025, s, s])
        addIcon(modelInfo, fig, "wrist_flexion.png",   [0.320 + d, b - 0.025, s, s])
        addIcon(modelInfo, fig, "wrist_flexion.png",   [0.320 + d + d, b - 0.025, s, s])
    else:
        b = 0.25
        d = 0.273
        addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.095, b, s, s])
        addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.095 + d, b, s, s])
        addIcon(modelInfo, fig, "wrist_ulnar_deviation.png", [0.095 + d + d, b, s, s])
        addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.320, b, s, s])
        addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.320 + d, b,  s, s])
        addIcon(modelInfo, fig, "wrist_radial_deviation.png",  [0.320 + d + d, b, s, s])
    
    fig.patch.set_facecolor('white')
    fileName = '\\' + modelInfo['experimentID'] + '_bone_kinematics_mean_wSD_' + dataIn[0]['type']
    plt.savefig(
        fname = outputFolders()['graphics'] + fileName,
        dpi=600,
        facecolor=fig.get_facecolor(),
        transparent=False)
    plt.close() # prevent Jupyter from showing plot

def plotBoneplugsDisplacement(modelInfo, dataIn):
    fig = plt.figure()
    fig.set_size_inches(6, 6)
    gs0 = gridspec.GridSpec(1, 1, figure=fig)#, height_ratios=[3,3,1])
    ax1_0 = fig.add_subplot(gs0[0])
    
    ylimMin = dataIn[0]['difference'][0]
    ylimMax = dataIn[0]['difference'][0]
    
    l2c, l2n, l2s = 0, 0, 0
    for data in dataIn:
        if ('cut' in data['title']):
            l2c += 1
        if ('normal' in data['title']):
            l2n += 1
        if ('scaffold' in data['title']):
            l2s += 1
            
    l1 = dataIn[0]['difference'].shape[0]
    arrN = np.zeros((l2n, l1), dtype=float)
    arrC = np.zeros((l2c, l1), dtype=float)
    arrS = np.zeros((l2s, l1), dtype=float)
    for data in dataIn:
        r1 = data['difference']
        if ('normal' in data['title']):
            l2n -= 1
            arrN[l2n, :] = r1
        if ('cut' in data['title']):
            l2c -= 1
            arrC[l2c, :] = r1
        if ('scaffold' in data['title']):
            l2s -= 1
            arrS[l2s, :] = r1

        # raw rot data is in positive direction for both FE and UR
        # UR rotation was clock wise first
        rot = deepcopy(data['rot'])
        if (dataIn[0]['type'] == 'FE'):
            rot *= -1.0
        if (data['type'] == 'UR' and not modelInfo['isLeftHand']):
            rot = rot * -1.0
        
        #r1 = keepInBounds(r1)
        ax1_0.plot(rot, r1, color=figColours[data['title']], linewidth=0.9, alpha=0.7, label=figNames[data['title']])
        
        ylimMin = min(min(r1) * 0.95, ylimMin)
        ylimMax = max(max(r1) * 1.05, ylimMax)
    
    ax1_0.hlines([modelInfo['scaffoldBPtoBPlength']], 0, 1, transform=ax1_0.get_yaxis_transform(), colors='r')
    
    meanN = np.mean(arrN / modelInfo['scaffoldBPtoBPlength'], axis=0)
    meanC = np.mean(arrC / modelInfo['scaffoldBPtoBPlength'], axis=0)
    meanS = np.mean(arrS / modelInfo['scaffoldBPtoBPlength'], axis=0)
    rmseNC = np.sqrt(np.mean((meanN - meanC)**2))
    rmseNS = np.sqrt(np.mean((meanN - meanS)**2))
    ax1_0.text(0.84, 0.10, "$R_1$: {:.2f}".format(rmseNC),
    verticalalignment='center',
    rotation='horizontal', fontsize=14, transform=ax1_0.transAxes)
    ax1_0.text(0.84, 0.05, "$R_2$: {:.2f}".format(rmseNS),
    verticalalignment='center',
    rotation='horizontal', fontsize=14, transform=ax1_0.transAxes)

    note = "$R_1$: Root-mean-squared-error between {} and {}\n$R_2$: Root-mean-squared-error between {} and {}\n$R_1$ and $R_2$ normalized to implant size.".format(figNames['normal'], figNames['cut'], figNames['normal'], figNames['scaffold'])
    ax1_0.text(0.0, -0.3, note,
        verticalalignment='center',
        rotation='horizontal', fontsize=12, transform=ax1_0.transAxes)
    
    ax1_0.spines['right'].set_visible(False) # Hide the right and top spines
    ax1_0.spines['top'].set_visible(False)
    ax1_0.grid(True)
    if 'FE' in dataIn[0]['type']:
        ax1_0.set(ylabel='Bone-Plug to Bone-Plug displacement (mm)', title='Flexion-Extension')
    else:
        ax1_0.set(ylabel='Bone-Plug to Bone-Plug displacement (mm)', title='Ulnar-Radial Deviation')

    ax1_0.set_ylim([ylimMin,ylimMax])
    
    #fig.legend((, l2), ('Line 1', 'Line 2'), 'upper left')
    fig.patch.set_facecolor('white')
    plt.savefig(
        fname = outputFolders()['graphics'] + '\\' + modelInfo['experimentID'] + '_BPtoBP_vs_wrist_angle' + dataIn[0]['type'],
        dpi=600,
        facecolor=fig.get_facecolor(),
        transparent=False,
        bbox_inches="tight")
    plt.close() # prevent Jupyter from showing plot

def calcAndPlotRotaionAxies(modelInfo, dataIn):
    # 3D plot showing rotation of metacarpal bone for combined rotation axies and position relative to radius
    # Also computes the rotation matrix to offset this
    # dataIn = dataFE
    show = True
    show = False
    if show:
        fig = plt.figure()
        fig.set_size_inches(6, 6)
        gs0 = gridspec.GridSpec(1, 1, figure=fig)#, height_ratios=[3,3,1])
        ax1_0 = fig.add_subplot(gs0[0], projection='3d')

        ax1_0.set(xlabel="x", ylabel="z", zlabel="y")

    #initalPositons, names = fo.getInitalBonePositons(modelInfo)
    print("running")
    positons, names = fo.getBonePositonsFromIK(modelInfo, dataIn)

    # rotations calculated are valid as long as the rotation matrix below is the same for all models
    # rotation matrix should be the same for all models for radius to ground
    #radiusRot = fo.getRadiusReferenceRotationMatrix(modelInfo)

    everyNth = 20
    
    #initialP = initalPositons[names.index('radius')]
    #x = dataIn[2]['kinematics']['uln_xtran'].to_numpy()[1::everyNth] + initialP[0]
    #y = dataIn[2]['kinematics']['uln_ytran'].to_numpy()[1::everyNth] + initialP[1]
    #z = dataIn[2]['kinematics']['uln_ztran'].to_numpy()[1::everyNth] + initialP[2]
    nameI = names.index('radius')
    x = positons[nameI, 1::everyNth, 0]
    y = positons[nameI, 1::everyNth, 1]
    z = positons[nameI, 1::everyNth, 2]
    t = np.arange(0, x.shape[0])
    if show:
        ax1_0.scatter(x, z, y, marker='.')

    #initialP = initalPositons[names.index('hand_complete')]
    #x = dataIn[2]['kinematics']['hand_xtrans'].to_numpy()[1::everyNth] + initialP[0]
    #y = dataIn[2]['kinematics']['hand_ytrans'].to_numpy()[1::everyNth] + initialP[1]
    #z = dataIn[2]['kinematics']['hand_ztrans'].to_numpy()[1::everyNth] + initialP[2]
    nameI = names.index('hand_complete')
    x = positons[nameI, 1::everyNth, 0]
    y = positons[nameI, 1::everyNth, 1]
    z = positons[nameI, 1::everyNth, 2]
    t = np.arange(0, x.shape[0])

    if show:
        ax1_0.scatter(x, z, y, marker='o')

    if show:
        x2 = []
        y2 = []
        z2 = []
        nPoints = 50
        for ii in range(nPoints):
            x2.append(positons[nameI, int(np.shape(positons)[1]/nPoints*ii):int(np.shape(positons)[1]/nPoints*ii + 1), 0])
            y2.append(positons[nameI, int(np.shape(positons)[1]/nPoints*ii):int(np.shape(positons)[1]/nPoints*ii + 1), 1])
            z2.append(positons[nameI, int(np.shape(positons)[1]/nPoints*ii):int(np.shape(positons)[1]/nPoints*ii + 1), 2])
        ax1_0.scatter(x2, z2, y2, marker='o')

    # From: https://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
    data = np.concatenate((x[:, np.newaxis],
                       y[:, np.newaxis],
                       z[:, np.newaxis]),
                      axis=1)
    datamean = data.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data - datamean)
    linepts = vv[0] * np.mgrid[-1.*abs(data.max() - data.min()):abs(data.max() - data.min()):2j][:, np.newaxis]
    linepts += datamean

    def convertOpenSim2Graph(a):
        b = np.array(a)
        a0 = np.array(b[1,:])
        b[1,:] = b[2,:]
        b[2,:] = a0
        return b

    if show:
        #ax.scatter3D(*data.T)
        ax1_0.plot3D(*convertOpenSim2Graph(linepts.T))


    zPlaneLine = np.array(linepts.T)
    zPlaneLine[2, 0] = zPlaneLine[2,1]
    if show:
        ax1_0.plot3D(*convertOpenSim2Graph(zPlaneLine))

    verticalLine1 = np.array(linepts.T)
    verticalLine1[0, 0] = verticalLine1[0,1]
    verticalLine1[2, 0] = verticalLine1[2,1]
    if show:
        ax1_0.plot3D(*convertOpenSim2Graph(verticalLine1))
    
    horizontalLine1 = np.array(linepts.T)
    horizontalLine1[2, 0] = horizontalLine1[2,1]
    horizontalLine1[1, 0] = horizontalLine1[1,1]
    if show:
        ax1_0.plot3D(*convertOpenSim2Graph(horizontalLine1))

    #d1 = zPlaneLine - verticalLine1
    #d2 = zPlaneLine - horizontalLine1
    
    v1 = fm.normalizeVector(zPlaneLine.T[0, :] - zPlaneLine.T[1, :])
    v2 = fm.normalizeVector(verticalLine1.T[0, :] - verticalLine1.T[1, :])
    v3 = fm.normalizeVector(horizontalLine1.T[0, :] - horizontalLine1.T[1, :])
    if show:
        v1Plot = (np.array([[0.,0.,0.],v1*0.015]) + zPlaneLine.T[1, :]).T
        ax1_0.plot3D(*convertOpenSim2Graph(v1Plot))
        v1Plot = (np.array([[0.,0.,0.],v3*0.015]) + zPlaneLine.T[1, :]).T
        ax1_0.plot3D(*convertOpenSim2Graph(v1Plot))

        m0 = min(min(ax1_0.get_zlim()), min(ax1_0.get_xlim()))
        m1 = max(max(ax1_0.get_zlim()), max(ax1_0.get_xlim()))
        ax1_0.set_zlim(m0, m1)
        ax1_0.set_xlim(m0, m1)

    rotM = fm.rotation_matrix_from_vectors2(v1, v2) # Used with FE motions
    rotM2 = fm.rotation_matrix_from_vectors2(v3, v1)
    v4 = fm.rotateVector(v1, rotM)
    if show:
        v1Plot = (np.array([[0.,0.,0.],v4*0.015]) + zPlaneLine.T[1, :]).T
        ax1_0.plot3D(*v1Plot)

    # flip direction of rotation
    rotM = rotM * [[1., -1., 0.], [-1., 1., 0.], [0., 0., 1.]]

    # save rotations to pickel
    from slil.common.cache import loadCache, saveCache
    dictContents = {
        'vectorMean': linepts.T,
        'vectorOnZPlane': v1,
        'vectorVertical': v2,
        'vectorHorizontal': v3,
        'rot1': rotM,
        'rot2': rotM2,
        'angle1': np.rad2deg(fm.angleBetweenVectors(v1, v2)),
        'angle2': np.rad2deg(fm.angleBetweenVectors(v1, v3))
    }

    try:
        dataRotations = loadCache('dataRotations')
    except FileNotFoundError:
        dataRotations = []

    name = modelInfo['experimentID'] + '_' + modelInfo['currentModel']
    
    exisitngIndex = [i for i, x in enumerate(dataRotations) if name in x.keys()]
    if exisitngIndex:
        dataRotations[exisitngIndex[0]] = {
            name: dictContents
        }
    else:
        dataRotations.append({
            name: dictContents
        })
    
    saveCache(dataRotations, 'dataRotations')

    if show:
        #initialP = initalPositons[names.index('lunate')]
        #x = dataIn[2]['kinematics']['lunate_xtrans'].to_numpy()[1::everyNth] + initialP[0]
        #y = dataIn[2]['kinematics']['lunate_ytrans'].to_numpy()[1::everyNth] + initialP[1]
        #z = dataIn[2]['kinematics']['lunate_ztrans'].to_numpy()[1::everyNth] + initialP[2]
        nameI = names.index('lunate')
        x = positons[nameI, 1::everyNth, 0]
        y = positons[nameI, 1::everyNth, 1]
        z = positons[nameI, 1::everyNth, 2]
        t = np.arange(0, x.shape[0])

        ax1_0.scatter(x, z, y, marker='x')

        #initialP = initalPositons[names.index('scaphoid')]
        #x = dataIn[2]['kinematics']['sca_xtran'].to_numpy()[1::everyNth] + initialP[0]
        #y = dataIn[2]['kinematics']['sca_ytran'].to_numpy()[1::everyNth] + initialP[1]
        #z = dataIn[2]['kinematics']['sca_ztran'].to_numpy()[1::everyNth] + initialP[2]
        nameI = names.index('scaphoid')
        x = positons[nameI, 1::everyNth, 0]
        y = positons[nameI, 1::everyNth, 1]
        z = positons[nameI, 1::everyNth, 2]
        t = np.arange(0, x.shape[0])

        ax1_0.scatter(x, z, y, marker='x')

#%%
