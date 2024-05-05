# Author: Alastair Quinn 2022
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.cbook import get_sample_data
from slil.common.cache import saveCache, loadCache
from slil.common.plotting_functions import figColours, figNames, cropData3
import copy
from slil.common.data_configs import outputFolders
from slil.common.math import nan_helper

def plotGroupAvg(models):
    
    for i in range(2):
        if i > 0:
            motionType = 'UR'
        else:
            motionType = 'FE'
            
        l2c, l2n, l2s = 0, 0, 0
        for model in models:
            dataIn = model['data' + motionType + 'cropped']
            for data in dataIn:
                if ('cut' in data['title']):
                    l2c += 1
                if ('normal' in data['title']):
                    l2n += 1
                if ('scaffold' in data['title']):
                    l2s += 1
        
        def toDegreesAndFlip(data, xyz):
            # flexion extension
            r1 = data['kinematics'][xyz[0]].to_numpy()*(180.0/np.pi)
            # ulnar radial deviation
            r2 = data['kinematics'][xyz[1]].to_numpy()*(180.0/np.pi)
            # pronation supination
            r3 = data['kinematics'][xyz[2]].to_numpy()*(180.0/np.pi)
            return [r1, r2, r3]

        l1 = models[0]['data' + motionType + 'cropped'][0]['kinematics']['lunate_flexion'].shape[0]
        arrN = np.zeros((9, l2n, l1), dtype=float)
        arrC = np.zeros((9, l2c, l1), dtype=float)
        arrS = np.zeros((9, l2s, l1), dtype=float)
        for model in models:
            dataIn = model['data' + motionType + 'cropped']
            for data in dataIn:
                [r1_1, r1_2, r1_3] = toDegreesAndFlip(data, ['lunate_flexion', 'lunate_deviation', 'lunate_rotation'])
                [r2_1, r2_2, r2_3] = toDegreesAndFlip(data, ['sca_xrot', 'sca_yrot', 'sca_zrot'])
                [r3_1, r3_2, r3_3] = toDegreesAndFlip(data, ['hand_flexion', 'hand_deviation', 'hand_rotation'])
                if (not model['isLeftHand']):
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

        useTime = True

        if useTime:
            rot = models[0]['data' + motionType + 'cropped'][0]['kinematics']['time'].to_numpy()
            # make sure time starts at zero. May not due to data cropping
            if (rot[0] != 0.0):
                rot = rot - rot[0]
        else:
            # raw rot data is in positive direction for both FE and UR
            # UR rotation was clock wise first
            rot = models[0]['data' + motionType + 'cropped'][0]['rot']
            if (models[0]['data' + motionType + 'cropped'][0]['type'] == 'FE'):
                rot *= -1.0
            if (models[0]['data' + motionType + 'cropped'][0]['type'] == 'UR' and not models[0]['isLeftHand']):
                rot *= -1.0


        fig = plt.figure()
        fig.set_size_inches(14.5, 14.5)
        gs0 = gridspec.GridSpec(4, 3, figure=fig)#, height_ratios=[3,3,1])
        ax = []
        for i in range(0, 9):
            ax.append(fig.add_subplot(gs0[i]))


        for cat in ['normal', 'cut', 'scaffold']:
            lineColour = figColours[cat]
            label1 = figNames[cat]
            if ( cat =='normal' ):
                arr = arrN
            if ( cat =='cut'):
                arr = arrC
            if ( cat =='scaffold' ):
                arr = arrS

            for ind, ax0 in enumerate(ax):
                mean = arr[ind].mean(axis=0)
                ax0.plot(rot, mean, color=lineColour, linewidth=0.9, alpha=0.9, label=label1)
                ax0.fill_between(rot,
                    mean - 1*arr[i].std(axis=0),
                    mean + 1*arr[i].std(axis=0),
                    color=lineColour, alpha=0.2)
        
        for ind, ax0 in enumerate(ax):
            meanN = arrN[ind].mean(axis=0)
            meanC = arrC[ind].mean(axis=0)
            meanS = arrS[ind].mean(axis=0)
            rmseNC = np.sqrt(np.mean((meanN-meanC)**2))
            rmseNS = np.sqrt(np.mean((meanN-meanS)**2))
            #mC = (meanC-meanN).mean()
            #SsqNC = np.sum(((meanC-meanN) - mC)**2)/(arrC.shape[2]-1)
            #mS = (meanS-meanN).mean()
            #SsqNS = np.sum(((meanS-meanN) - mS)**2)/(arrS.shape[2]-1)
            #SsqNC = np.sum((meanN-meanC)**2)/(arrC.shape[2]-1)
            #SsqNS = np.sum((meanN-meanS)**2)/(arrS.shape[2]-1)
            ax0.text(0.65, 0.15, "$RMSE_T$: {:.2f}".format(rmseNC),
            verticalalignment='center',
            rotation='horizontal', fontsize=14, transform=ax0.transAxes)
            ax0.text(0.65, 0.05, "$RMSE_I$ : {:.2f}".format(rmseNS),
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
        axAdjust(ax[0], ylabel = "Rotation (degrees)", title = "Flexion-Extension")
        axAdjust(ax[1], title = "Radial-Ulnar Deviation")
        axAdjust(ax[2], title = "Pro-Supination")
        axAdjust(ax[3], ylabel = "Rotation (degrees)")
        axAdjust(ax[4])
        axAdjust(ax[5])
        axAdjust(ax[6], ylabel = "Rotation (degrees)", xlabel = "Time (seconds)")
        axAdjust(ax[7], xlabel = "Time (seconds)")
        axAdjust(ax[8], xlabel = "Time (seconds)")
        #ax1_0.set_xlim([minTime,maxTime])
        #ax1_0.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
        #ax1_0.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
        figTitle = "Individual Bone Motions for Wrist Flexion-Extension"
        if 'UR' in dataIn[0]['type']:
            figTitle = "Individual Bone Motions for Wrist Radial-Ulnar Deviation"
        ax[1].text(-0.5, 1.3, figTitle,
            verticalalignment='center',
            rotation='horizontal', fontsize=20, transform=ax[1].transAxes)

        ax[0].text(-0.35, 0.5, 'Lunate',
            verticalalignment='center',
            rotation='vertical', fontsize=18, transform=ax[0].transAxes)
        ax[3].text(-0.35, 0.5, 'Scaphoid',
            verticalalignment='center',
            rotation='vertical', fontsize=18, transform=ax[3].transAxes)
        ax[6].text(-0.35, 0.5, '$3^{rd}$ Metacarpal',
            verticalalignment='center',
            rotation='vertical', fontsize=18, transform=ax[6].transAxes)

        note = "$RMSE_T$: Root-mean-squared-error between {} and {}\n$RMSE_I$ : Root-mean-squared-error between {} and {}".format(figNames['normal'], figNames['cut'], figNames['normal'], figNames['scaffold'])
        ax[6].text(-0.3, -0.8, note,
            verticalalignment='center',
            rotation='horizontal', fontsize=18, transform=ax[6].transAxes)

        ax[2].legend(bbox_to_anchor=(0.75, 1.05))
        
        def addIcon(img, coord):
            im = plt.imread(get_sample_data(models[0]['graphicsInputDir'] + "\\" + img))
            newax = fig.add_axes(coord, anchor='NE', zorder=-1)
            newax.imshow(im)
            newax.axis('off')
            
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
                addIcon("wrist_extension.png", [0.160, b, s, s])
                addIcon("wrist_extension.png", [0.160 + d, b, s, s])
                addIcon("wrist_extension.png", [0.160 + d + d, b, s, s])
                addIcon("wrist_flexion.png",   [0.269, b - 0.025, s, s])
                addIcon("wrist_flexion.png",   [0.269 + d, b - 0.025, s, s])
                addIcon("wrist_flexion.png",   [0.269 + d + d, b - 0.025, s, s])
            else:
                b = 0.23
                d = 0.273
                addIcon("wrist_ulnar_deviation.png", [0.155, b, s, s])
                addIcon("wrist_ulnar_deviation.png", [0.155 + d, b, s, s])
                addIcon("wrist_ulnar_deviation.png", [0.155 + d + d, b, s, s])
                addIcon("wrist_radial_deviation.png",  [0.269, b, s, s])
                addIcon("wrist_radial_deviation.png",  [0.269 + d, b,  s, s])
                addIcon("wrist_radial_deviation.png",  [0.269 + d + d, b, s, s])
        else:
            b = 0.255
            d = 0.273
            if (dataIn[0]['type'] == 'FE'):
                addIcon("wrist_extension.png", [0.100, b, s, s])
                addIcon("wrist_extension.png", [0.100 + d, b, s, s])
                addIcon("wrist_extension.png", [0.100 + d + d, b, s, s])
                addIcon("wrist_flexion.png",   [0.320, b - 0.025, s, s])
                addIcon("wrist_flexion.png",   [0.320 + d, b - 0.025, s, s])
                addIcon("wrist_flexion.png",   [0.320 + d + d, b - 0.025, s, s])
            else:
                b = 0.25
                d = 0.273
                addIcon("wrist_ulnar_deviation.png", [0.095, b, s, s])
                addIcon("wrist_ulnar_deviation.png", [0.095 + d, b, s, s])
                addIcon("wrist_ulnar_deviation.png", [0.095 + d + d, b, s, s])
                addIcon("wrist_radial_deviation.png",  [0.320, b, s, s])
                addIcon("wrist_radial_deviation.png",  [0.320 + d, b,  s, s])
                addIcon("wrist_radial_deviation.png",  [0.320 + d + d, b, s, s])
        
        #fig.legend((, l2), ('Line 1', 'Line 2'), 'upper left')
        fig.patch.set_facecolor('white')
        fileName = r'\all_bone_kinematics_wSD_' + dataIn[0]['type']
        plt.savefig(
            fname = outputFolders()['graphics'] + fileName,
            dpi=600,
            facecolor=fig.get_facecolor(),
            transparent=False)
        plt.close() # prevent Jupyter from showing plot

def calcAndCacheMeans(models, motionType):

    # find min max closes to 0.05 of a degree
    # find indecies in groups of 0.05 degrees
    # group kinematics by index groups
    inc = 0.1 # split data into groups on this sized angle
    if ('UR' in motionType):
        minX = round(float(-30.1) / inc) * inc
        maxX = round(float(30.1) / inc) * inc
    else:
        minX = round(float(-40.1) / inc) * inc
        maxX = round(float(40.1) / inc) * inc
    xnew = np.array(np.arange(minX, maxX, inc))

    kinematicAngles = [
        'lunate_flexion', 'lunate_deviation', 'lunate_rotation',
        'sca_xrot', 'sca_yrot', 'sca_zrot',
        'hand_flexion', 'hand_deviation', 'hand_rotation'
    ]
    kinematicAnglesFlip = [
        'lunate_deviation', 'lunate_rotation',
        'sca_yrot', 'sca_zrot',
        'hand_deviation', 'hand_rotation'
    ]
    
    l = 14 * 2 * len(models)
    y2new = np.empty((3, len(kinematicAngles), l, len(xnew)), dtype=float)

    # these loops are really inefficient...
    for ind2, kinematicAngle in enumerate(kinematicAngles):
        l2c, l2n, l2s = 0, 0, 0
        for model in models:
            dataIn = cropData3(model['data' + motionType], 0, 2083)
            for ind1, data in enumerate(dataIn):
                # raw rot data is in positive direction for both FE and UR
                # UR rotation was clock wise first
                rot = copy.deepcopy(data['rot'])
                if (data['type'] == 'FE'):
                    rot *= -1.0
                if ((data['type'] == 'UR') and (not model['isLeftHand'])):
                    rot *= -1.0
                y = copy.deepcopy(data['kinematics'][kinematicAngle].to_numpy()*(180.0/np.pi))
                if (kinematicAngle in kinematicAnglesFlip) and (not model['isLeftHand']):
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
    saveCache([xnew, y2new, kinematicAngles], 'dataAllMeans_' + motionType)

def plotGroupAvgMeans(models):
    
    for i in range(2):
        if i > 0:
            motionType = 'UR'
        else:
            motionType = 'FE'
        
        m = loadCache('dataAllMeans_' + motionType)
        xnew = m[0]
        y2new = m[1]
        inc = xnew[1] - xnew[0]
        minX = min(xnew)
        maxX = max(xnew) + inc
    

        fig = plt.figure()
        fig.set_size_inches(14.5, 14.5)
        gs0 = gridspec.GridSpec(4, 3, figure=fig)#, height_ratios=[3,3,1])
        ax = []
        for i in range(0, 9):
            ax.append(fig.add_subplot(gs0[i]))
        
        xnew = np.arange(minX, maxX, inc)

        for indFig, cat in enumerate(['normal', 'cut', 'scaffold']):
            for indAx, ax0 in enumerate(ax):
                mean = np.nanmean(y2new[indFig,indAx,:,:], axis=0)
                ax0.plot(xnew, mean, color=figColours[cat], linewidth=0.9, alpha=0.9, label=figNames[cat])
                # + - one standard deviation
                ax0.fill_between(xnew,
                    mean - 1*np.nanstd(y2new[indFig,indAx,:,:], axis=0),
                    mean + 1*np.nanstd(y2new[indFig,indAx,:,:], axis=0),
                    color=figColours[cat], alpha=0.2)

        for indAx, ax0 in enumerate(ax):
            meanN = np.nanmean(y2new[0,indAx,:,:], axis=0)
            nans, x1= nan_helper(meanN)
            meanN[nans]= 0.0
            meanC = np.nanmean(y2new[1,indAx,:,:], axis=0)
            nans, x1= nan_helper(meanC)
            meanC[nans]= 0.0
            meanS = np.nanmean(y2new[2,indAx,:,:], axis=0)
            nans, x1= nan_helper(meanS)
            meanS[nans]= 0.0
            rmseNC = np.sqrt(np.mean((meanN-meanC)**2))
            rmseNS = np.sqrt(np.mean((meanN-meanS)**2))
            ax0.text(0.65, 0.15, "$RMSE_T$: {:.2f}".format(rmseNC),
            verticalalignment='center',
            rotation='horizontal', fontsize=14, transform=ax0.transAxes)
            ax0.text(0.65, 0.05, "$RMSE_I$ : {:.2f}".format(rmseNS),
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
            ax.set_ylim([minX - 5.0, maxX + 5.0])
            ax.set_xlim([minX - 5.0, maxX + 5.0])
            ax.xaxis.label.set_size(15)
            ax.yaxis.label.set_size(15)
            ax.title.set_size(18)
            ax.xaxis.set_tick_params(labelsize=13)
            ax.yaxis.set_tick_params(labelsize=13)
        axAdjust(ax[0], ylabel = "Rotation (degrees)", title = "Flexion-Extension")
        axAdjust(ax[1], title = "Radial-Ulnar Deviation")
        axAdjust(ax[2], title = "Pro-Supination")
        axAdjust(ax[3], ylabel = "Rotation (degrees)")
        axAdjust(ax[4])
        axAdjust(ax[5])
        axAdjust(ax[6], ylabel = "Rotation (degrees)", xlabel = "Wrist Angle (degrees)")
        axAdjust(ax[7], xlabel = "Wrist Angle (degrees)")
        axAdjust(ax[8], xlabel = "Wrist Angle (degrees)")
        #ax1_0.set_xlim([minTime,maxTime])
        #ax1_0.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
        #ax1_0.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
        figTitle = "Individual Bone Motions for Wrist Flexion-Extension"
        if 'UR' in motionType:
            figTitle = "Individual Bone Motions for Wrist Radial-Ulnar Deviation"
        ax[1].text(-0.5, 1.3, figTitle,
            verticalalignment='center',
            rotation='horizontal', fontsize=20, transform=ax[1].transAxes)

        ax[0].text(-0.35, 0.5, 'Lunate',
            verticalalignment='center',
            rotation='vertical', fontsize=18, transform=ax[0].transAxes)
        ax[3].text(-0.35, 0.5, 'Scaphoid',
            verticalalignment='center',
            rotation='vertical', fontsize=18, transform=ax[3].transAxes)
        ax[6].text(-0.35, 0.5, '$3^{rd}$ Metacarpal',
            verticalalignment='center',
            rotation='vertical', fontsize=18, transform=ax[6].transAxes)

        note = "$RMSE_T$: Root-mean-squared-error between {} and {}\n$RMSE_I$ : Root-mean-squared-error between {} and {}".format(figNames['normal'], figNames['cut'], figNames['normal'], figNames['scaffold'])
        ax[6].text(-0.3, -0.8, note,
            verticalalignment='center',
            rotation='horizontal', fontsize=18, transform=ax[6].transAxes)

        ax[2].legend(bbox_to_anchor=(0.75, 1.05))
        
        def addIcon(img, coord):
            im = plt.imread(get_sample_data(models[0]['graphicsInputDir'] + "\\" + img))
            newax = fig.add_axes(coord, anchor='NE', zorder=-1)
            newax.imshow(im)
            newax.axis('off')
            
        s = 0.05 # figure size
        b = 0.255
        d = 0.273
        if ('FE' in motionType):
            addIcon("wrist_extension.png", [0.100, b, s, s])
            addIcon("wrist_extension.png", [0.100 + d, b, s, s])
            addIcon("wrist_extension.png", [0.100 + d + d, b, s, s])
            addIcon("wrist_flexion.png",   [0.320, b - 0.025, s, s])
            addIcon("wrist_flexion.png",   [0.320 + d, b - 0.025, s, s])
            addIcon("wrist_flexion.png",   [0.320 + d + d, b - 0.025, s, s])
        else:
            b = 0.25
            d = 0.273
            addIcon("wrist_ulnar_deviation.png", [0.095, b, s, s])
            addIcon("wrist_ulnar_deviation.png", [0.095 + d, b, s, s])
            addIcon("wrist_ulnar_deviation.png", [0.095 + d + d, b, s, s])
            addIcon("wrist_radial_deviation.png",  [0.320, b, s, s])
            addIcon("wrist_radial_deviation.png",  [0.320 + d, b,  s, s])
            addIcon("wrist_radial_deviation.png",  [0.320 + d + d, b, s, s])
        
        #fig.legend((, l2), ('Line 1', 'Line 2'), 'upper left')
        fig.patch.set_facecolor('white')
        fileName = r'\all_bone_kinematics_mean_wSD_' + motionType
        plt.savefig(
            fname = outputFolders()['graphics'] + fileName,
            dpi=600,
            facecolor=fig.get_facecolor(),
            transparent=False)
        plt.close() # prevent Jupyter from showing plot

def plotGroupAvgMeansRelative(models):
    # relative to normal
    # not sure if it makes sense to compare without relative
    
    for i in range(2):
        if i > 0:
            motionType = 'UR'
        else:
            motionType = 'FE'
        
        m = loadCache('dataAllMeans_' + motionType)
        xnew = m[0]
        y2new = m[1]
        kinematicAngles = m[2]
        inc = xnew[1] - xnew[0]
        minX = min(xnew)
        maxX = max(xnew) + inc

        l = 14 * 2 * len(models)
        
        for ind2, kinematicAngle in enumerate(kinematicAngles):
            for ind1 in range(l):
                y2new[1, ind2, ind1, :] = y2new[1, ind2, ind1, :] - y2new[0, ind2, ind1, :]
                y2new[2, ind2, ind1, :] = y2new[2, ind2, ind1, :] - y2new[0, ind2, ind1, :]
    
        fig = plt.figure()
        fig.set_size_inches(14.5, 14.5)
        gs0 = gridspec.GridSpec(4, 3, figure=fig)#, height_ratios=[3,3,1])
        ax = []
        for i in range(0, 9):
            ax.append(fig.add_subplot(gs0[i]))
        
        xnew = np.arange(minX, maxX, inc)

        for indFig, cat in enumerate(['cut', 'scaffold']): # 'normal', 
            for indAx, ax0 in enumerate(ax):
                mean = np.nanmean(y2new[indFig + 1,indAx,:,:], axis=0)
                ax0.plot(xnew, mean, color=figColours[cat], linewidth=0.9, alpha=0.9, label=figNames[cat])
                # + - one standard deviation
                ax0.fill_between(xnew,
                    mean - 1*np.nanstd(y2new[indFig + 1,indAx,:,:], axis=0),
                    mean + 1*np.nanstd(y2new[indFig + 1,indAx,:,:], axis=0),
                    color=figColours[cat], alpha=0.2)

        for indAx, ax0 in enumerate(ax):
            meanN = np.nanmean(y2new[0,indAx,:,:], axis=0)
            meanN[:]= 0.0
            meanC = np.nanmean(y2new[1,indAx,:,:], axis=0)
            nans, x1= nan_helper(meanC)
            meanC[nans]= 0.0
            meanS = np.nanmean(y2new[2,indAx,:,:], axis=0)
            nans, x1= nan_helper(meanS)
            meanS[nans]= 0.0
            rmseNC = np.sqrt(np.mean((meanN-meanC)**2))
            rmseNS = np.sqrt(np.mean((meanN-meanS)**2))
            ax0.text(0.65, 0.15, "$RMSE_T$: {:.2f}".format(rmseNC),
            verticalalignment='center',
            rotation='horizontal', fontsize=14, transform=ax0.transAxes)
            ax0.text(0.65, 0.05, "$RMSE_I$ : {:.2f}".format(rmseNS),
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
            ax.set_ylim([minX - 5.0, maxX + 5.0])
            ax.set_xlim([minX - 5.0, maxX + 5.0])
            ax.xaxis.label.set_size(15)
            ax.yaxis.label.set_size(15)
            ax.title.set_size(18)
            ax.xaxis.set_tick_params(labelsize=13)
            ax.yaxis.set_tick_params(labelsize=13)
        axAdjust(ax[0], ylabel = "Rotation (degrees)", title = "Flexion-Extension")
        axAdjust(ax[1], title = "Radial-Ulnar Deviation")
        axAdjust(ax[2], title = "Pro-Supination")
        axAdjust(ax[3], ylabel = "Rotation (degrees)")
        axAdjust(ax[4])
        axAdjust(ax[5])
        axAdjust(ax[6], ylabel = "Rotation (degrees)", xlabel = "Wrist Angle (degrees)")
        axAdjust(ax[7], xlabel = "Wrist Angle (degrees)")
        axAdjust(ax[8], xlabel = "Wrist Angle (degrees)")
        #ax1_0.set_xlim([minTime,maxTime])
        #ax1_0.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
        #ax1_0.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
        figTitle = "Individual Bone Motions for Wrist Flexion-Extension - Relative to " + figNames['normal']
        if 'UR' in motionType:
            figTitle = "Individual Bone Motions for Wrist Radial-Ulnar Deviation - Relative to " + figNames['normal']
        ax[1].text(-1.0, 1.3, figTitle,
            verticalalignment='center',
            rotation='horizontal', fontsize=20, transform=ax[1].transAxes)

        ax[0].text(-0.35, 0.5, 'Lunate',
            verticalalignment='center',
            rotation='vertical', fontsize=18, transform=ax[0].transAxes)
        ax[3].text(-0.35, 0.5, 'Scaphoid',
            verticalalignment='center',
            rotation='vertical', fontsize=18, transform=ax[3].transAxes)
        ax[6].text(-0.35, 0.5, '$3^{rd}$ Metacarpal',
            verticalalignment='center',
            rotation='vertical', fontsize=18, transform=ax[6].transAxes)

        note = "$RMSE_T$: Root-mean-squared-error between {} and {}\n$RMSE_I$ : Root-mean-squared-error between {} and {}".format(figNames['normal'], figNames['cut'], figNames['normal'], figNames['scaffold'])
        ax[6].text(-0.3, -0.8, note,
            verticalalignment='center',
            rotation='horizontal', fontsize=18, transform=ax[6].transAxes)

        ax[2].legend(bbox_to_anchor=(0.75, 1.05))
        
        def addIcon(img, coord):
            im = plt.imread(get_sample_data(models[0]['graphicsInputDir'] + "\\" + img))
            newax = fig.add_axes(coord, anchor='NE', zorder=-1)
            newax.imshow(im)
            newax.axis('off')
            
        s = 0.05 # figure size
        b = 0.255
        d = 0.273
        if ('FE' in motionType):
            addIcon("wrist_extension.png", [0.100, b, s, s])
            addIcon("wrist_extension.png", [0.100 + d, b, s, s])
            addIcon("wrist_extension.png", [0.100 + d + d, b, s, s])
            addIcon("wrist_flexion.png",   [0.320, b - 0.025, s, s])
            addIcon("wrist_flexion.png",   [0.320 + d, b - 0.025, s, s])
            addIcon("wrist_flexion.png",   [0.320 + d + d, b - 0.025, s, s])
        else:
            b = 0.25
            d = 0.273
            addIcon("wrist_ulnar_deviation.png", [0.095, b, s, s])
            addIcon("wrist_ulnar_deviation.png", [0.095 + d, b, s, s])
            addIcon("wrist_ulnar_deviation.png", [0.095 + d + d, b, s, s])
            addIcon("wrist_radial_deviation.png",  [0.320, b, s, s])
            addIcon("wrist_radial_deviation.png",  [0.320 + d, b,  s, s])
            addIcon("wrist_radial_deviation.png",  [0.320 + d + d, b, s, s])
        
        #fig.legend((, l2), ('Line 1', 'Line 2'), 'upper left')
        fig.patch.set_facecolor('white')
        fileName = r'\all_bone_kinematics_mean_relativeNormal_wSD_' + motionType
        plt.savefig(
            fname = outputFolders()['graphics'] + fileName,
            dpi=600,
            facecolor=fig.get_facecolor(),
            transparent=False)
        plt.close() # prevent Jupyter from showing plot

def plotGroupAvgMeansRelativeSL(models):
    # remove 3rd metacarpal
    # relative to normal
    # not sure if it makes sense to compare without relative
    
    for i in range(2):
        if i > 0:
            motionType = 'UR'
        else:
            motionType = 'FE'
        
        m = loadCache('dataAllMeans_' + motionType)
        xnew = m[0]
        y2new = m[1]
        kinematicAngles = m[2]
        inc = xnew[1] - xnew[0]
        minX = min(xnew)
        maxX = max(xnew) + inc

        l = 14 * 2 * len(models)
        
        for ind2, kinematicAngle in enumerate(kinematicAngles):
            for ind1 in range(l):
                y2new[1, ind2, ind1, :] = y2new[1, ind2, ind1, :] - y2new[0, ind2, ind1, :]
                y2new[2, ind2, ind1, :] = y2new[2, ind2, ind1, :] - y2new[0, ind2, ind1, :]
    
        fig = plt.figure()
        fig.set_size_inches(14.5, 14.5)
        gs0 = gridspec.GridSpec(4, 3, figure=fig)#, height_ratios=[3,3,1])
        ax = []
        for i in range(0, 6):
            ax.append(fig.add_subplot(gs0[i]))
        
        xnew = np.arange(minX, maxX, inc)

        for indFig, cat in enumerate(['cut', 'scaffold']): # 'normal', 
            for indAx, ax0 in enumerate(ax):
                mean = np.nanmean(y2new[indFig + 1,indAx,:,:], axis=0)
                ax0.plot(xnew, mean, color=figColours[cat], linewidth=0.9, alpha=0.9, label=figNames[cat])
                # + - one standard deviation
                ax0.fill_between(xnew,
                    mean - 1*np.nanstd(y2new[indFig + 1,indAx,:,:], axis=0),
                    mean + 1*np.nanstd(y2new[indFig + 1,indAx,:,:], axis=0),
                    color=figColours[cat], alpha=0.2)

        for indAx, ax0 in enumerate(ax):
            meanN = np.nanmean(y2new[0,indAx,:,:], axis=0)
            meanN[:]= 0.0
            meanC = np.nanmean(y2new[1,indAx,:,:], axis=0)
            nans, x1= nan_helper(meanC)
            meanC[nans]= 0.0
            meanS = np.nanmean(y2new[2,indAx,:,:], axis=0)
            nans, x1= nan_helper(meanS)
            meanS[nans]= 0.0
            rmseNC = np.sqrt(np.mean((meanN-meanC)**2))
            rmseNS = np.sqrt(np.mean((meanN-meanS)**2))
            ax0.text(0.65, 0.15, "$RMSE_T$: {:.2f}".format(rmseNC),
            verticalalignment='center',
            rotation='horizontal', fontsize=14, transform=ax0.transAxes)
            ax0.text(0.65, 0.05, "$RMSE_I$ : {:.2f}".format(rmseNS),
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
            ax.set_ylim([minX - 5.0, maxX + 5.0])
            ax.set_xlim([minX - 5.0, maxX + 5.0])
            ax.xaxis.label.set_size(15)
            ax.yaxis.label.set_size(15)
            ax.title.set_size(18)
            ax.xaxis.set_tick_params(labelsize=13)
            ax.yaxis.set_tick_params(labelsize=13)
        axAdjust(ax[0], ylabel = "Rotation (degrees)", title = "Flexion-Extension")
        axAdjust(ax[1], title = "Radial-Ulnar Deviation")
        axAdjust(ax[2], title = "Pro-Supination")
        axAdjust(ax[3], ylabel = "Rotation (degrees)", xlabel = "Wrist Angle (degrees)")
        axAdjust(ax[4], xlabel = "Wrist Angle (degrees)")
        axAdjust(ax[5], xlabel = "Wrist Angle (degrees)")
        #ax1_0.set_xlim([minTime,maxTime])
        #ax1_0.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
        #ax1_0.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
        figTitle = "Individual Bone Motions for Wrist Flexion-Extension - Relative to " + figNames['normal']
        if 'UR' in motionType:
            figTitle = "Individual Bone Motions for Wrist Radial-Ulnar Deviation - Relative to " + figNames['normal']
        ax[1].text(-1.0, 1.3, figTitle,
            verticalalignment='center',
            rotation='horizontal', fontsize=20, transform=ax[1].transAxes)

        ax[0].text(-0.35, 0.5, 'Lunate',
            verticalalignment='center',
            rotation='vertical', fontsize=18, transform=ax[0].transAxes)
        ax[3].text(-0.35, 0.5, 'Scaphoid',
            verticalalignment='center',
            rotation='vertical', fontsize=18, transform=ax[3].transAxes)

        note = "$RMSE_T$: Root-mean-squared-error between {} and {}\n$RMSE_I$ : Root-mean-squared-error between {} and {}".format(figNames['normal'], figNames['cut'], figNames['normal'], figNames['scaffold'])
        ax[3].text(-0.3, -0.8, note,
            verticalalignment='center',
            rotation='horizontal', fontsize=18, transform=ax[3].transAxes)

        ax[2].legend(bbox_to_anchor=(0.75, 1.05))
        
        def addIcon(img, coord):
            im = plt.imread(get_sample_data(models[0]['graphicsInputDir'] + "\\" + img))
            newax = fig.add_axes(coord, anchor='NE', zorder=-1)
            newax.imshow(im)
            newax.axis('off')
            
        s = 0.05 # figure size
        b = 0.455
        d = 0.273
        if ('FE' in motionType):
            addIcon("wrist_extension.png", [0.100, b, s, s])
            addIcon("wrist_extension.png", [0.100 + d, b, s, s])
            addIcon("wrist_extension.png", [0.100 + d + d, b, s, s])
            addIcon("wrist_flexion.png",   [0.320, b - 0.025, s, s])
            addIcon("wrist_flexion.png",   [0.320 + d, b - 0.025, s, s])
            addIcon("wrist_flexion.png",   [0.320 + d + d, b - 0.025, s, s])
        else:
            b = 0.45
            d = 0.273
            addIcon("wrist_ulnar_deviation.png", [0.095, b, s, s])
            addIcon("wrist_ulnar_deviation.png", [0.095 + d, b, s, s])
            addIcon("wrist_ulnar_deviation.png", [0.095 + d + d, b, s, s])
            addIcon("wrist_radial_deviation.png",  [0.320, b, s, s])
            addIcon("wrist_radial_deviation.png",  [0.320 + d, b,  s, s])
            addIcon("wrist_radial_deviation.png",  [0.320 + d + d, b, s, s])
        
        #fig.legend((, l2), ('Line 1', 'Line 2'), 'upper left')
        fig.patch.set_facecolor('white')
        fileName = r'\all_S&L_bone_kinematics_mean_relativeNormal_wSD_' + motionType
        plt.savefig(
            fname = outputFolders()['graphics'] + fileName,
            dpi=600,
            facecolor=fig.get_facecolor(),
            transparent=False)
        plt.close() # prevent Jupyter from showing plot