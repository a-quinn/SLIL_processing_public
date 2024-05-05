# Author: Alastair Quinn 2022
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from slil.common.plotting_functions import figColours, figNames
import slil.common.data_configs as dc

def plotLengthChangeVsRotation(Xes, all, direction='FE'):
    fig = plt.figure()
    fig.set_size_inches(7, 6)
    gs0 = gridspec.GridSpec(1, 1, figure=fig)#, height_ratios=[3,3,1])
    figAxies = [
        fig.add_subplot(gs0[0])
    ]

    time = Xes[direction]['time']
    rot = Xes[direction]['rot']
    #time = modelInfo['data' + direction + 'cropped'][0]['time']
    time = time - time[0]
    #time = time[:4166]
    #rot = modelInfo['data' + direction + 'cropped'][0]['rot'] * -1.0
    #rot = rot[:4166]
    #rot = time

    def subplotSD(ax, x, y, label, colour):
        ax.plot(x, y.mean(axis=0), alpha=0.5, color=colour, label=label, linewidth = 1.0)
        ax.fill_between(x,
            y.mean(axis=0) - 2*y.std(axis=0),
            y.mean(axis=0) + 2*y.std(axis=0),
            color=colour, alpha=0.2) 
        ax.legend(loc='upper right')
        ax.set_ylabel("Length Change (%)", fontsize=18)
        ax.set_ylabel("Length (mm)", fontsize=18)
        degree_sign = u"\N{DEGREE SIGN}"
        ax.set_xlabel("Wrist rotation ("+degree_sign+")", fontsize=18)
        ax.set_xlabel("Wrist Angle (degrees)", fontsize=18)

    groupNormal = []
    groupCut = []
    groupScaffold = []
    dLen1 = all[0][list(all[0].keys())[0]].shape[0] * len(all)
    dLen2 = all[0]['normal'].shape[1]

    groupNormal = np.empty((dLen1, dLen2), float)
    groupCut = np.empty((dLen1, dLen2), float)
    groupScaffold = np.empty((dLen1, dLen2), float)

    for i, b in enumerate(all):
        jj = b['normal'].shape[0] * (i+1)
        j = b['normal'].shape[0] * i
        groupNormal[j:jj]=b['normal']
        groupCut[j:jj]=b['cut']
        groupScaffold[j:jj]=b['scaffold']
        #groupNormal = np.append(groupNormal, b['normal'], axis=1)
        #groupCut = np.append(groupNormal, b['cut'], axis=1)
        #groupScaffold = np.append(groupNormal, b['scaffold'], axis=1)

    subplotSD(figAxies[0], rot, groupNormal, figNames['normal'], figColours['normal'])
    subplotSD(figAxies[0], rot, groupCut, figNames['cut'], figColours['cut'])
    subplotSD(figAxies[0], rot, groupScaffold, figNames['scaffold'], figColours['scaffold'])

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    if (direction == 'FE'):
        figAxies[0].set_title('Wrist Motion: Flexion-Extension', fontsize=18)
    if (direction == 'UR'):
        figAxies[0].set_title('Wrist Motion: Radial-Ulnar Deviation', fontsize=18)

    note = "Bone-Plug to Bone-Plug Distance, Normalized to Implant Size".format()
    figAxies[0].text(0.0, -0.2, note,
        verticalalignment='center',
        rotation='horizontal', fontsize=12, transform=figAxies[0].transAxes)
    note = "$R_1$: Root-mean-squared-error between {} and {}\n$R_2$: Root-mean-squared-error between {} and {}".format(figNames['normal'], figNames['cut'], figNames['normal'], figNames['scaffold'])
    figAxies[0].text(0.0, -0.3, note,
        verticalalignment='center',
        rotation='horizontal', fontsize=12, transform=figAxies[0].transAxes)
    
    meanN = groupNormal.mean(axis=0)
    meanC = groupCut.mean(axis=0)
    meanS = groupScaffold.mean(axis=0)
    rmseNC = np.sqrt(np.mean((meanN - meanC)**2))
    rmseNS = np.sqrt(np.mean((meanN-meanS)**2))
    figAxies[0].text(0.84, 0.10, "$R_1$: {:.2f}".format(rmseNC),
    verticalalignment='center',
    rotation='horizontal', fontsize=14, transform=figAxies[0].transAxes)
    figAxies[0].text(0.84, 0.05, "$R_2$: {:.2f}".format(rmseNS),
    verticalalignment='center',
    rotation='horizontal', fontsize=14, transform=figAxies[0].transAxes)

    plt.savefig(
        fname = dc.outputFolders()['graphics'] + r'\all_LengthChangeVsWristRotation_wSD_' + direction,
        dpi = 600,
        facecolor=fig.get_facecolor(),
        transparent=False,
        bbox_inches="tight")

def plotRelativeLengthChangeVsRotation(Xes, allRel, direction='FE', showSD = False):
    # showSD = True
    fig = plt.figure()
    fig.set_size_inches(7, 6)
    gs0 = gridspec.GridSpec(1, 1, figure=fig)#, height_ratios=[3,3,1])
    figAxies = [
        fig.add_subplot(gs0[0])
    ]

    time = Xes[direction]['time']
    rot = Xes[direction]['rot']
    #time = modelInfo['data' + direction + 'cropped'][0]['time']
    #rot = modelInfo['data' + direction + 'cropped'][0]['rot'] * -1.0
    #time = time[:4166]
    rot = rot[:4166]

    dLen1 = allRel[0][list(allRel[0].keys())[0]].shape[0] * len(allRel)
    dLen2 = allRel[0]['cut'].shape[1]

    groupCut = np.empty((dLen1, dLen2), float)
    groupScaffold = np.empty((dLen1, dLen2), float)

    def subplotSD(ax, x, y, label, colour):
        ax.plot(x, y.mean(axis=0), alpha=0.5, color=colour, label=label, linewidth = 1.0)
        if (showSD):
            ax.fill_between(x,
                y.mean(axis=0) - 1*y.std(axis=0),
                y.mean(axis=0) + 1*y.std(axis=0),
                color=colour, alpha=0.2)
        ax.legend(loc='best')
        ax.set_xlabel('Wrist Angle (degrees)')
        degree_sign = u"\N{DEGREE SIGN}"
        #ax.set_xlabel("Wrist rotation ("+degree_sign+")", fontsize=18)
        ax.set_ylabel('Length change (mm)')
        ax.set_ylabel("Length (mm)", fontsize=18)

    for i, b in enumerate(allRel):
        jj = b['cut'].shape[0] * (i+1)
        j = b['cut'].shape[0] * i
        groupCut[j:jj]=b['cut']
        groupScaffold[j:jj]=b['scaffold']

    # find mean for each experiment
    dLen1 = len(allRel)
    dLen2 = allRel[0]['cut'].shape[1]

    groupCut2 = np.empty((dLen1, dLen2), float)
    groupScaffold2 = np.empty((dLen1, dLen2), float)
    for i, b in enumerate(allRel):
        jj = b['cut'].shape[0] * (i+1)
        j = b['cut'].shape[0] * i
        groupCut2[i] = groupCut[j:jj].mean(axis=0)
        groupScaffold2[i] = groupScaffold[j:jj].mean(axis=0)

    if False:
    #if (showSD):
        
        axisInd = [i for i, x in enumerate(kinematicAngles) if x == angleName][0]

        d1 = y2new[0, axisInd].shape[0]
        y2merge = np.empty((d1*3, groupCut2.shape[1]))
        #y2merge[:d1] = y2new[0, axisInd]
        y2merge[d1:d1*2] = y2new[1, axisInd]
        y2merge[d1*2:] = y2new[2, axisInd] 
        y2mergeInds = np.empty((d1*3)) # categories: normal, cut, scaffold
        y2mergeInds[:d1] = 0
        y2mergeInds[d1:d1*2] = 1
        y2mergeInds[d1*2:]= 2

        indToRemove = []
        for ind0, i0 in enumerate(y2merge):
            for ind1, i1 in enumerate(i0):
                if np.isnan(i1) and not (ind0 in indToRemove):
                    indToRemove.append(ind0)

        for ind, i in enumerate(indToRemove[::-1]):
            y2merge = np.delete(y2merge, i, axis=0)
            y2mergeInds = np.delete(y2mergeInds, i, axis=0)

        alpha=0.05

        t = spm1d.stats.ttest2( y2merge[y2mergeInds == 1], y2merge[y2mergeInds == 2] ) # c to s
        ti = t.inference(alpha, two_tailed=True, interp=False)

    subplotSD(figAxies[0], rot, groupCut2, figNames['cut'], figColours['cut'])
    subplotSD(figAxies[0], rot, groupScaffold2, figNames['scaffold'], figColours['scaffold'])
    plt.hlines(0, min(rot), max(rot), colors='black', linestyles='dotted', label='')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    if (direction == 'FE'):
        figAxies[0].set_title('Wrist Motion: Flexion-Extension', fontsize=18)
    if (direction == 'UR'):
        figAxies[0].set_title('Wrist Motion: Radial-Ulnar Deviation', fontsize=18)
    #figAxies[0].set_title('Change relative to intact', fontsize=18)

    note = "Bone-Plug to Bone-Plug Distance, Normalized to\nImplant Size and Relative to Intact".format()
    figAxies[0].text(0.0, -0.2, note,
        verticalalignment='center',
        rotation='horizontal', fontsize=12, transform=figAxies[0].transAxes)

    fileName = r'\all_RelativeLengthChangeVsWristRotation_'
    if (showSD):
        fileName = r'\all_RelativeLengthChangeVsWristRotation_wSD_'
    plt.savefig(
        fname = dc.outputFolders()['graphics'] + fileName + direction,
        dpi = 600,
        facecolor=fig.get_facecolor(),
        transparent=False,
        bbox_inches="tight")

def plotDistance(exps, all, direction='FE'):
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 4)

    trialTypes = ('noraml', 'cut', 'scaffold')
    x_pos = np.arange(len(exps))
    width = 0.15  # the width of the bars

    meanGroupNormal = []
    meanGroupCut = []
    meanGroupScaffold = []
    stdGroupNormal = []
    stdGroupCut = []
    stdGroupScaffold = []
    for i, b in enumerate(all):
        for j, c in enumerate(b):
            if (c == 'normal'):
                meanGroupNormal.append(b[c].mean())
                stdGroupNormal.append(b[c].std())
            if (c == 'cut'):
                meanGroupCut.append(b[c].mean())
                stdGroupCut.append(b[c].std())
            if (c == 'scaffold'):
                meanGroupScaffold.append(b[c].mean())
                stdGroupScaffold.append(b[c].std())

    groupNormal = ax.bar(x_pos - width, meanGroupNormal, width=width, yerr=stdGroupNormal,
        align='center', label=figNames['normal'], color=figColours['normal'])
    groupCut = ax.bar(x_pos, meanGroupCut, width=width, yerr=stdGroupCut,
        align='center', label=figNames['cut'], color=figColours['cut'])
    groupScaffold = ax.bar(x_pos + width, meanGroupScaffold, width=width, yerr=stdGroupScaffold,
        align='center', label=figNames['scaffold'], color=figColours['scaffold'])

    ax.set_xticks(x_pos)
    nums = np.arange(1,len(exps)+1)
    ax.set_xticklabels(nums)
    ax.set_xticklabels(exps)
    ax.set_xlabel('Experiment Number')
    #ax.set_title('Change relative to intact')
    if (direction == 'FE'):
        ax.set_title('Wrist Motion: Flexion-Extension')
    if (direction == 'UR'):
        ax.set_title('Wrist Motion: Radial-Ulnar Deviation')
    ax.set_ylabel('Length (mm)')
    ax.legend()

    note = "Bone-Plug to Bone-Plug Distance, Normalized to Implant Size".format()
    ax.text(0.0, -0.2, note,
        verticalalignment='center',
        rotation='horizontal', fontsize=8, transform=ax.transAxes)

    plt.savefig(
        fname = dc.outputFolders()['graphics'] + r'\all_Distance_' + direction,
        dpi = 600,
        facecolor=fig.get_facecolor(),
        transparent=False,
        bbox_inches="tight")

def plotLengthChange(exps, allRel, direction='FE'):
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 4)

    trialTypes = ('cut', 'scaffold')
    x_pos = np.arange(len(exps))
    width = 0.15  # the width of the bars

    meanGroupCut = []
    meanGroupScaffold = []
    stdGroupCut = []
    stdGroupScaffold = []
    for i, b in enumerate(allRel):
        for j, c in enumerate(b):
            if (c == 'cut'):
                meanGroupCut.append(b[c].mean())
                stdGroupCut.append(b[c].std())
            if (c == 'scaffold'):
                meanGroupScaffold.append(b[c].mean())
                stdGroupScaffold.append(b[c].std())

    groupCut = ax.bar(x_pos -width/2, meanGroupCut, width=width, yerr=stdGroupCut,
        align='center', label=figNames['cut'], color=figColours['cut'])
    groupScaffold = ax.bar(x_pos + width/2, meanGroupScaffold, width=width, yerr=stdGroupScaffold,
        align='center', label=figNames['scaffold'], color=figColours['scaffold'])

    ax.set_xticks(x_pos)
    nums = np.arange(1,len(exps)+1)
    ax.set_xticklabels(nums)
    ax.set_xticklabels(exps)
    ax.set_xlabel('Experiment Number')
    if (direction == 'FE'):
        ax.set_title('Wrist Motion: Flexion-Extension')
    if (direction == 'UR'):
        ax.set_title('Wrist Motion: Radial-Ulnar Deviation')
    ax.set_ylabel('Change of Length (mm)')
    ax.legend()
    
    note = "Bone-Plug to Bone-Plug Distance, Normalized to\nImplant Size and Relative to Intact".format()
    ax.text(0.0, -0.2, note,
        verticalalignment='center',
        rotation='horizontal', fontsize=8, transform=ax.transAxes)

    plt.savefig(
        fname = dc.outputFolders()['graphics'] + r'\all_RelativeLengthChange_' + direction,
        dpi = 600,
        facecolor=fig.get_facecolor(),
        transparent=False,
        bbox_inches="tight")

def plotRMSD(exps, all, allUR):
    #RMSE
    fig, (axFE, axUR) = plt.subplots(nrows = 2, ncols = 1)

    x_pos = np.arange(len(exps))
    width = 0.15  # the width of the bars
    fig.set_size_inches(6, 5)

    dLen1 = len(all)
    dLen2 = all[0]['normal'].shape[1]

    groupCut = np.empty((dLen1), float)
    groupScaffold = np.empty((dLen1), float)
    groupURCut = np.empty((dLen1), float)
    groupURScaffold = np.empty((dLen1), float)

    for i, b in enumerate(all):
        jj = b['cut'].shape[0] * (i+1)
        j = b['cut'].shape[0] * i
        #groupCut[i]=np.sqrt(np.mean((b['cut'].mean()-b['normal'].mean())**2))
        #groupScaffold[i]=np.sqrt(np.mean((b['scaffold'].mean()-b['normal'].mean())**2))
        groupCut[i]=np.sqrt(np.mean((b['normal']-b['cut'])**2))
        groupScaffold[i]=np.sqrt(np.mean((b['normal']-b['scaffold'])**2))
        #groupCut[i]=np.mean(b['normal']-b['cut'])
        #groupScaffold[i]=np.mean(b['normal']-b['scaffold'])

    for i, b in enumerate(allUR):
        jj = b['cut'].shape[0] * (i+1)
        j = b['cut'].shape[0] * i
        #groupURCut[i]=np.sqrt(np.mean((b['cut'].mean()-b['normal'].mean())**2))
        #groupURScaffold[i]=np.sqrt(np.mean((b['scaffold'].mean()-b['normal'].mean())**2))
        groupURCut[i]=np.sqrt(np.mean((b['normal']-b['cut'])**2))
        groupURScaffold[i]=np.sqrt(np.mean((b['normal']-b['scaffold'])**2))
        #groupURCut[i]=np.mean(b['normal']-b['cut'])
        #groupURScaffold[i]=np.mean(b['normal']-b['scaffold'])
    
    #from scipy import stats
    #tTestRes, tTestResP = stats.ttest_rel( groupCut, groupScaffold, axis=0)
    #tTestRes, tTestResP = stats.ttest_rel( groupURCut, groupURScaffold, axis=0)
    #dStat = {}
    #dStat['kurt'] = stats.kurtosis(groupCut, axis=None)
    #dStat['skew'] = stats.skew(groupCut, axis=None)
    #dStat['varianceB'] = stats.bartlett(groupCut, groupScaffold).pvalue
    #dStat['varianceF'] = stats.fligner(groupCut, groupScaffold).pvalue
    #dStat['normality'] = stats.shapiro(groupCut).pvalue # needs to be greater than 5% for normally distributed data
    #plt.hist(groupCut, bins=len(groupCut))
    #stats.wilcoxon( groupCut, groupScaffold)
    #stats.wilcoxon( groupURCut, groupURScaffold)


    # for only positive RMSD
    if (False):
        for i, b in enumerate(all):
            groupCut[i] = np.abs(groupCut[i])
            groupScaffold[i] = np.abs(groupScaffold[i])
        for i, b in enumerate(allUR):
            groupURCut[i] = np.abs(groupURCut[i])
            groupURScaffold[i] = np.abs(groupURScaffold[i])

    axFE.bar(x_pos - width/2, groupCut, color=figColours['cut'], width=width, align='center', label=figNames['cut'])
    axFE.bar(x_pos + width/2, groupScaffold, color=figColours['scaffold'], width=width, align='center', label=figNames['scaffold'])
    axUR.bar(x_pos - width/2, groupURCut, color=figColours['cut'], width=width, align='center', label='UR ' + figNames['cut'])
    axUR.bar(x_pos + width/2, groupURScaffold, color=figColours['scaffold'], width=width, align='center', label='UR ' + figNames['scaffold'])
    
    #groupCut = axFE.bar(x_pos - width*1.5, groupCut, color=figColours['cut'], width=width, align='center', label='FE ' + figNames['cut'])
    #groupScaffold = axFE.bar(x_pos - width/2, groupScaffold, color=figColours['scaffold'], width=width, align='center', label='FE ' + figNames['scaffold'])
    #groupURCut = axUR.bar(x_pos + width/2, groupURCut, color=figColours['cut'], width=width, align='center', label='UR ' + figNames['cut'])
    #groupURScaffold = axUR.bar(x_pos + width*1.5, groupURScaffold, color=figColours['scaffold'], width=width, align='center', label='UR ' + figNames['scaffold'])
    
    #groupCut = ax.bar(x_pos - width*1.5, groupCut, color=['blue'], width=width, align='center', label='FE Severed')
    #groupScaffold = ax.bar(x_pos - width/2, groupScaffold, color=['cornflowerblue'], width=width, align='center', label='FE Implant')
    #groupURCut = ax.bar(x_pos + width/2, groupURCut, color=['red'], width=width, align='center', label='UR Severed')
    #groupURScaffold = ax.bar(x_pos + width*1.5, groupURScaffold, color=['lightcoral'], width=width, align='center', label='UR Implant')
    yMax = max(axFE.get_ylim()[1], axUR.get_ylim()[1])
    axFE.set_ylim([0, yMax])
    axUR.set_ylim([0, yMax])

    steps = 0.1
    ticksY = np.arange(0, np.ceil(float(yMax) / steps) * steps, steps)
    axFE.set_yticks(ticksY)
    axUR.set_yticks(ticksY)
    #ticksYLables = np.arange(0, np.ceil(float(yMax) / steps) * steps, steps)
    #axFE.set_xticklabels(ticksYLables)
    #axUR.set_xticklabels(ticksYLables)

    axFE.set_xticks(x_pos)
    axUR.set_xticks(x_pos)
    nums = np.arange(1,len(exps)+1)
    axFE.set_xticklabels([])
    axUR.set_xticklabels(nums)
    #ax.set_xticklabels(experiments)
    axUR.set_xlabel('Experiment Number')
    #ax.set_title('Error relative to intact')
    axFE.set_title('Flexion-Extension')
    axUR.set_title('Radial-Ulnar Deviation', )
    axFE.set_ylabel('RMSE (mm)')
    axUR.set_ylabel('RMSE (mm)')
    axFE.legend(loc='upper right')
    #axUR.legend(loc='none')

    #plt.tight_layout()
    plt.savefig(
        fname = dc.outputFolders()['graphics'] + r'\all_RMSD',
        dpi = 600,
        facecolor=fig.get_facecolor(),
        transparent=False,
        bbox_inches="tight")
