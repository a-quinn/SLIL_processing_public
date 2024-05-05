# Author: Alastair Quinn 2022
#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.cbook import get_sample_data
from slil.common.plotting_functions import figColours, figNames, cropData3
from slil.common.cache import loadCache, saveCache, loadOnlyExps
from slil.common.data_configs import outputFolders
from slil.common.math import nan_helper
import copy
import spm1d

#%%
DEBUG = False

motionType = 'UR'
#motionType = 'FE'
    
relative2Metacarpal = True
makeRelativeToNormal = False

if 'FE' in motionType:
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
else:
    experiments = [
        #'11525', # good data but short UR
        #'11526', # only UR is bad
        '11527',
        '11534',
        '11535',
        '11536',
        #'11537', # only UR is bad
        '11538',
        '11539'
        ]
exps = experiments

if True:
#def plotGroupAvgMeansRelativeSLStats(exps):
    # remove 3rd metacarpal
    # relative to normal
    # not sure if it makes sense to compare without relative
    models = loadOnlyExps(exps)
    
    # find min max closes to 0.05 of a degree
    # find indecies in groups of 0.05 degrees
    # group kinematics by index groups
    inc = 0.1 # split data into groups on this sized angle
    # similar to sampling all points every incriment and finding mean for each kinematic axis
    if ('UR' in motionType):
        minX = round(float(-30.1) / inc) * inc
        maxX = round(float(30.1) / inc) * inc
    else:
        minX = round(float(-40.1) / inc) * inc
        maxX = round(float(40.1) / inc) * inc

    rotMin = []
    rotMax = []
    for model in models:
        dataIn = cropData3(model['data' + motionType], 2083, 4165)
        for ind1, data in enumerate(dataIn):
            rot = copy.deepcopy(data['rot'])
            if (data['type'] == 'FE'):
                rot *= -1.0
            if ((data['type'] == 'UR') and (not model['isLeftHand'])):
                rot *= -1.0
            rotMin.append(min(rot))
            rotMax.append(max(rot))
    minX = round(float(max(rotMin)) / inc) * inc
    maxX = round(float(min(rotMax)) / inc) * inc

    xnew = np.array(np.arange(minX, maxX, inc))
    
    # 3 types: noraml, cut, sacffold
    # x angles
    # 14 sections * 2 trials of each type
    kinematicAngles = [
        'lunate_flexion', 'lunate_deviation', 'lunate_rotation',
        'sca_xrot', 'sca_yrot', 'sca_zrot'#,
        #'hand_flexion', 'hand_deviation', 'hand_rotation'
    ]
    # flip these if model is right handed
    kinematicAnglesFlip = [
        'lunate_deviation', 'lunate_rotation',
        'sca_yrot', 'sca_zrot'#,
        #'hand_deviation', 'hand_rotation'
    ]
    l = 14 * 2 * len(models)
    l = 6 * 2 * len(models)
    
    if (relative2Metacarpal):
        if (True):
            
            rotMin = []
            rotMax = []
            for model in models:
                dataIn = cropData3(model['data' + motionType], 2083, 4165)
                for ind1, data in enumerate(dataIn):
                    if (data['type'] == 'FE'):
                        rot = copy.deepcopy(data['kinematics']['hand_flexion'].to_numpy()*(180.0/np.pi))
                        #rot *= -1.0
                    if (data['type'] == 'UR'):
                        rot = copy.deepcopy(data['kinematics']['hand_deviation'].to_numpy()*(180.0/np.pi))
                        if (not model['isLeftHand']):
                            rot *= -1.0
                    rotMin.append(min(rot))
                    rotMax.append(max(rot))
            #minX = round(float(max(rotMin)) / inc) * inc
            #maxX = round(float(min(rotMax)) / inc) * inc

            # keep those within 1 std of mean
            stdPos = np.mean(rotMax) - 2 * np.std(rotMax)
            rotMaxV = min([x for x in rotMax if x > stdPos])
            stdNeg = np.mean(rotMin) + 2 * np.std(rotMin)
            rotMinV = max([x for x in rotMin if x < stdNeg])
            minX = round(float(rotMinV) / inc) * inc
            maxX = round(float(rotMaxV) / inc) * inc

            #plt.hist(rotMin, bins=252)

            xnew = np.array(np.arange(minX, maxX, inc))

            y2new = np.empty((3, len(kinematicAngles), l, len(xnew)), dtype=float)

            # these loops are really inefficient...
            for ind2, kinematicAngle in enumerate(kinematicAngles):
                l2c, l2n, l2s = 0, 0, 0
                for model in models:
                    dataIn = cropData3(model['data' + motionType], 2083, 4165)
                    for ind1, data in enumerate(dataIn): # n1, n2, c1, c2, s1, s2
                        # raw rot data is in positive direction for both FE and UR
                        # UR rotation was clock wise first
                        if (data['type'] == 'FE'):
                            rot = copy.deepcopy(data['kinematics']['hand_flexion'].to_numpy()*(180.0/np.pi))
                            #rot *= -1.0
                        if (data['type'] == 'UR'):
                            rot = copy.deepcopy(data['kinematics']['hand_deviation'].to_numpy()*(180.0/np.pi))
                            if (not model['isLeftHand']):
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

            saveCache(y2new, 'statsTemp3_' + motionType)
        else:
            y2new = loadCache('statsTemp3_' + motionType)
    
    # 0: normal
    # 1: cut
    # 2: scaffold
    #
    # Each trial is split into 14 or 6 parts (zero to min or max angle)
    # 2 trials
    # multiply by number of cadavers
    # eg. 14 * 2 * 8 = 244
    #
    # array structure is:
    # [ cadaver state, axis angle of a bone, segment (combined all cadavers), angle value]
    #
    # https://www.tandfonline.com/doi/full/10.1080/23335432.2019.1597643
    # The definition of the SPM p-values can be stated as: ‘the probability that smooth, random continua would produce a supra-threshold cluster as broad as the observed cluster’ (spm1d.org, © T. Pataky).
    # Critical thresholds are usually calculated with the Type I error α = 0.05. 
    #
    # What is a p value?
    # The probability that a random
    # process will yield a particular result.
    # 
    # t star is critical threshold

    # TODO:
    # remove data which does not go to full range
    # check motions
    # check dirstribution
    # use certain stat test/check

    if makeRelativeToNormal and False:
        # make relative to normal
        for ind2, kinematicAngle in enumerate(kinematicAngles):
            for ind1 in range(l):
                y2new[1, ind2, ind1, :] = y2new[1, ind2, ind1, :] - y2new[0, ind2, ind1, :]
                y2new[2, ind2, ind1, :] = y2new[2, ind2, ind1, :] - y2new[0, ind2, ind1, :]

    def getStats(angleName, y):
        axisInd = [i for i, x in enumerate(kinematicAngles) if x == angleName][0]

        d1 = y[0, axisInd].shape[0]
        y2merge = np.empty((d1*3, y[0, axisInd].shape[1]))
        y2merge[:d1] = y[0, axisInd] 
        #y2merge[:d1] = y[1, axisInd] 
        y2merge[d1:d1*2] = y[1, axisInd] 
        #y2merge[d1:d1*2] = y[2, axisInd] 
        y2merge[d1*2:] = y[2, axisInd] 
        y2mergeInds = np.empty((d1*3)) # categories: normal, cut, scaffold
        y2mergeInds[:d1] = 0
        y2mergeInds[d1:d1*2] = 1
        y2mergeInds[d1*2:]= 2

        indToRemove = []
        tooManyNaNs = []
        for ind0, i0 in enumerate(y2merge):
            for ind1, i1 in enumerate(i0):
                if np.isnan(i1) and not (ind0 in indToRemove):
                    nanIndexes = [i for i, x in enumerate(y2merge[ind0]) if np.isnan(y2merge[ind0, i])]
                    
                    if len(nanIndexes) < 20:
                        nans, x1 = nan_helper(y2merge[ind0])
                        y2merge[ind0][nans]= np.interp(x1(nans), x1(~nans), y2merge[ind0][~nans])
                    else:
                        tooManyNaNs.append(len(nanIndexes))
                        indToRemove.append(ind0)
        print('Too many NaNs: {} containing: {}'.format(len(tooManyNaNs), tooManyNaNs))

        for ind, i in enumerate(indToRemove[::-1]):
            y2merge = np.delete(y2merge, i, axis=0)
            y2mergeInds = np.delete(y2mergeInds, i, axis=0)

        alpha=0.05


        t = spm1d.stats.ttest2( y2merge[y2mergeInds == 1], y2merge[y2mergeInds == 2] ) # c to s
        ti = t.inference(alpha, two_tailed=True, interp=False)
        clusters = []
        for clu in ti.clusters:
            clusters.append(clu.endpoints)

        def p2string(p):
            pr = '= {:.03f}'.format(p)
            if p < 0.00005:
                pr = '< 0.0001'
            else:
                if p < 0.0005:
                    pr = '< 0.001'
                else:
                    if p < 0.005:
                        pr = '< 0.01'
                    else:
                        if p < 0.05:
                            pr = '< 0.1'
            return pr

        dataObj = {
            'alpha': alpha,
            'p_value': p2string(ti.p_set),
            'H0 Rejected': ti.h0reject,
            'zstar': ti.zstar,
            'clusters': clusters,
        }
        return dataObj

    angleStats = []
    for angle in kinematicAngles:
        angleStats.append( getStats(angle, y2new) )

##%%
    if DEBUG:
        dataset = spm1d.data.uv1d.anova2rm.SPM1D_ANOVA2RM_2x2()
        Y,A,B,SUBJ   = dataset.get_data()
        dataset      = spm1d.data.uv1d.anova1rm.SpeedGRFcategoricalRM()
        Y,A,SUBJ     = dataset.get_data()
        # dependent variable: rotation andgle
        # factors: flexion-extension, extension-flexion
        # groups: normal, cut, scaffold
        # subjects: 7

        #'lunate_flexion', 'lunate_deviation', 'lunate_rotation',
        #'sca_xrot', 'sca_yrot', 'sca_zrot'
        axisInd = [i for i, x in enumerate(kinematicAngles) if x == 'sca_xrot'][0]

        nTimePoints = y2new.shape[0]
        d1 = y2new[0, axisInd].shape[0]
        y2merge = np.empty((d1*nTimePoints, y2new[0, axisInd].shape[1]))
        y2merge[:d1] = y2new[0, axisInd] 
        #y2merge[:d1] = y2new[1, axisInd] 
        y2merge[d1:d1*2] = y2new[1, axisInd] 
        #y2merge[d1:d1*2] = y2new[2, axisInd] 
        y2merge[d1*2:] = y2new[2, axisInd] 
        y2mergeInds = np.empty((d1*nTimePoints)) # categories: normal, cut, scaffold
        y2mergeInds[:d1] = 0
        y2mergeInds[d1:d1*2] = 1
        y2mergeInds[d1*2:]= 2
        
        y2mergeDirectionInds = np.tile([0,1], int((d1*nTimePoints)/2)) # direction: flexion to extension or extension to flexion

        y2Subs = np.empty((d1*nTimePoints)) # cadavers: 1 to 7
        nCycles = 12
        for i in range(len(exps)):
            y2Subs[nCycles*i:i*nCycles+nCycles] = i
            y2Subs[d1 + nCycles*i:d1 + i*nCycles+nCycles] = i
            y2Subs[d1*2 + nCycles*i:d1*2 + i*nCycles+nCycles] = i

        indToRemove = []
        tooManyNaNs = []
        for ind0, i0 in enumerate(y2merge):
            for ind1, i1 in enumerate(i0):
                if np.isnan(i1) and not (ind0 in indToRemove):
                    nanIndexes = [i for i, x in enumerate(y2merge[ind0]) if np.isnan(y2merge[ind0, i])]
                    if len(nanIndexes) < 20:
                        nans, x1= nan_helper(y2merge[ind0])
                        y2merge[ind0][nans]= np.interp(x1(nans), x1(~nans), y2merge[ind0][~nans])
                    else:
                        print(nanIndexes)
                        tooManyNaNs.append(nanIndexes)
                        indToRemove.append(ind0)
        print('Too many NaNs: {} containing: {}'.format(len(tooManyNaNs), tooManyNaNs))

    #plt.hist([x[0] for x in tooManyNaNs], bins=200)
    #indexCutOff = int(np.mean([x[0] for x in tooManyNaNs if x[0] > max(max(tooManyNaNs))/2]) - np.std([x[0] for x in tooManyNaNs if x[0] > max(max(tooManyNaNs))/2]))
    #xnew[indexCutOff]
        for ind, i in enumerate(indToRemove[::-1]):
            y2merge = np.delete(y2merge, i, axis=0)
            y2Subs = np.delete(y2Subs, i, axis=0)
            y2mergeInds = np.delete(y2mergeInds, i, axis=0)
            y2mergeDirectionInds = np.delete(y2mergeDirectionInds, i, axis=0)


        alpha=0.05

        # check if distribution is normal. Answer: It's not normal for all kinematic angles!
        #plt.hist(y2merge[y2mergeInds == 0], bins=50)
        a = np.histogram(y2merge[y2mergeInds == 0].ravel(), bins=20)
        plt.hist(np.concatenate(y2merge[y2mergeInds == 0]), bins=20)
        plt.hist(y2merge[y2mergeInds == 0], bins=20)
        #plt.ylim(top=a[0].max)
        #plt.hist(y2merge[y2mergeInds == 1], bins=50)
        #plt.hist(y2merge[y2mergeInds == 2], bins=50)
        plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
        plt.show()
        print("")

        from scipy import stats
        dStat = {}
        dStat['kurt'] = [np.nan, np.nan, np.nan]
        # For normally distributed data, the skewness should be about zero
        dStat['skew'] = [np.nan, np.nan, np.nan]
        dStat['variance'] = [np.nan, np.nan, np.nan]
        for i in range(3):
            dStat['kurt'][i] = stats.kurtosis(y2merge[y2mergeInds == i], axis=None)
            dStat['skew'][i] = stats.skew(y2merge[y2mergeInds == i], axis=None)
            dStat['variance'][i] = stats.bartlett(*y2merge[y2mergeInds == i][:30]).pvalue
            dStat['variance'][i] = stats.fligner(*y2merge[y2mergeInds == i][:30]).pvalue
        #kw = stats.kruskal( # for independent samples
        #    #y2merge[y2mergeInds == 0],
        #    y2merge[y2mergeInds == 1],
        #    y2merge[y2mergeInds == 2])
        dStat['fs'] = stats.friedmanchisquare(
            y2merge[y2mergeInds == 0],
            y2merge[y2mergeInds == 1],
            y2merge[y2mergeInds == 2])
        dStat['wc'] = stats.mannwhitneyu(
            y2merge[y2mergeInds == 0],
            y2merge[y2mergeInds == 1])
        dStat['wc'] = stats.wilcoxon(
            y2merge[y2mergeInds == 0],
            y2merge[y2mergeInds == 2])
        dStat['spNC'] = stats.spearmanr(
            y2merge[y2mergeInds == 0],
            y2merge[y2mergeInds == 1])

        # interp yes or no?

        # predict no significant difference in two samples
        # if true then there's a difference
        t = spm1d.stats.ttest2( y2merge[y2mergeInds == 0], y2merge[y2mergeInds == 1] ) # n to c
        t = spm1d.stats.ttest2( y2merge[y2mergeInds == 0], y2merge[y2mergeInds == 2] ) # n to s
        t = spm1d.stats.ttest2( y2merge[y2mergeInds == 1], y2merge[y2mergeInds == 2] ) # c to s
        # If normalised to intact
        #t = spm1d.stats.ttest( y2merge[y2mergeInds == 1] ) # n to c
        #t = spm1d.stats.ttest( y2merge[y2mergeInds == 2] ) # n to s
        #t = spm1d.stats.ttest_paired( y2merge[y2mergeInds == 1], y2merge[y2mergeInds == 2] ) # c to s
        ti = t.inference(alpha, two_tailed=True, interp=False)
        ti.plot()
        ti.plot_threshold_label(fontsize=8)
        ti.plot_p_values(size=10, offsets=[(0,0.3)])
        plt.show()
        print("H0 rejected: {}".format(ti.h0reject))

        t  = spm1d.stats.regress(y2merge, y2mergeInds)
        ti = t.inference(alpha=0.05)
        ti.plot_p_values(size=10, offsets=[(0,0.3)])
        ti.plot()
        plt.show()
        print("")

        # not actually equal var
        F = spm1d.stats.anova1( y2merge, y2mergeInds, equal_var=True)
        F = spm1d.stats.anova1rm( y2merge, y2mergeInds, y2Subs, equal_var=True)
        # A one-way ANOVA with two levels is equivalent to a two-sample t test. The F statistic is equal to the square of the t statistic.
        #F = spm1d.stats.anova1(
        #    (y2merge[y2mergeInds == 1],
        #    y2merge[y2mergeInds == 2])
        #    , equal_var=True)
        F = spm1d.stats.anova1rm(
            (y2merge[y2mergeInds == 0],
            y2merge[y2mergeInds == 1],
            y2merge[y2mergeInds == 2]),
            equal_var=True)
        F = spm1d.stats.anova2rm(
            y2merge,
            y2mergeInds,
            y2mergeDirectionInds,
            y2Subs)
        Fi = F.inference(alpha)
        #plt.close('all')
        Fi.plot()
        #Fi.plot_threshold_label(bbox=dict(facecolor='w'))
        #Fi.plot_p_values(size=10, offsets=[(0,0.3)])
        #plt.ylim(-1, 500)
        plt.xlabel('Time (%)', size=20)
        plt.title(r'Critical threshold at $\alpha$=%.2f:  $F^*$=%.3f' %(alpha, Fi.zstar))
        plt.show()
        print("")



    # merge all cycles into one for each experiment
    nCycles = int(y2new[0, 0].shape[0]/len(models))
    y3 = np.empty((3, len(kinematicAngles), len(models), len(xnew)), dtype=float)
    for i0 in range(y2new.shape[0]):
        for i1 in range(y2new.shape[1]):
            for i2 in range(len(models)):
                y3[i0,i1,i2,:] = np.nanmean(y2new[i0, i1, i2*nCycles:(i2+1)*nCycles,:], axis=0)
    
    if DEBUG:
        axisInd = [i for i, x in enumerate(kinematicAngles) if x == 'sca_xrot'][0]
        axisInd = 5

        y3nTimePoints = y3.shape[0]
        y3d1 = y3[0, axisInd].shape[0]
        y3merge = np.empty((y3d1*y3nTimePoints, y3[0, axisInd].shape[1]))
        y3merge[:y3d1] = y3[0, axisInd]  
        y3merge[y3d1:y3d1*2] = y3[1, axisInd] 
        y3merge[y3d1*2:] = y3[2, axisInd] 
        y3mergeInds = np.empty((y3d1*y3nTimePoints)) # categories: normal, cut, scaffold
        y3mergeInds[:y3d1] = 0
        y3mergeInds[y3d1:y3d1*2] = 1
        y3mergeInds[y3d1*2:]= 2


        y3Subs = np.empty((y3d1*y3nTimePoints)) # cadavers: 1 to 7
        nCycles = 1
        for i in range(len(exps)):
            y3Subs[nCycles*i:i*nCycles+nCycles] = i
            y3Subs[y3d1 + nCycles*i:y3d1 + i*nCycles+nCycles] = i
            y3Subs[y3d1*2 + nCycles*i:y3d1*2 + i*nCycles+nCycles] = i


        indToRemove = []
        tooManyNaNs = []
        for ind0, i0 in enumerate(y3merge):
            for ind1, i1 in enumerate(i0):
                if np.isnan(i1) and not (ind0 in indToRemove):
                    nanIndexes = [i for i, x in enumerate(y3merge[ind0]) if np.isnan(y3merge[ind0, i])]
                    if len(nanIndexes) < 20:
                        nans, x1= nan_helper(y3merge[ind0])
                        y3merge[ind0][nans]= np.interp(x1(nans), x1(~nans), y3merge[ind0][~nans])
                    else:
                        print(nanIndexes)
                        tooManyNaNs.append(nanIndexes)
                        indToRemove.append(ind0)
        print('Too many NaNs: {} containing: {}'.format(len(tooManyNaNs), tooManyNaNs))

    #plt.hist([x[0] for x in tooManyNaNs], bins=200)
    #indexCutOff = int(np.mean([x[0] for x in tooManyNaNs if x[0] > max(max(tooManyNaNs))/2]) - np.std([x[0] for x in tooManyNaNs if x[0] > max(max(tooManyNaNs))/2]))
    #xnew[indexCutOff]
        for ind, i in enumerate(indToRemove[::-1]):
            y3merge = np.delete(y3merge, i, axis=0)
            y3Subs = np.delete(y3Subs, i, axis=0)
            y3mergeInds = np.delete(y3mergeInds, i, axis=0)
            

    # make relative to normal
    if makeRelativeToNormal:
        for ind2, kinematicAngle in enumerate(kinematicAngles):
            for ind1 in range(len(models)):
                y3[1, ind2, ind1, :] = y3[1, ind2, ind1, :] - y3[0, ind2, ind1, :]
                y3[2, ind2, ind1, :] = y3[2, ind2, ind1, :] - y3[0, ind2, ind1, :]
        
    angleStats = []
    for angle in kinematicAngles:
        #angleStats.append( getStats(angle, y2new) )
        angleStats.append( getStats(angle, y3) )

    if DEBUG:
        F = spm1d.stats.anova1rm(
            y3merge,
            y3mergeInds,
            y3Subs,
            equal_var=True)
        Fi = F.inference(alpha)
        #plt.close('all')
        Fi.plot()
        #Fi.plot_threshold_label(bbox=dict(facecolor='w'))
        #Fi.plot_p_values(size=10, offsets=[(0,0.3)])
        #plt.ylim(-1, 500)
        plt.xlabel('Time (%)', size=20)
        plt.title(r'Critical threshold at $\alpha$=%.2f:  $F^*$=%.3f' %(alpha, Fi.zstar))
        plt.show()
        print("")


##%%
if True:
    ## make relative to normal
    #for ind2, kinematicAngle in enumerate(kinematicAngles):
    #    for ind1 in range(l):
    #        y2new[1, ind2, ind1, :] = y2new[1, ind2, ind1, :] - y2new[0, ind2, ind1, :]
    #        y2new[2, ind2, ind1, :] = y2new[2, ind2, ind1, :] - y2new[0, ind2, ind1, :]

    fig = plt.figure()
    fig.set_size_inches(14.5, 14.5)
    gs0 = gridspec.GridSpec(4, 3, figure=fig)#, height_ratios=[3,3,1])
    ax = []
    for i in range(0, 6):
        ax.append(fig.add_subplot(gs0[i]))
    
    remapGraphs = [
        0, 3, 6,
        1, 4, 7,
        2, 5, 8
    ]
    # add stats
    for indAx, ax0 in enumerate(ax):
        for clu in angleStats[remapGraphs[indAx]]['clusters']:
            ax0.axvspan(xnew[clu[0]], xnew[clu[1]], alpha=0.25, color='grey')

    xnew = np.arange(minX, maxX, inc)

    rotYMin = []
    rotYMax = []
    if makeRelativeToNormal:
        bones = ['cut', 'scaffold']
        offest = 1
    else:
        bones = ['normal', 'cut', 'scaffold']
        offest = 0

    for indFig, cat in enumerate(bones): # 'normal', 
        for indAx, ax0 in enumerate(ax):
            mean = np.nanmean(y3[indFig + offest,remapGraphs[indAx],:,:], axis=0)
            ax0.plot(xnew, mean, color=figColours[cat], linewidth=0.9, alpha=0.9, label=figNames[cat])
            # + - one standard deviation
            negSD = mean - 1*np.nanstd(y3[indFig + offest,remapGraphs[indAx],:,:], axis=0)
            posSD = mean + 1*np.nanstd(y3[indFig + offest,remapGraphs[indAx],:,:], axis=0)
            ax0.fill_between(xnew, negSD, posSD,
                color=figColours[cat], alpha=0.2)
            rotYMin.append(min(negSD))
            rotYMax.append(max(posSD))
    minY = min(rotYMin)
    maxY = max(rotYMax)


    for indAx, ax0 in enumerate(ax):
        #meanN = np.nanmean(y2new[0,indAx,:,:], axis=0)
        #meanN[:]= 0.0
        #meanC = np.nanmean(y2new[1,indAx,:,:], axis=0)
        #nans, x1= nan_helper(meanC)
        #meanC[nans]= 0.0
        #meanS = np.nanmean(y2new[2,indAx,:,:], axis=0)
        #nans, x1= nan_helper(meanS)
        #meanS[nans]= 0.0
        #rmseNC = np.sqrt(np.mean((meanN-meanC)**2))
        #rmseNS = np.sqrt(np.mean((meanN-meanS)**2))
        #ax0.text(0.65, 0.15, "$RMSE_T$: {:.2f}".format(rmseNC),
        #verticalalignment='center',
        #rotation='horizontal', fontsize=14, transform=ax0.transAxes)
        #ax0.text(0.65, 0.05, "$RMSE_I$ : {:.2f}".format(rmseNS),
        #verticalalignment='center',
        #rotation='horizontal', fontsize=14, transform=ax0.transAxes)

        if len(angleStats[remapGraphs[indAx]]['clusters']) > 0:
            ax0.text(0.40, 0.95, "p {}".format(angleStats[remapGraphs[indAx]]['p_value']),
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
        ax.set_ylim([minY - 5.0, maxY + 5.0])
        ax.set_xlim([minX - 5.0, maxX + 5.0])
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.title.set_size(18)
        ax.xaxis.set_tick_params(labelsize=13)
        ax.yaxis.set_tick_params(labelsize=13)
    #axAdjust(ax[0], ylabel = "Rotation (degrees)", title = "Flexion-Extension")
    #axAdjust(ax[1], title = "Radial-Ulnar Deviation")
    #axAdjust(ax[2], title = "Pro-Supination")
    #axAdjust(ax[3], ylabel = "Rotation (degrees)", xlabel = "Wrist Angle (degrees)")
    #axAdjust(ax[4], xlabel = "Wrist Angle (degrees)")
    #axAdjust(ax[5], xlabel = "Wrist Angle (degrees)")
    
    axAdjust(ax[0], ylabel = "Flexion-Extension\n(degrees)", title = "Lunate")
    axAdjust(ax[1], title = "Scaphoid")
    #axAdjust(ax[2], title = "Scaphoid relative\nto Lunate")
    axAdjust(ax[3], ylabel = "Radial-Ulnar Deviation\n(degrees)")
    axAdjust(ax[4])
    #axAdjust(ax[5])
    axAdjust(ax[6], ylabel = "Pro-Supination\n(degrees)", xlabel = "Wrist Angle (degrees)")
    axAdjust(ax[7], xlabel = "Wrist Angle (degrees)")
    #axAdjust(ax[8], xlabel = "Wrist Angle (degrees)")

    #ax1_0.set_xlim([minTime,maxTime])
    #ax1_0.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
    #ax1_0.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
    figTitle = "Wrist Motion: Flexion-Extension - Relative to " + figNames['normal']
    if ('UR' in motionType):
        figTitle = "Wrist Motion: Radial-Ulnar Deviation - Relative to " + figNames['normal']
    ax[1].text(-1.0, 1.3, figTitle,
        verticalalignment='center',
        rotation='horizontal', fontsize=20, transform=ax[1].transAxes)

    #ax[0].text(-0.35, 0.5, 'Lunate',
    #    verticalalignment='center',
    #    rotation='vertical', fontsize=18, transform=ax[0].transAxes)
    #ax[3].text(-0.35, 0.5, 'Scaphoid',
    #    verticalalignment='center',
    #    rotation='vertical', fontsize=18, transform=ax[3].transAxes)

#    note = "$RMSE_T$: Root-mean-squared-error between {} and {}\n$RMSE_I$ : Root-mean-squared-error between {} and {}".format(figNames['normal'], figNames['cut'], figNames['normal'], figNames['scaffold'])
#    ax[3].text(-0.3, -0.8, note,
#        verticalalignment='center',
#        rotation='horizontal', fontsize=18, transform=ax[3].transAxes)

    #ax[2].legend(bbox_to_anchor=(0.75, 1.05))
    
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
    if makeRelativeToNormal:
        if relative2Metacarpal:
            fileName = r'\all_S&L_bone_kinematics2_1_mean_relativeNormal_wSD_stats_' + motionType
        else:
            fileName = r'\all_S&L_bone_kinematics_mean_relativeNormal_wSD_stats_' + motionType
    else:
        if relative2Metacarpal:
            fileName = r'\all_S&L_bone_kinematics2_1_mean_wSD_stats_' + motionType
        else:
            fileName = r'\all_S&L_bone_kinematics_mean_wSD_stats_' + motionType
    
    plt.savefig(
        fname = outputFolders()['graphics'] + fileName,
        dpi=600,
        facecolor=fig.get_facecolor(),
        transparent=False)
    plt.close() # prevent Jupyter from showing plot
# %%
