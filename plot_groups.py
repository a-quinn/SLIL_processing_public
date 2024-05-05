#%%
# Author: Alastair Quinn 2022
from slil.common.cache import loadCache, loadOnlyExps
import slil.plotting.groups_A as pg_A
from slil.plotting.groups_B import *
from slil.cache_results_plot import createSLILGapCache

#%%

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
    
models = loadOnlyExps(experiments)
pg_A.plotGroupAvg(models)

pg_A.calcAndCacheMeans(models, 'UR')
pg_A.calcAndCacheMeans(models, 'FE')
pg_A.plotGroupAvgMeans(models)
pg_A.plotGroupAvgMeansRelative(models)
pg_A.plotGroupAvgMeansRelativeSL(models)

#%%
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

createSLILGapCache(experiments)

#%%
Xes = loadCache('dataXes')
all = loadCache('dataAll')
allUR = loadCache('dataAllUR')
allRel = loadCache('dataAllRel')
allURRel = loadCache('dataAllURRel')

#%%

plotRMSD(experiments, all, allUR)

plotLengthChangeVsRotation(Xes, all)
plotLengthChangeVsRotation(Xes, allUR, 'UR')

plotDistance(experiments, all)
plotDistance(experiments, allUR, 'UR')

plotRelativeLengthChangeVsRotation(Xes, allRel)
plotRelativeLengthChangeVsRotation(Xes, allURRel, 'UR')

plotRelativeLengthChangeVsRotation(Xes, allRel, showSD=True)
plotRelativeLengthChangeVsRotation(Xes, allURRel, 'UR', showSD=True)


plotLengthChange(experiments, allRel)
plotLengthChange(experiments, allURRel, 'UR')

# %%
