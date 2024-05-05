#%%
import slil.plotting.plots as pfp
from slil.common.cache import loadCache, saveCache, loadOnlyExps

#%% To allow re-importing modules without restarting jupyter
%load_ext autoreload
%autoreload 2

#%matplotlib qt

#%%
experiments = [
    #'11525',
    #'11526',
    #'11527',
    '11534',
    #'11535',
    #'11536',
    #'11537',
    #'11538',
    #'11539'
    ]
models = loadOnlyExps(experiments)

#%%
experiments = [
    '11525',
    '11526',
    '11527',
    '11534',
    '11535',
    '11536',
    #'11537',
    '11538',
    '11539'
    ]

import plot_groups_stats as pfpgs
pfpgs.plotGroupAvgMeansRelativeSLStats(models, experiments)

#%%
# run after first running all data through IK
# geneartes rotations to better align markers
# uses FE, finds rotation arc
# angle is from rotation arc 'plane' to flexion-extension model rotation axis
# i.e. so that running IK again there is more flexion extension rather than UR
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

import ray

@ray.remote
def testRun(modelInfo, dataFE):
    pfp.calcAndPlotRotaionAxies(modelInfo, dataFE)

ray.init(num_cpus=2, ignore_reinit_error=True)
models = loadOnlyExps(experiments)
tasksRunning = []
for ind, modelInfo in enumerate(models):
    dataFE = modelInfo['dataFE']
    for modelType in [ 'normal', 'cut', 'scaffold']:
        modelInfo['currentModel'] = modelType
        tasksRunning.append(testRun.remote(modelInfo, dataFE))
        #pfp.calcAndPlotRotaionAxies(modelInfo, dataFE)

while tasksRunning:
    finished, tasksRunning = ray.wait(tasksRunning, num_returns=1, timeout=None)
    for task in finished:
        result = ray.get(task)
        #print('result:', result)
    print('Tasks remaining:', len(tasksRunning))
print('Finished tasks.')

# %%
import slil.common.opensim as fo
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
modelsKinematics = {}
for ind, modelInfo in enumerate(models):
    modelsKinematics[modelInfo['experimentID']] = {}
    for trial in modelInfo['trialsRawData_only_normal'] + modelInfo['trialsRawData_only_cut'] + modelInfo['trialsRawData_only_scaffold']:
        rotations, names = fo.getBoneRotationsFromIK(modelInfo, trial)
        rotX = {}
        for name in names:
            rotX[name] = rotations[names.index(name), :, :, :]
        modelsKinematics[modelInfo['experimentID']][trial] = rotX

saveCache(modelsKinematics, 'dataModelsKinematics')
print('Done!')
#%%
modelsKinematics = loadCache('dataModelsKinematics')

experiments = [
    '11525',
    #'11526',
    #'11527',
    #'11534',
    #'11535',
    #'11536',
    #'11537',
    #'11538',
    #'11539'
    ]
models = loadOnlyExps(experiments)
for ind, modelInfo in enumerate(models):
    
    dataURcrop = modelInfo['dataURcropped']
    dataFEcrop = modelInfo['dataFEcropped']
    dataUR = modelInfo['dataUR']
    dataFE = modelInfo['dataFE']
    dataKinematics = modelsKinematics[modelInfo['experimentID']]

    pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataFEcrop, dataKinematics)
# %%
