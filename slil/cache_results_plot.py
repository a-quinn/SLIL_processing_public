# Author: Alastair Quinn 2022

import numpy as np
import slil.common.plotting_functions as pf
from slil.common.cache import loadCache, saveCache, loadOnlyExps

def createModelsCache(experiments, localParallel = True):

    if not localParallel:
        for experiment in experiments:
            _generateModelData(experiment)
    else:
        import ray

        ray.init(ignore_reinit_error=True)

        @ray.remote
        def generateModelData(experiment):
            _generateModelData(experiment)

        tasksRunning = []
        for experiment in experiments:
            tasksRunning.append(generateModelData.remote(experiment))

        while tasksRunning:
            finished, tasksRunning = ray.wait(tasksRunning, num_returns=1, timeout=None)
            for task in finished:
                result = ray.get(task)
                #print('result:', result)
            print('Tasks remaining:', len(tasksRunning))

    createSLILGapCache(experiments)
    print('Finished creating main model caches.')

def _generateModelData(experiment):
    import slil.common.data_configs as dc
    modelInfo = dc.load(experiment)

    #strain = (data['difference'][0] - data['difference'])/data['difference'][0]
    #fig, ax1 = plt.subplots()
    #ax1.plot(lunateBoneplug['time'],strain, linewidth=0.8, alpha=0.9, label="d")
    #ax1.spines['right'].set_visible(False) # Hide the right and top spines
    #ax1.spines['top'].set_visible(False)
    ##ax1.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
    ##ax1.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
    #ax1.legend(edgecolor='None')
    #plt.title('Strain')
    #plt.xlabel('Time (sec)')
    #plt.ylabel('Strain (mm/mm)', fontsize=10, position = (-0.8,0.5))

    dataUR = pf.createDataD(modelInfo, pf.groupInfoUR) # should really be RU not UR... too late now, it's everywhere!
    dataFE = pf.createDataD(modelInfo, pf.groupInfoFE)

    # kinematics only
    dataUR = pf.convertRelativeToRadius(dataUR)
    dataFE = pf.convertRelativeToRadius(dataFE)
    modelInfo['dataUR'] = dataUR
    modelInfo['dataFE'] = dataFE

    # Removes little bit at start and ends of data which are not a full cycle
    #dataURcrop = pf.cropDataMinLoss(dataUR)
    #dataFEcrop = pf.cropDataMinLoss(dataFE)

    # I think this is only needed when comparing samples
    # we want same wrist motion direction for begining of all trials
    # radial-ulnar deviation always started in clockwise motion of robot
    # shift data of left hands so that initial direction is towards ulnar deviation
    if (modelInfo['isLeftHand']):
        dataURcrop = pf.cropDataRemoveBeginning(dataUR)
    else:
        dataURcrop = dataUR
    #dataFE = pf.cropDataRemoveBeginning(dataFE)
    # Removes end but keeps start
    dataURcrop = pf.cropDataToWholeCycles(dataURcrop)
    dataFEcrop = pf.cropDataToWholeCycles(dataFE)
    
    modelInfo['dataURcropped'] = dataURcrop
    modelInfo['dataFEcropped'] = dataFEcrop

    saveCache(modelInfo, 'dataModel_' + modelInfo['experimentID'])


#import slil.common.data_configs as dc
#        models = []
#        for i, experiment in enumerate(experiments):
#            modelInfo = dc.load(experiment)
#            models += [modelInfo]
#
#        for ind, model in enumerate(models):
#            modelInfo = model
#                #strain = (data['difference'][0] - data['difference'])/data['difference'][0]
#                #fig, ax1 = plt.subplots()
#                #ax1.plot(lunateBoneplug['time'],strain, linewidth=0.8, alpha=0.9, label="d")
#                #ax1.spines['right'].set_visible(False) # Hide the right and top spines
#                #ax1.spines['top'].set_visible(False)
#                ##ax1.set_xticks(np.arange(-0.10, 0.01, 0.50), minor=False)
#                ##ax1.xaxis.set_major_formatter(FormatStrFormatter('%0.01f'))
#                #ax1.legend(edgecolor='None')
#                #plt.title('Strain')
#                #plt.xlabel('Time (sec)')
#                #plt.ylabel('Strain (mm/mm)', fontsize=10, position = (-0.8,0.5))
#
#            dataUR = pf.createDataD(modelInfo, pf.groupInfoUR)
#            dataFE = pf.createDataD(modelInfo, pf.groupInfoFE)
#
#            # kinematics only
#            dataUR = pf.convertRelativeToRadius(dataUR)
#            dataFE = pf.convertRelativeToRadius(dataFE)
#            models[ind]['dataUR'] = dataUR
#            models[ind]['dataFE'] = dataFE
#
#            # Removes little bit at start and ends of data which are not a full cycle
#            #dataURcrop = pf.cropDataMinLoss(dataUR)
#            #dataFEcrop = pf.cropDataMinLoss(dataFE)
#
#            # we want same wrist motion direction for begining of all trials
#            # ulnar-radius deviation always started in clockwise motion of robot
#            # shift data of left hands so that initial direction is towards ulnar deviation
#            if (modelInfo['isLeftHand']):
#                dataURcrop = pf.cropDataRemoveBeginning(dataUR)
#            else:
#                dataURcrop = dataUR
#            #dataFE = pf.cropDataRemoveBeginning(dataFE)
#            # Removes end but keeps start
#            dataURcrop = pf.cropDataToWholeCycles(dataURcrop)
#            dataFEcrop = pf.cropDataToWholeCycles(dataFE)
#            
#            models[ind]['dataURcropped'] = dataURcrop
#            models[ind]['dataFEcropped'] = dataFEcrop
#
#

def createSLILGapCache(exps):
    models = loadOnlyExps(exps)
    
    #for jj, model in enumerate(models):
    #    model['dataUR'] = pf.createDataD_woutKinematics(model, pf.groupInfoUR)
    #    model['dataFE'] = pf.createDataD_woutKinematics(model, pf.groupInfoFE)

    for i, model in enumerate(models):
        dataUR = model['dataUR']
        dataFE = model['dataFE']

        # make relative to scaffold size
        for data in dataUR:
            data['difference'] = data['difference'] / model['scaffoldBPtoBPlength']
        for data in dataFE:
            data['difference'] = data['difference'] / model['scaffoldBPtoBPlength']

        #models[i]['dataURcropped'] = dataUR
        #models[i]['dataFEcropped'] = dataFE
        #models[i]['dataURcropped'] = cropData3(dataUR, offset = 2083, cycleLength = 8333)
        #models[i]['dataFEcropped'] = cropData3(dataFE, offset = 2083, cycleLength = 8333)
        #models[i]['dataURcropped'] = pf.cropDataMinLoss_woutKinematics(dataUR)
        #models[i]['dataFEcropped'] = pf.cropDataMinLoss_woutKinematics(dataFE)

    for i, model in enumerate(models):
        #models[i]['dataURcropped'] = model['dataUR']
        #models[i]['dataFEcropped'] = model['dataFE']
        models[i]['dataURcropped'] = pf.cropData3_noKinematics(model['dataUR'], offset = 2083, cycleLength = 4166)
        models[i]['dataFEcropped'] = pf.cropData3_noKinematics(model['dataFE'], offset = 2083, cycleLength = 4166)

    #%% 'difference'

    def split(models, direction):
        modelInfo = models[0]
        from copy import deepcopy
        all = []
        dLen1 = int(len(modelInfo['data' + direction + 'cropped'])/3)
        dLen2 = len(modelInfo['data' + direction + 'cropped'][0]['difference'])
        flip = False
        for i, model in enumerate(models):
            all.append({
                'normal': np.empty((dLen1, dLen2), float),
                'cut': np.empty((dLen1, dLen2), float),
                'scaffold': np.empty((dLen1, dLen2), float)
            })
            k = 0
            for j, data in enumerate(model['data' + direction + 'cropped']):
                if (j == 12 or j == 24):
                    k = 0
                r1 = deepcopy(data['difference'])
                if (flip):
                    r1 = r1[::-1]
                    flip = False
                else:
                    flip = True
                all[i][data['title']][k] = r1
                k += 1
        return all
    
    def relative(models, all, direction):
        # relative to normal
        modelInfo = models[0]
        #from copy import deepcopy
        allRel = []
        dLen1 = int(len(modelInfo['data' + direction + 'cropped'])/3)
        dLen2 = len(modelInfo['data' + direction + 'cropped'][0]['difference'])
        for j, model in enumerate(models):
            allRel.append({
                'cut': np.empty((dLen1, dLen2), float),
                'scaffold': np.empty((dLen1, dLen2), float)
            })

            for i, a in enumerate(all[j]['cut']):
                allRel[j]['cut'][i] = a - all[j]['normal'][i]
            
            for i, a in enumerate(all[j]['scaffold']):
                allRel[j]['scaffold'][i] = a - all[j]['normal'][i]
        return allRel

    def getX(models, direction):
        modelInfo = models[0]
        time = modelInfo['data' + direction + 'cropped'][0]['time']
        rot = modelInfo['data' + direction + 'cropped'][0]['rot'] * -1.0
        return {'rot': rot, 'time': time}

    all = split(models, 'FE')
    allUR = split(models, 'UR')
    allRel = relative(models, all, 'FE')
    allURRel = relative(models, allUR, 'UR')
    x = getX(models, 'FE')
    xUR = getX(models, 'UR')
    xes = {'FE': x, 'UR': xUR}

    #%%
    saveCache(xes, 'dataXes')
    saveCache(all, 'dataAll')
    saveCache(allUR, 'dataAllUR')
    saveCache(allRel, 'dataAllRel')
    saveCache(allURRel, 'dataAllURRel')