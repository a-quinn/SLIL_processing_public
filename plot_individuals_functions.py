import slil.plotting.plots as pfp
from slil.common.cache import loadOnlyExps

def generateIndividualGraphics(experiments, localParallel = True):
    
    models = loadOnlyExps(experiments)
    
    if not localParallel:
        for modelInfo in models:
            _generate(modelInfo)
    else:
        import ray

        @ray.remote
        def generate(modelInfo):
            _generate(modelInfo)

        if ray.is_initialized(): # must not run on remote instance
            ray.shutdown()

        ray.init(address = 'local', ignore_reinit_error=True)
        tasksRunning = []
        for modelInfo in models:
            tasksRunning.append(generate.remote(modelInfo))

        while tasksRunning:
            finished, tasksRunning = ray.wait(tasksRunning, num_returns=1, timeout=None)
            for task in finished:
                result = ray.get(task)
                #print('result:', result)
            print('Tasks remaining:', len(tasksRunning))
        print('Finished tasks.')


def _generate(modelInfo):
    dataUR = modelInfo['dataUR']
    dataFE = modelInfo['dataFE']
    
    #pfp.deleteAllPreviousCache(modelInfo['experimentID'])

    #pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataUR, ScaRel2Lun = True)
    #pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataFE, ScaRel2Lun = True)
    #pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataUR, ScaRel2Lun = False)
    #pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataFE, ScaRel2Lun = False)
    #pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataUR, ScaRel2Lun = True, isMeanSD = True)
    #pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataFE, ScaRel2Lun = True, isMeanSD = True)
    #pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataUR, ScaRel2Lun = False, isMeanSD = True)
    #pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataFE, ScaRel2Lun = False, isMeanSD = True)

    pfp.plotSLILGapvWristAngle(modelInfo, dataUR, isMeanSD = False)
    pfp.plotSLILGapvWristAngle(modelInfo, dataFE, isMeanSD = False)
    pfp.plotSLILGapvWristAngle(modelInfo, dataUR, isMeanSD = True)
    pfp.plotSLILGapvWristAngle(modelInfo, dataFE, isMeanSD = True)
    #pfp.plotSLILGapvTime(modelInfo, dataUR)
    #pfp.plotSLILGapvTime(modelInfo, dataFE)

def _generate2(modelInfo):
    dataURcrop = modelInfo['dataURcropped']
    dataFEcrop = modelInfo['dataFEcropped']
    dataUR = modelInfo['dataUR']
    dataFE = modelInfo['dataFE']
    
    pfp.deleteAllPreviousCache(modelInfo['experimentID'])

    pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataUR)
    pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataFE)
    pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataUR, ScaRel2Lun = True)
    pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataFE, ScaRel2Lun = True)

    pfp.plotSLILGapvWristAngle(modelInfo, dataUR)

    pfp.plotSLILGapvTime(modelInfo, dataUR)
    pfp.plotSLILGapvTime(modelInfo, dataFE)
##%%
    pfp.plotBPtoBPvTime(modelInfo, dataUR)
    pfp.plotBPtoBPvTime(modelInfo, dataFE)
##%%
    #pfp.plotBPtoBPvTime(modelInfo, dataURcrop)
    #pfp.plotBPtoBPvTime(modelInfo, dataFEcrop)

##%%
    pfp.plotKinematicsBreakDown(modelInfo, dataUR)
    pfp.plotKinematicsBreakDown(modelInfo, dataFE)

#    print('Plotting UR {}'.format(modelInfo['experimentID']))
#    pfp.plotKinematicsBreakDown(modelInfo, dataURcrop)
#    print('Plotting FE {}'.format(modelInfo['experimentID']))
#    pfp.plotKinematicsBreakDown(modelInfo, dataFEcrop)

    pfp.plotKinematicsBreakDownMean(modelInfo, dataUR)
    pfp.plotKinematicsBreakDownMean(modelInfo, dataFE)

    pfp.plotKinematicsBreakDownSD(modelInfo, dataURcrop)
    pfp.plotKinematicsBreakDownSD(modelInfo, dataFEcrop)

    pfp.plotBoneplugsDisplacement(modelInfo, dataURcrop)
    pfp.plotBoneplugsDisplacement(modelInfo, dataFEcrop)

#for ind, modelInfo in enumerate(models):
#    pfp.calcAndPlotRotaionAxies(modelInfo, dataFE)
#    dataURcrop = modelInfo['dataURcropped']
#    dataFEcrop = modelInfo['dataFEcropped']
#    dataUR = modelInfo['dataUR']
#    dataFE = modelInfo['dataFE']
#    pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataUR)
#    pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataFE)
#    pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataUR, ScaRel2Lun = True)
#    pfp.plotKinematicsBreakDownSD_RelativeToMet(modelInfo, dataFE, ScaRel2Lun = True)
#
###%%
#    pfp.plotBPtoBPvTime(modelInfo, dataUR)
#    pfp.plotBPtoBPvTime(modelInfo, dataFE)
###%%
#    #pfp.plotBPtoBPvTime(modelInfo, dataURcrop)
#    #pfp.plotBPtoBPvTime(modelInfo, dataFEcrop)
#
###%%
#    pfp.plotKinematicsBreakDown(modelInfo, dataUR)
#    pfp.plotKinematicsBreakDown(modelInfo, dataFE)
#
##    print('Plotting UR {}'.format(modelInfo['experimentID']))
##    pfp.plotKinematicsBreakDown(modelInfo, dataURcrop)
##    print('Plotting FE {}'.format(modelInfo['experimentID']))
##    pfp.plotKinematicsBreakDown(modelInfo, dataFEcrop)
#
#    pfp.plotKinematicsBreakDownMean(modelInfo, dataUR)
#    pfp.plotKinematicsBreakDownMean(modelInfo, dataFE)
#
#    pfp.plotKinematicsBreakDownSD(modelInfo, dataURcrop)
#    pfp.plotKinematicsBreakDownSD(modelInfo, dataFEcrop)
##
##
##    # I don't understand what this is for... maybe different directions?
##    #pfp.plotKinematicsBreakDown(modelInfo, dataURcrop, mode = 'individual')
##    #pfp.plotKinematicsBreakDown(modelInfo, dataFEcrop, mode = 'individual')
#
#    pfp.plotBoneplugsDisplacement(modelInfo, dataURcrop)
#    pfp.plotBoneplugsDisplacement(modelInfo, dataFEcrop)