# Author: Alastair Quinn 2022

#from slil.process.align_by_surface import runOptimization
from slil.process.align_by_slil_gap import runOptimization

#runOptimization('11525', [ 'normal', 'cut', 'scaffold'])
#runOptimization('11527', [ 'cut' ])
#runOptimization('11534', [ 'cut' ])
#runOptimization('11535', [ 'normal' ])
#runOptimization('11539', [ 'scaffold' ])

experiments = [
    '11524',
    '11525',
    '11526',
    '11527', # bad, do not run
    '11534',
    '11535',
    '11536',
    '11537',
    '11538',
    '11539'
    ]
clusterAddress = 'ray://100.99.142.105:10001'

for i, experiment in enumerate(experiments):
    runOptimization(experiment, [ 'normal' ], clusterAddress)

#for i, experiment in enumerate(experiments):
#    runOptimization(experiment, [ 'cut', 'scaffold' ], clusterAddress)