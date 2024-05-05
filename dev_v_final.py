# Author: Alastair Quinn 2019 - 2024

#%%
##%% To allow re-importing modules without restarting jupyter
#%load_ext autoreload
#%autoreload 2
from slil.process import generate_model
import slil.filtering.filter_v4 as frmv4
import slil.process.transpose_c3d as tc3d

runOnlyOne = False # used for testing
experiments = [
    '11525', # D
    '11526', # B
    '11527', # C # bad, do not run for model generation.
    '11534', # A
    '11535', # B
    '11536', # A
    '11537', # A
    '11538', # B
    '11539'  # B
    ]
models = [ 'normal', 'cut', 'scaffold']
#models = [ 'cut' ]
#models = [ 'normal' ]
#models = [ 'scaffold' ]

#clusterAddress = 'ray://100.99.142.105:10001' # 6dof
#clusterAddress = 'ray://132.234.133.156:10001' # biospine beefy

# Use this to start ray on local machine
# ray start --head --block --node-ip-address=localhost
#clusterAddress = 'localhost:10001' # find or creates local
clusterAddress = 'auto' # find or creates local


#%% # Check output folders exist
from slil.common.data_configs import checkBaseOutputFoldersExist
checkBaseOutputFoldersExist(experiments)


#%% # Filter raw motion capture data from .log files and creates .c3d files
frmv4.run(experiments, models + ['static'], runOnlyOne, clusterAddress)

# Only needed if data is on a network drive or OneDrive
#from slil.common.data_configs import checkBadFiles, checkBadFiles2
#checkBadFiles()
##checkBadFiles2() # for debugging


# %% # Generate OpenSim models and cache files
import slil.mesh.interface as smi
mi = smi.MeshInterface(1) # uses 3-Matic!

generate_model.generate_STLs(mi, experiments)
generate_model.generate_model(mi, experiments)


#%% # 1. Transposes all trials to align to the radius of the respective static trial (found during generate_model)
# 2. Make all other bones move relative to the radius (i.e., radius will not move, helpful for visualizing motions)
# Warning: this is a slow process, especially for long trials. Takes over an hour on a decent computer.
# A c++ implementation was planned and would be faster.
tc3d.run(experiments, models, runOnlyOne, runWithRay = True, clusterAddress = clusterAddress)


#%% # align markers using optimization method
# Poor results so don't use.
if False:
    #from slil.process.align_by_slil_gap import runOptimization
    from slil.process.align_by_surface import runOptimization
    runOptimization('11534', [ 'normal', 'cut', 'scaffold' ], clusterAddress = clusterAddress)

    modelsOptimized = [ 'normal' ] #, 'cut']#, 'scaffold']
    from slil.process import generate_model
    generate_model.setOptimizedMarkers(experiments, modelsOptimized)

    # For viewing results of optimizations
    import dev_generateVideo as slilgv
    for experiment in experiments:
        slilgv.generateCut2ScaffoldVideo(experiment)


# %% # Manually find aligment of markers (normal to cut and scaffold trials) by choosing parameters and visualising resultant alignment

import slil.common.data_configs as dc
from slil.process.align_viewer import AlignViewer
from slil.process.align_viewer_settings import settings

#Groups
# D: Couldn't align any normal to cut
# C: Couldn't align scaphoid and/or lunate normal to cut
# B: Couldn't align scaphoid and/or lunate noraml to implanted
# A: All found reasonable alignment

# CAD # Group # Comments
# 11524 C # But scaffold to normal looks fine...
          # Could be A with right initial aligment.. tried many times
# 11525 D # scaffold to normal looks almost fine...
# 11526 B # scaffold: scaphoid pin looks moved... just
# 11527 C # cut and scaffold look very similar... Could try a different static file or timestamp
# 11534 A # lunate is twisted, could be real and not marker error...
# 11535 B # scaffold: lunate looks moved
# 11536 A
# 11537 A
# 11538 B # scaffold: lunate pin looks moved
# 11539 B # scaffold: lunate and/or scaphoid pins look moved

for experiment in experiments:
    experiment = '11537'
    modelInfo = dc.load(experiment)
    # only needed if missing some geometry (e.g. capitate)
    #mi.open_project(modelInfo['3_matic_file'])

    scene = AlignViewer(modelInfo)
    scene.loadScene()
    flip, possiblePinSet, alignmentMethod, mapMarkerChangesN2C, mapMarkerChangesN2S = settings()
    scene.setConfigurations(flip, possiblePinSet, alignmentMethod,
        mapMarkerChangesN2C, mapMarkerChangesN2S)
    scene.align()

    # this can be called manually in the GUI
    scene.setMarkersInModels(['normal', 'cut', 'scaffold'])

#%%    
scene.setMarkersInModels(['scaffold'])


#%% # Run inverse kinematics
# This is calls an external program written with OpenSim C++ API and parallelises the inverse kinematics and point kinematics. It's the fastest method to run IK.
import slil.process.inverse_kinematics as pik
pik.run(experiments, models, '', runOnlyOne)


# %% # Generate the primary output data. Used for plotting and tabulation.
from slil.cache_results_plot import createModelsCache
createModelsCache(experiments, localParallel = False)


#%%
from plot_individuals_functions import generateIndividualGraphics
generateIndividualGraphics(experiments, localParallel = False)


# %% # Tabulate
from slil.common.tabulation import DiscreteResults
dr = DiscreteResults()
dr.generate(experiments)
dr.format('RU')
dr.format('FE')
from slil.common.data_configs import outputFolders
dr.save(outputFolders()['root'] + "\discrete_results.xlsx")
