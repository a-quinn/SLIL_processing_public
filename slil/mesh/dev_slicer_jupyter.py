# Author: Alastair Quinn 2022
# This script was for developing and testing 3D Slicer using jupyter notebooks.

# %% To allow re-importing modules without restarting jupyter
%load_ext autoreload
%autoreload 2
# %%

# Run in terminal to install 3D Slicer kernel in jupyter server
# jupyter-kernelspec install "C:/Users/Griffith/AppData/Local/NA-MIC/Slicer 5.0.3/NA-MIC/Extensions-30893/SlicerJupyter/share/Slicer-5.0/qt-loadable-modules/JupyterKernel/Slicer-5.0" --replace --user
# or run the notebook

# test if connected
import JupyterNotebooksLib as slicernb
slicernb.ViewDisplay()

import slicer
slicer.util._executePythonModule('jupyter', ['notebook', 'list'])

#
from platform import python_version
print(python_version())

#
import rpyc
rpyc.version.version

# %%
import slil.mesh.slicerConn.listener as listener
listener.toggle_listener()

# %%
#import slicer
import slil.mesh.interface as smi
mi = smi.MeshInterface(1)
mi.client.fun('util.warningDisplay', 'Test text2')
mi.client.fun('util.showStatusMessage', 'Test text2')
mi.client.check_connection()
mi.client.fun('util.exit', 0)
mi.client.fun('util.restart', 0)
mi.closeSession()




# %%
import slil.mesh.interface as smi
mi = smi.MeshInterface(1)
import slil.common.data_configs as dc

#%%
experiments = [
    '11525',
    '11526',
    #'11527', # bad, do not run
    '11534',
    '11535',
    '11536',
    '11537',
    '11538',
    '11539'
    ] 
#for experiment in experiments:
if True:
    experiment = '11525'
    modelInfo = dc.load(experiment)


    path2volumes = modelInfo['rootFolder'] + '\\' + 'models' + '\\' \
        + modelInfo['experimentID'] + '\\' + 'bones' + '\\'

    #%%
    mi.import_volume(path2volumes + modelInfo['experimentID'] + '_reg_lun.stl')
    mi.import_volume(path2volumes + modelInfo['experimentID'] + '_reg_sca.stl')
    mi.import_volume(path2volumes + modelInfo['experimentID'] + '_reg_rad.stl')
    mi.import_volume(path2volumes + modelInfo['experimentID'] + '_reg_mc3.stl')
    mi.import_volume(path2volumes + modelInfo['experimentID'] + '_reg_cap.stl')
# %%

#import slicer
#mi.create_point([1,2,3])

mi.launchSession()
mi.client.fun('util.showStatusMessage', 'Test text3')
mi.open_project('s')

mi.create_line()
mi.client.fun('util.warningDisplay', 'Test text2')
mi.client.fun('util.showStatusMessage', 'Test text2')
mi.client.fun('util.exit', 0)
mi.client.fun('util.restart', 0)
mi.closeSession()
# %%
mi.create_line([1,2,3], [3,6,3])
# %%


#%%


import slil.mesh.interface as smi
mi = smi.MeshInterface(1)
mi.load_remote_functions(globals()) # to enable the slicer functions in the global namespace
slicer.util.selectModule('Data')


#mi.closeSession()

a = mi.client.fun('slicer.mrmlScene.AddNewNodeByClass', "vtkMRMLModelNode")

dir(a)
a.SetName('6')

c = mi.find_part('11525 Sensor Guide2. 05.13')
dir(c)
c.SetName('7')


a = mi.client.fun('slicer.util.selectedModule')
b = mi.client.fun('slicer.util.exportNode', 'Model', 'tessssst')


a = mi.client.fun('slicer.mrmlScene.Clear', 0)
mi.close_project()

slicer.util.loadModel('testse.mrml')

slicer.util.loadScene('testse.mrml')
slicer.util.saveScene('testse.mrml')
slicer.util.restart()

#%% Start creating scene/collection to be used later.

from pathlib import Path
from slil.common.data_configs import load

exp_ID = '11527'
exp = load(exp_ID)

model_folder = exp['modelFolder']

if Path(model_folder + r'\\collection.mrml').is_file():
    mi.open_project(model_folder + r'\\collection.mrml')
else:
    files_to_load = []
    stl_folder = model_folder + r'\\' + 'stl'
    files_to_load.append(stl_folder + r'\\' + exp['sensorGuideName'] + '.stl')
    files_to_load.append(stl_folder + r'\\' + exp['placedScaffoldName'] + '.stl')
    files_to_load.append(stl_folder + r'\\' + exp['surgicalGuideSca'] + '.stl')
    files_to_load.append(stl_folder + r'\\' + exp['surgicalGuideLun'] + '.stl')

    bone_names = ['lunate', 'scaphoid', 'radius', 'metacarp3']
    for bone in bone_names:
        files_to_load.append(stl_folder + r'\\' + exp['names'][bone] + '.stl') #+ exp['experimentID'] + '_reg_lun.stl'
    files_to_load.append(stl_folder + r'\\' + exp['otherHandBones'][0] + '.stl')
    for file in files_to_load:
        mi.import_volume(file)



def scale_and_save(name, scale = 1.0):
    mi.scale_factor(name, scale)
    mi.export_stl(name, stl_folder + r'\\' + name + '.stl')

stls_to_fix = [
    exp['placedScaffoldName'],
    exp['surgicalGuideSca'],
    exp['surgicalGuideLun'],
    exp['sensorGuideName'],
]
for bone in bone_names:
    stls_to_fix.append(exp['names'][bone])


for name in stls_to_fix:
    scale_and_save(name, 0.001)

scale_and_save(exp['placedScaffoldName'], 0.001)
scale_and_save(exp['surgicalGuideSca'], 0.001)
scale_and_save(exp['surgicalGuideLun'], 0.001)
scale_and_save(exp['sensorGuideName'], 0.001)
scale_and_save(exp['otherHandBones'][0], 1000.0)


slicer.util.resetThreeDViews()

mi.save_project(model_folder + r'\\collection.mrml')
# %%
