# Author: Alastair Quinn 2023

#%%
#'11525',
#'11526',
#'11527',
#'11534',
#'11535',
#'11536',
#'11537',
#'11538',
#'11539'
from gui import GUI
gui = GUI(experiment = '11538')


self = gui
self.angle_tool.SLangles
self.angle_tool.calAngle()
self.angle_tool.updateLines()

self.distance_tool.distances
measure = self.angle_tool.SLangles
measure = self.distance_tool.distances

for exp in measure:
    trialNames = ""
    trialValues = ""
    for trial in measure[exp]:
        value = measure[exp][trial]
        trialValues += str(value) + '\t'
        trial = trial.replace('_40_40','')
        trial = trial.replace('_30_30','')
        trial = trial.replace('\\','')
        trialNames += trial + '\t'
    print(f"{exp:.4f}" + '\t' + trialNames)
    print(f"{exp:.4f}" + '\t' + trialValues)
print('\r\n')


from slil.process.qt_plotter_multi import MultiScene
from pyvistaqt import BackgroundPlotter
scene = MultiScene()

plotter = BackgroundPlotter()

self.scene.plotter[0, self.scene.plotR].line_widgets.remove(self.angle_tool.line_widget['lunate'])


import vtk
line = vtk.vtkLineWidget()
line.SetClampToBounds



self.angle_tool.angle_between

boneSetInd = [i for i, x in enumerate(self.radiusCoords) if x['name'] == self.scene.modelInfo['experimentID']]
if len(boneSetInd) == 0:
    print(f"Error, no point indexes set for experiment {self.scene.modelInfo['experimentID']}")
else:
    boneSetInd = boneSetInd[0]


radiusCoord = self.radiusCoords[boneSetInd]



import slil.common.math as fm
import numpy as np

from pyvistaqt import BackgroundPlotter
def plotAxis(plot: BackgroundPlotter, rot, center=np.array([0,0,0]), name='', scale=20.0):
    rot *= scale
    lines = np.array([
        [center, rot[:3, 0]],
        [center, rot[:3, 1]],
        [center, rot[:3, 2]]
    ])
    plot.add_lines(
        lines = lines[0],
        name='Axis_' + name + '_x',
        width=3.0,
        color='red')
    plot.add_lines(
        lines = lines[1],
        name='Axis_' + name + '_y',
        width=3.0,
        color='green')
    plot.add_lines(
        lines = lines[2],
        name='Axis_' + name + '_z',
        width=3.0,
        color='blue')

rot = np.eye(3)
plotAxis(self.scene.plotter[0, self.scene.plotL], rot, center=np.array([0,0,0]), name='global')


#%%




#%%
gui.scene.plotter[0, gui.scene.plotL].enable_point_picking(callback=pickedPoint, use_mesh=True)


#gui.scene.plotter[0, gui.scene.plotL].picked_point
selected_points = gui.scene.plotter[0, gui.scene.plotL].picked_points[0]



# %%

# add dropdown to choose trial
# add slider to move through trial
# update position of bones at set trial time
# maybe automate measures
