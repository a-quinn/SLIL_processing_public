

import os
os.environ["QT_API"] = "pyqt5"

import pyvista as pv
from copy import deepcopy
import numpy as np
from pyvistaqt import MultiPlotter
import slil.process.functions as fn

class MultiScene():
    def __init__(self, modelInfo):

        self.plotter = MultiPlotter(window_size=(1500, 1000), nrows=1, ncols=2, show=True)

        self.plotL = 0
        self.plotR = 1

        self.plotter[0, self.plotL].background_color = 'white'
        self.plotter[0, self.plotR].background_color = 'white'
        self._setup(modelInfo)
        self.actors = [{}, {}]

    def _reset(self):
        for indPlot in range(2):
            while len(self.plotter[0, indPlot].actors) > 0:
                actor_name = list(self.plotter[0, indPlot].actors.keys())[0]
                self.plotter[0, indPlot].remove_actor(actor_name)
        del self.boneGeometry
        del self.modelCacheExtra
        del self.bonePolyData

    def _setup(self, modelInfo):
        self.modelInfo = modelInfo
        extra = fn.getModelCacheExtra(self.modelInfo, True)
        if extra != False:
            self.modelCacheExtra = extra
        else:
            print('Error: No extra cache model found.')

    def loadScene(self, plotInd):
        if not hasattr(self,'boneGeometry'):
            names = fn.boneModelNames(self.modelInfo) #[lunateName, scaphoidName, radiusName, metacarp3Name] =
            names = names + self.modelInfo['otherHandBones']
            self.boneGeometry = {}
            for name in names:
                self.boneGeometry[name] = fn.getGeometry(self.modelInfo, name)
        if not hasattr(self,'bonePolyData'):
            self.bonePolyData = [{}, {}]
        self.createSurfaces(plotInd)
        self.addMeshes(plotInd)

    def createSurfaces(self, plotInd):
        def converToPolyData(geometry):
            faces0 = geometry[0]
            vertices = geometry[1]
            vertices = np.array(vertices)
            facesTemp = np.array(faces0)
            faces = np.empty((facesTemp.shape[0], facesTemp.shape[1]+1), dtype=int)
            faces[:, 1:] = facesTemp
            faces[:, 0] = 3
            return pv.PolyData(vertices, faces)
        for geometry in self.boneGeometry:
            self.bonePolyData[plotInd][geometry] = converToPolyData(self.boneGeometry[geometry])
        if hasattr(self.modelCacheExtra,'sensorGuide'):
            self.bonePolyData[plotInd]['sensorGuide'] = converToPolyData(self.modelCacheExtra['sensorGuide'])
        #self.bonePolyData[plotInd]['placedScaffold'] = converToPolyData(self.modelCacheExtra['placedScaffold'])
        #self.bonePolyData[plotInd]['surgicalGuideSca'] = converToPolyData(self.modelCacheExtra['surgicalGuideSca'])
        #self.bonePolyData[plotInd]['surgicalGuideLun'] = converToPolyData(self.modelCacheExtra['surgicalGuideLun'])
        for boneName in ['lunate', 'scaphoid']:
            self.bonePolyData[plotInd][self.modelInfo['names'][boneName]+'_moved'] = deepcopy(self.bonePolyData[plotInd][self.modelInfo['names'][boneName]])

    def addMeshes(self, plotInd):
        for geometry in self.bonePolyData[plotInd]:
            surf = self.bonePolyData[plotInd][geometry]
            self.actors[plotInd][geometry] = self.plotter[0, plotInd].add_mesh(
                mesh = surf,
                name = geometry,
                #scalars = np.arange(surf.n_faces),
                show_scalar_bar = False,
                color='blanchedalmond',
                specular=1.0,
                specular_power=10,
                opacity=0.5
            )
        if hasattr(self.modelCacheExtra,'sensorGuidePins'):
            lines = self.modelCacheExtra['sensorGuidePins']
            for line in lines:
                l = self.modelCacheExtra['sensorGuidePins'][line]
                mesh = pv.Line(l['point1'], l['point2'])
                self.actors[plotInd][line] = self.plotter[0, plotInd].add_mesh(
                    mesh,
                    name = line,
                    color = 'k',
                    line_width = 3,
                    calar_bar_args = None,
                    opacity=0.5
                    )

    def addPoints(self, plotInd, points, name = 'tempMarkerSpheres_static', color = 'grey'):
        self.actors[plotInd][name] = self.plotter[0, plotInd].add_points(
            points,
            name = name,
            color = color,
            render_points_as_spheres = True,
            point_size = 20.0,
            opacity = 0.9
        )
        
    def setOpacity(self, plotInd, name, opacity):
        self.actors[plotInd][name] = self.plotter[0, plotInd].add_mesh(
                        mesh = self.bonePolyData[plotInd][name],
                        name = name,
                        #scalars = np.arange(self.bonePolyData[name].n_faces),
                        color = 'blanchedalmond',
                        specular = 1.0,
                        specular_power = 10,
                        show_scalar_bar = False,
                        opacity = opacity
                    )

    def viewScene(self, plotInd):
        self.plotter[0, plotInd].show_axes()
        self.plotter[0, plotInd].view_vector([0, 0, -1])
        self.plotter[0, plotInd].set_viewup([-1, 0, 0])
        #self.plotter[0, plotInd].set_position((1.0, 1.0, 1.0))
        self.plotter.show()