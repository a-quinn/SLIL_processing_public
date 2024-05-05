# Author: Alastair Quinn 2023
import pyvista as pv
#pv.set_jupyter_backend('pythreejs')
import slil.process.functions as fn
import numpy as np
from pyvistaqt import BackgroundPlotter

class Scene():
    def __init__(self, modelInfo, useBackgroundPlotter = True):
        self.modelInfo = modelInfo
        if useBackgroundPlotter:
            self.plotter = BackgroundPlotter(window_size=(1000, 1000))
        else:
            self.plotter = pv.Plotter(window_size=[1000, 1000])
        self.plotter.background_color = 'white'
        self.plotter.show_axes_all()
        #elf.plotter = pv.Plotter(window_size=[1000, 1000])
        #self.sceneObjects = {}
        extra = fn.getModelCacheExtra(modelInfo, True)
        if extra != False:
            self.modelCacheExtra = extra
        else:
            print('Error: No extra cache model found.')
        self.actors = {}

    def loadGeometry(self):
        names = fn.boneModelNames(self.modelInfo) #[lunateName, scaphoidName, radiusName, metacarp3Name] =
        names = names + self.modelInfo['otherHandBones']
        self.boneGeometry = {}
        for name in names:
            self.boneGeometry[name] = fn.getGeometry(self.modelInfo, name)
        self.createSurfaces()

    def createSurfaces(self):
        def converToPolyData(geometry):
            faces0 = geometry[0]
            vertices = geometry[1]
            vertices = np.array(vertices)
            facesTemp = np.array(faces0)
            faces = np.empty((facesTemp.shape[0], facesTemp.shape[1]+1), dtype=int)
            faces[:, 1:] = facesTemp
            faces[:, 0] = 3
            return pv.PolyData(vertices, faces)
        self.bonePolyData = {}
        for geometry in self.boneGeometry:
            self.bonePolyData[geometry] = converToPolyData(self.boneGeometry[geometry])
        if hasattr(self.modelCacheExtra,'sensorGuide'):
            self.bonePolyData['sensorGuide'] = converToPolyData(self.modelCacheExtra['sensorGuide'])
        #self.bonePolyData['placedScaffold'] = converToPolyData(self.modelCacheExtra['placedScaffold']) # Redacted
        #self.bonePolyData['surgicalGuideSca'] = converToPolyData(self.modelCacheExtra['surgicalGuideSca'])
        #self.bonePolyData['surgicalGuideLun'] = converToPolyData(self.modelCacheExtra['surgicalGuideLun'])

    def addMeshes(self):
        for geometry in self.bonePolyData:
            surf = self.bonePolyData[geometry]
            self.actors[geometry] = self.plotter.add_mesh(
                mesh = surf,
                name = geometry,
                #scalars = np.arange(surf.n_faces),
                show_scalar_bar = False,
                color='blanchedalmond',
                specular=1.0,
                specular_power=10
            )
        if hasattr(self.modelCacheExtra,'sensorGuidePins'):
            lines = self.modelCacheExtra['sensorGuidePins']
            for line in lines:
                l = self.modelCacheExtra['sensorGuidePins'][line]
                mesh = pv.Line(l['point1'], l['point2'])
                self.actors[line] = self.plotter.add_mesh(
                    mesh,
                    name = line,
                    color = 'k',
                    line_width = 3,
                    calar_bar_args = None
                    )

    def addPoints(self, points, name = 'tempMarkerSpheres_static', color = 'grey'):
        self.actors[name] = self.plotter.add_points(
            points,
            name = name,
            color = color,
            render_points_as_spheres = True,
            point_size = 20.0,
            opacity = 0.9
        )
    
    def setOpacity(self, name, opacity):
        self.actors[name] = self.plotter.add_mesh(
                        mesh = self.bonePolyData[name],
                        name = name,
                        #scalars = np.arange(self.bonePolyData[name].n_faces),
                        color = 'blanchedalmond',
                        specular = 1.0,
                        specular_power = 10,
                        show_scalar_bar = False,
                        opacity = opacity
                    )

    def viewScene(self):
        self.plotter.show_axes_all()
        self.plotter.view_vector([0, 1, 0])
        self.plotter.set_viewup([0, 0, -1])
        #self.plotter.set_position((1.0, 1.0, 1.0))
        self.plotter.show()