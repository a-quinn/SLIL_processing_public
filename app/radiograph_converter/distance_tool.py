# Author: Alastair Quinn 2023

import numpy as np
import slil.common.math as fm
from slil.common.cache import loadCache, saveCache

class DistanceTool():
    def __init__(self, scene, start_active=True):
        self.scene = scene
        self.active = False
        self.distance_between = 0.0

        self.line_widget = {}

        def updatLine(pointa, pointb):
            self.updateLine()

        self.line_widget['distancetool'] = self.scene.plotter[0, self.scene.plotR].add_line_widget(
            callback=updatLine,
            use_vertices=True,
            color='green')
        self.line_widget['distancetool'].SetEnabled(0)
        
        cache = loadCache('points_picker_distances')
        if cache:
            self.distances = cache
        else:
            self.distances = {}

        if start_active:
            self.addTool()
    
    def addTool(self):
        self.line_widget['distancetool'].SetPoint1(-10, 20, 0)
        self.line_widget['distancetool'].SetPoint2(10, 20, 0)

        self.active = True
        
        self.line_widget['distancetool'].SetEnabled(1)
        self.updateLines()

        self.scene.plotter[0, self.scene.plotR].add_text(
            f"Distance: {self.distance_between:.3f} mm", position = 'upper_right',
            color = 'black', name = 'distance_tool')
    
    def removeTool(self):
        self.active = False
        self.line_widget['distancetool'].SetEnabled(0)
        self.scene.plotter[0, self.scene.plotR].remove_actor('distance_tool')

    def setupScene(self):
        self.updateLine()

    def saveDistances(self, current_trial):
        print('Saving Distances')
        if not self.scene.modelInfo['experimentID'] in self.distances:
            self.distances[self.scene.modelInfo['experimentID']] = {}
        self.distances[self.scene.modelInfo['experimentID']][current_trial] = self.distance_between
        saveCache(self.distances, 'points_picker_distances')
    
    def updateLine(self):
        if self.active:
            point1 = list(self.line_widget['distancetool'].GetPoint1())
            point1[2] = 0.0
            self.line_widget['distancetool'].SetPoint1(point1)
            point2 = list(self.line_widget['distancetool'].GetPoint2())
            point2[2] = 0.0
            self.line_widget['distancetool'].SetPoint2(point2)

            self.distance_between = self.calcDistance()
            self.scene.plotter[0, self.scene.plotR].add_text(
                f"Distance: {self.distance_between:.3f} mm", position = 'upper_right',
                color = 'black', name = 'distance_tool')

    def calcDistance(self):
        point1 = np.array(self.line_widget['distancetool'].GetPoint1())
        point2 = np.array(self.line_widget['distancetool'].GetPoint2())
        return fm.calcDist(point1, point2)