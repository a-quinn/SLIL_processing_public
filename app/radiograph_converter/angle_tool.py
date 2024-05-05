# Author: Alastair Quinn 2023

import numpy as np
import slil.common.math as fm
from slil.common.cache import loadCache, saveCache

class AngleTool():
    def __init__(self, scene, start_active=True):
        self.scene = scene
        self.active = False
        self.angle_between = 0.0

        def updatLine(pointa, pointb):
            self.updateLines()

        self.line_widget = {}
        self.line_widget['lunate'] = self.scene.plotter[0, self.scene.plotR].add_line_widget(
            callback=updatLine,
            use_vertices=True,
            color='blue')
        self.line_widget['scaphoid'] = self.scene.plotter[0, self.scene.plotR].add_line_widget(
            callback=updatLine,
            use_vertices=True,
            color='green')
        self.line_widget['lunate'].SetEnabled(0)
        self.line_widget['scaphoid'].SetEnabled(0)
        
        cache = loadCache('points_picker_SLangles')
        if cache:
            self.SLangles = cache
        else:
            self.SLangles = {}
        
        if start_active:
            self.addTool()
    
    def addTool(self):
        self.line_widget['lunate'].SetPoint1(-10, 20, 0)
        self.line_widget['lunate'].SetPoint2(10, 20, 0)
        self.line_widget['scaphoid'].SetPoint1(-10, 10, 0)
        self.line_widget['scaphoid'].SetPoint2(10, 10, 0)

        self.active = True

        self.line_widget['lunate'].SetEnabled(1)
        self.line_widget['scaphoid'].SetEnabled(1)
        self.updateLines()

        self.scene.plotter[0, self.scene.plotR].add_text(
            f"Angle: {self.angle_between:.3f} deg", position = 'upper_right',
            color = 'black', name = 'angle_tool')
        
    def removeTool(self):
        self.active = False
        self.line_widget['lunate'].SetEnabled(0)
        self.line_widget['scaphoid'].SetEnabled(0)
        self.scene.plotter[0, self.scene.plotR].remove_actor('angle_tool')

    def setupScene(self):
        self.updateLines()

    def saveAngles(self, current_trial):
        print('Saving Angles')
        if not self.scene.modelInfo['experimentID'] in self.SLangles:
            self.SLangles[self.scene.modelInfo['experimentID']] = {}
        self.SLangles[self.scene.modelInfo['experimentID']][current_trial] = self.angle_between
        saveCache(self.SLangles, 'points_picker_SLangles')
    
    def updateLines(self):
        if self.active:
            for widget in self.line_widget:
                point1 = list(self.line_widget[widget].GetPoint1())
                point1[2] = 0.0
                self.line_widget[widget].SetPoint1(point1)
                point2 = list(self.line_widget[widget].GetPoint2())
                point2[2] = 0.0
                self.line_widget[widget].SetPoint2(point2)

            self.angle_between = self.calcAngle()
            self.scene.plotter[0, self.scene.plotR].add_text(
                f"Angle: {self.angle_between:.3f} deg", position = 'upper_right',
                color = 'black', name = 'angle_tool')

    def calcAngle(self):
        point1 = np.array(self.line_widget['lunate'].GetPoint1())
        point2 = np.array(self.line_widget['lunate'].GetPoint2())
        v1 = fm.normalizeVector(point1 - point2)
        point1 = np.array(self.line_widget['scaphoid'].GetPoint1())
        point2 = np.array(self.line_widget['scaphoid'].GetPoint2())
        v2 = fm.normalizeVector(point1 - point2)
        return np.rad2deg(fm.angleBetweenVectors(v1, v2))
