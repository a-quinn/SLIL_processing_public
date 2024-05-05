

import os
os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets
from pyvistaqt import QtInteractor, MainWindow

import pyvista as pv
#pv.set_jupyter_backend('pythreejs')
import slil.process.functions as fn
import slil.process.align_main as fnam
from slil.process.inverse_kinematics import findMarkerToPointsAlignmentOnce
from copy import deepcopy
import slil.common.math as fm
import numpy as np
from pyvistaqt import MultiPlotter
#from pyvistaqt import BackgroundPlotter

#import sys
#qtApp = QtWidgets.QApplication(sys.argv)
from qtpy.QtWidgets import (QSpinBox, QSlider, QCheckBox, QVBoxLayout,
    QGroupBox, QGridLayout, QHBoxLayout)
from qtpy.QtCore import Qt


def visualizeMarkerPins(scene, possiblePinSet, currentModel, plotInd, modelInfo, framePointsRaw, color = 'black'):
    # this function is slow because it takes time to draw the lines...

    for boneName in fn.boneNames():
        plateAssignment = modelInfo['plateAssignment_' + currentModel]
        boneModelName = modelInfo['names'][boneName]
        points = np.array([
            framePointsRaw[plateAssignment[boneModelName][0]],
            framePointsRaw[plateAssignment[boneModelName][1]],
            framePointsRaw[plateAssignment[boneModelName][2]] # middle marker
            ])
        pinsSets = fnam.generatePinsFromMarkerPoints(points)
        pinsSets = pinsSets[possiblePinSet[modelInfo['experimentID']][currentModel][boneName]]
        if hasattr(scene, 'actorsLines'):
            if currentModel in scene.actorsLines[plotInd] and boneName in scene.actorsLines[plotInd][currentModel]:
                for actor in scene.actorsLines[plotInd][currentModel][boneName]:
                    scene.plotter[0, plotInd].remove_actor(actor)
        else:
            scene.actorsLines= [{}, {}]
        scene.actorsLines[plotInd][currentModel] = {}
        scene.actorsLines[plotInd][currentModel][boneName] = {}
        for indSet, pins in enumerate(pinsSets):
            name1 = 'line_0_' + str(currentModel) + '_' + str(boneName) + '_' + str(indSet)
            scene.actorsLines[plotInd][currentModel][boneName][name1] = scene.plotter[0, plotInd].add_lines(
                    lines = pins[0:2],
                    color = color,
                    width = 2.0,
                    name = name1
                )
            name2 = 'line_1_' + str(currentModel) + '_' + str(boneName) + '_' + str(indSet)
            scene.actorsLines[plotInd][currentModel][boneName][name2] = scene.plotter[0, plotInd].add_lines(
                    lines = pins[2:4],
                    color = color,
                    width = 2.0,
                    name = name2
                )

class AlignViewer():
    def __init__(self, modelInfo, parent=None, show=True):

        self.modelInfo = modelInfo
        self.plotter = MultiPlotter(window_size=(1500, 1000), nrows=1, ncols=2, show=True)

        self.plotter[0, 0].background_color = 'white'
        self.plotter[0, 1].background_color = 'white'
        self.plotter[0, 0].show_axes_all()
        self.plotter[0, 1].show_axes_all()
        
        #self.sceneObjects = {}
        self.modelCache = fn.getModelCache(self.modelInfo)
        extra = fn.getModelCacheExtra(modelInfo, True)
        if extra != False:
            self.modelCacheExtra = extra
        else:
            print('Error: No extra cache model found.')
        self.actors = [{}, {}]

        self.translationOffset = [0.0, 0.0, 0.0]
        self.rotationOffset = [0.0, 0.0, 0.0]
        self.alignedPoints = {}
        self.mapMarkerChangesN2 = {}

        self.markersFinal = {}
        
        windowSelf = self.plotter._window
        mainMenu = windowSelf.menuBar()
        # allow adding a sphere
        meshMenu = mainMenu.addMenu('Mesh')
        windowSelf.add_sphere_action = QtWidgets.QAction('Add Sphere', windowSelf)
        windowSelf.add_sphere_action.triggered.connect(self.add_sphere)
        meshMenu.addAction(windowSelf.add_sphere_action)
        
        #user_menu_experiments = self.plotter[0, 0].main_menu.addMenu('Save Markers In Model')
        def set_markers_in_model():
            self.setMarkersInModels(['normal', 'cut', 'scaffold'])
        windowSelf.set_markers_in_models = QtWidgets.QAction('Save Markers In Model', windowSelf)
        windowSelf.set_markers_in_models.triggered.connect(set_markers_in_model)
        mainMenu.addAction(windowSelf.set_markers_in_models)
        #user_menu_experiments.addAction('setMarkersInModels', self.setMarkersInModels(['normal', 'cut', 'scaffold']))


        layout = self.plotter._layout
        groupBox = self.plotter._central_widget

        layoutBottomLeft = QGroupBox()
        layoutLeft = QVBoxLayout()

        self.n2cLunate = QCheckBox("n2cLunate")
        self.n2cLunate.toggled.connect(self.n2cLunate_toggled)
        layoutLeft.addWidget(self.n2cLunate)
        self.n2cScaphoid = QCheckBox("n2cScaphoid")
        self.n2cScaphoid.toggled.connect(self.n2cScaphoid_toggled)
        layoutLeft.addWidget(self.n2cScaphoid)
        self.n2cRadius = QCheckBox("n2cRadius")
        self.n2cRadius.toggled.connect(self.n2cRadius_toggled)
        layoutLeft.addWidget(self.n2cRadius)
        self.n2cMetacarp3 = QCheckBox("n2cMetacarp3")
        self.n2cMetacarp3.toggled.connect(self.n2cMetacarp3_toggled)
        layoutLeft.addWidget(self.n2cMetacarp3)

        self.n2cPinsLunate = QCheckBox("Pins Lunate")
        self.n2cPinsLunate.toggled.connect(self.n2cPinsLunate_toggled)
        layoutLeft.addWidget(self.n2cPinsLunate)
        self.n2cPinsScaphoid = QCheckBox("Pins Scaphoid")
        self.n2cPinsScaphoid.toggled.connect(self.n2cPinsScaphoid_toggled)
        layoutLeft.addWidget(self.n2cPinsScaphoid)
        self.n2cPinsRadius = QCheckBox("Pins Radius")
        self.n2cPinsRadius.toggled.connect(self.n2cPinsRadius_toggled)
        layoutLeft.addWidget(self.n2cPinsRadius)
        self.n2cPinsMetacarp3 = QCheckBox("Pins Metacarp3")
        self.n2cPinsMetacarp3.toggled.connect(self.n2cPinsMetacarp3_toggled)
        layoutLeft.addWidget(self.n2cPinsMetacarp3)

        callbackFuncs1 = [
            self.spinboxX_value_changed,
            self.spinboxY_value_changed,
            self.spinboxZ_value_changed,
        ]
        callbackFuncs1R = [
            self.spinboxXr_value_changed,
            self.spinboxYr_value_changed,
            self.spinboxZr_value_changed,
        ]
        callbackFuncs2 = [
            self.sliderX_value_changed,
            self.sliderY_value_changed,
            self.sliderZ_value_changed,
        ]
        self.spinBox = []
        self.slider = []
        for indAxies in range(0,3):
            self.spinBox.append(QSpinBox(groupBox))
            self.spinBox[indAxies].setValue(0)
            self.spinBox[indAxies].setMinimum(-50)
            self.spinBox[indAxies].setMaximum(50)
            self.spinBox[indAxies].valueChanged.connect(callbackFuncs1[indAxies])
            layoutLeft.addWidget(self.spinBox[indAxies])

            #self.slider.append(QSlider(Qt.Orientation.Horizontal, groupBox))
            #self.slider[indAxies].setValue(0)
            #self.slider[indAxies].setMinimum(-20)
            #self.slider[indAxies].setMaximum(20)
            #self.slider[indAxies].valueChanged.connect(callbackFuncs2[indAxies])
            #layout.addWidget(self.slider[indAxies])
            #QSlider(Qt.Orientation.Horizontal, groupBox).sliderReleased()
        self.spinBoxR = []
        for indAxies in range(0,3):
            self.spinBoxR.append(QSpinBox(groupBox))
            self.spinBoxR[indAxies].setValue(0)
            self.spinBoxR[indAxies].setMinimum(-180)
            self.spinBoxR[indAxies].setMaximum(180)
            self.spinBoxR[indAxies].valueChanged.connect(callbackFuncs1R[indAxies])
            layoutLeft.addWidget(self.spinBoxR[indAxies])

        layoutBottomRight = QGroupBox()
        layoutRight = QVBoxLayout()
        
        self.n2sLunate = QCheckBox("n2sLunate")
        self.n2sLunate.toggled.connect(self.n2sLunate_toggled)
        layoutRight.addWidget(self.n2sLunate)
        self.n2sScaphoid = QCheckBox("n2sScaphoid")
        self.n2sScaphoid.toggled.connect(self.n2sScaphoid_toggled)
        layoutRight.addWidget(self.n2sScaphoid)
        self.n2sRadius = QCheckBox("n2sRadius")
        self.n2sRadius.toggled.connect(self.n2sRadius_toggled)
        layoutRight.addWidget(self.n2sRadius)
        self.n2sMetacarp3 = QCheckBox("n2sMetacarp3")
        self.n2sMetacarp3.toggled.connect(self.n2sMetacarp3_toggled)
        layoutRight.addWidget(self.n2sMetacarp3)

        self.n2sPinsLunate = QCheckBox("Pins Lunate")
        self.n2sPinsLunate.toggled.connect(self.n2sPinsLunate_toggled)
        layoutRight.addWidget(self.n2sPinsLunate)
        self.n2sPinsScaphoid = QCheckBox("Pins Scaphoid")
        self.n2sPinsScaphoid.toggled.connect(self.n2sPinsScaphoid_toggled)
        layoutRight.addWidget(self.n2sPinsScaphoid)
        self.n2sPinsRadius = QCheckBox("Pins Radius")
        self.n2sPinsRadius.toggled.connect(self.n2sPinsRadius_toggled)
        layoutRight.addWidget(self.n2sPinsRadius)
        self.n2sPinsMetacarp3 = QCheckBox("Pins Metacarp3")
        self.n2sPinsMetacarp3.toggled.connect(self.n2sPinsMetacarp3_toggled)
        layoutRight.addWidget(self.n2sPinsMetacarp3)
        
        layoutBottomLeft.setLayout(layoutLeft)
        layoutBottomRight.setLayout(layoutRight)
        layout.addWidget(layoutBottomLeft)
        layout.addWidget(layoutBottomRight)

    def _toggleInList(self, listName, boneName, shouldBeInList):
        if shouldBeInList and not boneName in self.alignmentMethod[self.modelInfo['experimentID']][listName]:
            self.alignmentMethod[self.modelInfo['experimentID']][listName].append(boneName)
        if not shouldBeInList and boneName in self.alignmentMethod[self.modelInfo['experimentID']][listName]:
            self.alignmentMethod[self.modelInfo['experimentID']][listName].remove(boneName)

    def n2cLunate_toggled(self, value):
        self._toggleInList('normal2cut', 'lunate', value)
        self.alignLeft()
    def n2cScaphoid_toggled(self, value):
        self._toggleInList('normal2cut', 'scaphoid', value)
        self.alignLeft()
    def n2cRadius_toggled(self, value):
        self._toggleInList('normal2cut', 'radius', value)
        self.alignLeft()
    def n2cMetacarp3_toggled(self, value):
        self._toggleInList('normal2cut', 'metacarp3', value)
        self.alignLeft()
    def n2cPinsLunate_toggled(self, value):
        self._toggleInList('byPins_normal2cut', 'lunate', value)
        self.alignLeft()
    def n2cPinsScaphoid_toggled(self, value):
        self._toggleInList('byPins_normal2cut', 'scaphoid', value)
        self.alignLeft()
    def n2cPinsRadius_toggled(self, value):
        self._toggleInList('byPins_normal2cut', 'radius', value)
        self.alignLeft()
    def n2cPinsMetacarp3_toggled(self, value):
        self._toggleInList('byPins_normal2cut', 'metacarp3', value)
        self.alignLeft()

    def n2sLunate_toggled(self, value):
        self._toggleInList('normal2scaffold', 'lunate', value)
        self.alignRight()
    def n2sScaphoid_toggled(self, value):
        self._toggleInList('normal2scaffold', 'scaphoid', value)
        self.alignRight()
    def n2sRadius_toggled(self, value):
        self._toggleInList('normal2scaffold', 'radius', value)
        self.alignRight()
    def n2sMetacarp3_toggled(self, value):
        self._toggleInList('normal2scaffold', 'metacarp3', value)
        self.alignRight()
    def n2sPinsLunate_toggled(self, value):
        self._toggleInList('byPins_normal2scaffold', 'lunate', value)
        self.alignRight()
    def n2sPinsScaphoid_toggled(self, value):
        self._toggleInList('byPins_normal2scaffold', 'scaphoid', value)
        self.alignRight()
    def n2sPinsRadius_toggled(self, value):
        self._toggleInList('byPins_normal2scaffold', 'radius', value)
        self.alignRight()
    def n2sPinsMetacarp3_toggled(self, value):
        self._toggleInList('byPins_normal2scaffold', 'metacarp3', value)
        self.alignRight()
        
    def setBoneToggles(self):
        if 'lunate' in self.alignmentMethod[self.modelInfo['experimentID']]['normal2cut']:
            self.n2cLunate.setChecked(True)
        if 'scaphoid' in self.alignmentMethod[self.modelInfo['experimentID']]['normal2cut']:
            self.n2cScaphoid.setChecked(True)
        if 'radius' in self.alignmentMethod[self.modelInfo['experimentID']]['normal2cut']:
            self.n2cRadius.setChecked(True)
        if 'metacarp3' in self.alignmentMethod[self.modelInfo['experimentID']]['normal2cut']:
            self.n2cMetacarp3.setChecked(True)
        if 'lunate' in self.alignmentMethod[self.modelInfo['experimentID']]['normal2scaffold']:
            self.n2sLunate.setChecked(True)
        if 'scaphoid' in self.alignmentMethod[self.modelInfo['experimentID']]['normal2scaffold']:
            self.n2sScaphoid.setChecked(True)
        if 'radius' in self.alignmentMethod[self.modelInfo['experimentID']]['normal2scaffold']:
            self.n2sRadius.setChecked(True)
        if 'metacarp3' in self.alignmentMethod[self.modelInfo['experimentID']]['normal2scaffold']:
            self.n2sMetacarp3.setChecked(True)
        if 'lunate' in self.alignmentMethod[self.modelInfo['experimentID']]['byPins_normal2cut']:
            self.n2cPinsLunate.setChecked(True)
        if 'scaphoid' in self.alignmentMethod[self.modelInfo['experimentID']]['byPins_normal2cut']:
            self.n2cPinsScaphoid.setChecked(True)
        if 'radius' in self.alignmentMethod[self.modelInfo['experimentID']]['byPins_normal2cut']:
            self.n2cPinsRadius.setChecked(True)
        if 'metacarp3' in self.alignmentMethod[self.modelInfo['experimentID']]['byPins_normal2cut']:
            self.n2cPinsMetacarp3.setChecked(True)
        if 'lunate' in self.alignmentMethod[self.modelInfo['experimentID']]['byPins_normal2scaffold']:
            self.n2sPinsLunate.setChecked(True)
        if 'scaphoid' in self.alignmentMethod[self.modelInfo['experimentID']]['byPins_normal2scaffold']:
            self.n2sPinsScaphoid.setChecked(True)
        if 'radius' in self.alignmentMethod[self.modelInfo['experimentID']]['byPins_normal2scaffold']:
            self.n2sPinsRadius.setChecked(True)
        if 'metacarp3' in self.alignmentMethod[self.modelInfo['experimentID']]['byPins_normal2scaffold']:
            self.n2sPinsMetacarp3.setChecked(True)

    def sliderX_value_changed(self, value):
        self.spinBox[0].setValue(value)
        self.translationOffset[0] = value
        self.align()
    def spinboxX_value_changed(self, value):
        #self.slider[0].setValue(value)
        self.translationOffset[0] = value
        self.align()

    def sliderY_value_changed(self, value):
        self.spinBox[1].setValue(value)
        self.translationOffset[1] = value
        self.align()
    def spinboxY_value_changed(self, value):
        #self.slider[1].setValue(value)
        self.translationOffset[1] = value
        self.align()

    def sliderZ_value_changed(self, value):
        self.spinBox[2].setValue(value)
        self.translationOffset[2] = value
        self.align()
    def spinboxZ_value_changed(self, value):
        #self.slider[2].setValue(value)
        self.translationOffset[2] = value
        self.align()

    def spinboxXr_value_changed(self, value):
        self.rotationOffset[0] = np.deg2rad(value)
        self.align()
    def spinboxYr_value_changed(self, value):
        self.rotationOffset[1] = np.deg2rad(value)
        self.align()
    def spinboxZr_value_changed(self, value):
        self.rotationOffset[2] = np.deg2rad(value)
        self.align()

    def setOffset(self, x, y, z, rx, ry, rz):
        self.translationOffset[0] = x
        self.translationOffset[1] = y
        self.translationOffset[2] = z
        self.rotationOffset[0] = np.deg2rad(rx)
        self.rotationOffset[1] = np.deg2rad(ry)
        self.rotationOffset[2] = np.deg2rad(rz)

    def add_sphere(self):
        """ add a sphere to the pyqt frame """
        sphere = pv.Sphere()
        self.plotter[0, 0].add_mesh(sphere, show_edges=True)
        self.plotter[0, 0].reset_camera()

    def setFlip(self, flip):
        self.flip = flip

    def setAlignmentMethod(self, alignmentMethod):
        self.alignmentMethod = alignmentMethod
        self.setBoneToggles()

    def setPossiblePinSet(self, possiblePinSet):
        self.possiblePinSet = possiblePinSet

    def setMapMarkerChangesN2C(self, mapMarkerChanges):
        self.mapMarkerChangesN2['cut'] = mapMarkerChanges

    def setMapMarkerChangesN2S(self, mapMarkerChanges):
        self.mapMarkerChangesN2['scaffold'] = mapMarkerChanges

    def setConfigurations(self, flip, possiblePinSet, alignmentMethod, mapMarkerChangesN2C, mapMarkerChangesN2S):
        self.setMapMarkerChangesN2C(mapMarkerChangesN2C)
        self.setMapMarkerChangesN2S(mapMarkerChangesN2S)
        self.setFlip(flip)
        self.setPossiblePinSet(possiblePinSet)
        self.setAlignmentMethod(alignmentMethod)

    def swapMarker(self, mapMarkerChanges, inputPoints, boneName, model2):

        currentModel = model2
        plateAssignment = self.modelInfo['plateAssignment_' + currentModel]
        boneModelName = self.modelInfo['names'][boneName]
        points = np.array([
                inputPoints[plateAssignment[boneModelName][0]],
                inputPoints[plateAssignment[boneModelName][1]],
                inputPoints[plateAssignment[boneModelName][2]] # middle marker
                ])
        pinsSets = fnam.generatePinsFromMarkerPoints(points)
        pinsSets = pinsSets[self.possiblePinSet[self.modelInfo['experimentID']][currentModel][boneName]]
        pointsCut = np.empty((3, 3))
        if boneName in mapMarkerChanges[self.modelInfo['experimentID']]:
            mapMarkerChanges = mapMarkerChanges[self.modelInfo['experimentID']][boneName]

            if mapMarkerChanges == 0:
                pointsCut[:2, :] = pinsSets[0, 0:2, :]
                pointsCut[2, :] = pinsSets[0, 2, :]
            if mapMarkerChanges == 1:
                pointsCut[:2, :] = pinsSets[0, 2:4, :]
                pointsCut[2, :] = pinsSets[0, 0, :]
            if mapMarkerChanges == 2:
                pointsCut[:2, :] = pinsSets[1, 0:2, :]
                pointsCut[2, :] = pinsSets[1, 2, :]
            if mapMarkerChanges == 3:
                pointsCut[:2, :] = pinsSets[1, 2:4, :]
                pointsCut[2, :] = pinsSets[1, 0, :]

            markerSetsNearNormal = fnam.generateMarkerPointsFromPins(pointsCut)[0]
        else:
            markerSetsNearNormal = points
        return markerSetsNearNormal

    def alignSide(self, currentModel):
        framePoints = self.modelCache['markers']

        initTransMarkers = self.modelCache['initialTmatRadiusMarker']
        #transfromAdjustedNormal = fn.getOptimizationResultsAsTransMat(modelInfo, '_SLILGap')
        transfromAdjustedNormal = fm.createTransformationMatFromPosAndEuler(
                self.translationOffset[0],
                self.translationOffset[1],
                self.translationOffset[2],
                self.rotationOffset[0],
                self.rotationOffset[1],
                self.rotationOffset[2],
            )

        indPlot = 0
        if 'scaffold' in currentModel:
            indPlot = 1
        
        framePointsSwapped = deepcopy(framePoints[currentModel])
        if self.modelInfo['experimentID'] in self.mapMarkerChangesN2[currentModel]:
            for boneName in self.mapMarkerChangesN2[currentModel][self.modelInfo['experimentID']]:
                aligneedPointsTemp = self.swapMarker(self.mapMarkerChangesN2[currentModel], framePointsSwapped, boneName, currentModel)
                plateAssignment = self.modelInfo['plateAssignment_' + currentModel]
                boneModelName = self.modelInfo['names'][boneName]
                framePointsSwapped[plateAssignment[boneModelName][0]] = aligneedPointsTemp[0]
                framePointsSwapped[plateAssignment[boneModelName][1]] = aligneedPointsTemp[1]
                framePointsSwapped[plateAssignment[boneModelName][2]] = aligneedPointsTemp[2]
        
        if self.modelInfo['experimentID'] in self.alignmentMethod:
            for boneName in self.alignmentMethod[self.modelInfo['experimentID']]['pinDepth_normal2'+currentModel]:
                vecTowardBone = fnam.getPinVector(self, self.mapMarkerChangesN2[currentModel], framePointsSwapped, boneName, currentModel)
                depth = self.alignmentMethod[self.modelInfo['experimentID']]['pinDepth_normal2'+currentModel][boneName]
                
                plateAssignment = self.modelInfo['plateAssignment_' + currentModel]
                boneModelName = self.modelInfo['names'][boneName]
                framePointsSwapped[plateAssignment[boneModelName][0]] = framePointsSwapped[plateAssignment[boneModelName][0]] + vecTowardBone * depth
                framePointsSwapped[plateAssignment[boneModelName][1]] = framePointsSwapped[plateAssignment[boneModelName][1]] + vecTowardBone * depth
                framePointsSwapped[plateAssignment[boneModelName][2]] = framePointsSwapped[plateAssignment[boneModelName][2]] + vecTowardBone * depth

        self.normalPoints, self.alignedPoints[currentModel] = fnam.align(self, indPlot, self.modelInfo, 'normal', currentModel,
            framePoints, transfromAdjustedNormal, initTransMarkers, self.alignmentMethod,
            self.flip, framePointsSwapped)

        # show any swapped points
        if self.modelInfo['experimentID'] in self.mapMarkerChangesN2[currentModel]:
            for boneName in self.mapMarkerChangesN2[currentModel][self.modelInfo['experimentID']]:
                aligneedPointsTemp = self.swapMarker(self.mapMarkerChangesN2[currentModel], self.alignedPoints[currentModel], boneName, currentModel)
                self.plotter[0, indPlot].add_points(
                    aligneedPointsTemp,
                    name = 'aligneedPoints' + currentModel + 'Temp_' + boneName,
                    color = 'yellow',
                    render_points_as_spheres=True,
                    point_size=15.0,
                    opacity = 0.9
                )
        
        visualizeMarkerPins(self, self.possiblePinSet, 'normal', indPlot, self.modelInfo, self.normalPoints,
            color = 'red')
        visualizeMarkerPins(self, self.possiblePinSet, currentModel, indPlot, self.modelInfo, self.alignedPoints[currentModel],
            color = 'blue')

    def alignLeft(self):
        self.alignSide('cut')

    def alignRight(self):
        self.alignSide('scaffold')

    def align(self):
        self.alignLeft()
        self.alignRight()

    def setMarkersInModels(self, modelsToSetMarkersIn):
        modelInfo = self.modelInfo
        modelCache = self.modelCache
        for modelType in modelsToSetMarkersIn:
            markersFinal = self.markersFinal[modelType]

            # set model
            # don't change bone positions, instead change marker locations
            # this will make the zero angles correct across model types (normal, cut, scaffold)
            modelInfo['currentModel'] = modelType
            coms = {
                modelInfo['names']['lunate']: modelCache['lunateCOM'],
                modelInfo['names']['scaphoid']: modelCache['scaphoidCOM'],
                modelInfo['names']['radius']: modelCache['radiusCOM'],
                modelInfo['names']['metacarp3']: modelCache['metacarp3COM'],
            }
            fn.setMarkersInModel(modelInfo, coms, markersFinal)

    def addGuidePins(self):
        pins = self.modelCacheExtra['sensorGuidePins']
        for pin in pins:
            self.plotter[0, 0].add_lines(
                lines = np.array([pins[pin]['point1'],pins[pin]['point2']]),
                color = 'black',
                width = 2.0,
                name = 'sensorGuidePins_' + pin
            )
            self.plotter[0, 1].add_lines(
                lines = np.array([pins[pin]['point1'],pins[pin]['point2']]),
                color = 'black',
                width = 2.0,
                name = 'sensorGuidePins_' + pin
            )

    def loadScene(self):
        names = fn.boneModelNames(self.modelInfo) #[lunateName, scaphoidName, radiusName, metacarp3Name] =
        names = names + self.modelInfo['otherHandBones']
        self.boneGeometry = {}
        for name in names:
            self.boneGeometry[name] = fn.getGeometry(self.modelInfo, name)
        self.bonePolyData = [{}, {}]
        self.createSurfaces(0)
        self.createSurfaces(1)

        self.addMeshes(0)
        self.addMeshes(1)
        self.addGuidePins()
        self.viewScene(0)
        self.viewScene(1)

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
                specular_power=10
            )
        if hasattr(self.modelCacheExtra,'sensorGuidePins'):
            lines = self.modelCacheExtra['sensorGuidePins']
            for line in lines:
                l = self.modelCacheExtra['sensorGuidePins'][line]
                mesh = pv.Line(l['point1'], l['point2'])
                self.actors[plotInd][line] = self.plotter[0, plotInd].add_mesh(
                    mesh,
                    name = line,
                    color='k',
                    line_width=3,
                    calar_bar_args = None
                    )

    def addPoints(self, plotInd, points, name = 'tempMarkerSpheres_static', color = 'grey'):
        self.actors[plotInd][name] = self.plotter[0, plotInd].add_points(
            points,
            name = name,
            color = color,
            render_points_as_spheres=True,
            point_size=20.0,
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
        self.plotter[0, plotInd].show_axes_all()
        self.plotter[0, plotInd].view_vector([0, 0, -1])
        self.plotter[0, plotInd].set_viewup([-1, 0, 0])
        #self.plotter[0, plotInd].set_position((1.0, 1.0, 1.0))
        self.plotter.show()