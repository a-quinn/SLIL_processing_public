# Author: Alastair Quinn 2023

import os
os.environ["QT_API"] = "pyqt5"
import numpy as np
import slil.common.data_configs as dc
import slil.common.math as fm
from copy import deepcopy
from slil.process.qt_plotter_multi import MultiScene
import alphashape
import pyvista as pv
from functions import *
from points_picker import enable_points_picking
from slil.common.cache import loadCache, saveCache
import bone_transformer
from slil.process.functions import getModelCache
from angle_tool import AngleTool
from distance_tool import DistanceTool

default_radiusCoords = [
    {
    'name': '11524',
    'sigmoidNotch': 1401,
    'radialStyloid': 878
    },
    {
    'name': '11525',
    'sigmoidNotch': 952,
    'radialStyloid': 1386
    },
    {
    'name': '11526',
    'sigmoidNotch': 808,
    'radialStyloid': 1345
    },
    {
    'name': '11527',
    'sigmoidNotch': 677,
    'radialStyloid': 1106
    },
    {
    'name': '11534',
    'sigmoidNotch': 6729,
    'radialStyloid': 11405
    },
    {
    'name': '11535',
    'sigmoidNotch': 5688,
    'radialStyloid': 14839
    },
    {
    'name': '11536',
    'sigmoidNotch': 8237,
    'radialStyloid': 16421
    },
    {
    'name': '11537',
    'sigmoidNotch': 7831,
    'radialStyloid': 14107
    },
    {
    'name': '11538',
    'sigmoidNotch': 10904,
    'radialStyloid': 20231
    },
    {
    'name': '11539',
    'sigmoidNotch': 7721,
    'radialStyloid': 14377
    },
]

experiments =[
    '11524',
    '11525',
    '11526',
    '11527',
    '11534',
    '11535',
    '11536',
    '11537',
    '11538',
    '11539'
]

def findCenterLineExtra(scene, cutPlanePoints, show = False):
    # rotates the points so taht the normal of the cut plane is vertical
    # finds center line
    # rotate back to original frame

    planeNormal, planeCenter = fitPlaneNormal(cutPlanePoints)

    # make vector point away from carpals
    lunatePolyData = scene.bonePolyData[scene.plotL][scene.modelInfo['names']['lunate']]
    lunateCOM = fm.calcCOM(lunatePolyData.points, [], method='points')
    if fm.calcDist(planeCenter + planeNormal, lunateCOM) < fm.calcDist(planeCenter - planeNormal, lunateCOM):
        planeNormal *= -1.0

    if show:
        ve = np.array([planeNormal * 3 + planeCenter, planeCenter])
        scene.plotter[0, scene.plotL].add_points(
            points = ve,
            name='p_22',
            render_points_as_spheres=True,
            point_size=15.0,
            color='red',
            pickable=False)

    rX = fm.normalizeVector(planeNormal)
    rZ = np.array([0.0, 0.0, 1.0])
    rY = fm.normalizeVector(np.cross(rX, rZ)) # TODO: Might need to make sure this is pointing volarly
    rZ = fm.normalizeVector(np.cross(rY, rX))

    # make sure it's similar to global
    if fm.calcDist(rY, [0,1,0]) > fm.calcDist(rY * -1.0, [0,1,0]):
        rY *= -1.0
    if fm.calcDist(rZ, [0,0,1]) > fm.calcDist(rZ * -1.0, [0,0,1]):
        rZ *= -1.0

    if show:
        center = planeCenter
        mag = 3
        coords = np.array([
            (rX * mag) + center,
            (rY * mag) + center,
            (rZ * mag) + center
        ])
        scene.plotter[0, scene.plotL].add_points(
                                points=coords,
                                name='radius_coords222',
                                render_points_as_spheres=True,
                                point_size=10.0,
                                color='red',
                                pickable = False)

    rotMat = np.array([rX, rY, rZ]).T
    trans = np.eye(4)
    trans[:3,:3] = rotMat
    trans = fm.inverseTransformationMat(trans) # rotate opposite direction

    radiusPolyData = scene.bonePolyData[scene.plotL][scene.modelInfo['names']['radius']]
    pointsTransformed = fm.transformPoints_1(radiusPolyData.points, trans)

    # remove some top points to stop scewed line
    x_range = abs(max(pointsTransformed[:, 0]) - min(pointsTransformed[:, 0]))
    remove_percentage = 0.55
    x_cutoff = min(pointsTransformed[:, 0]) + x_range * remove_percentage
    pointsTransformed = np.array([point for point in pointsTransformed if point[0] >= x_cutoff])

    if show:
        scene.plotter[0, scene.plotL].add_points(
                                points=pointsTransformed,
                                name='pointsTransformed222',
                                render_points_as_spheres=True,
                                point_size=5.0,
                                color='red',
                                pickable = False)

    centerLine = findLine(pointsTransformed)

    centerLines = points_to_line_segments(centerLine)

    if show:
        scene.plotter[0, scene.plotL].add_lines(
                                lines = centerLines,
                                name='centerLines222',
                                width=3.0,
                                color='green')


    trans = fm.inverseTransformationMat(trans) # back to original frame
    centerLineTransformed = fm.transformPoints_1(centerLine, trans)
    
    if show:
        scene.plotter[0, scene.plotL].add_lines(
                                lines = centerLineTransformed,
                                name='centerLineTransformed222',
                                width=3.0,
                                color='red')
    return centerLineTransformed

def generateRI(scene, radiusCoord, showLine=False):
    """
    Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7172869/pdf/bio-19-1204_064501.pdf
    RI: radius intersection, the projection of RC on distal
    articular surface of radius.
    """
    radiusPolyData = scene.bonePolyData[scene.plotL][scene.modelInfo['names']['radius']]
    
    if not 'cutPoints' in radiusCoord:
        print('No cutPoints for this model, using global axis to find center line.')
        centerLine = findLine(radiusPolyData.points)
    else:
        print('Found cutPoints for this model, using to find center line.')
        cutPlanePoints = radiusPolyData.points[radiusCoord['cutPoints']]
        centerLine = findCenterLineExtra(scene, cutPlanePoints, show=False)

    if showLine:
        centerLines = points_to_line_segments(centerLine)
        #print(centerLines)
        scene.plotter[0, scene.plotL].add_lines(
                                lines = centerLines,
                                name='radiusCenterLine',
                                width=3.0,
                                color='green')

    direction = fm.normalizeVector(centerLine[0] - centerLine[-1])
    origin = centerLine[-1]
    #points, _ = radiusPolyData.ray_trace(origin, direction)
    points, _, _ = radiusPolyData.multi_ray_trace([origin], [direction])

    metacarpa3 = scene.bonePolyData[scene.plotL][scene.modelInfo['names']['metacarp3']]

    p, ind = find_closest_point(points, metacarpa3.center_of_mass())
    return p, np.delete(points, ind, axis=0)[0]

class GUI():
    def __init__(self, experiment = experiments[0]):
        
        self._setup_data_pre(experiment)

        user_menu_experiments = self.scene.plotter[0, self.scene.plotL].main_menu.addMenu('Load Experiment')
        self.experiment_funcs = {}
        funcStr = """
def exp_EXP_ID(self):
    self._load_experiment('EXP_ID')
GUI.exp_EXP_ID = exp_EXP_ID
self.experiment_funcs['EXP_ID'] = self.exp_EXP_ID
        """
        for experiment_add in experiments:
            exec(funcStr.replace('EXP_ID', experiment_add) )
            user_menu_experiments.addAction(experiment_add, self.experiment_funcs[experiment_add])


        self.bonePointColours = [
            'blue',
            'green',
            'red',
            'brown',
            'purple'
        ]
        
        leftPlot_plane_normal = np.array([0.5,0.5,0.5])
        leftPlot_plane_origin = np.array([0.0,0.0,0.0])
        self.planeDownward = np.array([1.0, 0.0, 0.0])
        self.contoursEnabled = False

        plotRight = self.scene.plotter[0, self.scene.plotR]
        self.scene.plotter._layout.addWidget(plotRight.app_window, 0, 1, 2, 1)

        def toggle2DContour(flag):
            self.contoursEnabled = flag
            if flag:
                self.updateChart()
            else:
                for i in range(len(self.boneList)):
                    a = self.scene.plotter[0, self.scene.plotR].add_lines(
                        lines = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0]]),
                        name='planeLines_'+(self.boneList[i]),
                        width=3.0,
                        color=self.bonePointColours[i])
                    a.VisibilityOff()
        self.viewPALateral = False
        def togglePALateral(flag):
            self.viewPALateral = flag
            self.updateRadiusCoord()

        self.scene.plotter[0, self.scene.plotL].add_plane_widget(
            self.updatePlane,
            normal=leftPlot_plane_normal)
        
        self.angle_tool = AngleTool(self.scene)
        self.distance_tool = DistanceTool(self.scene, start_active=False)

        self.scene.plotter[0, self.scene.plotR].add_text(
            f"ID: {experiment}", position = 'upper_left',
            color = 'black', name = 'experimentID')

        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QToolBar, QAction, QGridLayout, QGroupBox, QCheckBox, QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QLabel

        main_window = [
            self.scene.plotter[0, 0].app_window,
            self.scene.plotter[0, 1].app_window
            ]
        
        for window in main_window:
            toolbars = window.findChildren(QToolBar)
            for toolbar in toolbars:
                window.removeToolBar(toolbar)

        toolbar = [
            main_window[0].addToolBar('User Toolbar'),
            main_window[1].addToolBar('User Toolbar')
        ]


        self.trials_combo_box = QComboBox()
        self.trials_combo_box.currentTextChanged.connect(self.changeTrial)

        toolbar[0].addWidget(self.trials_combo_box)

        self.bottomLeftGroupBox = QGroupBox("Trial Data")
        self.timeSlider = QSlider(Qt.Orientation.Horizontal, self.bottomLeftGroupBox)
        self.timeSlider.orientation()
        self.timeSlider.setMinimum(0)
        self.timeSlider.setMaximum(2000)
        self.timeSlider.valueChanged.connect(self.changeTrialTimeSilder)
        #slider.actionTriggered.connect(changeTime)

        self.timeFrameBox = QSpinBox(self.bottomLeftGroupBox)
        self.timeFrameBox.setMinimum(0)
        self.timeFrameBox.setMaximum(2000)
        self.timeFrameBox.setValue(0)
        self.timeFrameBox.valueChanged.connect(self.changeTrialTimeFrameBox)
        styleLabel = QLabel("Frame:")
        styleLabel.setBuddy(self.timeFrameBox)

        self.timeSecondsBox = QDoubleSpinBox(self.bottomLeftGroupBox)
        self.timeSecondsBox.setDecimals(3)
        self.timeSecondsBox.setMinimum(0)
        self.timeSecondsBox.setMaximum(2000)
        self.timeSecondsBox.setValue(0)
        self.timeSecondsBox.valueChanged.connect(self.changeTrialTimeSecondsBox)
        styleLabel = QLabel("Time:")
        styleLabel.setBuddy(self.timeSecondsBox)

        import pyqtgraph as pg
        plot = pg.PlotWidget()
        self.scatterPlot = pg.ScatterPlotItem(pen=None, symbol='o')
        plot.addItem(self.scatterPlot)

        layout = QGridLayout()
        layout.addWidget(self.timeSlider, 0, 0, 1, 2)
        layout.addWidget(self.timeFrameBox, 1, 0, 1, 1)
        layout.addWidget(self.timeSecondsBox, 1, 1, 1, 1)
        layout.addWidget(plot, 2, 0, 1, 2)
        self.bottomLeftGroupBox.setLayout(layout)

        #plotLeft = self.scene.plotter[0, self.scene.plotL]
        #self.scene.plotter._layout.addWidget(plotLeft.app_window, 0, 0, 2, 1)
        
        self.scene.plotter._layout.addWidget(self.bottomLeftGroupBox, 1, 0, 1, 1)
        self.scene.plotter._layout.setRowStretch(0,3)
        self.scene.plotter._layout.setRowStretch(1,1)

        def add_button(key, method, sceneID):
            action = QAction(key, main_window[sceneID])
            action.triggered.connect(method)
            toolbar[sceneID].addAction(action)
            return
        def add_toggle(key, method, sceneID):
            #toggle = QCheckBox(key, main_window[sceneID])
            #toggle.toggled.connect(method)
            #toolbar[sceneID].addWidget(toggle)
            
            action = QAction(key, main_window[sceneID])
            action.setCheckable(True)
            action.toggled.connect(method)
            toolbar[sceneID].addAction(action)
            return

        add_button(
            'Default Pose',
            self.defaultPose,
            self.scene.plotL)
        add_button(
            'Auto Align',
            self.updateRadiusCoord,
            self.scene.plotL)
        add_toggle(
            'Toggle Point/Points Select',
            self.pointOrPointsPicking,
            self.scene.plotL)
        add_button(
            'Set Cut Points',
            self.setCutPoints,
            self.scene.plotL)
        add_button(
            'Save Cache',
            self.saveCache,
            self.scene.plotL)
        
        add_toggle(
            'Toggle PA/Lateral',
            togglePALateral,
            self.scene.plotR)
        add_toggle(
            'Toggle Contour',
            toggle2DContour,
            self.scene.plotR)
        add_button(
            'Reset View',
            self.setRightView,
            self.scene.plotR)
        add_button(
            'Auto Place Lines',
            self.autoplaceLines,
            self.scene.plotR)
        add_button(
            'Save Angles',
            self.saveAngle,
            self.scene.plotR)
        
        def toggleDistanceAngleTools(flag):
            if flag:
                self.angle_tool.removeTool()
                self.distance_tool.addTool()
            else:
                self.distance_tool.removeTool()
                self.angle_tool.addTool()

        add_toggle(
            'Toggle Distance/Angle Tools',
            toggleDistanceAngleTools,
            self.scene.plotR)
        add_button(
            'Save Distances',
            self.saveDistances,
            self.scene.plotR)
        
        self._setup_data_post()
    
    def _setup_data_pre(self, experiment):
        modelInfo = dc.load(experiment)
        # only needed if missing some geometry (e.g. capitate)
        #mi.open_project(modelInfo['3_matic_file'])
        print(f"Loading experiment: {experiment}")
        if not hasattr(self, 'scene'):
            self.scene = MultiScene(modelInfo)
        else:
            self.scene._reset()
            self.scene._setup(modelInfo)
        self.scene.loadScene(self.scene.plotL)

        self.scene.plotter[0, self.scene.plotR].add_text(
            f"ID: {experiment}", position = 'upper_left',
            color = 'black', name = 'experimentID')
        
        extrasToRemove = ['lunate', 'scaphoid']
        for removeBone in extrasToRemove:
            name = self.scene.modelInfo['names'][removeBone] + '_moved'
            self.scene.plotter[0, self.scene.plotL].remove_actor(name)

        self.scene.plotter[0, self.scene.plotL].show_axes()
        self.scene.plotter[0, self.scene.plotL].view_vector([0, 1, 0])
        self.scene.plotter[0, self.scene.plotL].set_viewup([0, 0, -1])
        self.scene.plotter.show()
        self.scene.viewScene(self.scene.plotR)

        self.radiusCoords = default_radiusCoords
        cache = loadCache('points_picker_radiusCoords')
        if cache:
            self.radiusCoords = cache

        boneNames = deepcopy(self.scene.modelInfo['names'])
        del boneNames['boneplugLunate']
        del boneNames['boneplugScaphoid']
        self.points_2d = {}
        self.boneList = list(boneNames.values()) + self.scene.modelInfo['otherHandBones']

    def _setup_data_post(self):
        self.angle_tool.setupScene()
        self.distance_tool.setupScene()

        self.bt = bone_transformer.BoneTransformer(self, self.scene.modelInfo['experimentID'])

        trials = deepcopy(self.scene.modelInfo['trialsRawData_only_normal'])
        trials += self.scene.modelInfo['trialsRawData_only_cut']
        trials += self.scene.modelInfo['trialsRawData_only_scaffold']
        self.trials_combo_box.clear()
        for trial in trials:
            self.trials_combo_box.addItem(trial)
        
        self.updateChart()
        self.setRightView()
        

    def _load_experiment(self, experiment):
        self._setup_data_pre(experiment)
        self._setup_data_post()
        
    def changeTrial(self, trial):
        if len(trial) > 0:
            print(f"Loading trial: {trial}")
            self.currentTrial = trial
            self.trialSampleRate = 250
            self.bt.updateTrial(self.currentTrial)
            currentFrame = 0
            self.bt.updateFrame(currentFrame)
            maxFrames = len(self.bt.transMats['scaphoid'])
            self.timeSlider.setValue(currentFrame)
            self.timeSlider.setMaximum(maxFrames)
            self.timeFrameBox.setValue(currentFrame)
            self.timeFrameBox.setMaximum(maxFrames)
            self.timeSecondsBox.setValue(currentFrame)
            self.timeSecondsBox.setMaximum(maxFrames * (1/self.trialSampleRate))
            
            x, y = self.loadTrialRaw(self.currentTrial)
            self.scatterPlot.setData(x, y)
            self.scatterPlot.getViewWidget().autoRange()
            self.updateRadiusCoord()
            
            #boneSetInd = [i for i, x in enumerate(self.radiusCoords) if x['name'] == self.scene.modelInfo['experimentID']]
            #if len(boneSetInd) != 0:
            #    self.updateRadiusCoord()
        
    def changeTrialTime(self, pos):
        #print(f"Time: {pos}")
        self.bt.updateFrame(pos)

    def changeTrialTimeSilder(self, pos):
        self.timeFrameBox.setValue(pos)
        self.timeSecondsBox.setValue(pos * (1/self.trialSampleRate))
        self.changeTrialTime(pos)

    def changeTrialTimeFrameBox(self, pos):
        self.timeSlider.setValue(pos)
        self.timeSecondsBox.setValue(pos * (1/self.trialSampleRate))
        self.changeTrialTime(pos)

    def changeTrialTimeSecondsBox(self, pos):
        pos = int(pos / (1/self.trialSampleRate))
        self.timeSlider.setValue(pos)
        self.timeFrameBox.setValue(pos)
        self.changeTrialTime(pos)
    
    def defaultPose(self):
        if self.bt:
            self.bt.defaultPose()

    def loadTrialRaw(self, trial):
        from slil.common.plotting_functions import loadTest
        trialRaw = loadTest(self.scene.modelInfo['dataInputDir'] + trial, skiprows = 0)
        if 'ur' in self.currentTrial:
            y = trialRaw['feedback/rot/x'].to_numpy() *(180.0/np.pi)
            # could be if left hand
            if not self.scene.modelInfo['isLeftHand']:
                y *= -1.0
        else:
            y = trialRaw['feedback/rot/y'].to_numpy() *(180.0/np.pi)

        return trialRaw['time'].to_numpy(), y
        
    def saveCache(self):
        print('Saving Cache')
        saveCache(self.radiusCoords, 'points_picker_radiusCoords')
    
    def setCutPoints(self):
        selected_points = self.scene.plotter[0, self.scene.plotL].picked_points[0]
        boneSetInd = [i for i, x in enumerate(self.radiusCoords) if x['name'] == self.scene.modelInfo['experimentID']]
        self.radiusCoords[boneSetInd[0]].update({'cutPoints': selected_points})

    def pointOrPointsPicking(self, flag):
        print(flag)
        self.scene.plotter[0, self.scene.plotL].disable_picking()
        if flag:
            self.scene.plotter[0, self.scene.plotL].enable_point_picking(callback=self.pickedPoint, use_mesh=True)
        else:
            enable_points_picking(
                self = self.scene.plotter[0, self.scene.plotL],
                callback=self.pickedPoints
            )
    
    def updatePlane(self, plane_normal, plane_origin):
        global leftPlot_plane_normal, leftPlot_plane_origin
        leftPlot_plane_normal = np.array(plane_normal)
        leftPlot_plane_origin = np.array(plane_origin)
        
        leftPlot_plane_normal = fm.normalizeVector(leftPlot_plane_normal)
        right_vector = np.cross(leftPlot_plane_normal, [0,0,1])
        self.planeDownward = np.cross(leftPlot_plane_normal, right_vector)
        self.planeDownward = np.cross(leftPlot_plane_normal, self.planeDownward) * -1.0

        self.updateChart()
        self.setRightView()
        return
    
    def autoplaceLines(self):
        for boneName in ['lunate', 'scaphoid']:
            slope, y_intercept = linear_regression(self.points_2d[self.scene.modelInfo['names'][boneName]])
            min_x = np.min(self.points_2d[self.scene.modelInfo['names'][boneName]], axis=0)[0]
            max_x = np.max(self.points_2d[self.scene.modelInfo['names'][boneName]], axis=0)[0]
            self.angle_tool.line_widget[boneName].SetPoint1(min_x, slope*min_x + y_intercept, 0)
            self.angle_tool.line_widget[boneName].SetPoint2(max_x, slope*max_x + y_intercept, 0)
    
    def saveAngle(self):
        self.angle_tool.saveAngles(self.currentTrial)

    def saveDistances(self):
        self.distance_tool.saveDistances(self.currentTrial)

    def updateRadiusCoord(self):
        boneSetInd = [i for i, x in enumerate(self.radiusCoords) if x['name'] == self.scene.modelInfo['experimentID']]
        if len(boneSetInd) == 0:
            print(f"Error, no point indexes set for experiment {self.scene.modelInfo['experimentID']}")
            return
        else:
            boneSetInd = boneSetInd[0]
        print("Updating...")

        radiusCoord = self.radiusCoords[boneSetInd]

        ri, rc = generateRI(self.scene, radiusCoord, True)
        radiusCoord.update({'RI': ri})
        radiusCoord.update({'RC': rc})

        radiusPolyData = self.scene.bonePolyData[self.scene.plotL][self.scene.modelInfo['names']['radius']]
        radiusCoord.update({'SN': radiusPolyData.points[radiusCoord['sigmoidNotch']]})
        radiusCoord.update({'RS': radiusPolyData.points[radiusCoord['radialStyloid']]})


        points = np.array([
            radiusCoord['RI'],
            radiusCoord['RC'],
            radiusCoord['RS'],
            radiusCoord['SN']
        ])
        self.scene.plotter[0, self.scene.plotL].add_points(
                                points=points,
                                name='radius_coord_points',
                                render_points_as_spheres=True,
                                point_size=10.0,
                                color='blue',
                                pickable = False)

        rY = fm.normalizeVector(radiusCoord['RC'] - radiusCoord['RI'])
        rZ = fm.normalizeVector(radiusCoord['SN'] - radiusCoord['RS'])
        rX = fm.normalizeVector(np.cross(rY, rZ)) # TODO: Might need to make sure this is pointing volarly
        rZ = fm.normalizeVector(np.cross(rY, rX))

        center = radiusCoord['RI']
        mag = 3
        coords = np.array([
            (rX * mag) + center,
            (rY * mag) + center,
            (rZ * mag) + center
        ])
        self.scene.plotter[0, self.scene.plotL].add_points(
                                points=coords,
                                name='radius_coords',
                                render_points_as_spheres=True,
                                point_size=10.0,
                                color='red',
                                pickable = False)

        plane = self.scene.plotter[0, self.scene.plotL].plane_widgets[0]
        if self.viewPALateral:
            plane.SetNormal(rX)
            plane.SetOrigin(radiusCoord['RI'])
            self.planeDownward = rZ * -1.0
        else:
            plane.SetNormal(rZ)
            plane.SetOrigin(radiusCoord['RI'])
            self.planeDownward = rX
        self.updatePlane(plane.GetNormal(), plane.GetOrigin())
        print("Finshed Updating.")
    
    def setRightView(self):
        #scene.plotter[0, scene.plotR].view_vector([0, 0, 1])
        #scene.plotter[0, scene.plotR].set_viewup([0, 1, 0])
        self.scene.plotter[0, self.scene.plotR].view_yx()
        return
    
    def updateChart(self):
        plane_normal = leftPlot_plane_normal
        plane_origin = leftPlot_plane_origin
        d = shortest_distance_to_origin(plane_origin, plane_normal)
        plane_model = np.append(np.array(plane_normal), d)
        
        for indBone, boneName in enumerate(self.boneList):
            bonePoints = np.array(self.scene.bonePolyData[self.scene.plotL][boneName].points)
            points = project_points_onto_plane3D(bonePoints, plane_model)
            
            # alternatively, this could be calculated based on all meshes in scene
            #bounds = np.array(scene.plotter[0, scene.plotL].bounds).reshape([2,3])
            #planeSize = np.sqrt(np.sum(np.power(np.diff(bounds, axis = 0), 2))) * 2
            #points = np.array(get_plane_corners(plane_normal, plane_origin, planeSize) )
            
            self.scene.plotter[0, self.scene.plotL].add_points(points,
                                name='planePoints_'+(self.boneList[indBone]),
                                render_points_as_spheres=True,
                                point_size=3.0,
                                color=self.bonePointColours[indBone],
                                pickable = False)
            
            #points_2d = convert_points_to_2d(bonePoints, plane_normal)
            self.points_2d[boneName] = project_points_onto_plane2D(bonePoints, plane_normal, plane_origin, self.planeDownward)

            #print(self.points_2d[boneName])

            self.scene.plotter[0, self.scene.plotR].add_points(
                                np.hstack((self.points_2d[boneName], np.zeros((self.points_2d[boneName].shape[0], 1)))),
                                name='planePoints_'+(self.boneList[indBone]),
                                render_points_as_spheres=True,
                                point_size=3.0,
                                color=self.bonePointColours[indBone])
        
        if self.contoursEnabled:
            for indBone, boneName in enumerate(self.boneList):

                alpha_shape = alphashape.alphashape(self.points_2d[boneName], alpha=0.9)
                outer_contour_points = np.array(alpha_shape.boundary.coords)
                
                outer_contour_points = points_to_line_segments(outer_contour_points)

                if (len(outer_contour_points) & 0x1): # if odd
                    outer_contour_points = outer_contour_points[:-1]
                line = np.hstack((outer_contour_points, np.zeros((outer_contour_points.shape[0], 1))))
                
                self.scene.plotter[0, self.scene.plotR].add_lines(
                                    lines = line,
                                    name='planeLines_'+(self.boneList[indBone]),
                                    width=3.0,
                                    color=self.bonePointColours[indBone], )
                
    
    
    def pickedPoint(self, polydata, vertID):
        self.scene.plotter[0, self.scene.plotL].add_text(
            f"Vertex ID: {vertID}", position = 'upper_right',
            color = 'black', name = 'vertID')
        return True

    def pickedPoints(self, picked_points):
        #self.scene.plotter[0, self.scene.plotL].add_text(
        # f"polydata: {picked_cells_multi}", position = 'upper_left',
        # color = 'black', name = 'polydata')
        if picked_points:
            radiusPolyData = self.scene.bonePolyData[self.scene.plotL][self.scene.modelInfo['names']['radius']]
            points = radiusPolyData.points[picked_points[0]]
            print(f"sets of picked_points: {len(picked_points)}")
            #picked_points = np.hstack(picked_points)
            self.scene.plotter[0, self.scene.plotL].add_points(
                points = points,
                name='pickedPoints',
                render_points_as_spheres=True,
                point_size=15.0,
                color='red',
                pickable=False)
            
            planeNormal, planeCenter = fitPlaneNormal(points)

            ve = np.array([planeNormal * 3 + planeCenter, planeCenter])
            self.scene.plotter[0, self.scene.plotL].add_points(
                points = ve,
                name='p_',
                render_points_as_spheres=True,
                point_size=15.0,
                color='red',
                pickable=False)
        else:
            self.scene.plotter[0, self.scene.plotL].remove_actor('pickedPoints')
            
        return True
