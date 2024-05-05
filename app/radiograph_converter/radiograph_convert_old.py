# Author: Alastair Quinn 2023

#%%
import os
os.environ["QT_API"] = "pyqt5"
import numpy as np
import slil.process.functions as fn
import slil.common.data_configs as dc
import slil.common.math as fm
from copy import deepcopy
from slil.process.qt_plotter import Scene
from slil.process.qt_plotter_multi import MultiScene
from slil.common.data_configs import outputFolders
import alphashape
import pyvista as pv
from functions import *

#'11525',
#'11526',
#'11527',
#'11534',
#'11535',
#'11536',
#'11537',
#'11538',
#'11539'
experiment = '11538'
modelInfo = dc.load(experiment)
# only needed if missing some geometry (e.g. capitate)
#mi.open_project(modelInfo['3_matic_file'])

#%%
scene = MultiScene(modelInfo)
scene.loadScene(scene.plotL)
#modelCache = fn.getModelCache(modelInfo)
scene.viewScene(scene.plotL)
scene.viewScene(scene.plotR)



boneNames = deepcopy(scene.modelInfo['names'])
del boneNames['boneplugLunate']
del boneNames['boneplugScaphoid']
boneList = list(boneNames.values()) + scene.modelInfo['otherHandBones']

allBonePoints = []
for boneName in boneList:
    allBonePoints.append(np.array(scene.bonePolyData[scene.plotL][boneName].points))
bonePointColours = [
    'blue',
    'green',
    'red',
    'brown',
    'purple'
]

chat2dEnabled = False
if chat2dEnabled:
    chart = pv.Chart2D()
    scene.plotter[0, scene.plotR].add_chart(chart)

    x = np.linspace(0, 2*np.pi, 20)
    y = np.sin(x)
    plotScatter = []
    plotLine = []
    for indBone in range(len(allBonePoints)):
        plotScatter.append(chart.scatter(x, y, color = bonePointColours[indBone], size=4))
        plotLine.append(chart.line(x, y, color = bonePointColours[indBone], width = 2))


chartMinX = 0.0
chartMaxX = 0.0
chartMinY = 0.0
chartMaxY = 0.0

def updateAxies():
    #Doesn't work for some reason.
    #xAxisMin = chart.x_range[0]
    #xAxisMax = chart.x_range[1]
    #yAxisMin = chart.y_range[0]
    #yAxisMax = chart.y_range[1]

    xAxisMin = chartMinX
    xAxisMax = chartMaxX
    yAxisMin = chartMinY
    yAxisMax = chartMaxY

    xMag = abs(xAxisMax - xAxisMin)
    yMag = abs(yAxisMax - yAxisMin)

    #print(f"{xAxisMin}, {xAxisMax}, {yAxisMin}, {yAxisMax}, {xMag}, {yMag}")
    
    if xMag > yMag:
        mag = xMag
    else:
        mag = yMag
    y_centre = yAxisMin + (mag/2)
    newMax = y_centre + mag/2
    newMin = y_centre - mag/2
    chart.y_range = [newMin, newMax]
    x_centre = xAxisMin + (mag/2)
    newMax = x_centre + mag/2
    newMin = x_centre - mag/2
    chart.x_range = [newMin, newMax]
 
leftPlot_plane_normal = np.array([0.5,0.5,0.5])
leftPlot_plane_origin = np.array([0.0,0.0,0.0])
planeDownward = np.array([1.0, 0.0, 0.0])
contour_detection_resolution = 1.0
contoursEnabled = False

def setRightView(temp=None):
    #scene.plotter[0, scene.plotR].view_vector([0, 0, 1])
    #scene.plotter[0, scene.plotR].set_viewup([0, 1, 0])
    scene.plotter[0, scene.plotR].view_yx()
    return

def updateChart():
    global chartMinX, chartMaxX, chartMinY, chartMaxY
    plane_normal = leftPlot_plane_normal
    plane_origin = leftPlot_plane_origin
    d = shortest_distance_to_origin(plane_origin, plane_normal)
    plane_model = np.append(np.array(plane_normal), d)
    i = 0
    points_2d_all = []
    
    if chat2dEnabled:
        chartMinXTemp = 1000.0
        chartMaxXTemp = -1000.0
        chartMinYTemp = 1000.0
        chartMaxYTemp = -1000.0

    for bonePoints in allBonePoints:
        points = project_points_onto_plane3D(bonePoints, plane_model)
        
        # alternatively, this could be calculated based on all meshes in scene
        #bounds = np.array(scene.plotter[0, scene.plotL].bounds).reshape([2,3])
        #planeSize = np.sqrt(np.sum(np.power(np.diff(bounds, axis = 0), 2))) * 2
        #points = np.array(get_plane_corners(plane_normal, plane_origin, planeSize) )
        
        scene.plotter[0, scene.plotL].add_points(points,
                             name='planePoints_'+(boneList[i]),
                             render_points_as_spheres=True,
                             point_size=3.0,
                             color=bonePointColours[i],
                             pickable = False)
        
        #points_2d = convert_points_to_2d(bonePoints, plane_normal)
        points_2d = project_points_onto_plane2D(bonePoints, plane_normal, plane_origin, planeDownward)

        #print(points_2d)
        points_2d_all.append(points_2d)

        if chat2dEnabled:
            plotScatter[i].update(points_2d[:,0], points_2d[:,1])
        
        scene.plotter[0, scene.plotR].add_points(
                             np.hstack((points_2d, np.zeros((points_2d.shape[0], 1)))),
                             name='planePoints_'+(boneList[i]),
                             render_points_as_spheres=True,
                             point_size=3.0,
                             color=bonePointColours[i])
        
        if chat2dEnabled:
            chartMinXTemp = np.min([chartMinXTemp, np.min(points_2d[:,0])])
            chartMaxXTemp = np.max([chartMaxXTemp, np.max(points_2d[:,0])])
            chartMinYTemp = np.min([chartMinYTemp, np.min(points_2d[:,1])])
            chartMaxYTemp = np.max([chartMaxYTemp, np.max(points_2d[:,1])])

        i=i+1
    
    if chat2dEnabled:
        chartMinX = chartMinXTemp
        chartMaxX = chartMaxXTemp
        chartMinY = chartMinYTemp
        chartMaxY = chartMaxYTemp
    if contoursEnabled:
        i = 0
        for indBones in range(len(points_2d_all)):
            # offset and scale up for OpenCV
            #scale = contour_detection_resolution
            #points_2d = points_2d_all[indBones]
            #offsetX = (np.min(points_2d_all[indBones][:,0]) * -1.0) + 1
            #offsetY = (np.min(points_2d_all[indBones][:,1]) * -1.0) + 1
            #points_2d[:,0] += offsetX
            #points_2d[:,1] += offsetY
            #points_2d *= scale
            #resolution = int(np.max(points_2d))
            #outer_contour_points = outer_contour(points_2d, resolution)
            #outer_contour_points = np.array(outer_contour_points, dtype=float)
            #outer_contour_points /= scale
            #outer_contour_points[:,0] -= offsetX
            #outer_contour_points[:,1] -= offsetY

            alpha_shape = alphashape.alphashape(points_2d_all[indBones], alpha=0.9)
            outer_contour_points = np.array(alpha_shape.boundary.coords)
            
            if chat2dEnabled:
                plotLine[i].update(outer_contour_points[:,0], outer_contour_points[:,1])
            
            outer_contour_points = points_to_line_segments(outer_contour_points)

            if (len(outer_contour_points) & 0x1): # if odd
                outer_contour_points = outer_contour_points[:-1]
            line = np.hstack((outer_contour_points, np.zeros((outer_contour_points.shape[0], 1))))
            
            scene.plotter[0, scene.plotR].add_lines(
                                lines = line,
                                name='planeLines_'+(boneList[i]),
                                width=3.0,
                                color=bonePointColours[i], )
            
            i=i+1
    if chat2dEnabled:
        updateAxies()
    

def updatePlane(plane_normal, plane_origin):
    global leftPlot_plane_normal, leftPlot_plane_origin
    leftPlot_plane_normal = np.array(plane_normal)
    leftPlot_plane_origin = np.array(plane_origin)
    updateChart()
    setRightView()
    return

def updateLines():
    for ind in range(len(scene.plotter[0, scene.plotR].line_widgets)):
        point1 = list(scene.plotter[0, scene.plotR].line_widgets[ind].GetPoint1())
        point1[2] = 0.0
        scene.plotter[0, scene.plotR].line_widgets[ind].SetPoint1(point1)
        point2 = list(scene.plotter[0, scene.plotR].line_widgets[ind].GetPoint2())
        point2[2] = 0.0
        scene.plotter[0, scene.plotR].line_widgets[ind].SetPoint2(point2)
    if len(scene.plotter[0, scene.plotR].line_widgets) > 0:
        angle_between = calAngle()
        scene.plotter[0, scene.plotR].add_text(f"Angle: {angle_between:.3f} deg", position = 'upper_right', color = 'black', name = 'angle')
    return

def calAngle():
    if len(scene.plotter[0, scene.plotR].line_widgets) > 1:
        point1 = np.array(scene.plotter[0, scene.plotR].line_widgets[0].GetPoint1())
        point2 = np.array(scene.plotter[0, scene.plotR].line_widgets[0].GetPoint2())
        v1 = fm.normalizeVector(point1 - point2)
        point1 = np.array(scene.plotter[0, scene.plotR].line_widgets[1].GetPoint1())
        point2 = np.array(scene.plotter[0, scene.plotR].line_widgets[1].GetPoint2())
        v2 = fm.normalizeVector(point1 - point2)
        return np.rad2deg(fm.angleBetweenVectors(v1, v2))
    else:
        return 0.0

def updatLine1(pointa, pointb):
    updateLines()
    return
def updatLine2(pointa, pointb):
    updateLines()
    return

def pickedPoint(polydata, vertID):
    scene.plotter[0, scene.plotL].add_text(f"Vertex ID: {vertID}", position = 'upper_right', color = 'black', name = 'vertID')
    return True

def pickedCell(picked_cells_multi):
    scene.plotter[0, scene.plotL].add_text(f"polydata: {picked_cells_multi}", position = 'upper_left', color = 'black', name = 'polydata')
    if picked_cells_multi:
        p = []
        for picked_cells in picked_cells_multi:
            if picked_cells.n_cells > 0:
                p.append(picked_cells)
        if len(p) > 1:
            print('Error, multiple obejct selected! Using first found.')
        
        planeNormal, planeCenter = fitPlaneNormal(p[0].points)

        ve = np.array([planeNormal * 3 + planeCenter, planeCenter])
        scene.plotter[0, scene.plotL].add_points(
            points = ve,
            name='p_',
            render_points_as_spheres=True,
            point_size=15.0,
            color='red')
    return True


def fitPlaneNormal(data):
    # from: https://stackoverflow.com/questions/38754668/plane-fitting-in-a-3d-point-cloud
    mean = np.mean(data, axis=0)
    data_adjust = data - mean
    
    matrix = np.cov(data_adjust.T) 

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    #: sort eigenvalues and eigenvectors
    sort = eigenvalues.argsort()[::-1]
    #eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:,sort]

    return eigenvectors[:,2], mean # normal, and center


mb1 = scene.plotter[0, scene.plotL].picked_cells
mb = pv.MultiBlock(mb1)

type(scene.plotter[0, scene.plotL].picked_mesh)

p= []
for block in mb:
    if block.n_cells > 0:
        print(block)
        p.append(block)


planeNormal, planeCenter = fitPlaneNormal(p[0].points)
type(p[0])
p0 = pv.UnstructuredGrid(p[0])
p0.FIELD_ASSOCIATION_VERTICES

pointIDs = []
for cellID in p0.cells:
    pointIDs.append(p0.get_cell(cellID).point_ids)
p0.get_cell(cellID).GetPointIds()
pointIDs = np.hstack(np.array(pointIDs))
pointIDs = np.unique(pointIDs)


pv.DataSet.get_cell

radiusPolyData = scene.bonePolyData[scene.plotL][scene.modelInfo['names']['radius']]
radiusPolyData = pv.PolyData(radiusPolyData)
radiusPolyData.cell_point_ids(0)
radiusPolyData.surface_indices()

ve = np.array([planeNormal * 3 + planeCenter, planeCenter])
scene.plotter[0, scene.plotL].add_points(
    points = ve,
    name='p_',
    render_points_as_spheres=True,
    point_size=15.0,
    color='red')
scene.plotter[0, scene.plotL].add_points(
    points = radiusPolyData.points[pointIDs],
    name='p_',
    render_points_as_spheres=True,
    point_size=15.0,
    color='red')
scene.plotter[0, scene.plotL].add_points(
    points = p0.points,
    name='p_',
    render_points_as_spheres=True,
    point_size=15.0,
    color='red')


def try_callback(func, *args):
    """Wrap a given callback in a try statement.

    Parameters
    ----------
    func : callable
        Callable object.

    *args
        Any arguments.

    """
    import sys, traceback, warnings
    try:
        func(*args)
    except Exception:
        etype, exc, tb = sys.exc_info()
        stack = traceback.extract_tb(tb)[1:]
        formatted_exception = 'Encountered issue in callback (most recent call last):\n' + ''.join(
            traceback.format_list(stack) + traceback.format_exception_only(etype, exc)
        ).rstrip('\n')
        warnings.warn(formatted_exception)

def enable_cell_picking(
    self,
    callback=None,
    show=True,
    style='wireframe',
    line_width=5,
    color='pink',
    **kwargs,
):
    import weakref, pyvista
    from pyvista import _vtk
    self_ = weakref.ref(self)

    # make sure to consistently use renderer
    renderer_ = weakref.ref(self.renderer)
    picker = _vtk.vtkRenderedAreaPicker()
    picker.GetSelectionPoint

    def end_pick_helper(picker, event_id):
        # Merge the selection into a single mesh
        picked = self_().picked_cells
        if isinstance(picked, pyvista.MultiBlock):
            if picked.n_blocks > 0:
                picked = picked.combine()
            else:
                picked = pyvista.UnstructuredGrid()
        # Check if valid
        is_valid_selection = picked.n_cells > 0

        if show and is_valid_selection:
            # Select the renderer where the mesh is added.
            active_renderer_index = self_().renderers._active_index
            for index in range(len(self.renderers)):
                renderer = self.renderers[index]
                for actor in renderer._actors.values():
                    mapper = actor.GetMapper()
                    if isinstance(mapper, _vtk.vtkDataSetMapper):
                        loc = self_().renderers.index_to_loc(index)
                        self_().subplot(*loc)
                        break

            # Use try in case selection is empty
            self_().add_mesh(
                picked,
                name='_cell_picking_selection',
                style=style,
                color=color,
                line_width=line_width,
                pickable=False,
                reset_camera=False,
                **kwargs,
            )

            # Reset to the active renderer.
            loc = self_().renderers.index_to_loc(active_renderer_index)
            self_().subplot(*loc)

            # render here prior to running the callback
            self_().render()
        elif not is_valid_selection:
            self.remove_actor('_cell_picking_selection')
            self_().picked_cells = None

        if callback is not None:
            try_callback(callback, self_().picked_cells)

        # TODO: Deactivate selection tool
        return

    def through_pick_call_back(picker, event_id):
        picked = pyvista.MultiBlock()
        for actor in renderer_().actors.values():
            if actor.GetMapper() and actor.GetPickable():
                input_mesh = pyvista.wrap(actor.GetMapper().GetInputAsDataSet())
                input_mesh.cell_data['orig_extract_id'] = np.arange(input_mesh.n_cells)
                extract = _vtk.vtkExtractGeometry()
                extract.SetInputData(input_mesh)
                extract.SetImplicitFunction(picker.GetFrustum())
                extract.Update()
                picked.append(pyvista.wrap(extract.GetOutput()))

        if len(picked) == 1:
            self_().picked_cells = picked[0]
        else:
            self_().picked_cells = picked
        return end_pick_helper(picker, event_id)

    self._picker = _vtk.vtkRenderedAreaPicker()
    self._picker.AddObserver(_vtk.vtkCommand.EndPickEvent, through_pick_call_back)

    self.enable_rubber_band_style()
    self.iren.set_picker(self._picker)






def toggle2DContour(flag):
    global contoursEnabled
    contoursEnabled = flag
    if flag:
        updateChart()

def generateRI(scene, showLine=False):
    """
    RI: radius intersection, the projection of RC on distal
    articular surface of radius.
    """
    radiusPolyData = scene.bonePolyData[scene.plotL][scene.modelInfo['names']['radius']]
    centerLine = findLine(radiusPolyData.points)

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
    points, _ = radiusPolyData.ray_trace(origin, direction)

    metacarpa3 = scene.bonePolyData[scene.plotL][scene.modelInfo['names']['metacarp3']]

    p, ind = find_closest_point(points, metacarpa3.center_of_mass())
    return p, np.delete(points, ind, axis=0)[0]

def updateRadiusCoord(flag):
    global planeDownward
    boneSetInd = [i for i, x in enumerate(radiusCoords) if x['name'] == scene.modelInfo['experimentID']]
    if len(boneSetInd) == 0:
        print(f"Error, no point indexes set for experiment {scene.modelInfo['experimentID']}")
        return
    else:
        boneSetInd = boneSetInd[0]

    radiusCoord = radiusCoords[boneSetInd]

    ri, rc = generateRI(scene, True)
    radiusCoord.update({'RI': ri})
    radiusCoord.update({'RC': rc})

    radiusPolyData = scene.bonePolyData[scene.plotL][scene.modelInfo['names']['radius']]
    radiusCoord.update({'SN': radiusPolyData.points[radiusCoord['sigmoidNotch']]})
    radiusCoord.update({'RS': radiusPolyData.points[radiusCoord['radialStyloid']]})


    points = np.array([
        radiusCoord['RI'],
        radiusCoord['RC'],
        radiusCoord['RS'],
        radiusCoord['SN']
    ])
    scene.plotter[0, scene.plotL].add_points(
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
    scene.plotter[0, scene.plotL].add_points(
                            points=coords,
                            name='radius_coords',
                            render_points_as_spheres=True,
                            point_size=10.0,
                            color='red',
                            pickable = False)

    plane = scene.plotter[0, scene.plotL].plane_widgets[0]
    plane.SetNormal(rZ)
    plane.SetOrigin(radiusCoord['RI'])
    planeDownward = rX
    updatePlane(plane.GetNormal(), plane.GetOrigin())

#def change_resolution(value):
#    global contour_detection_resolution
#    contour_detection_resolution = value
#    updateChart()
#    return

#scene.plotter[0, scene.plotR].add_slider_widget(change_resolution, [1, 10],
#                                                value = 1.0, title='Resolution',
#                                                color = 'black')
scene.plotter[0, scene.plotL].add_plane_widget(updatePlane,
                                               normal=leftPlot_plane_normal)

scene.plotter[0, scene.plotR].add_text("Angle: 0.00 deg", position = 'upper_right', color = 'black', name = 'angle')
scene.plotter[0, scene.plotR].add_line_widget(callback=updatLine1,
                                              use_vertices=True,
                                              color='red')
scene.plotter[0, scene.plotR].add_line_widget(callback=updatLine2,
                                              use_vertices=True,
                                              color='green')


#scene.plotter[0, scene.plotR].add_callback(setRightView, interval=100)

scene.plotter[0, scene.plotL].enable_point_picking(callback=pickedPoint, use_mesh=True)
#scene.plotter[0, scene.plotL].picked_point
scene.plotter[0, scene.plotL].enable_cell_picking(callback=pickedCell)

scene.plotter[0, scene.plotL].add_checkbox_button_widget(
    updateRadiusCoord, color_on = 'grey', position = (20, 70),
    color_off = 'grey', value=False)
scene.plotter[0, scene.plotL].add_text(
    "Auto\nAlign", font_size = 12, position = (15,10),
    color = 'black', name = 'button_title_updateRadiusCoord')

scene.plotter[0, scene.plotR].add_checkbox_button_widget(
    toggle2DContour, color_on = 'green', position = (30, 70),
    color_off = 'red', value=False)
scene.plotter[0, scene.plotR].add_text(
    " Toggle\nContour", font_size = 12, position = (10,10),
    color = 'black', name = 'button_title_toggle2DContour')

scene.plotter[0, scene.plotR].add_checkbox_button_widget(
    setRightView, color_on = 'grey', position = (130, 70),
    color_off = 'grey', value=False)
scene.plotter[0, scene.plotR].add_text(
    "Reset\nView", font_size = 12, position = (130,10),
    color = 'black', name = 'button_title_setRightView')


# vertex index for points on radius
radiusCoords = [
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


updateChart()
setRightView()

boneSetInd = [i for i, x in enumerate(radiusCoords) if x['name'] == scene.modelInfo['experimentID']]
if len(boneSetInd) != 0:
    updateRadiusCoord(True)
    
# %%

# add dropdown to choose trial
# add slider to move through trial
# update position of bones at set trial time
# maybe automate measures
