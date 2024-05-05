# Author: Alastair Quinn 2022
# These scripts work with the 3-Matic API Trimatic and Python Jupyter

#%% To allow re-importing modules without restarting jupyter
%load_ext autoreload
%autoreload 2
#%matplotlib qt
#%%
import slil.mesh.interface as smi
mi = smi.MeshInterface(1)

import numpy as np

import slil.process.functions as fn
import slil.common.data_configs as dc
import slil.common.math as fm

import pyvista as pv
#pv.set_jupyter_backend('pythreejs')

#py -m pip install pyvista trimesh optimparallel pythreejs


#%%

experiments = [
    '11524',
    '11525',
    '11526',
    '11527', # bad, do not run
    '11534',
    '11535',
    '11536',
    '11537',
    '11538',
    '11539'
    ]

limitPos = 19.0 # mm
limitRot = 45 # degrees
from slil.process.align_functions import checkResultsCloseToBounds
checkResultsCloseToBounds(experiments, limitPos, limitRot, '_SLILGap')

#%%
from slil.process.align_by_surface import viewResult
viewResult('11535', 'normal')
#%%
from slil.process.align_by_slil_gap import viewResult
viewResult('11538', 'normal')

#%%

import slil.process.functions as fn
import slil.common.data_configs as dc
import numpy as np

experiments = [
    #'11524',
    #'11525',
    #'11526',
    #'11527', # bad, do not run
    '11534',
    #'11535',
    #'11536',
    #'11537',
    #'11538',
    #'11539'
    ]

for i, experiment in enumerate(experiments):
    modelInfo = dc.load(experiment)
    modelTypes = [ 'normal', 'cut', 'scaffold']
    modelInfo['currentModel'] = modelTypes[0]

    trial = '\\' + modelInfo['currentModel'] + r'_fe_40_40\log1'
    print("Using trial: {}".format(trial))
    fileToImport = modelInfo['dataOutputDir'] + trial + '.c3d'

    #from slil.process.align_by_surface import SurfaceOpimizer
    #surfOpt = SurfaceOpimizer(modelInfo)
    from slil.process.align_by_slil_gap import SLILGapOpimizer, getFrameNs
    opt = SLILGapOpimizer(modelInfo)
    opt.loadTrial(fileToImport)

    alignmentFrames = getFrameNs()
    opt.selectFrames(alignmentFrames[experiment][modelInfo['currentModel']]['fe'])

    trial = '\\' + modelInfo['currentModel'] + r'_ur_30_30\log1'
    fileToImport = modelInfo['dataOutputDir'] + trial + '.c3d'
    opt.loadTrial2(fileToImport)
    opt.selectFrames2(alignmentFrames[experiment][modelInfo['currentModel']]['ur'])

    modelCache = fn.getModelCache(modelInfo)
    opt.modelCache = modelCache

    opt.loadGeometry()

    boundsPos = 35 # mm
    boundsRot = np.deg2rad(20) # radians
    boundsRot1 = np.deg2rad(15) # radians
    # x, y, z, rx, ry, rz
    bounds = [
        (-1.0 * boundsPos, boundsPos), \
        (-1.0 * boundsPos, boundsPos), \
        (-1.0 * boundsPos, boundsPos), \
        (-1.0 * boundsRot, boundsRot), \
        (-1.0 * boundsRot, boundsRot), \
        (-1.0 * boundsRot1, boundsRot1)
        ]
    opt.setBound(bounds)

    x0 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    opt.setStartingTransformation(x0)

    import timeit
    start_time = timeit.default_timer()
    opt.runOptFun_noVis(x0, showErrors = False)
    print('One evaluation takes ~{:.3f} milliseconds'.format((timeit.default_timer() - start_time)*1000.0))


    resultC = opt.runOnCluster(
        clusterAddress = 'ray://100.99.142.105:10001', # 6DoF
        #clusterAddress = 'ray://localhost:10003', # HPC
        optimizerName = 'mystic_0')
        #optimizerName = 'L-BFGS-B')
    print(resultC)
    fn.saveLoadOptimizationResults(modelInfo, resultC, '_SLILGap')



#%%

modelInfo = dc.load(experiment)
modelTypes = [ 'normal', 'cut', 'scaffold']
modelInfo['currentModel'] = modelTypes[0]
adjustTransMarkers = fn.getOptimizationResults(modelInfo)
x0 = adjustTransMarkers[modelInfo['currentModel']]


#%%

x0 = ( 5.323302,   0.000289,   3.836420,   0.000095,  -0.000164,   0.000581)
x0 = ( 12.135814,    10.727236,    2.103416,   -0.542047,   -0.317778,   -0.286396)
x0 = (1.64609053, 14.48559671, -0.65843621, -0.09815981,  0.55065258, 0.78527846)
x0 = (3.32228602e+00,  8.72727001e+00,  1.17381831e+00, -4.02231839e-01, 3.37557983e-03,  1.25637289e-01)
x0 = (7.57519737,  2.84121626, -1.10120177, -0.12869181,  0.28174716, 0.12674859)

x0 = (2.00712342, 14.86173315, -0.37368196, -0.20102396,  0.64183582, 0.52220445)

x0 = ( 5.0, 0.0, 5.0, 0.0, 0.0, 0.3)

# 11534
#         Current function value: 46.356141
#         Iterations: 8
#         Function evaluations: 1488
#         Total function evaluations: 18859
#STOP("NormalizedChangeOverGeneration with {'tolerance': 0.0001, 'generations': 5}")
x0 = (6.82714361,   6.16466476, -11.23535839,  -0.35867641, 0.67946307,   0.50914366)

# 11539
#fun: 99.85580792069152
# hess_inv: <6x6 LbfgsInvHessProduct with dtype=float64>
#      jac: array([-1.93568895e-01, -2.74390266e-01,  3.78378216e-01,  3.69403587e+03,
#        8.37854051e+02, -7.13353760e+02])
#  message: 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
#     nfev: 378
#      nit: 9
#     njev: 54
#   status: 0
#  success: True
x0 = (1.64729777, 14.48579403, -0.65910727, -0.10560831,  0.55330904, 0.76680171)

opt.visualize(x0, showErrors = True, showLines = False)


#%%

from pyvistaqt import BackgroundPlotter
opt.displayViz = BackgroundPlotter(window_size=(1000, 1000))

optimizerName = 'fmin'
#optimizerName = 'fmin_powell'
#optimizerName = 'differential_evolution'
optimizerName = 'L-BFGS-B'
#optimizerName = 'minimize_parallel'
#optimizerName = 'mystic_0'
opt.run(optimizerName)

#fn.saveLoadOptimizationResults(modelInfo, opt.results, '_SLILGap')

#%%
opt.run(optimizerName)

x0 = opt.results['cluster']['t']
x0 = opt.results['t']
import slil.common.math as fm
print('Overall angle change = {}'.format(
    np.rad2deg(fm.rotMat2AxisAngle(fm.eulerAnglesToRotationMatrix((x0[3], x0[4], x0[5])))[1])))

opt.visualize(x0, showErrors = True, newWindow = True, showLines = False)

#%%

modelsOptimized = [ 'normal' ] #, 'cut']#, 'scaffold']
from slil.process import generate_model
generate_model.setOptimizedMarkers([ experiment ], modelsOptimized)


#%%

from slil.process.align_by_surface_opt_fun import optFun
import pickle

b = pickle.dumps(optFun)

#%% View some results

import slil.process.functions as fn
import slil.common.data_configs as dc
import slil.common.math as fm
import pyvista as pv

[lunateName, scaphoidName, radiusName, metacarp3Name] = fn.boneModelNames(modelInfo)
surfLun = [x['surface'] for x in opt.checkBones if x['name'] == lunateName][0]
surfSca = [x['surface'] for x in opt.checkBones if x['name'] == scaphoidName][0]
surfRad = opt.referenceBone['surface']


surfs = []
surfs.append(surfLun)
surfs.append(surfSca)
surfs.append(surfRad)

outputPairs = []
outputPairs.append([modelCache['SLILpointS'],modelCache['SLILpointL']])

from slil.process.align_by_slil_gap import findPairedPointsWithin
pointsFoundLun, pointsFoundSca, pointsFoundDistance = findPairedPointsWithin(surfLun.points, surfSca.points, maxDist = 4.0)

outputPairs = []
for i in range(len(pointsFoundDistance)):
    outputPairs.append([pointsFoundLun[i], pointsFoundSca[i]])


from pyvistaqt import BackgroundPlotter
plotter = BackgroundPlotter(window_size=(1000, 1000))

for ind, surf in enumerate(surfs):
    plotter.add_mesh(
        surf,
        name = 'surf_' + str(ind),
        scalars = np.arange(surf.n_faces),
        color = 'silver',
        specular = 1.0,
        specular_power = 10,
        opacity = 0.5
    )
#for ind, frame in enumerate(framePointsRaw):
#    plotter.add_points(
#        points = np.array(frame),
#        name = 'frame_' + str(ind),
#        render_points_as_spheres = True,
#        point_size = 20.0
#        )
for ind, pair in enumerate(outputPairs):
    plotter.add_lines(
        lines = np.array(pair),
        width = 2.0,
        name = 'line_' + str(ind)
    )

plotter.show_axes_all()
plotter.view_vector([0, 1, 0])
plotter.set_viewup([0, 0, -1])

origin = (0.0, 0.0, 0.0) # for axies lines at origin
plotter.add_mesh(pv.Line(origin, (1.0, 0.0, 0.0)), color='red')
plotter.add_mesh(pv.Line(origin, (0.0, 1.0, 0.0)), color='green')
plotter.add_mesh(pv.Line(origin, (0.0, 0.0, 1.0)), color='blue')

