# Author: Alastair Quinn 2022

import slil.process.functions as fn
import slil.common.math as fm
import numpy as np
from copy import deepcopy
import pyvista as pv
import slil.common.io as fio
from slil.process.align_by_surface_opt_fun import optFun
import ray
from scipy import optimize

Nfeval = 1

def runOptimization(experiment, modelTypes, clusterAddress = 'auto'):
    import slil.common.data_configs as dc
    #for i, experiment in enumerate(experiments):
    modelInfo = dc.load(experiment)
    #modelTypes = [ 'normal', 'cut', 'scaffold']
    for modelType in modelTypes:
        modelInfo['currentModel'] = modelType

        trial = '\\' + modelInfo['currentModel'] + r'_fe_40_40\log1'
        print("Using trial: {}".format(trial))
        fileToImport = modelInfo['dataOutputDir'] + trial + '.c3d'

        surfOpt = SurfaceOpimizer(modelInfo)
        surfOpt.loadTrial(fileToImport)

        frameNs = [
            2,
            2083,
            2083 * 3,
            2083 * 2,
            2083 * 3 + 1000,
            ]
        surfOpt.selectFrames(frameNs)
        surfOpt.loadGeometry()

        boundsPos = 35 # mm
        boundsRot = np.deg2rad(15) # radians
        # x, y, z, rx, ry, rz
        bounds = [
            (-1.0 * boundsPos, boundsPos), \
            (-1.0 * boundsPos, boundsPos), \
            (-1.0 * boundsPos, boundsPos), \
            (-1.0 * boundsRot, boundsRot), \
            (-1.0 * boundsRot, boundsRot), \
            (-1.0 * np.deg2rad(25), np.deg2rad(25))
            ]
        surfOpt.setBound(bounds)

        x0 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        surfOpt.setStartingTransformation(x0)

        import timeit
        start_time = timeit.default_timer()
        surfOpt.runOptFun_noVis(x0, showErrors = False)
        print('One evaluation takes ~{:.3f} milliseconds'.format((timeit.default_timer() - start_time)*1000.0))

        resultC = surfOpt.runOnCluster(
            clusterAddress = clusterAddress,
            #clusterAddress = 'ray://100.99.142.105:10001', # 6DoF
            #clusterAddress = 'ray://localhost:10003', # Griffith HPC
            optimizerName = 'mystic_0')
        print(resultC)
        fn.saveLoadOptimizationResults(modelInfo, resultC)

def viewResult(experiment, modelType):
    import slil.common.data_configs as dc
    modelInfo = dc.load(experiment)
    modelInfo['currentModel'] = modelType

    trial = '\\' + modelInfo['currentModel'] + r'_fe_40_40\log1'
    print("Experiment {} trial {}".format(experiment, trial))
    fileToImport = modelInfo['dataOutputDir'] + trial + '.c3d'

    surfOpt = SurfaceOpimizer(modelInfo)
    surfOpt.loadTrial(fileToImport)

    frameNs = [
        2,
        2083,
        2083 * 3,
        2083 * 2,
        2083 * 3 + 1000,
        ]
    surfOpt.selectFrames(frameNs)
    surfOpt.loadGeometry()

    boundsPos = 20 # mm
    boundsRot = np.deg2rad(50) # radians
    # x, y, z, rx, ry, rz
    bounds = [
        (-1.0 * boundsPos, boundsPos), \
        (-1.0 * boundsPos, boundsPos), \
        (-1.0 * boundsPos, boundsPos), \
        (-1.0 * boundsRot, boundsRot), \
        (-1.0 * boundsRot, boundsRot), \
        (-1.0 * boundsRot, boundsRot)
        ]
    surfOpt.setBound(bounds)

    x0 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    surfOpt.setStartingTransformation(x0)

    import timeit
    start_time = timeit.default_timer()
    surfOpt.runOptFun_noVis(x0, showErrors = False)
    print('One evaluation takes ~{:.3f} milliseconds'.format((timeit.default_timer() - start_time)*1000.0))
    adjustTransMarkers = fn.getOptimizationResults(modelInfo)
    x0 = adjustTransMarkers[modelInfo['currentModel']]
    
    surfOpt.visualize(x0, showErrors = True)

class SurfaceOpimizer():
    def __init__(self, modelInfo):
        self.frames = []

        self.modelInfo = modelInfo
        self.referenceBone = {}
        self.checkBones = []
        self.framesPointsRaw = []
        self.display = False
        self.displayViz = None
        self.optimizedT = []

        self.results = {}
        
        modelCache = fn.getModelCache(self.modelInfo)
        self.M_O2Rad_orig = modelCache['initialTmatRadiusMarker'][self.modelInfo['currentModel']]

        self.bounds = [(-20,20), (-20,20), (-20,20), \
            (-1.0 * np.deg2rad(50), np.deg2rad(50)), \
            (-1.0 * np.deg2rad(50), np.deg2rad(50)), \
            (-1.0 * np.deg2rad(50), np.deg2rad(50))
            ]
        self.x0 = (5,0,5,0,0,0)

    def getExtraArgs(self):
        return {
            'modelInfo': self.modelInfo,
            'referenceBone': self.referenceBone,
            'checkBones': self.checkBones,
            'M_O2Rad_orig': self.M_O2Rad_orig,
            'framesPointsRaw': self.framesPointsRaw,
            'display': self.display,
            'displayViz': self.displayViz
        }

    def setBound(self, newBounds):
        self.bounds = newBounds

    def setStartingTransformation(self, newX0):
        self.x0 = newX0

    def loadTrial(self, fileToImport):
        self.frames = np.array(fio.readC3D(fileToImport))

    def selectFrames(self, frameNs):
        self.framesPointsRaw = []
        for n in frameNs:
            self.framesPointsRaw.append(np.array(deepcopy(self.frames[n,:,0:3])))

    def loadGeometry(self):
        [lunateName, scaphoidName, radiusName, metacarp3Name] = fn.boneModelNames(self.modelInfo)
        
        self.geometryLun = fn.getGeometry(self.modelInfo, lunateName)
        self.geometrySca = fn.getGeometry(self.modelInfo, scaphoidName)
        self.geometryRad = fn.getGeometry(self.modelInfo, radiusName)
        #surfMeta = getGeometry(modelInfo, metacarp3Name)
        self.createSurfaces() # seperate as ray.io doesn't like pickling PolyData

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

        surfLun = converToPolyData(self.geometryLun)
        surfSca = converToPolyData(self.geometrySca)
        eID = self.modelInfo['experimentID']
        # there's a better way to determine surface resolution... impliment it when you know it
        if eID != '11524' and eID != '11525' and eID != '11526' and eID != '11527':
            surfLun = surfLun.decimate(target_reduction = 0.9, volume_preservation=True)
            surfSca = surfSca.decimate(target_reduction = 0.9, volume_preservation=True)

        surfRad_original = converToPolyData(self.geometryRad)
        #surfMeta = getGeometry(modelInfo, metacarp3Name)

        [lunateName, scaphoidName, radiusName, metacarp3Name] = fn.boneModelNames(self.modelInfo)
        
        # this is a slow operation!
        lunCOM = fm.calcCOM(surfLun.points, surfLun.faces.reshape(-1, 4), method = 'mesh')

        t_WL1 = np.eye(4)
        t_WL1[:3, 3] = lunCOM.T
        
        scaCOM = fm.calcCOM(surfSca.points, surfSca.faces.reshape(-1, 4), method = 'mesh')

        t_WLxSca = np.eye(4)
        t_WLxSca[:3, 3] = scaCOM.T

        self.checkBones = [
            { 'name': lunateName,'surface': surfLun, 't_WLx': t_WL1},
            { 'name': scaphoidName,'surface': surfSca, 't_WLx': t_WLxSca},
            #{ 'name': metacarp3Name,'surface': surfMeta },
            ]
        
        radCOM = fm.calcCOM(surfRad_original.points, surfRad_original.faces.reshape(-1, 4), method = 'mesh')

        t_WLxRad = np.eye(4)
        t_WLxRad[:3, 3] = radCOM.T

        # reduce radius by splitting model. Could be done a better way
        plane = pv.Plane()
        plane.scale((50,50,0), inplace=True)
        plane.rotate_y(90, inplace=True)
        plane.translate((6.5,0,0), inplace=True)
        plane = plane.triangulate()
        # maybe split the radius vectors to only have part of the surface we need.
        # same could be done for the lunate and scaphoid
        surfRad = surfRad_original.boolean_difference(plane) # this is a slow operation!
        if eID != '11524' and eID != '11525' and eID != '11526' and eID != '11527':
            surfRad = surfRad.decimate(target_reduction = 0.9, volume_preservation=True)

        self.referenceBone = {
            'name': radiusName,
            'surface': surfRad,
            't_WLx': t_WLxRad
            }
        

    def visualize(self, x0, showErrors = True):

        argsExtra = self.getExtraArgs()
        argsExtra['display'] = showErrors
        argsExtra['displayViz'] = True

        return optFun(x0, argsExtra)

    def runOptFun_noVis(self, x0, showErrors = True):

        argsExtra = self.getExtraArgs()
        argsExtra['display'] = showErrors
        argsExtra['displayViz'] = None

        return optFun(x0, argsExtra)

    #def saveT(self, PR):
    #    s = PR
    #    t_optimized = np.eye(4)
    #    t_optimized[:3, :3] = fm.eulerAnglesToRotationMatrix((s[3], s[4], s[5]))
    #    t_optimized[:3, 3] = [s[0], s[1], s[2]]
    #    fn.modifyModelCache_adjTmatRadiusMarker(self.modelInfo, t_optimized)
        
    #def saveOptT(self):
    #    self.saveT(self.optimizedT)
    
    
    def run(self, optimizerName):
        global Nfeval

        argsExtra = self.getExtraArgs()

        Nfeval = 1
        def callbackF(Xi):
            global Nfeval
            #print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], f(Xi, argsExtra)))
            print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}   {7: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5], optFun(Xi, argsExtra)))
            Nfeval += 1
        #print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)'))

        #import timeit
        #start_time_global = timeit.default_timer()
        
        def callbackFde(Xi, convergence):
            global Nfeval
            #global start_time_global
            print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}   {7: 3.6f}   {8: 3.6f}'.format(Nfeval, convergence, Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5], optFun(Xi, argsExtra)))
            Nfeval += 1
            #if (timeit.default_timer() - start_time_global) > 30.0:
            #    return True # stops optimizer
            return False

        return self._run(optimizerName, callbackF, callbackFde)

    def _run(self, optimizerName, callbackF, callbackFde):
        import timeit

        argsExtra = self.getExtraArgs()
        
        print('Starting Optimizer: {}'.format(optimizerName))

        if optimizerName == 'fmin':
            startTime = timeit.default_timer()
            result = optimize.fmin( \
                func = optFun, \
                x0 = self.x0, \
                args = (argsExtra, ), \
                callback = callbackF, \
                full_output = True, \
                retall = True, \
                maxiter = 10)
            self.results['optimizationDuration'] = (timeit.default_timer() - startTime)
            self.results['finished'] = True
            self.results['t'] = result[0] # xopt
            self.results['succeded'] = None
            self.results['fopt'] = result[1] # fopt
            self.results['iter'] = result[2] # iter
            self.results['funcalls'] = result[3] # funcalls
            self.results['solutionHistory'] = result[5] # allvecs

        if optimizerName == 'fmin_powell':
            startTime = timeit.default_timer()
            result = optimize.fmin_powell( \
                func = optFun, \
                x0 = self.x0, \
                args = (argsExtra, ), \
                callback = callbackF, \
                full_output = True, \
                retall = True, \
                maxiter = 20000)
            self.results['optimizationDuration'] = (timeit.default_timer() - startTime)
            self.results['finished'] = True
            self.results['t'] = result[0] # xopt
            self.results['succeded'] = None
            self.results['fopt'] = result[1] # fopt
            self.results['iter'] = result[3] # iter
            self.results['funcalls'] = result[4] # funcalls
            self.results['solutionHistory'] = result[6] # allvecs

        if optimizerName == 'differential_evolution' or \
            optimizerName == 'L-BFGS-B' or \
            optimizerName == 'minimize_parallel':

            if optimizerName == 'differential_evolution':
                startTime = timeit.default_timer()
                result = optimize.differential_evolution( \
                    func = optFun, \
                    bounds = self.bounds, \
                    x0 = self.x0, \
                    args = (argsExtra, ), \
                    callback = callbackFde, \
                    disp = True, \
                    workers = -1, \
                    maxiter = 1000)

            if optimizerName == 'L-BFGS-B':
                startTime = timeit.default_timer()
                result = optimize.minimize( \
                    fun = optFun, \
                    bounds = self.bounds, \
                    x0 = self.x0, \
                    args = (argsExtra, ), \
                    method = 'L-BFGS-B', \
                    callback = callbackF, \
                        options= {
                            'disp': False,
                            'maxcor': 40,
                            'maxiter': 3000,
                            },
                    )
            if optimizerName == 'minimize_parallel':
                from optimparallel import minimize_parallel
                startTime = timeit.default_timer()
                result = minimize_parallel( \
                    fun = optFun, \
                    bounds = self.bounds, \
                    x0 = self.x0, \
                    #jac = None, \
                    args = (argsExtra, ), \
                    callback = callbackF, \
                    parallel = {
                        #'max_workers': 60, \
                        'verbose': True, \
                        'loginfo': True, \
                        #'forward': False \
                        }, \
                    )
            
            self.results['optimizationDuration'] = (timeit.default_timer() - startTime)
            self.results['finished'] = True
            self.results['t'] = result.x
            self.results['succeded'] = result.success
            self.results['status'] = result.status
            self.results['fopt'] = result.fun
            self.results['iter'] = result.nit
            self.results['funcalls'] = result.nfev

        if optimizerName == 'mystic_0':

            from mystic.solvers import BuckshotSolver

            # Powell's Directonal solver
            from mystic.solvers import PowellDirectionalSolver

            # if available, use a pathos worker pool
            try:
                from pathos.pools import ProcessPool as Pool
            #from pathos.pools import ParallelPool as Pool
            except ImportError:
                from mystic.pools import SerialPool as Pool

            # tools
            from mystic.termination import NormalizedChangeOverGeneration as NCOG
            from mystic.monitors import VerboseLoggingMonitor
            from pathos.helpers import freeze_support, shutdown
            freeze_support() # help Windows use multiprocessing

            print("Powell's Method")
            print("===============")

            # dimensional information
            from mystic.tools import random_seed
            random_seed(123)
            ndim = 6 # x, y, z, rx, ry, rz
            npts = 12 # 'workers'

            # configure monitor
            #stepmon = VerboseLoggingMonitor(1,2)

            # use buckshot-Powell to solve 8th-order Chebyshev coefficients
            solver = BuckshotSolver(ndim, npts)
            solver.SetNestedSolver(PowellDirectionalSolver)
            #solver.SetInitialPoints(x0)
            solver.SetMapper(Pool().map)
            #solver.SetGenerationMonitor(stepmon)
            solver.SetStrictRanges(min=[x[0] for x in self.bounds], max=[x[1] for x in self.bounds])
            
            startTime = timeit.default_timer()
            solver.Solve(
                cost = optFun,
                # doesn't seem to much change after generation 5...
                termination = NCOG(tolerance = 1e-4, generations = 5), # ("NormalizedChangeOverGeneration with {'tolerance': 0.0001, 'generations': 10}")
                ExtraArgs = (argsExtra, ),
                disp = 1)
            result = {}
            result['message'] = 'Method used: {}'.format(optimizerName)
            self.results['optimizationDuration'] = (timeit.default_timer() - startTime)

            self.results['t'] = solver.Solution()
            self.results['solutionHistory'] = solver.solution_history
            self.results['population'] = solver.population
            self.results['evaluations'] = solver.evaluations
            #self.results['bestFunctionValue'] = solver.bestEnergy
            shutdown() # help multiprocessing shutdown all workers

            # write 'convergence' support file
            #from mystic.munge import write_support_file
            #write_support_file(solver._stepmon) #XXX: only saves the 'best'

        self.results['bestFunctionValue'] = optFun(self.results['t'], argsExtra)

        print(self.results)
        print(result)
        self.optimizedT = self.results['t']
        return self

    @ray.remote
    def _runOnRemote(self, optimizerName, callbackF, callbackFde):
        # doesn't work
        #import logging
        #logging.basicConfig(level=logging.INFO, format='%(message)s')
        #logger = logging.getLogger()
        #logger.addHandler(logging.FileHandler('test.log', 'a'))
        #print = logger.info

        self.createSurfaces()

        return self._run(optimizerName, callbackF, callbackFde)

    def runOnCluster(self, clusterAddress, optimizerName):
        
        argsExtra = self.getExtraArgs()

        global Nfeval
        Nfeval = 1
        def callbackF(Xi):
            global Nfeval
            #print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], f(Xi, argsExtra)))
            print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}   {7: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5], optFun(Xi, argsExtra)))
            Nfeval += 1
        #print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)'))

        #import timeit
        #start_time_global = timeit.default_timer()
        
        def callbackFde(Xi, convergence):
            global Nfeval
            #global start_time_global
            print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}   {7: 3.6f}   {8: 3.6f}'.format(Nfeval, convergence, Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], Xi[5], optFun(Xi, argsExtra)))
            Nfeval += 1
            #if (timeit.default_timer() - start_time_global) > 30.0:
            #    return True # stops optimizer
            return False
        
        #clusterAddress='ray://100.99.142.105:10001'
        #clusterAddress='ray://localhost:10003'
        if not ray.is_initialized():
            ray.init(address=clusterAddress, \
                runtime_env={ \
                    "working_dir": ".", \
                    "excludes": [
                        #"/cadaver results 03_08_2021.zip", \
                        "/slil/rayTest/*", \
                        "/slil/3-matic_backup/*", \
                        "*.zip", \
                        "*.PNG", \
                        "*.png", \
                        "*.whl", \
                        "*.pptx", \
                        "*.xlsx", \
                        "*.log", \
                        "*.m", \
                        "**/__pycache__", \
                        #"./*.*",
                        ],
                    }, \
                ignore_reinit_error=True,
                )

        #ray.shutdown()

        tasksRunning = []
        try:
            tasksRunning.append(
                self._runOnRemote.remote(
                    self, optimizerName, callbackF, callbackFde
                    )
                )
        except:
            print('Cluster run failed.')

        resultsClust = []
        while tasksRunning:
            finished, tasksRunning = ray.wait(tasksRunning, num_returns=1, timeout=None)
            for task in finished:
                result = ray.get(task)
                print('Cluster result:', result)
                resultsClust.append(result)
            print('Tasks remaining:', len(tasksRunning))
        print('Finished tasks.')

        self.results['cluster'] = resultsClust[0].results
        return resultsClust