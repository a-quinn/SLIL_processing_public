import sys
from os import add_dll_directory, path
from pathlib import Path
import subprocess
from slil.common.data_configs import getModelPath

# set manually or add to path environemnt variable  and find_opensim() will find it.
_opensim_root = Path(r"C:\local\opensim_python_3_10\opensim_install")

def find_opensim():
    try:
        result = subprocess.check_output(['where.exe', 'simbody-visualizer.exe'], shell=True)
        return result.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        print("Failed to run 'where.exe' command.")
        return None

def is_API_setup(python_folder):
    # OpenSim uses a silly method to install Python API. Check if it's been done.
    import fileinput
    file = str(python_folder.joinpath('opensim','__init__.py'))
    with fileinput.FileInput(file, inplace=False) as file:
        for line in file:
            if 'DLL_PATH' in line:
                return False
    return True

if not _opensim_root.exists():
    print("Searching for OpenSim")
    _opensim_root = Path(find_opensim()).parent.parent
    print(f"Found OpenSim at: {_opensim_root}")

python_folder = _opensim_root.joinpath('sdk', 'Python')
DLL_PATH = str(_opensim_root.joinpath('bin'))

if not python_folder.exists():
    print("Failed to find OpenSim Python folder.")
else:
    if not is_API_setup(python_folder):
        print('Running one time Python setup.')
        setup_file = python_folder.joinpath('setup_win_python38.py')
        subprocess.run(['python', str(setup_file)], cwd=str(python_folder))
    print("Adding OpenSim dlls to Python path.")
    sys.path.extend([str(python_folder)])
    try:
        add_dll_directory(DLL_PATH) # some OpenSim versions don't do this automatically
    except:
        print('Count no add OpenSim dlls to python.')

try:
    import opensim
except:
    print("Count no add OpenSim dlls to python. Ensure the same Python version used here has built OpenSim.")
    
import numpy as np
from tqdm import tqdm

def rootFolderFix(modelInfo):
    return modelInfo
    # this is a bad fix TODO: fix fix...
    dirRootOD = path.join(path.expandvars("%userprofile%"),"OneDrive - Griffith University")
    dirRootOD += r'\Projects\MTP - SLIL project\cadaver experiements'

    dataInputDir = dirRootOD + r'\Data'
    dataOutPutFolderPath = dirRootOD + r'\Data processed_v6'
    experimentID = modelInfo['experimentID']
    modelInfo['rootFolder'] = dirRootOD
    modelInfo['dataInputDir'] = dataInputDir + '\\' + experimentID + r'\vel 10 acc 10 dec 10 jer 3.6'
    modelInfo['dataOutputDir'] = dataOutPutFolderPath + '\\' + experimentID

    modelInfo['opensimModel'] = modelInfo['modelFolder'] + r'\wrist.osim' # old and unused
    return modelInfo

def getInitalBonePositons(modelInfo):
    # Names are usually:
    # radius
    # scaphoid
    # lunate
    # hand_complete

    modelInfo = rootFolderFix(modelInfo)

    filePath = getModelPath(modelInfo)

    model = opensim.Model(filePath)
    state = model.initSystem()
    bodies = model.get_BodySet()

    boneInitialPositions = []
    boneNameOrder = []
    for i in range(bodies.getSize()):
        body1 = bodies.get(i)
        body = bodies.updComponent('/bodyset/' + str(body1.getName()))
        c = list(body.getComponentsList())
        
        bComp = None
        bComp = next((x for x in c if isinstance(x, opensim.simulation.PhysicalOffsetFrame)), None)
        if (bComp):
            bComp2 = body.updComponent(bComp.getName())
            p = bComp.getPositionInGround(state).to_numpy()
            boneInitialPositions.append(p)
            boneNameOrder.append(body1.getName())
        del c

    return boneInitialPositions, boneNameOrder

def getBonePositonsFromIK(modelInfo, dataIn):
    #modelInfo = rootFolderFix(modelInfo)

    filePath = getModelPath(modelInfo)
    print("Reading model file.")
    model = opensim.Model(filePath)
    state = model.initSystem()
    bodies = model.get_BodySet()

    file = [x['file'] for x in dataIn if modelInfo['currentModel'] in x['file']][0]
    filePath = modelInfo['dataOutputDir'] + file + r'_model_kinematics.sto'
    print("Reading states file.")
    statesTable = opensim.TimeSeriesTable(filePath)
    statesTable.addTableMetaDataString('inDegrees','no')
 
    #initialTime = statesTable.getIndependentColumn()[0]
    #finalTime = statesTable.getIndependentColumn()[-1]
    #duration = finalTime - initialTime
    print("Converting states time series table to StatesTrajectory.")
    statesTraj = opensim.StatesTrajectory.createFromStatesTable(
            model, statesTable, True, True, False)

    print("Finding positions.")
    p = np.empty((bodies.getSize(), statesTraj.getSize(), 3))

    # realize position for all states
    for ii in tqdm(range(statesTraj.getSize())):
    #for ii in range(statesTraj.getSize()):
        model.realizePosition(statesTraj[ii])

    boneNameOrder = []
    for i in range(bodies.getSize()):
        body1 = bodies.get(i)
        body = bodies.updComponent('/bodyset/' + str(body1.getName()))
        c = list(body.getComponentsList())
        
        bComp = None
        bComp = next((x for x in c if isinstance(x, opensim.simulation.PhysicalOffsetFrame)), None)
        if (bComp):
            #for ii in range(statesTraj.getSize()):
            for ii in tqdm(range(statesTraj.getSize())):
                p[i, ii, :] = bComp.getPositionInGround(statesTraj[ii]).to_numpy()
            boneNameOrder.append(body1.getName())
        del c

    return p, boneNameOrder

def getRotationAsNumpy(r):
    return np.array([
        [r.R().get(0, 0), r.R().get(0, 1), r.R().get(0, 2)],
        [r.R().get(1, 0), r.R().get(1, 1), r.R().get(1, 2)],
        [r.R().get(2, 0), r.R().get(2, 1), r.R().get(2, 2)]
        ])

def getRadiusReferenceRotationMatrix(modelInfo):
    modelInfo = rootFolderFix(modelInfo)

    filePath = getModelPath(modelInfo)
    print("Reading model file.")
    model = opensim.Model(filePath)
    state = model.initSystem()
    bodies = model.get_BodySet()

    rot = None
    for i in range(bodies.getSize()):
        body1 = bodies.get(i)
        body = bodies.updComponent('/bodyset/' + str(body1.getName()))
        c = list(body.getComponentsList())
        
        bComp = None
        bComp = next((x for x in c if isinstance(x, opensim.simulation.PhysicalOffsetFrame)), None)
        if (bComp and body1.getName() == 'radius'):
            rot = getRotationAsNumpy(body.getTransformInGround(state))
        del c

    return rot

def getBoneRotationsFromIK(modelInfo, trial):
    # relative to ground
    #modelInfo = rootFolderFix(modelInfo)

    filePath = getModelPath(modelInfo)
    print("Reading model file.")
    model = opensim.Model(filePath)
    state = model.initSystem()
    bodies = model.get_BodySet()

    #file = [x['file'] for x in dataIn if modelInfo['currentModel'] in x['file']][0]
    filePath = modelInfo['dataOutputDir'] + trial + r'_model_kinematics.sto'
    print("Reading states file.")
    statesTable = opensim.TimeSeriesTable(filePath)
    statesTable.addTableMetaDataString('inDegrees','no')
 
    #initialTime = statesTable.getIndependentColumn()[0]
    #finalTime = statesTable.getIndependentColumn()[-1]
    #duration = finalTime - initialTime
    print("Converting states time series table to StatesTrajectory.")
    statesTraj = opensim.StatesTrajectory.createFromStatesTable(
            model, statesTable, True, True, False)

    print("Finding rotations.")
    rot = np.empty((bodies.getSize(), statesTraj.getSize(), 3, 3))

    # realize position for all states
    for ii in tqdm(range(statesTraj.getSize())):
    #for ii in range(statesTraj.getSize()):
        model.realizePosition(statesTraj[ii])

    boneNameOrder = []
    for i in range(bodies.getSize()):
        body1 = bodies.get(i)
        body = bodies.updComponent('/bodyset/' + str(body1.getName()))
        c = list(body.getComponentsList())
        
        bComp = None
        bComp = next((x for x in c if isinstance(x, opensim.simulation.PhysicalOffsetFrame)), None)
        if (bComp):
            #for ii in range(statesTraj.getSize()):
            for ii in tqdm(range(statesTraj.getSize())):
                rot[i, ii, :, :] = getRotationAsNumpy(body.getTransformInGround(statesTraj[ii]))
            boneNameOrder.append(body1.getName())
        del c

    return rot, boneNameOrder

def getMarkerPositionsInModelGlobal(modelInfo):

    filePath = getModelPath(modelInfo)

    model = opensim.Model(filePath)
    markers = model.get_MarkerSet()
    state = model.initSystem()
    
    markerPositions = []
    for i in range(markers.getSize()):
        #name = markers.get('marker0').getParentFrameName()
        #name = name.split('/bodyset/')[1]
        markerPositions.append(markers.get('marker' + str(i)).getLocationInGround(state).to_numpy())
    return markerPositions

def setBonesInModel(modelInfo, jointsToDistances, openSimDistances):
    # doesn't set mesh_filename properly :(
    filePath = getModelPath(modelInfo)

    print('Setting bone locations in file: ' + str(filePath))
    model = opensim.Model(filePath)
    
    bodies = model.get_BodySet()
    bodiesToMeshNames = [
        modelInfo['names']['radius'],
        modelInfo['names']['scaphoid'],
        modelInfo['names']['lunate'],
        modelInfo['names']['metacarp3']
        ]
    for i in range(len(bodiesToMeshNames)):
        body1 = bodies.get(i)
        body = bodies.updComponent('/bodyset/' + str(body1.getName()))
        c = list(body.getComponentsList())
        
        bComp = None
        bComp = next((x for x in c if isinstance(x, opensim.simulation.PhysicalOffsetFrame)), None)
        if (bComp):
            bComp2 = body.updComponent(bComp.getName())
            bGeom = bComp2.upd_attached_geometry(0)
            bMesh = bGeom.updPropertyByName('mesh_file')
            bMesh = bodiesToMeshNames[i] + '.stl'
            #bGeom.updPropertyByName('mesh_file') = bodiesToMeshNames[i] + '.stl'
        del c

    #jointsToDistances = [
    #    'radial_scaphoid',
    #    'scaphoid_lunate',
    #    'hand_wrist'
    #]
    joints = model.get_JointSet()
    for i in range(joints.getSize()):
        jointTemp = joints.get(i)
        if (jointTemp.getName() in jointsToDistances):
            index = jointsToDistances.index(jointTemp.getName())
            frame = jointTemp.upd_frames(0)
            #frame.getName() should be name of the first frame
            frame.set_translation(
                opensim.Vec3(
                    openSimDistances[index][0],
                    openSimDistances[index][1],
                    openSimDistances[index][2]
                    )
                )

    if (not model.printToXML(filePath)):
        print('Error saving model file!')

def setMarkersInModel(modelInfo, openSimDistances):
    plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
    
    m = []
    m.extend(plateAssignment[modelInfo['names']['lunate']])
    m.extend(plateAssignment[modelInfo['names']['scaphoid']])
    m.extend(plateAssignment[modelInfo['names']['radius']])
    m.extend(plateAssignment[modelInfo['names']['metacarp3']])

    filePath = getModelPath(modelInfo)

    print('Setting markers in file: ' + str(filePath))
    model = opensim.Model(filePath)
    markers = model.get_MarkerSet()
    
    for i in range(len(openSimDistances)):
        markerTemp = markers.get('marker' + str(m[i]))
        markerTemp.set_location(opensim.Vec3(
            openSimDistances[i][0],
            openSimDistances[i][1],
            openSimDistances[i][2]
            ))
        #print('marker' + str(m[i]) + ' = ' \
        #+ str(openSimDistances[i][0]) + ' '\
        #+ str(openSimDistances[i][1]) + ' '\
        #+ str(openSimDistances[i][2]))
        if (m[i] in plateAssignment[modelInfo['names']['lunate']]):
            markerTemp.setParentFrameName('/bodyset/lunate')
        if (m[i] in plateAssignment[modelInfo['names']['scaphoid']]):
            markerTemp.setParentFrameName('/bodyset/scaphoid')
        if (m[i] in plateAssignment[modelInfo['names']['radius']]):
            markerTemp.setParentFrameName('/bodyset/radius')
        if (m[i] in plateAssignment[modelInfo['names']['metacarp3']]):
            markerTemp.setParentFrameName('/bodyset/hand_complete')

    if (not model.printToXML(filePath)):
        print('Error saving model file!')