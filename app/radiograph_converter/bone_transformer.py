# Author: Alastair Quinn 2023

import slil.common.math as fm
import slil.common.plotting_functions as pf

def convertRelativeToRadius_missing(dataIn):
    # originally imported data is missing translations and radius->ground
    for data in dataIn:
        dataK = data['kinematics']
        dataK['sca_xrot']     += dataK['uln_xrot']
        dataK['sca_yrot']     += dataK['uln_yrot']
        dataK['sca_zrot']     += dataK['uln_zrot']
        dataK['sca_xtran']    += dataK['uln_xtran']
        dataK['sca_ytran']    += dataK['uln_ytran']
        dataK['sca_ztran']    += dataK['uln_ztran']

        dataK['lunate_flexion']    += dataK['uln_xrot']
        dataK['lunate_deviation']  += dataK['uln_yrot']
        dataK['lunate_rotation']   += dataK['uln_zrot']
        #dataK['lunate_flexion']   += dataK['sca_xrot']
        #dataK['lunate_deviation'] += dataK['sca_yrot']
        #dataK['lunate_rotation']  += dataK['sca_zrot']
        dataK['lunate_xtrans']     += dataK['sca_xtran']
        dataK['lunate_ytrans']     += dataK['sca_ytran']
        dataK['lunate_ztrans']     += dataK['sca_ztran']

        dataK['hand_flexion']    += dataK['uln_xrot']
        dataK['hand_deviation']  += dataK['uln_yrot']
        dataK['hand_rotation']   += dataK['uln_zrot']
        #dataK['hand_flexion']   += dataK['sca_xrot']
        #dataK['hand_deviation'] += dataK['sca_yrot']
        #dataK['hand_rotation']  += dataK['sca_zrot']
        dataK['hand_xtrans']     += dataK['sca_xtran']
        dataK['hand_ytrans']     += dataK['sca_ytran']
        dataK['hand_ztrans']     += dataK['sca_ztran']

        trans = [
            'uln_xtran', 'uln_ytran', 'uln_ztran',
            'sca_xtran', 'sca_ytran', 'sca_ztran',
            'lunate_xtrans', 'lunate_ytrans', 'lunate_ztrans',
            'hand_xtrans', 'hand_ytrans', 'hand_ztrans'
            ]
        for tran in trans:
            dataK[tran] *= 1000.0

    return dataIn

def convertRelativeToRadius_all(dataK):
    dataK['sca_xrot']         += dataK['uln_xrot']
    dataK['sca_yrot']         += dataK['uln_yrot']
    dataK['sca_zrot']         += dataK['uln_zrot']
    dataK['sca_xtran']        += dataK['uln_xtran']
    dataK['sca_ytran']        += dataK['uln_ytran']
    dataK['sca_ztran']        += dataK['uln_ztran']
 
    dataK['lunate_flexion']   += dataK['sca_xrot']
    dataK['lunate_deviation'] += dataK['sca_yrot']
    dataK['lunate_rotation']  += dataK['sca_zrot']
    dataK['lunate_xtrans']    += dataK['sca_xtran']
    dataK['lunate_ytrans']    += dataK['sca_ytran']
    dataK['lunate_ztrans']    += dataK['sca_ztran']

    dataK['hand_flexion']     += dataK['sca_xrot']
    dataK['hand_deviation']   += dataK['sca_yrot']
    dataK['hand_rotation']    += dataK['sca_zrot']
    dataK['hand_xtrans']      += dataK['sca_xtran']
    dataK['hand_ytrans']      += dataK['sca_ytran']
    dataK['hand_ztrans']      += dataK['sca_ztran']

    trans = [
        'uln_xtran', 'uln_ytran', 'uln_ztran',
        'sca_xtran', 'sca_ytran', 'sca_ztran',
        'lunate_xtrans', 'lunate_ytrans', 'lunate_ztrans',
        'hand_xtrans', 'hand_ytrans', 'hand_ztrans'
        ]
    for tran in trans:
        dataK[tran] *= 1000.0

    return dataK

def scaleTranslation(dataK, scale = 1000.0):
    trans = [
        'uln_xtran', 'uln_ytran', 'uln_ztran',
        'sca_xtran', 'sca_ytran', 'sca_ztran',
        'lunate_xtrans', 'lunate_ytrans', 'lunate_ztrans',
        'hand_xtrans', 'hand_ytrans', 'hand_ztrans'
        ]
    for tran in trans:
        dataK[tran] *= scale

    return dataK

def getKinematics(experiment):
    from slil.common.cache import loadOnlyExps
    from copy import deepcopy
    expData = loadOnlyExps([experiment])[0]

    trialKinematics = {}

    dataUR = convertRelativeToRadius_missing(expData['dataUR'])
    #dataUR = expData['dataUR']
    for data in dataUR:
        trialKinematics.update({data['file']: data['kinematics']})

    dataFE = convertRelativeToRadius_missing(expData['dataFE'])
    #dataFE = expData['dataFE']
    for data in dataFE:
        trialKinematics.update({data['file']: data['kinematics']})

    # IK files were not generated for all these.
    #kinematicsExtra = []
    #fileExtModelKinematics = r'_model_kinematics.sto'
    #dataOutputDir = expData['dataOutputDir']
    #for session in expData['trialsRawData_only_static']:
    #    kinematicsExtra.append({'kinematics': pf.getDataIK(dataOutputDir + session + fileExtModelKinematics)})
    #
    #dataExtra = convertRelativeToRadius_missing(kinematicsExtra)
    #for data in dataExtra:
    #    trialKinematics.update({data['file']: data['kinematics']})

    return deepcopy(trialKinematics)
from numpy import eye, dot
import numpy as np

def rot3Matic2Qt():
    t1 = fm.createTransformationMatFromPosAndEuler(
        0, 0, 0, np.deg2rad(180), np.deg2rad(0), np.deg2rad(0))
    t2 = fm.createTransformationMatFromPosAndEuler(
        0, 0, 0, np.deg2rad(0), np.deg2rad(90), np.deg2rad(0))
    t3 = fm.createTransformationMatFromPosAndEuler(
        0, 0, 0, np.deg2rad(0), np.deg2rad(0), np.deg2rad(90))
    return np.dot(t3, np.dot(t1, t2))

def jointDef(translation, rotation):
    return fm.createTransformationMatFromPosAndEuler(
        translation[0],
        translation[1],
        translation[2],
        rotation[0],
        rotation[1],
        rotation[2]
    )

class BoneTransformer():
    def __init__(self, gui, experimentID):
        self.gui = gui

        self.bonesToMove = [
            'scaphoid',
            'lunate',
            'metacarp3',
            'radius'
        ]
        self.allBones = self.bonesToMove + ['capitate']
        self.boneFileNames = {}
        for boneName in self.bonesToMove:
            self.boneFileNames[boneName] = gui.scene.modelInfo['names'][boneName]
        self.boneFileNames['capitate'] = [x for x in gui.scene.modelInfo['otherHandBones'] if 'cap' in x][0]

        # convert all meshes to same orientation as OpenSim GUI
        convert3Matic2Qt = rot3Matic2Qt()
        for geometry in list(gui.scene.bonePolyData[gui.scene.plotL].keys()):
            gui.scene.bonePolyData[gui.scene.plotL][geometry].points = fm.transformPoints_1(
                gui.scene.bonePolyData[gui.scene.plotL][geometry].points,
                convert3Matic2Qt)
        
        #self.allTrialKinematics = getKinematics(experimentID)
        
        # COM was used as joint centers
        self.bonesCOM = {}
        for boneName in self.allBones:
            polydata = gui.scene.bonePolyData[gui.scene.plotL][self.boneFileNames[boneName]]
            self.bonesCOM[boneName] = fm.calcCOM(polydata.points, polydata.faces.reshape(-1, 4), method = 'mesh')
        
        # move all bones to world origin
        for boneName in self.allBones:
            polydata = gui.scene.bonePolyData[gui.scene.plotL][self.boneFileNames[boneName]]
            polydata.points = polydata.points - self.bonesCOM[boneName]

        from copy import deepcopy
        newOrigin = deepcopy(self.bonesCOM['radius'])
        for boneName in self.bonesCOM:
            self.bonesCOM[boneName] -= newOrigin
        
        # duplicate to use as original
        from copy import deepcopy
        self.geometryOriginal = {}
        for boneName in self.allBones:
            #if not self.boneFileNames[boneName]+'_original' in gui.scene.bonePolyData[gui.scene.plotL]:
            #    gui.scene.bonePolyData[gui.scene.plotL][self.boneFileNames[boneName]+'_original'] = deepcopy(gui.scene.bonePolyData[gui.scene.plotL][self.boneFileNames[boneName]])
            self.geometryOriginal[boneName] = np.array(deepcopy(gui.scene.bonePolyData[gui.scene.plotL][self.boneFileNames[boneName]].points))


        # this could be read from the .osim file
        zero = [0,0,0]
        self.joints = {}
        self.joints['ground_radius'] = jointDef(
            self.bonesCOM['radius'],
            [0, 0, -np.pi])
        self.joints['radius_scaphoid'] = jointDef(
            self.bonesCOM['scaphoid'] - self.bonesCOM['radius'],
            zero)
        self.joints['scaphoid_lunate'] = jointDef(
            self.bonesCOM['lunate'] - self.bonesCOM['scaphoid'],
            zero)
        self.joints['scaphoid_metacarp3'] = jointDef(
            self.bonesCOM['metacarp3'] - self.bonesCOM['scaphoid'],
            zero)
        self.joints['metacarp3_capitate'] = jointDef(
            self.bonesCOM['capitate'] - self.bonesCOM['metacarp3'],
            zero)
    
    def updateTrial(self, trial):
        self.transMats = {}
        
        kinematicsRaw = pf.getDataIK(
            self.gui.scene.modelInfo['dataOutputDir']
            + trial
            + r'_model_kinematics.sto')
        kinematicsRaw = scaleTranslation(kinematicsRaw, 1000.0)

        for boneName in self.bonesToMove:
            self.transMats[boneName], _, _ = fm.getTransformationMatrix(kinematicsRaw, boneName)
            #self.transMats[boneName], _, _ = fm.getTransformationMatrix(self.allTrialKinematics[trial], boneName)

    def getTinGlobal(self, boneName):
        # get all transforms down chain
        foundGround = False
        trans = []
        while not foundGround:
            jointName = [x for x in self.joints if '_'+boneName in x][0]
            trans.append(self.joints[jointName])
            jointParent = jointName.split('_')[0]
            if jointParent == 'ground':
                foundGround = True
            boneName = jointParent
        endTran = np.eye(4)
        for tran in trans:
            endTran = dot(tran, endTran)
        return endTran
    
    def getTinGlobalAtFrame(self, boneName, frame_index = 0):
        # get all transforms down chain
        foundGround = False
        trans = []
        while not foundGround:
            jointName = [x for x in self.joints if '_'+boneName in x][0]
            if boneName in self.bonesToMove:
                trans.append(dot(self.joints[jointName], self.transMats[boneName][frame_index]))
            else:
                trans.append(self.joints[jointName])
            jointParent = jointName.split('_')[0]
            if jointParent == 'ground':
                foundGround = True
            boneName = jointParent
        endTran = np.eye(4)
        for tran in trans:
            endTran = dot(tran, endTran)
            
        return endTran
        
    def defaultPose(self):
        for boneName in self.allBones:
            polyDataName = self.boneFileNames[boneName]
            #bonePD = self.gui.scene.bonePolyData[self.gui.scene.plotL][polyDataName+'_original'].points
            bonePD = self.geometryOriginal[boneName]
            
            tranGlobal = self.getTinGlobal(boneName)
            bonePointsMoved = fm.transformPoints_1(bonePD, tranGlobal)
            self.gui.scene.bonePolyData[self.gui.scene.plotL][polyDataName].points = bonePointsMoved
            #self.gui.scene.bonePolyData[self.gui.scene.plotL][polyDataName].points = bonePD

        self.gui.scene.plotter[0, self.gui.scene.plotL].render()

    def updateFrame(self, frameIndex):
        for boneName in self.allBones:
            polyDataName = self.boneFileNames[boneName]
            #bonePD = self.gui.scene.bonePolyData[self.gui.scene.plotL][polyDataName+'_original'].points
            bonePD = self.geometryOriginal[boneName]
            tranGlobal = self.getTinGlobalAtFrame(boneName, frameIndex)
            bonePointsMoved = fm.transformPoints_1(bonePD, tranGlobal)
            self.gui.scene.bonePolyData[self.gui.scene.plotL][polyDataName].points = bonePointsMoved
            
        self.gui.scene.plotter[0, self.gui.scene.plotL].render()
