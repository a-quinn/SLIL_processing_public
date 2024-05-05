from os import path

from pathlib import Path
def project_data_root():
    # this is an absolute folder path! Almost all the code uses this... so be careful!
    return r'C:\local\projects\SLIL_processing_public\data'
    dirRootOD = path.join(path.expandvars("%userprofile%"),"OneDrive - Griffith University")
    dirRootOD += r'\Projects\MTP - SLIL project\cadaver experiements'
    return dirRootOD

def outputFolderPath():
    return project_data_root() + '\\' + 'data_processed_0'

def checkBadFiles():
    import os
    searchIn = outputFolderPath()
    print('Looking for OneDrive duplicate files in: {}'.format(searchIn))
    for path, subdirs, files in os.walk(searchIn):
        for name in files:
            if os.path.getsize(os.path.join(path, name)) == 0:
                foundFile = os.path.join(path, name)
                foundFile = foundFile.split(searchIn)[1]
                print('File \'{}\' has zero size.'.format(foundFile))
            if 'PC' in name:
                foundFile = os.path.join(path, name)
                foundFile = foundFile.split(searchIn)[1]
                print('Found: {}'.format(foundFile))

def checkBadFiles2():
    import os
    searchIn = project_data_root() + r'\data_cleaned'
    print('Looking for OneDrive duplicate files in: {}'.format(searchIn))
    for path, subdirs, files in os.walk(searchIn):
        for name in files:
            fileSize = os.path.getsize(os.path.join(path, name))
            if fileSize >= 1_000_000 and fileSize <= 13_000_000: #3 bytes
                foundFile = os.path.join(path, name)
                foundFile = foundFile.split(searchIn)[1]
                print('File \'{}\' has {} bytes size.'.format(foundFile, os.path.getsize(os.path.join(path, name))))

def outputFolders():
    folders = {
        'root': outputFolderPath() + r'\outputs',
        'graphics': outputFolderPath() + r'\outputs' + r'\graphics',
        'pickles': outputFolderPath() + r'\outputs' + r'\pickles',
        'geometry': outputFolderPath() + r'\outputs' + r'\pickles' + r'\geometry'
    }
    checkFoldersExist(folders)
    return folders

def checkFoldersExist(checkPaths):
    import os
    for p in checkPaths:
        path = checkPaths[p]
        if not os.path.exists(path):
            print('Creating path {}'.format(path))
            os.makedirs(path)

def all_experiments():
    return [
        '11524', # actually not working...
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

def all_model_types():
    return [ 'normal', 'cut', 'scaffold']

def getModelPath(modelInfo, model = None):
    if model == None:
        model = modelInfo['currentModel']
    filePath = modelInfo['dataOutputDir'] + '\\' + 'wrist_' + model + '.osim'
    return filePath

def setup_opensim_model(modelInfo, model = None):
    import fileinput
    import shutil

    if model == None:
        model = modelInfo['currentModel']
        
    template_file = modelInfo['rootFolder'] + r'\wrist_template.osim'
    output_file = getModelPath(modelInfo, model)
    shutil.copy(template_file, output_file)

    replace_dict = {
        'CAD_template': modelInfo['experimentID'] + "_" + model,
        'rad_stl': modelInfo['names']['radius'],
        'sca_stl': modelInfo['names']['scaphoid'],
        'lun_stl': modelInfo['names']['lunate'],
        'mc3_stl': modelInfo['names']['metacarp3']
    }
    with fileinput.FileInput(output_file, inplace=True) as file:
        for line in file:
            for key in replace_dict:
                line = line.replace(key, replace_dict[key])
            print(line, end='')

def checkBaseOutputFoldersExist(experiments):
    data_output = Path(outputFolderPath())
    if not data_output.exists():
        print(f'Creating folder: {data_output}')
        data_output.mkdir()

    for experiment in experiments:
        modelInfo = load(experiment)
        trials = modelInfo['trialsRawData']
        data_folder = Path(modelInfo['dataOutputDir'])

        if not data_folder.exists():
            print(f'Creating folder: {data_folder}')
            data_folder.mkdir()

        for trial in trials:
            trial_folder = data_folder.joinpath(trial.split('\\')[1])
            if not trial_folder.exists():
                print(f'Creating folder: {trial_folder}')
                trial_folder.mkdir()
                
        geometry_folder = data_folder.joinpath('Geometry')
        if not geometry_folder.exists():
            print(f'Creating folder: {geometry_folder}')
            geometry_folder.mkdir()
            
        for model in all_model_types():
            setup_opensim_model(modelInfo, model)

    outputFolders()

def load(experimentID = '11535'):

    experimentID = str(experimentID)
    dataInputDir = project_data_root() + r'\data_cleaned'

    # default values
    modelInfo = {
        'experimentID': experimentID,
        'names': {
            'lunate': '',
            'scaphoid': '',
            'radius': '',
            'metacarp3': '',
            'boneplugLunate': 'lunate boneplug',
            'boneplugScaphoid': 'scaphoid boneplug'
        },
        'otherHandBones': []
    }
    
    modelInfo['rootFolder'] = project_data_root()
    modelInfo['rootModelFolder'] = modelInfo['rootFolder'] + r'\models_3_matic'
    modelInfo['modelFolder'] = modelInfo['rootModelFolder'] + r'\\' + experimentID
    modelInfo['graphicsInputDir'] = project_data_root() + r'\graphics_input'
    modelInfo['dataInputDir'] = dataInputDir + '\\' + experimentID + r'\vel 10 acc 10 dec 10 jer 3.6'
    modelInfo['dataOutputDir'] = outputFolderPath() + '\\' + experimentID

    modelInfo['3_matic_file'] = modelInfo['modelFolder'] + r'\version2.mxp'

    modelInfo['currentModel'] = 'normal'
    modelInfo['isLeftHand'] = False
    modelInfo['numMarkers'] = 12

    modelInfo['moCapPins'] = [
        'pin_scaphoid1',
        'pin_scaphoid2',
        'pin_lunate1',
        'pin_lunate2',
        'pin_radius1',
        'pin_radius2',
        'pin_metacarp31',
        'pin_metacarp32'
    ]

    modelInfo['trialsRawData_only_normal'] = [
        r'\normal_fe_40_40\log1',
        r'\normal_fe_40_40\log2',
        r'\normal_ur_30_30\log1',
        r'\normal_ur_30_30\log2'
        ]
    modelInfo['trialsRawData_only_cut'] = [
        r'\cut_fe_40_40\log1',
        r'\cut_fe_40_40\log2',
        r'\cut_ur_30_30\log1',
        r'\cut_ur_30_30\log2'
        ]
    modelInfo['trialsRawData_only_scaffold'] = [
        r'\scaffold_fe_40_40\log1',
        r'\scaffold_fe_40_40\log2',
        r'\scaffold_ur_30_30\log1',
        r'\scaffold_ur_30_30\log2'
        ]
    
    
    # this experiment was bad for many reasons...
    if (experimentID == '11524'):
        modelInfo['scaffoldBPtoBPlength'] = 12 #mm not correct, unknown

        modelInfo['isLeftHand'] = True
        
        # I don't think any of these are correct... meaning what was acutally used in the experiment
        modelInfo['sensorGuideName'] = '11526 Sensor Guide 2. 27.05.21'
        modelInfo['placedScaffoldName'] = 'Placed_scaffold_11526_Date190521'
        modelInfo['surgicalGuideSca'] = '11526 Guide S 21.05.21'
        modelInfo['surgicalGuideLun'] = '11526 Guide L 21.05.21'
        
        modelInfo['names']['lunate'] =    '11524_reg_lun_mirrored'
        modelInfo['names']['scaphoid'] =  '11524_reg_sca_mirrored'
        modelInfo['names']['radius'] =    '11524_reg_rad_mirrored'
        modelInfo['names']['metacarp3'] = '11524_reg_mc3_mirrored'
        modelInfo['otherHandBones'] = ['11524_reg_cap_mirrored']

        # left and right marker id's could be wrong way around
        modelInfo['plateAssignment_normal'] = {
            modelInfo['names']['lunate']: [8, 9, 7],
            modelInfo['names']['scaphoid']: [0, 3, 2],
            #modelInfo['names']['radius']: [8, 9, 5],
            modelInfo['names']['radius']: [11, 6, 10],
            modelInfo['names']['metacarp3']: [4, 5, 1],
        }
        modelInfo['plateAssignment_cut'] = {
            modelInfo['names']['lunate']: [8, 9, 7],
            modelInfo['names']['scaphoid']: [0, 3, 2],
            modelInfo['names']['radius']: [11, 6, 10],
            modelInfo['names']['metacarp3']: [4, 5, 1],
        }
        modelInfo['plateAssignment_scaffold'] = {
            modelInfo['names']['lunate']: [8, 9, 7],
            modelInfo['names']['scaphoid']: [0, 3, 2],
            modelInfo['names']['radius']: [5, 11, 10],
            modelInfo['names']['metacarp3']: [4, 6, 1],
        }

        modelInfo['trialsRawData_only_static'] = [
            r'\normal_static\log',
            r'\cut_static\log',
            r'\scaffold_static\log',
            r'\scaffold_static\log2',
            #r'\scaffold_manual_rotation\log1'
        ]
        modelInfo['trialsRawData'] = modelInfo['trialsRawData_only_normal'] + modelInfo['trialsRawData_only_cut'] + modelInfo['trialsRawData_only_scaffold'] + modelInfo['trialsRawData_only_static']


    if (experimentID == '11525'):
        modelInfo['scaffoldBPtoBPlength'] = 11.1 #mm

        modelInfo['sensorGuideName'] = '11525 Sensor Guide2. 05.13'
        modelInfo['placedScaffoldName'] = 'Placed_scaffold_v3'
        modelInfo['surgicalGuideSca'] = '11525 Guide.1 05.13 V3'
        modelInfo['surgicalGuideLun'] = '11525 Guide.2 05.13 V3'
        
        modelInfo['names']['lunate'] =    '11525_reg_lun'
        modelInfo['names']['scaphoid'] =  '11525_reg_sca'
        modelInfo['names']['radius'] =    '11525_reg_rad'
        modelInfo['names']['metacarp3'] = '11525_reg_mc3'
        modelInfo['otherHandBones'] = ['11525_reg_cap']

        modelInfo['plateAssignment_normal'] = {
            modelInfo['names']['lunate']: [10, 11, 3],
            modelInfo['names']['scaphoid']: [0, 2, 1],
            modelInfo['names']['radius']: [8, 9, 5],
            modelInfo['names']['metacarp3']: [7, 6, 4],
        }
        modelInfo['plateAssignment_cut'] = {
            modelInfo['names']['lunate']: [10, 3, 11],
            modelInfo['names']['scaphoid']: [0, 2, 1],
            modelInfo['names']['radius']: [8, 9, 5],
            modelInfo['names']['metacarp3']: [4, 6, 7],
        }
        modelInfo['plateAssignment_scaffold'] = {
            modelInfo['names']['lunate']: [3, 10, 11],
            modelInfo['names']['scaphoid']: [0, 2, 1],
            modelInfo['names']['radius']: [8, 9, 5],
            modelInfo['names']['metacarp3']: [4, 6, 7],
        }

        modelInfo['trialsRawData_only_static'] = [
            r'\normal_static\log1',
            r'\cut_static\log1',
            r'\scaffold_static\log1',
            r'\cut_manual_rotation\log1'
            ]
        modelInfo['trialsRawData'] = modelInfo['trialsRawData_only_normal'] + modelInfo['trialsRawData_only_cut'] + modelInfo['trialsRawData_only_scaffold'] + modelInfo['trialsRawData_only_static']


    if (experimentID == '11526'):
        modelInfo['scaffoldBPtoBPlength'] = 12.00 #mm

        modelInfo['isLeftHand'] = True

        modelInfo['sensorGuideName'] = '11526 Sensor Guide 2. 27.05.21'
        modelInfo['placedScaffoldName'] = 'Placed_scaffold_11526_Date190521'
        modelInfo['surgicalGuideSca'] = '11526 Guide S 21.05.21'
        modelInfo['surgicalGuideLun'] = '11526 Guide L 21.05.21'
        
        modelInfo['names']['lunate'] =    '11526_reg_lun'
        modelInfo['names']['scaphoid'] =  '11526_reg_sca'
        modelInfo['names']['radius'] =    '11526_reg_rad'
        modelInfo['names']['metacarp3'] = '11526_reg_mc3'
        modelInfo['otherHandBones'] = ['11526_reg_cap']

        modelInfo['plateAssignment_normal'] = {
            modelInfo['names']['lunate']: [8, 6, 7],
            modelInfo['names']['scaphoid']: [4, 10, 5],
            modelInfo['names']['radius']: [0, 9, 1],
            modelInfo['names']['metacarp3']: [3, 2, 11],
        }
        modelInfo['plateAssignment_cut'] = {
            modelInfo['names']['lunate']: [8, 6, 7],
            modelInfo['names']['scaphoid']: [4, 10, 5],
            modelInfo['names']['radius']: [0, 9, 1],
            modelInfo['names']['metacarp3']: [2, 3, 11],
        }
        modelInfo['plateAssignment_scaffold'] = {
            modelInfo['names']['lunate']: [3, 2, 11],
            modelInfo['names']['scaphoid']: [4, 10, 5],
            modelInfo['names']['radius']: [8, 6, 7],
            modelInfo['names']['metacarp3']: [9, 0, 1],
        }

        modelInfo['trialsRawData_only_static'] = [
            r'\normal_static\log1',
            #r'\cut_static\log_wrong',
            r'\cut_static\log_fromFE',
            r'\cut_static\log_wrong_fixed',
            r'\scaffold_static\log1',
            r'\scaffold_manual_rotation\log1',
            r'\default\log_default',
            ]
        modelInfo['trialsRawData'] = modelInfo['trialsRawData_only_normal'] + modelInfo['trialsRawData_only_cut'] + modelInfo['trialsRawData_only_scaffold'] + modelInfo['trialsRawData_only_static']


    if (experimentID == '11527'):
        modelInfo['scaffoldBPtoBPlength'] = 13.00 #mm

        modelInfo['sensorGuideName'] = ''
        modelInfo['placedScaffoldName'] = 'Placed_scaffold'
        modelInfo['surgicalGuideSca'] = '11527 Guide 1'
        modelInfo['surgicalGuideLun'] = '11527 Guide 2'

        modelInfo['moCapPins'] = [] # because Kaecee never gave me the marker pin file
        
        modelInfo['names']['lunate'] =    '11527_reg_lun'
        modelInfo['names']['scaphoid'] =  '11527_reg_sca'
        modelInfo['names']['radius'] =    '11527_reg_rad'
        modelInfo['names']['metacarp3'] = '11527_reg_mc3'
        modelInfo['otherHandBones'] = ['11527_reg_cap']

        # for normal
        modelInfo['plateAssignment_normal'] = {
            modelInfo['names']['lunate']: [2, 4, 5],
            modelInfo['names']['scaphoid']: [8, 7, 9],
            modelInfo['names']['radius']: [11, 6, 10],
            modelInfo['names']['metacarp3']: [0, 1, 3],
        }
        # for cut
        modelInfo['plateAssignment_cut'] = {
            modelInfo['names']['lunate']: [4, 10, 5],
            modelInfo['names']['scaphoid']: [11, 6, 2],
            modelInfo['names']['radius']: [8, 7, 9],
            modelInfo['names']['metacarp3']: [0, 1, 3],
        }
        # for scaffold
        modelInfo['plateAssignment_scaffold'] = {
            modelInfo['names']['lunate']: [4, 10, 5],
            modelInfo['names']['scaphoid']: [11, 6, 2],
            modelInfo['names']['radius']: [8, 7, 9],
            modelInfo['names']['metacarp3']: [0, 1, 3],
        }

        modelInfo['trialsRawData_only_normal'].append(r'\normal_ur_30_30\log3')
        modelInfo['trialsRawData_only_scaffold'].append(r'\scaffold_ur_30_30\log3')
        modelInfo['trialsRawData_only_static'] = [
            r'\normal_static\log1',
            r'\cut_static\log1',
            r'\scaffold_static\log1',
            #r'\scaffold_manual_rotation\log1' # didn't do?
            ]
        modelInfo['trialsRawData'] = modelInfo['trialsRawData_only_normal'] + modelInfo['trialsRawData_only_cut'] + modelInfo['trialsRawData_only_scaffold'] + modelInfo['trialsRawData_only_static']


    if (experimentID == '11534'):
        modelInfo['scaffoldBPtoBPlength'] = 10.00 #mm

        modelInfo['isLeftHand'] = True

        modelInfo['sensorGuideName'] = '11534 Sensor Guide'
        modelInfo['placedScaffoldName'] = 'Placed_scaffold_11534_Date310521'
        modelInfo['surgicalGuideSca'] = '11534 Drill Guide S'
        modelInfo['surgicalGuideLun'] = '11534 Drill Guide L'
        
        modelInfo['names']['lunate'] =    '11534_reg_lun'
        modelInfo['names']['scaphoid'] =  '11534_reg_sca'
        modelInfo['names']['radius'] =    '11534_reg_rad'
        modelInfo['names']['metacarp3'] = '11534_reg_mc3'
        modelInfo['otherHandBones'] = ['11534_reg_cap']

        modelInfo['plateAssignment_normal'] = {
            modelInfo['names']['lunate']: [8, 6, 7],
            modelInfo['names']['scaphoid']: [4, 10, 5],
            modelInfo['names']['radius']: [3, 2, 11],
            modelInfo['names']['metacarp3']: [0, 1, 9],
        }
        modelInfo['plateAssignment_cut'] = {
            modelInfo['names']['lunate']: [8, 6, 7],
            modelInfo['names']['scaphoid']: [4, 10, 5],
            modelInfo['names']['radius']: [3, 2, 11],
            modelInfo['names']['metacarp3']: [0, 1, 9],
        }
        modelInfo['plateAssignment_scaffold'] = {
            modelInfo['names']['lunate']: [8, 6, 7],
            modelInfo['names']['scaphoid']: [4, 10, 5],
            modelInfo['names']['radius']: [3, 2, 11],
            modelInfo['names']['metacarp3']: [0, 1, 9],
        }

        modelInfo['trialsRawData_only_static'] = [
            r'\normal_static\log1',
            r'\cut_static\log1',
            r'\scaffold_static\log1',
            r'\scaffold_manual_rotation\log1'
            ]
        modelInfo['trialsRawData'] = modelInfo['trialsRawData_only_normal'] + modelInfo['trialsRawData_only_cut'] + modelInfo['trialsRawData_only_scaffold'] + modelInfo['trialsRawData_only_static']


    if (experimentID == '11535'):
        fileToImport = modelInfo['dataOutputDir'] + r'\normal_static\log.c3d'
        #fileToImport = modelInfo['dataOutputDir'] + r'\cut_static\log.c3d'
        #fileToImport = modelInfo['dataOutputDir'] + r'\scaffold_static\log1.c3d'
        modelInfo['scaffoldBPtoBPlength'] = 12.25 #mm

        modelInfo['sensorGuideName'] = '11535 Sensor Guide'
        modelInfo['placedScaffoldName'] = 'Placed_scaffold_11535_Date310521'
        modelInfo['surgicalGuideSca'] = '11535 Drill Guide S'
        modelInfo['surgicalGuideLun'] = '11535 Drill Guide L'
        
        modelInfo['names']['lunate'] =    '11535_reg_lun'
        modelInfo['names']['scaphoid'] =  '11535_reg_sca'
        modelInfo['names']['radius'] =    '11535_reg_rad'
        modelInfo['names']['metacarp3'] = '11535_reg_mc3'
        modelInfo['otherHandBones'] = ['11535_reg_cap']

        # for normal and cut
        modelInfo['plateAssignment_normal'] = {
            modelInfo['names']['lunate']: [0, 1, 9],
            modelInfo['names']['scaphoid']: [4, 10, 5],
            modelInfo['names']['radius']: [3, 2, 11],
            modelInfo['names']['metacarp3']: [8, 6, 7],
        }
        modelInfo['plateAssignment_cut'] = {
            modelInfo['names']['lunate']: [0, 1, 9],
            modelInfo['names']['scaphoid']: [4, 10, 5],
            modelInfo['names']['radius']: [3, 2, 11],
            modelInfo['names']['metacarp3']: [8, 6, 7],
        }
        ## for scaffold
        modelInfo['plateAssignment_scaffold'] = {
            modelInfo['names']['lunate']: [0, 1, 9],
            modelInfo['names']['scaphoid']: [3, 2, 11],
            #modelInfo['names']['radius']: [4, 10, 5],
            modelInfo['names']['radius']: [10, 4, 5],
            #modelInfo['names']['metacarp3']: [8, 6, 7],
            modelInfo['names']['metacarp3']: [6, 8, 7],
        }

        modelInfo['trialsRawData_only_static'] = [
            r'\normal_static\log',
            r'\cut_static\log',
            r'\scaffold_static\log1',
            r'\scaffold_manual_rotation\log1'
            ]
        modelInfo['trialsRawData'] = modelInfo['trialsRawData_only_normal'] + modelInfo['trialsRawData_only_cut'] + modelInfo['trialsRawData_only_scaffold'] + modelInfo['trialsRawData_only_static']


    if (experimentID == '11536'):
        fileToImport = modelInfo['dataOutputDir'] + r'\normal_static\log.c3d'
        #fileToImport = modelInfo['dataOutputDir'] + r'\cut_static\log.c3d'
        #fileToImport = modelInfo['dataOutputDir'] + r'\scaffold_static\log1.c3d'
        modelInfo['isLeftHand'] = True
        modelInfo['scaffoldBPtoBPlength'] = 12.25 #mm

        modelInfo['sensorGuideName'] = '11536 Sensor Guide 26. 06. 21'
        modelInfo['placedScaffoldName'] = 'Placed_scaffold_11536_Date310521'
        modelInfo['surgicalGuideSca'] = '11536 Drill Guide S'
        modelInfo['surgicalGuideLun'] = '11536 Drill Guide L'
        
        modelInfo['names']['lunate'] =    '11536_reg_lun'
        modelInfo['names']['scaphoid'] =  '11536_reg_sca'
        modelInfo['names']['radius'] =    '11536_reg_rad'
        modelInfo['names']['metacarp3'] = '11536_reg_mc3'
        modelInfo['otherHandBones'] = ['11536_reg_cap']

        # for normal
        modelInfo['plateAssignment_normal'] = {
            modelInfo['names']['lunate']: [9, 7, 8],
            modelInfo['names']['scaphoid']: [11, 3, 2],
            modelInfo['names']['radius']: [4, 5, 6],
            modelInfo['names']['metacarp3']: [0, 1, 10],
        }
        modelInfo['plateAssignment_cut'] = {
            modelInfo['names']['lunate']: [9, 7, 8],
            modelInfo['names']['scaphoid']: [11, 3, 2],
            modelInfo['names']['radius']: [4, 5, 6],
            modelInfo['names']['metacarp3']: [0, 1, 10],
        }
        ## for scaffold
        modelInfo['plateAssignment_scaffold'] = {
            modelInfo['names']['lunate']: [1, 0, 10],
            modelInfo['names']['scaphoid']: [11, 3, 2],
            #modelInfo['names']['radius']: [7, 9, 8],
            modelInfo['names']['radius']: [9, 7, 8],
            modelInfo['names']['metacarp3']: [4, 5, 6],
        }
        
        modelInfo['trialsRawData_only_static'] = [
            r'\normal_static\log',
            r'\normal_static_after\log',
            r'\cut_static\log',
            r'\cut_static_after\log',
            r'\scaffold_static\log1',
            r'\scaffold_static_after\log1',
            r'\scaffold_manual_rotation\log1'
            ]
        modelInfo['trialsRawData'] = modelInfo['trialsRawData_only_normal'] + modelInfo['trialsRawData_only_cut'] + modelInfo['trialsRawData_only_scaffold'] + modelInfo['trialsRawData_only_static']

    if (experimentID == '11537'):
        modelInfo['scaffoldBPtoBPlength'] = 11.00 #mm
        
        modelInfo['sensorGuideName'] = '11537 Sensor Guide 26. 06. 21'
        modelInfo['placedScaffoldName'] = 'Placed_scaffold_11537_Date160621'
        modelInfo['surgicalGuideSca'] = '11537 Drill Guide S'
        modelInfo['surgicalGuideLun'] = '11537 Drill Guide L'
        
        modelInfo['names']['lunate'] =    '11537_reg_lun'
        modelInfo['names']['scaphoid'] =  '11537_reg_sca'
        modelInfo['names']['radius'] =    '11537_reg_rad'
        modelInfo['names']['metacarp3'] = '11537_reg_mc3'
        modelInfo['otherHandBones'] = ['11537_reg_cap']

        # for normal
        modelInfo['plateAssignment_normal'] = {
            modelInfo['names']['lunate']: [5, 3, 4],
            modelInfo['names']['scaphoid']: [11, 10, 2],
            modelInfo['names']['radius']: [9, 0, 1],
            modelInfo['names']['metacarp3']: [6, 7, 8],
        }
        # cut
        modelInfo['plateAssignment_cut'] = {
            modelInfo['names']['lunate']: [5, 3, 4],
            modelInfo['names']['scaphoid']: [11, 10, 2],
            modelInfo['names']['radius']: [9, 0, 1],
            modelInfo['names']['metacarp3']: [6, 7, 8],
        }
        # for scaffold
        modelInfo['plateAssignment_scaffold'] = {
            modelInfo['names']['lunate']: [5, 3, 4],
            modelInfo['names']['scaphoid']: [11, 10, 2],
            modelInfo['names']['radius']: [9, 0, 1],
            modelInfo['names']['metacarp3']: [6, 7, 8],
        }
        
        modelInfo['trialsRawData_only_static'] = [
            r'\normal_static\log1',
            r'\normal_static_after\log1',
            r'\cut_static\log1',
            r'\cut_static_after\log1',
            r'\scaffold_static\log1',
            r'\scaffold_static_after\log1',
            r'\scaffold_manual_rotation\log1'
            ]
        modelInfo['trialsRawData'] = modelInfo['trialsRawData_only_normal'] + modelInfo['trialsRawData_only_cut'] + modelInfo['trialsRawData_only_scaffold'] + modelInfo['trialsRawData_only_static']

    if (experimentID == '11538'):
        fileToImport = modelInfo['dataOutputDir'] + r'\normal_static\log.c3d'
        #fileToImport = modelInfo['dataOutputDir'] + r'\cut_static\log.c3d'
        #fileToImport = modelInfo['dataOutputDir'] + r'\scaffold_static\log1.c3d'
        modelInfo['isLeftHand'] = True
        modelInfo['scaffoldBPtoBPlength'] = 12.25 #mm
        
        modelInfo['sensorGuideName'] = '11538 Sensor Guide'
        modelInfo['placedScaffoldName'] = 'Placed_scaffold_11538_Date170621'
        modelInfo['surgicalGuideSca'] = '11538 Drill Guide S'
        modelInfo['surgicalGuideLun'] = '11538 Drill Guide L'
        
        modelInfo['names']['lunate'] =    '11538_reg_lun'
        modelInfo['names']['scaphoid'] =  '11538_reg_sca'
        modelInfo['names']['radius'] =    '11538_reg_bones_rad'
        modelInfo['names']['metacarp3'] = '11538_reg_bones_mc3'
        modelInfo['otherHandBones'] = ['11538_reg_bones_cap']

        # for normal
        modelInfo['plateAssignment_normal'] = {
            modelInfo['names']['lunate']: [0, 1, 10],
            modelInfo['names']['scaphoid']: [11, 3, 2],
            modelInfo['names']['radius']: [4, 5, 6],
            modelInfo['names']['metacarp3']: [9, 7, 8],
        }
        # cut
        modelInfo['plateAssignment_cut'] = {
            modelInfo['names']['lunate']: [0, 1, 10],
            modelInfo['names']['scaphoid']: [11, 3, 2],
            modelInfo['names']['radius']: [4, 5, 6],
            modelInfo['names']['metacarp3']: [9, 7, 8],
        }
        # for scaffold
        modelInfo['plateAssignment_scaffold'] = {
            modelInfo['names']['lunate']: [7, 9, 8],
            modelInfo['names']['scaphoid']: [11, 3, 2],
            modelInfo['names']['radius']: [4, 5, 6],
            modelInfo['names']['metacarp3']: [1, 0, 10],
        }
        
        #modelInfo['trialsRawData_only_scaffold'].append(r'\scaffold_ur_30_30\log3')
        modelInfo['trialsRawData_only_static'] = [
            #r'\normal_static\log', # missing marker 8
            r'\normal_static_after\log',
            r'\cut_static\log',
            r'\cut_static_after\log',
            r'\scaffold_static\log1',
            r'\scaffold_static_after\log1',
            r'\scaffold_manual_rotation\log1'
            ]
        modelInfo['trialsRawData'] = modelInfo['trialsRawData_only_normal'] + modelInfo['trialsRawData_only_cut'] + modelInfo['trialsRawData_only_scaffold'] + modelInfo['trialsRawData_only_static']

    if (experimentID == '11539'):
        modelInfo['scaffoldBPtoBPlength'] = 12.00 #mm
        
        modelInfo['sensorGuideName'] = '11539 Sensor Guide 26. 06. 21'
        modelInfo['placedScaffoldName'] = 'Placed_scaffold_11539_Date100621_vs1'
        modelInfo['surgicalGuideSca'] = '11539 V1 Drill Guide S'
        modelInfo['surgicalGuideLun'] = '11539 V1 Drill Guide L'
        
        modelInfo['names']['lunate'] =    '11539_reg_lun'
        modelInfo['names']['scaphoid'] =  '11539_reg_sca'
        modelInfo['names']['radius'] =    '11539_reg_rad'
        modelInfo['names']['metacarp3'] = '11539_reg_mc3'
        modelInfo['otherHandBones'] = ['11539_reg_cap']

        # for normal
        modelInfo['plateAssignment_normal'] = {
            modelInfo['names']['lunate']: [6, 7, 8],
            modelInfo['names']['scaphoid']: [5, 4, 3],
            modelInfo['names']['radius']: [9, 0, 1],
            modelInfo['names']['metacarp3']: [11, 10, 2],
        }
        # cut
        modelInfo['plateAssignment_cut'] = {
            modelInfo['names']['lunate']: [6, 7, 8],
            modelInfo['names']['scaphoid']: [5, 4, 3],
            modelInfo['names']['radius']: [9, 0, 1],
            modelInfo['names']['metacarp3']: [11, 10, 2],
        }
        # for scaffold
        modelInfo['plateAssignment_scaffold'] = {
            #modelInfo['names']['lunate']: [5, 4, 3],
            modelInfo['names']['lunate']: [4, 5, 3],
            modelInfo['names']['scaphoid']: [6, 8, 7],
            modelInfo['names']['radius']: [9, 0, 1],
            modelInfo['names']['metacarp3']: [11, 10, 2],
        }
        
        modelInfo['trialsRawData_only_scaffold'].append(r'\scaffold_fe_40_40\log_old')
        modelInfo['trialsRawData_only_static'] = [
            r'\normal_static\log1',
            r'\normal_static\log2',
            r'\normal_static_after\log1',
            r'\cut_static\log1',
            r'\cut_static_after\log1',
            r'\scaffold_static\log1',
            r'\scaffold_static\log_old',
            r'\scaffold_static_after\log1',
            r'\scaffold_manual_rotation\log1'
            ]
        modelInfo['trialsRawData'] = modelInfo['trialsRawData_only_normal'] + modelInfo['trialsRawData_only_cut'] + modelInfo['trialsRawData_only_scaffold'] + modelInfo['trialsRawData_only_static']

    return modelInfo