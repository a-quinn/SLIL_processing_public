#%%
# Author: Alastair Quinn 2021
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from copy import deepcopy
import slil.common.data_configs as dc
import slil.common.plotting_functions as pf

experiments = [
    #'11535',
    #'11536',
    '11537',
    #'11538',
    #'11539'
    ]
models = []
for i, experiment in enumerate(experiments):
    modelInfo = dc.load(experiment)
    models += [modelInfo]
# %%

for jj, model in enumerate(models):
    modelInfo = model
    dataOutputDir = modelInfo['dataOutputDir']
    dataInputDir = modelInfo['dataInputDir']

    fileExtBadpoints = r'_badPoints.sto'
    fileExtModelKinematics = r'_model_kinematics.sto'
    fileExtPosLunateBoneplug = r'_pos_lunate_boneplug.sto'
    fileExtPosScaphoidBoneplug = r'_pos_scaphoid_boneplug.sto'

    groupFE = {
        'files': [
            r'\cut_fe_40_40\log1',
            r'\cut_fe_40_40\log2',
            r'\normal_fe_40_40\log1',
            r'\normal_fe_40_40\log2',
            r'\scaffold_fe_40_40\log1',
            r'\scaffold_fe_40_40\log2',
        ],
        'names': [
            'cut',
            'cut',
            'normal',
            'normal',
            'scaffold',
            'scaffold',
        ]
    }

    dataFE = []
    i = 0
    for session in groupFE['files']:
        testTemp = pf.loadTest(dataInputDir + session)
        dataFE.append({
            'file': session,
            'title': groupFE['names'][i],
            #'robotData': testTemp,
            'time': testTemp['time'].to_numpy(),
            'rot': testTemp['feedback/rot/y'].to_numpy() *(180.0/np.pi),
            'lunate': pf.getData(dataOutputDir + session + fileExtPosLunateBoneplug),
            'scaphoid': pf.getData(dataOutputDir + session + fileExtPosScaphoidBoneplug),
            'kinematics': pf.getData(dataOutputDir + session + fileExtModelKinematics),
            'type': 'FE'
        })
        dataFE[i].update({'difference': pf.calcDif(dataFE[i]['lunate'],dataFE[i]['scaphoid'])})
        i += 1


    groupUR = {
        'files': [
            r'\cut_ur_30_30\log1',
            r'\cut_ur_30_30\log2',
            r'\normal_ur_30_30\log1',
            r'\normal_ur_30_30\log2',
            r'\scaffold_ur_30_30\log1',
            r'\scaffold_ur_30_30\log2',
        ],
        'names': [
            'cut',
            'cut',
            'normal',
            'normal',
            'scaffold',
            'scaffold',
        ]
    }

    dataUR = []
    i = 0
    for session in groupUR['files']:
        testTemp = pf.loadTest(dataInputDir + session)
        dataUR.append({
            'file': session,
            'title': groupUR['names'][i],
            #'robotData': testTemp,
            'time': testTemp['time'].to_numpy(),
            'rot': testTemp['feedback/rot/x'].to_numpy() *(180.0/np.pi),
            'lunate': pf.getData(dataOutputDir + session + fileExtPosLunateBoneplug),
            'scaphoid': pf.getData(dataOutputDir + session + fileExtPosScaphoidBoneplug),
            'kinematics': pf.getData(dataOutputDir + session + fileExtModelKinematics),
            'type': 'UR'
        })
        
        dataUR[i].update({'difference': pf.calcDif(dataUR[i]['lunate'],dataUR[i]['scaphoid'])})
        i += 1

    models[jj]['dataUR'] = dataUR
    models[jj]['dataFE'] = dataFE

def cropDataMinLoss(dataIn, cycleLength = 8333):
    cl = cycleLength
    tmpCrop = []
    for data in dataIn:
        
        # first run may not be good data
        tmp = deepcopy(data)
        tmp['difference'] = tmp['difference'][:cl]
        tmp['time'] = tmp['time'][:cl]
        tmp['rot'] = tmp['rot'][:cl]
        tmp['kinematics'] = tmp['kinematics'][:cl]
        tmp['kinematics']['time'] = tmp['kinematics']['time']
        tmpCrop.append(tmp)

        tmp = deepcopy(data)
        tmp['difference'] = tmp['difference'][cl:cl*2]
        tmp['time'] = tmp['time'][cl:cl*2] - tmp['time'][cl]
        tmp['rot'] = tmp['rot'][cl:cl*2]
        tmp['kinematics'] = tmp['kinematics'][cl:cl*2]
        tmp['kinematics']['time'] = tmp['kinematics']['time'] - tmp['kinematics']['time'][cl]
        tmpCrop.append(tmp)
        
        tmp = deepcopy(data)
        if (len(tmp['time'])>cl*3 and len(tmp['kinematics'])>cl*3):
            tmp['difference'] = tmp['difference'][cl*2:cl*3]
            tmp['time'] = tmp['time'][cl*2:cl*3] - tmp['time'][cl*2]
            tmp['rot'] = tmp['rot'][cl*2:cl*3]
            tmp['kinematics'] = tmp['kinematics'][cl*2:cl*3]
            tmp['kinematics']['time'] = tmp['kinematics']['time'] - tmp['kinematics']['time'][cl*2]
            tmpCrop.append(tmp)
        
        #tmp = deepcopy(data)
        #if (len(tmp['time'])>cl*4 and len(tmp['kinematics'])>cl*4):
        #    tmp['difference'] = tmp['difference'][cl*3:cl*4]
        #    tmp['time'] = tmp['time'][cl*3:cl*4] - tmp['time'][cl*3]
        #    tmp['rot'] = tmp['rot'][cl*3:cl*4]
        #    tmp['kinematics'] = tmp['kinematics'][cl*3:cl*4]
        #    tmp['kinematics']['time'] = tmp['kinematics']['time'] - tmp['kinematics']['time'][cl*3]
        #    tmpCrop.append(tmp)
    return tmpCrop


def differenceToStrain(dataIn):
    for data in dataIn:
        data['difference'] = (data['difference'] - data['difference'][0])/data['difference'][0]

for i, model in enumerate(models):

    dataUR = model['dataUR']
    dataFE = model['dataFE']
    pf.convertRelativeToRadius(dataUR)
    pf.convertRelativeToRadius(dataFE)
    differenceToStrain(dataUR)
    differenceToStrain(dataFE)
    #models[i]['dataURcropped'] = dataUR
    #models[i]['dataFEcropped'] = dataFE
    #models[i]['dataURcropped'] = cropData3(dataUR, offset = 2083, cycleLength = 8333)
    #models[i]['dataFEcropped'] = cropData3(dataFE, offset = 2083, cycleLength = 8333)
    #models[i]['dataURcropped'] = cropDataMinLoss(dataUR)
    #models[i]['dataFEcropped'] = cropDataMinLoss(dataFE)

#%%

for i, model in enumerate(models):

    dataUR = model['dataUR']
    dataFE = model['dataFE']
    #models[i]['dataURcropped'] = dataUR
    #models[i]['dataFEcropped'] = dataFE
    models[i]['dataURcropped'] = pf.cropData3(dataUR, offset = 2083, cycleLength = 4166)
    models[i]['dataFEcropped'] = pf.cropData3(dataFE, offset = 2083, cycleLength = 4166)

#%%
direction = 'FE'
#direction = 'UR'
columns = [
    'lunate_flexion', 'lunate_deviation', 'lunate_rotation',
    'sca_xrot', 'sca_yrot', 'sca_zrot',
    'hand_flexion', 'hand_deviation', 'hand_rotation',
]
#if left hand, need to flip these axies
flipForColumns = [
    'lunate_deviation', 'lunate_rotation',
    'sca_yrot', 'sca_zrot',
    'hand_deviation', 'hand_rotation',
]

all = {
    'normal': { },
    'cut': { },
    'scaffold': { }
}
for i, modelType in enumerate(all):
    for ii, c in enumerate(columns):
        all[modelType][c] = np.array([[]])

for i, model in enumerate(models):
    for data in model['data' + direction + 'cropped']:
        for ii, c in enumerate(columns):
            r1  = data['kinematics'][c].to_numpy()*(180.0/np.pi)
            if ((c in flipForColumns)):
                r1 = r1 * -1.0
            if ((c in flipForColumns) and model['isLeftHand']):
                r1 = r1 * -1.0
            if (len(all[data['title']][c][0]) == 0):
                all[data['title']][c] = [r1]
            else:
                all[data['title']][c] = np.append(all[data['title']][c], [r1], axis=0)

#%%

colors = ['green', 'red', 'blue']
fig = plt.figure()
fig.set_size_inches(14.5, 14.5)
gs0 = gridspec.GridSpec(4, 3, figure=fig)#, height_ratios=[3,3,1])
figAxies = [
    fig.add_subplot(gs0[0]),
    fig.add_subplot(gs0[1]),
    fig.add_subplot(gs0[2]),
    fig.add_subplot(gs0[3]),
    fig.add_subplot(gs0[4]),
    fig.add_subplot(gs0[5]),
    fig.add_subplot(gs0[6]),
    fig.add_subplot(gs0[7]),
    fig.add_subplot(gs0[8])
]

time = models[0]['data' + direction + 'cropped'][0]['time']
rot = models[0]['data' + direction + 'cropped'][0]['rot'] * -1.0

def subplotSD(ax, x, y, label, colour):
    ax.plot(x, y.mean(axis=0), alpha=0.5, color=colour, label=label, linewidth = 1.0)
    ax.fill_between(x,
        y.mean(axis=0) - 2*y.std(axis=0),
        y.mean(axis=0) + 2*y.std(axis=0),
        color=colour, alpha=0.2) 
    ax.legend(loc='best')
    ax.set_ylabel("Rotation (degrees)")
    ax.set_xlabel("Time (seconds")

for i, t in enumerate(all):
    if (t=='normal'):
        lineColour = colors[0]
    if (t=='cut'):
        lineColour = colors[1]
    if (t=='scaffold'):
        lineColour = colors[2]
    for ii, c in enumerate(columns):
        subplotSD(figAxies[ii], time, all[t][c], t, lineColour)


#%% 'difference'

direction = 'FE'
#direction = 'UR'

all = {
    'normal': [],
    'cut': [],
    'scaffold': []
}
for i, modelType in enumerate(all):
    all[modelType] = np.array([[]])

flip = False
for i, model in enumerate(models):
    for data in model['data' + direction + 'cropped']:
        r1  = data['difference']
        if (flip):
            r1 = r1[::-1]
            flip = False
        else:
            flip = True
        if (len(all[data['title']][i]) == 0):
            all[data['title']] = [r1]
        else:
            all[data['title']] = np.append(all[data['title']], [r1], axis=0)

#%%

colors = ['green', 'red', 'blue']
fig = plt.figure()
fig.set_size_inches(6, 6)
gs0 = gridspec.GridSpec(1, 1, figure=fig)#, height_ratios=[3,3,1])
figAxies = [
    fig.add_subplot(gs0[0])
]

time = models[0]['data' + direction + 'cropped'][0]['time']
#time = time[:4166]
rot = models[0]['data' + direction + 'cropped'][0]['rot'] * -1.0
#rot = rot[:4166]

def subplotSD(ax, x, y, label, colour):
    ax.plot(x, y.mean(axis=0), alpha=0.5, color=colour, label=label, linewidth = 1.0)
    ax.fill_between(x,
        y.mean(axis=0) - 2*y.std(axis=0),
        y.mean(axis=0) + 2*y.std(axis=0),
        color=colour, alpha=0.2) 
    ax.legend(loc='best')
    ax.set_ylabel("Strain (%)")
    ax.set_xlabel("Rotation (degrees)")

for i, t in enumerate(all):
    if (t=='normal'):
        lineColour = colors[0]
    if (t=='cut'):
        lineColour = colors[1]
    if (t=='scaffold'):
        lineColour = colors[2]
    subplotSD(figAxies[0], rot, all[t], t, lineColour)

# %%
