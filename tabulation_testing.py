# Author: Alastair Quinn 2022
#%%
from slil.common.tabulation_compact import DiscreteResults_Compact
from slil.common.tabulation import DiscreteResults

#%%
dr = DiscreteResults()
#dr.insertSLILStrain('11525', 'normal', '0.9+-9', 9, 5)
#dr.insertRotation('11525', 'cut', 'Scaphoid', 'FE', '0.44', 9, 5)
#dr.insertRotation('11539', 'scaffold', 'Lunate', 'PS', '0.44', 'fff=-9', 5)
#dr.insertWristAngles('11539', 'scaffold', '0.44', 'fff=-9', 5)

experiments = [
    #'11525',
    #'11526',
    #'11527',
    '11534',
    #'11535',
    #'11536',
    '11537',
    #'11538',
    #'11539'
    ]

dr.generate(experiments)
dr.format('UR')
dr.format('FE')
from slil.common.data_configs import outputFolders
dr.save(outputFolders()['root'] + "\discrete results.xlsx")
#%%

def generate(modelInfo, dataIn):
    import numpy as np
    from copy import deepcopy
    motionType = dataIn[0]['type']
    dr.ws = dr.wb['UR']
    
experiments = [
    #'11525',
    #'11526',
    #'11527',
    #'11534',
    #'11535',
    #'11536',
    #'11537',
    '11538',
    #'11539'
    ]
from slil.common.cache import loadOnlyExps
models = loadOnlyExps(experiments)
for ind, modelInfo in enumerate(models):
    dataUR = modelInfo['dataUR']
    dataFE = modelInfo['dataFE']
    generate(modelInfo, dataUR)


#%%
dr.save("sample.xlsx")


# %%

import imgcompare

file = r"\11525_bone_kinematics_FE.png"
a = r"C:\Users\Griffith\OneDrive - Griffith University\Projects\MTP - SLIL project\cadaver experiements\Data processed_v5\outputs\graphics\individual experiments"
a += file

b = r"C:\Users\Griffith\OneDrive - Griffith University\Projects\MTP - SLIL project\cadaver experiements\Data processed_v5\outputs\graphics"
b += r"\11525_bone_kinematics_FE.png"
is_same = imgcompare.image_diff_percent(a, b)
is_same

imgcompare.is_equal(a, b)

# %% Old

experiments = [
    #'11525',
    #'11526',
    #'11527',
    #'11534',
    #'11535',
    #'11536',
    #'11537',
    '11538',
    #'11539'
    ]

drc = DiscreteResults_Compact()
drc.generate(experiments)
drc.save("sample_compact.xlsx")