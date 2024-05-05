# Author: Alastair Quinn 2022

from slil.common.data_configs import all_experiments

def settings():
    flip = {} # note this is specific to frames! meaning it's also specific to trials!
    possiblePinSet = {}
    alignmentMethod = {}
    mapMarkerChangesN2C = {}
    mapMarkerChangesN2S = {}
    for experiment in all_experiments(): # defaults
        alignmentMethod[experiment] = {
            'normal2cut': [
                'radius',
                'metacarp3'
            ],
            'normal2scaffold': [
                'radius',
                'metacarp3'
            ],
            'byPins_planned2normal': [ # default for all and should rarely be changed
                'lunate',
                'scaphoid'
            ],
            'byPins_normal2cut': [], # if not empty they will override other methods
            'byPins_normal2scaffold': [], # if not empty they will override other methods
            'pinDepth_normal': [],
            'pinDepth_normal2cut': {},
            'pinDepth_normal2scaffold': {},
            'cut2scaffold': [ # not actually used
                'radius'
            ],
        }
        possiblePinSet[experiment] = {
            'normal': {
                'lunate': [0, 1, 2, 3],
                'scaphoid': [0, 1, 2, 3],
                'radius': [0, 1, 2, 3],
                'metacarp3': [0, 1, 2, 3]
            },
            'cut': {
                'lunate': [0, 1, 2, 3],
                'scaphoid': [0, 1, 2, 3],
                'radius': [0, 1, 2, 3],
                'metacarp3': [0, 1, 2, 3]
            },
            'scaffold': {
                'lunate': [0, 1, 2, 3],
                'scaphoid': [0, 1, 2, 3],
                'radius': [0, 1, 2, 3],
                'metacarp3': [0, 1, 2, 3]
            }
        }
        mapMarkerChangesN2C[experiment] = {}
        mapMarkerChangesN2S[experiment] = {}

    flip['11524'] = {
        'scaffold': {
            #'metacarp3',
            #'radius'
        }
    }
    alignmentMethod['11524']['normal2scaffold'] = [
        #'radius',
        'metacarp3'
    ]
    mapMarkerChangesN2C['11524'] = {
        #'scaphoid': 3,
    }
    possiblePinSet['11524']['normal'] = {
        'lunate': [0, 1],
        'scaphoid': [0, 1],
        'radius': [2, 3],
        'metacarp3': [0, 1]
    }
    possiblePinSet['11524']['cut'] = {
        'lunate': [0, 1],
        'scaphoid': [0, 1],
        'radius': [2, 3],
        'metacarp3': [0, 1]
    }
    possiblePinSet['11524']['scaffold'] = {
        'lunate': [0, 1],
        'scaphoid': [0, 1],
        'radius': [0, 1],
        'metacarp3': [0, 1]
    }
    alignmentMethod['11524']['pinDepth_normal'] = {
        'lunate': 5.0,
        'scaphoid': 5.0
    }
    # 11524 possible manual adjustment
    # use
    # scene.setOffset(-20, 15, -10, 0, 0, 20)

    flip['11525'] = {
        'cut': {
            'radius',
            #'lunate'
        },
        'scaffold': {
            'lunate',
            #'radius'
        }
    }
    alignmentMethod['11525']['byPins_planned2normal'] = [
        'lunate',
        'scaphoid',
        'radius'
    ]
    alignmentMethod['11525']['normal2cut'] = [
        'radius',
        #'metacarp3'
    ]
    alignmentMethod['11525']['cut2scaffold'] = [
        #'radius',
        'metacarp3'
    ]
    possiblePinSet['11525']['normal'] = {
        'lunate': [0, 1],
        'scaphoid': [2, 3],
        'radius': [0, 1],
        'metacarp3': [0, 1]
    }
    possiblePinSet['11525']['cut'] = {
        'lunate': [0, 1],
        'scaphoid': [2, 3],
        'radius': [2,3],
        'metacarp3': [0, 1]
    }
    possiblePinSet['11525']['scaffold'] = {
        'lunate': [2, 3],
        'scaphoid': [2, 3],
        'radius': [0, 1],
        'metacarp3': [0, 1]
    }
    mapMarkerChangesN2C['11525'] = {
        'lunate': 3,
    }

    flip['11526'] = {
        'cut': {
            'metacarp3',
        },
        'scaffold': {
            #'metacarp3',
            'lunate',
            'radius'
        }
    }
    alignmentMethod['11526']['normal2cut'] = [
        #'radius',
        'metacarp3'
    ]
    alignmentMethod['11526']['normal2scaffold'] = [
        'radius',
        #'metacarp3'
    ]
    possiblePinSet['11526']['normal'] = {
        'lunate': [2, 3],
        'scaphoid': [2, 3],
        'radius': [0, 1],
        'metacarp3': [0, 1]
    }
    possiblePinSet['11526']['cut'] = {
        'lunate': [2, 3],
        'scaphoid': [2, 3],
        'radius': [0, 1],
        'metacarp3': [2, 3]
    }
    possiblePinSet['11526']['scaffold'] = {
        'lunate': [2, 3],
        'scaphoid': [2, 3],
        'radius': [2, 3],
        'metacarp3': [2, 3]
    }
    mapMarkerChangesN2S['11526'] = {
        'lunate': 2,
        'metacarp3': 3,
        #'scaphoid': 3,
    }
    alignmentMethod['11526']['byPins_normal2scaffold'] = [
        'lunate',
        'scaphoid'
    ]
    alignmentMethod['11526']['pinDepth_normal'] = {
        'lunate': 10.0,
        'scaphoid': 10.0
    }
    alignmentMethod['11526']['pinDepth_normal2scaffold'] = {
        'lunate': 5.0,
        'scaphoid': -5.0
    }


    flip['11527'] = {
        'cut': {
            'radius',
            'scaphoid'
        },
        'scaffold': {
            'radius',
            'scaphoid'
        },
    }
    alignmentMethod['11527']['normal2cut'] = [
        'radius',
        #'metacarp3'
    ]
    alignmentMethod['11527']['cut2scaffold'] = [
        #'radius',
        'metacarp3'
    ]
    possiblePinSet['11527']['normal'] = {
        'lunate': [2,3],
        'scaphoid': [3],
        'radius': [1],
        'metacarp3': [2]
    }
    possiblePinSet['11527']['cut'] = {
        'lunate': [3],
        'scaphoid': [0,1],
        'radius': [2,3],
        'metacarp3': [2]
    }
    possiblePinSet['11527']['scaffold'] = {
        'lunate': [3],
        'scaphoid': [0,1],
        'radius': [2,3],
        'metacarp3': [2]
    }
    mapMarkerChangesN2C['11527'] = {
        #'scaphoid': 3,
    }
    mapMarkerChangesN2S['11527'] = {
        #'scaphoid': 3,
    }
    alignmentMethod['11527']['pinDepth_normal2cut'] = {
        'lunate': -2.0,
        'scaphoid': -10.0
    }
    alignmentMethod['11527']['pinDepth_normal2scaffold'] = {
        'scaphoid': -10.0
    }
    # use
    # scene.setOffset(5, -9, -17, 13, -15, 3)


    alignmentMethod['11534']['normal2cut'] = [
        #'radius',
        'metacarp3'
    ]
    alignmentMethod['11534']['normal2scaffold'] = [
        #'radius',
        'metacarp3'
    ]
    possiblePinSet['11534']['normal'] = {
        'lunate': [3],
        'scaphoid': [3],
        'radius': [1],
        'metacarp3': [2]
    }
    possiblePinSet['11534']['cut'] = {
        'lunate': [3],
        'scaphoid': [3],
        'radius': [1],
        'metacarp3': [2]
    }
    possiblePinSet['11534']['scaffold'] = {
        'lunate': [3],
        'scaphoid': [3],
        'radius': [1],
        'metacarp3': [2]
    }
    alignmentMethod['11534']['pinDepth_normal'] = {
        'lunate': 10.0,
        'scaphoid': 10.0
    }
    alignmentMethod['11534']['byPins_normal2cut'] = [
        'lunate',
        'scaphoid'
    ]
    alignmentMethod['11534']['byPins_normal2scaffold'] = [
        'lunate',
        'scaphoid'
    ]
    alignmentMethod['11534']['pinDepth_normal2scaffold'] = {
        'lunate': -3.0,
    }


    flip['11535'] = {
        'scaffold': {
            'metacarp3',
            'lunate'
        }
    }
    alignmentMethod['11535']['normal2scaffold'] = [
        #'radius',
        #'metacarp3'
        'scaphoid'
    ]
    possiblePinSet['11535']['normal'] = {
        'lunate': [3],
        'scaphoid': [3],
        'radius': [1],
        'metacarp3': [2]
    }
    possiblePinSet['11535']['cut'] = {
        'lunate': [3],
        'scaphoid': [3],
        'radius': [1],
        'metacarp3': [2]
    }
    possiblePinSet['11535']['scaffold'] = {
        'lunate': [2, 3],
        'scaphoid': [2, 3],
        'radius': [1],
        'metacarp3': [0, 1]
    }
    alignmentMethod['11535']['pinDepth_normal'] = {
        'lunate': 8.0,
        'scaphoid': 8.0
    }
    mapMarkerChangesN2S['11535'] = {
        'lunate': 0, # or 3
        #'metacarp3': 3,
    }

    flip['11536'] = {
        'scaffold': {
            'lunate',
            #'metacarp3'
        }
    }
    alignmentMethod['11536']['normal2scaffold'] = [
        'radius',
        #'metacarp3'
    ]
    possiblePinSet['11536']['normal'] = {
        'lunate': [3],
        'scaphoid': [0, 1],
        'radius': [2, 3],
        'metacarp3': [0, 1]
    }
    possiblePinSet['11536']['cut'] = {
        'lunate': [3],
        'scaphoid': [0, 1],
        'radius': [2, 3],
        'metacarp3': [0, 1]
    }
    possiblePinSet['11536']['scaffold'] = {
        'lunate': [2, 3],
        'scaphoid': [0, 1],
        'radius': [2, 3],
        'metacarp3': [2, 3]
    }
    mapMarkerChangesN2S['11536'] = {
        'lunate': 3,
        'metacarp3': 3,
    }
    alignmentMethod['11536']['pinDepth_normal'] = {
        'lunate': 10.0,
        'scaphoid': 10.0
    }
    alignmentMethod['11536']['byPins_normal2scaffold'] = [
        'lunate',
        'scaphoid',
        'metacarp3'
    ]
    alignmentMethod['11536']['pinDepth_normal2scaffold'] = {
        'lunate': 7.0,
        'scaphoid': -5.0,
        'radius': 0.0,
        'metacarp3': -5.0
    }


    alignmentMethod['11537']['normal2scaffold'] = [
        'radius',
        'metacarp3'
    ]
    possiblePinSet['11537']['normal'] = {
        'lunate': [2],
        'scaphoid': [1],
        'radius': [0],
        'metacarp3': [0]
    }
    possiblePinSet['11537']['cut'] = {
        'lunate': [2],
        'scaphoid': [1],
        'radius': [0],
        'metacarp3': [0]
    }
    possiblePinSet['11537']['scaffold'] = {
        'lunate': [2],
        'scaphoid': [1],
        'radius': [0],
        'metacarp3': [0]
    }
    alignmentMethod['11537']['byPins_normal2scaffold'] = [
        'lunate',
        'metacarp3'
    ]

    flip['11538'] = {
        'scaffold': {
            #'lunate',
            #'metacarp3'
        }
    }
    possiblePinSet['11538']['normal'] = {
        'lunate': [0,1],
        'scaphoid': [0,1],
        'radius': [2,3],
        'metacarp3': [2,3]
    }
    possiblePinSet['11538']['cut'] = {
        'lunate': [0,1],
        'scaphoid': [0,1],
        'radius': [2,3],
        'metacarp3': [2,3]
    }
    possiblePinSet['11538']['scaffold'] = {
        'lunate': [0,1],
        'scaphoid': [0,1],
        'radius': [2,3],
        'metacarp3': [2,3]
    }
    alignmentMethod['11538']['normal2scaffold'] = [
        'lunate',
        'scaphoid',
        'radius',
        'metacarp3'
    ]
    alignmentMethod['11538']['pinDepth_normal'] = {
        'lunate': 3.0,
        'scaphoid': 3.0
    }
    alignmentMethod['11538']['pinDepth_normal2scaffold'] = {
        'lunate': -5.0
    }
    mapMarkerChangesN2S['11538'] = {
        #'lunate': 3,
        'metacarp3': 2,
    }

    possiblePinSet['11539']['normal'] = {
        'lunate': [0,1],
        'scaphoid': [2,3],
        'radius': [0,1],
        'metacarp3': [0,1]
    }
    possiblePinSet['11539']['cut'] = {
        'lunate': [0,1],
        'scaphoid': [2,3],
        'radius': [0,1],
        'metacarp3': [0,1]
    }
    possiblePinSet['11539']['scaffold'] = {
        'lunate': [0,1],
        'scaphoid': [2,3],
        'radius': [0,1],
        'metacarp3': [0,1]
    }
    alignmentMethod['11539']['pinDepth_normal2scaffold'] = {
        'lunate': -5.0
    }
    
    return flip, possiblePinSet, alignmentMethod, mapMarkerChangesN2C, mapMarkerChangesN2S
