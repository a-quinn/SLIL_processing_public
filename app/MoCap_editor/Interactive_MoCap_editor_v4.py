# Author: Alastair Quin 2021 - 2023
# use "Run Python FIle in Terminal"

from os import getcwd
from sys import path
path.append(getcwd()) # so slil module can be found

#from pickletools import string1
import PySimpleGUI as sg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fft import fft, fftfreq, fftshift
from scipy import signal
import scipy
import scipy.fftpack
from copy import deepcopy
import slil.filtering.filter_v4_functions as frm4f
import slil.common.plotting_functions as pf
import slil.common.data_configs as dc

# VARS CONSTS:
# Upgraded dataSize to global...
_VARS = {'window': False,
         'fig_agg': False,
         'pltFig': False,
         'pltFigCanvas': False,
         'pltFigAx': False,
         'fig_agg2': False,
         'pltFig2': False,
         'pltFig2Canvas': False,
         'pltFig2Ax': False,
         'markerX1': 20, # must be int
         'markerX2': 60, # must be int
         'markerY1': 0.0,
         'zoomX1': 0,
         'zoomX2': 10,
         'expCadID': 11537,
         'expMarker': 0,
         'expAxis': 'x',
         'expData': False,
         'dataInputDir': False,
         'dataFiles': False,
         'modelInfo': False,
         'maxX': 1000.0,
         'x': False,
         'y': False,
         'yToWrite': False,
         'nextBadpoint': 0.0,
         'isHorizontalTool': False,
         'horizontalToolOffset': 0.0,
         'horizontalToolAutoAdd': 0.0,
         'loadedFileContents': False
         }


#plt.style.use('Solarize_Light2')

# could auto generate but ehh
experiments = [
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


# Helper Functions


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


# \\  -------- PYSIMPLEGUI -------- //

AppFont = 'Any 16'
SliderFont = 'Any 14'
sg.theme('black')

class Canvas(FigureCanvasTkAgg):
    """
    Create a canvas for matplotlib pyplot under tkinter/PySimpleGUI canvas
    """
    def __init__(self, figure=None, master=None):
        super().__init__(figure=figure, master=master)
        self.canvas = self.get_tk_widget()
        self.canvas.pack(side='top', fill='both', expand=1)

exps = sg.Listbox(experiments,
    key='-CADA-LIST-',size=(10,20))
col2 = sg.Listbox(['Select a CAD'],
    key='-TRIAL-LIST-',size=(35,20))
col3 = sg.Listbox([''],
    key='-MARK-LIST-',size=(10,20))
col4 = sg.Listbox([''],
    key='-AXIS-LIST-',size=(10,20))

layout = [[
    sg.Column([[
        sg.Column([[sg.Text('Cadaver Exp:')], [exps]],size=(80,400)),
        sg.Column([[sg.Text('Trial:')],[ col2]],size=(300,400)),
        sg.Column([[sg.Text('Marker:')],[ col3]],size=(80,400)),
        sg.Column([[sg.Text('Axis:')],[ col4]],size=(80,400))],
        [sg.Button('Check Next', font=AppFont, pad=((100, 0), (0, 0)))],[
        #sg.Canvas(key='figCanvas2', background_color='#FDF6E3')
        sg.Graph((640, 480), (0, 0), (640, 480),
            key='figCanvas2', background_color='#FDF6E3')]
        ]),
    sg.Column([
        [
            sg.Text(text="Title",
                key='-TITLE-',
                font=SliderFont,
                background_color='#FDF6E3',
                pad=((0, 0), (10, 0)),
                text_color='Black'),
            sg.Button('Reload', font=AppFont, pad=((100, 0), (0, 0))),
            sg.Button('Check', font=AppFont, pad=((10, 0), (0, 0))),
            sg.Button('Save To Memory', font=AppFont, pad=((10, 0), (0, 0))),
            sg.Button('Save To Disk', font=AppFont, pad=((10, 0), (0, 0)))
            #sg.Button('Exit', font=AppFont, pad=((10, 0), (0, 0)))
        ],
        [
            sg.Column([
            [sg.Slider(range=(-0.15, 0.15), orientation='v', size=(34, 20),
                resolution=0.0001,
                default_value=_VARS['markerY1'],
                background_color='#FDF6E3',
                text_color='Black',
                key='-Slider3-',
                enable_events=True),
            sg.Slider(range=(-0.5, 0.5), orientation='v', size=(34, 20),
                resolution=0.0001,
                default_value=_VARS['horizontalToolOffset'],
                background_color='#FDF6E3',
                text_color='Black',
                key='-SliderHTool-',
                enable_events=True)
            ],
            [
                sg.Button('Zero', font=AppFont, pad=((10, 0), (0, 0))),
                sg.Input(default_text =str(0.0),
                    key='-SliderHToolMid-', size=(10,10))
            ]]),
            #sg.Canvas(key='figCanvas', background_color='#FDF6E3')
            sg.Graph((640, 480), (0, 0), (640, 480),
                key='figCanvas', background_color='#FDF6E3')
        ],
        [sg.Text(text="Zoom:    ",
                font=SliderFont,
                background_color='#FDF6E3',
                pad=((0, 0), (10, 0)),
                text_color='Black'),
        sg.Slider(range=(0, _VARS['maxX']), orientation='h', size=(50, 20),
                    resolution=1.0,
                    default_value=_VARS['zoomX1'],
                    background_color='#FDF6E3',
                    text_color='Black',
                    key='-SliderZoom1-',
                    enable_events=True),
        sg.Slider(range=(0, _VARS['maxX']), orientation='h', size=(50, 20),
                    resolution=1.0,
                    default_value=_VARS['zoomX2'],
                    background_color='#FDF6E3',
                    text_color='Black',
                    key='-SliderZoom2-',
                    enable_events=True)
        ],
        [sg.Text(text="Markers: ",
                font=SliderFont,
                background_color='#FDF6E3',
                pad=((0, 0), (10, 0)),
                text_color='Black'),
        sg.Slider(range=(0, _VARS['maxX']), orientation='h', size=(50, 20),
                    resolution=1.0,
                    default_value=_VARS['markerX1'],
                    background_color='#FDF6E3',
                    text_color='Black',
                    key='-Slider1-',
                    enable_events=True),
        sg.Slider(range=(0, _VARS['maxX']), orientation='h', size=(50, 20),
                    resolution=1.0,
                    default_value=_VARS['markerX2'],
                    background_color='#FDF6E3',
                    text_color='Black',
                    key='-Slider2-',
                    enable_events=True)
        ],
        [sg.Input(default_text =str(_VARS['markerX1']),
            key='-T-Slider1-', size=(10,10)),
        sg.Input(default_text =str(_VARS['markerX2']),
            key='-T-Slider2-', size=(10,10)),
        sg.Text(text="Toggle Horizontal Tool:",
                font=SliderFont,
                background_color='#FDF6E3',
                pad=((0, 0), (10, 0)),
                text_color='Black'),
        sg.Button('Off', size=(3,1), button_color=('white', 'red'), key='-ToggleHTool-'),
        sg.Button('Update',
                    font=AppFont,
                    pad=((4, 0), (10, 0))),
        sg.Button('To Next Bad Point',
                    font=AppFont,
                    pad=((4, 0), (10, 0))),
        sg.Button('Horizontal Auto Add',
                    font=AppFont,
                    pad=((4, 0), (10, 0)))]
    ],size=(1200,900))
]]

_VARS['window'] = sg.Window('Random Samples',
                            layout,
                            finalize=True,
                            resizable=True,
                            location=(100, 100),
                            element_justification="center",
                            background_color='#FDF6E3')


def drawChart():

    _VARS['pltFig'] = plt.figure(figsize=[10,6])
    _VARS['pltFigAx'] =[ _VARS['pltFig'].add_subplot(211),
        _VARS['pltFig'].add_subplot(212)]
    _VARS['pltFigAx'][0].legend(loc='right')
    _VARS['pltFigAx'][0].set(ylabel='Position (m)')
    _VARS['pltFig'].tight_layout()

    sg_canvas = _VARS['window']['figCanvas'].Widget
    _VARS['pltFigCanvas'] = Canvas(_VARS['pltFig'], sg_canvas)
    #_VARS['fig1_ax1'] = plt.subplot(2,1,1)
    #_VARS['fig1_ax2']  = plt.subplot(2,1,2)
    #_VARS['pltFig'] = _VARS['fig1_ax1'].get_figure()
    #dataXY = makeSynthData()
    #plt.plot(dataXY[0], dataXY[1], '.k')
    
    #plt.xlabel('Time (sec)')
    #_VARS['fig_agg'] = draw_figure(
    #    _VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])

    _VARS['pltFig2'] = plt.figure(figsize=[6,4])
    _VARS['pltFig2Ax'] =[
        _VARS['pltFig2'].add_subplot(311),
        _VARS['pltFig2'].add_subplot(312),
        _VARS['pltFig2'].add_subplot(313)
        ]
    _VARS['pltFig2'].tight_layout()
        
    sg_canvas2 = _VARS['window']['figCanvas2'].Widget
    _VARS['pltFig2Canvas'] = Canvas(_VARS['pltFig2'], sg_canvas2)
    #_VARS['fig_agg2'] = draw_figure(
    #    _VARS['window']['figCanvas2'].TKCanvas, _VARS['pltFig2'])

def updateExpList(cadaverID):
    _VARS['modelInfo'] = dc.load(cadaverID)
    _VARS['dataFiles'] = _VARS['modelInfo']['trialsRawData']
    _VARS['dataInputDir'] = _VARS['modelInfo']['dataInputDir']
    _VARS['window']['-TRIAL-LIST-'].Update(values=_VARS['dataFiles'])

def loadData(dataInStr):
    # dataInStr = '\\normal_ur_30_30\\log1'
    print("Loading {} {}".format(_VARS['modelInfo']['experimentID'], dataInStr))
    _VARS['loadedFileContents'] = pf.loadTest(_VARS['dataInputDir'] + dataInStr, skiprows = 0)
    _VARS['expData'] = pf.convertToNumpyArrays(_VARS['loadedFileContents'])
    _VARS['window']['-TITLE-'].Update(value="{} {} {} {}".format(_VARS['modelInfo']['experimentID'], dataInStr, _VARS['expMarker'], _VARS['expAxis']))
    updateMarkList()

def reloadData():
    _VARS['expData'] = pf.convertToNumpyArrays(_VARS['loadedFileContents'])

def updateMarkList():
    numMarkers = _VARS['modelInfo']['numMarkers'] # always 12
    _VARS['window']['-MARK-LIST-'].Update(values=[str(i) for i in range(1, numMarkers+1)])
    updateAxisList()

def updateAxisList():
    axies = ['x', 'y', 'z']
    _VARS['window']['-AXIS-LIST-'].Update(values=axies)

def updateChart():
    if (_VARS['expData']):
        updateChart2()

def modifyY(y):
    yModified = np.array(y)
    #print("{} {}".format(_VARS['markerX1'], _VARS['markerX2']))
    if (_VARS['isHorizontalTool'] and _VARS['markerY1'] != 0.0):
        if (_VARS['markerY1'] > 0):
            inds = np.where(yModified[_VARS['markerX1']:_VARS['markerX2']] < _VARS['horizontalToolOffset'])[0]
            yModified[_VARS['markerX1'] + inds] = yModified[_VARS['markerX1'] + inds] + _VARS['markerY1']
        else:
            inds = np.where(yModified[_VARS['markerX1']:_VARS['markerX2']] > _VARS['horizontalToolOffset'])[0]
            yModified[_VARS['markerX1'] + inds] = yModified[_VARS['markerX1'] + inds] + _VARS['markerY1']
    else:
        yModified[_VARS['markerX1']:_VARS['markerX2']] = yModified[_VARS['markerX1']:_VARS['markerX2']] + _VARS['markerY1']
    return yModified

def updateSliderRange(m):
    _VARS['maxX'] = m
    _VARS['window']['-SliderZoom1-'].Update(range=(0, m))
    _VARS['window']['-SliderZoom2-'].Update(range=(0, m))
    updateSliders()

def updateSliders():
    _VARS['window']['-Slider1-'].Update(range=(_VARS['zoomX1'], _VARS['zoomX2']))
    _VARS['window']['-Slider2-'].Update(range=(_VARS['zoomX1'], _VARS['zoomX2']))

def loadData2():
    _VARS['window']['-TITLE-'].Update(value="{} {} {} {}".format(_VARS['modelInfo']['experimentID'], prevTrial, _VARS['expMarker'], _VARS['expAxis']))
    _VARS['x'] = _VARS['expData']['time']
    _VARS['x'] = np.array(range(0,len(_VARS['expData']['time']),),dtype=float)
    updateSliderRange(len(_VARS['expData']['time']))
    name = 'marker/1_'+str(_VARS['expMarker'])+'/' + _VARS['expAxis']
    _VARS['y'] = _VARS['expData'][name]
    updateChart()

def updateChart22():
    #_VARS['fig_agg2'].get_tk_widget().forget()

    # plt.cla()
    #plt.clf()
    #plt.close(_VARS['pltFig2'])
    #_VARS['pltFig2'].clear()

    time = _VARS['expData']['time']
    y = _VARS['yToWrite']

    y1 = deepcopy( y)
    
    yAdjustRepeating = frm4f.fun3(y1)
    y1 = yAdjustRepeating

    velocityLimit1 = 40 # meters/sec
    [y1,
    yDot,
    badDataPoints,
    badDataPointsPos,
    badDataPointsNeg,
    badDataPointsX,
    badDataPointsY] = frm4f.fun1(y1, time, velocityLimit = velocityLimit1)

    velocityLimit2 = 5 # meters/sec
    maxYDot = max(abs(yDot))
    if (max(abs(yDot)) > 5.0):

        y1 = y
        for i in range(0,2):
            valLimitLargest = max(abs(yDot.max()) , abs(yDot.min()))
            valLimitLargest = valLimitLargest * 0.99
            [y1,
            yDot,
            badDataPoints,
            badDataPointsPos,
            badDataPointsNeg,
            badDataPointsX,
            badDataPointsY] = frm4f.fun1(y1, time, velocityLimit = valLimitLargest)

        [y2,
        yDot2,
        badDataPoints2,
        badDataPoints2Pos,
        badDataPoints2Neg,
        badDataPoints2X,
        badDataPoints2Y] = frm4f.fun1(y1, time, velocityLimit = velocityLimit2)

        #yRemoveIgnore = fun2(y2)
        #y2 = yRemoveIgnore
        
        [y3,
        yDot3,
        badDataPoints3,
        badDataPoints3Pos,
        badDataPoints3Neg,
        badDataPoints3X,
        badDataPoints3Y] = frm4f.fun1(y2, time, velocityLimit = velocityLimit2)
    else:
        y3 = y
        yDot2 = yDot
        yDot3 = yDot
        badDataPoints2X = badDataPointsX
        badDataPoints2Y = badDataPointsY
        badDataPoints3X = badDataPointsX
        badDataPoints3Y = badDataPointsY

    y3 = frm4f.fun2(y3)

    samplingFreq = 250 #Hz
    lowpassCutoffFreq = 0.5 #Hz
    #filt = frm4f.CriticallyDampedFilter(sampFreq = samplingFreq, cutoffFreq=lowpassCutoffFreq)
    #y3In = deepcopy(y3)
    #y3In = np.insert(y3In, 0, [y3In[0] for i in range(20)], axis=0)
    #y3In = np.append(y3In, [y3In[-1] for i in range(20)], axis=0)
    #yfilt = filt.runFilter(y3In)
    #yfilt = yfilt[20:-20]

    NyquistFreq = (samplingFreq)/2
    b, a = signal.butter(2, lowpassCutoffFreq/NyquistFreq, 'lowpass', fs = 1.0)
    y3In = deepcopy(y3)
    y3In = np.insert(y3In, 0, [y3In[0] for i in range(20)], axis=0)
    y3In = np.append(y3In, [y3In[-1] for i in range(20)], axis=0)
    yfilt = signal.filtfilt(b, a, y3In)
    yfilt = yfilt[20:-20]
    
    rep = frm4f.getNumRepeating(y)
    repInv = np.logical_not(rep)
    mp = len(np.where(np.isclose(y[repInv], yfilt[repInv], 0.001, 0.0001))[0])
    
    r = len(np.where(rep)[0])
    #print("Plotting: {}".format(
    #    str(_VARS['expCadID']) + ' /1_' + str(_VARS['expMarker']) + '/' + _VARS['expAxis'] +
    #    ' Matching points: ' + str(mp) + ' of ' + str(len(y))) +
    #    ' with missing ' + str((mp / (len(y) - r))*100)+ '%')
    

    #ax1 =  _VARS['pltFig2'].get_subplot(111)

    #_VARS['pltFig2'] = plt.figure(figsize=[8,4])
    #ax1 = _VARS['pltFig2'].add_subplot(211)
    #ax2 = _VARS['pltFig2'].add_subplot(212)
    ax1 = _VARS['pltFig2Ax'][0]
    ax2 = _VARS['pltFig2Ax'][1]
    ax3 = _VARS['pltFig2Ax'][2]
    ax1.cla()
    ax2.cla()
    ax3.cla()

    #ax =  _VARS['pltFig2'].get_axes()
    #print(ax)
    #ax1 = ax[0]

    time = _VARS['x']
    ax1.plot(time, y, linewidth=0.9, alpha=0.9, label='Raw')
    ax1.plot(time, yfilt, linewidth=0.9, alpha=0.9, label='Fixed')
    ax1.set_ylim( bottom = min(yfilt), top = max(yfilt) )
    ax1.set(ylabel='Position (m)')
    str1 = "{} /1_{}/{} MP: {} is {:.2f} %".format(_VARS['expCadID'], _VARS['expMarker'], _VARS['expAxis'], mp, (mp / (len(y) - r))*100)
    plt.title(str1)

    fs = 1.0/samplingFreq
    N = len(y)
   #yf = fft(y)
   #xf = fftfreq(N, fs)[:N//2]
   #ax2.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
   #ax2.set_ylim( bottom = -0.001, top = 0.001 )

    
    FFT = abs(scipy.fft.fft(y))
    freqs = scipy.fftpack.fftfreq(y.size, fs)
    #ax2.plot(freqs,20*np.lib.scimath.log10(FFT),'x')
    ax2.plot(freqs[:N//2], 20*np.lib.scimath.log10(FFT[0:N//2]))

    #f, t, Sxx = signal.spectrogram(y, fs)
    #ax2.pcolormesh(t, f, Sxx, shading='gouraud')
    #ax2.set(ylabel='Frequency [Hz]')
    #ax2.set(xlabel='Time [sec]')

    #ax2.plot(time[:-1], yDot2, linewidth=0.9, alpha=0.9, label='Raw')
    #ax2.axvline(x=_VARS['zoomX1'], color='green', alpha=0.4)
    #ax2.axvline(x=_VARS['zoomX2'], color='green', alpha=0.4)
    #ax2.axvline(x=_VARS['markerX1'], color='red', alpha=0.4)
    #ax2.axvline(x=_VARS['markerX2'], color='red', alpha=0.4)
    #sorter = np.argsort(_VARS['expData']['time'])
    #xInd = sorter[np.searchsorted(_VARS['expData']['time'], badDataPoints2X, sorter=sorter)]
    #ax2.plot(xInd, badDataPoints2Y, 'o', alpha=0.9, label='Raw bad points')
    #ax2.hlines([velocityLimit2, -1.0* velocityLimit2], 0, 1, transform=ax2.get_yaxis_transform(), colors='r')
    #ax2.legend(loc='right')
    #ax2.set(ylabel='Velocity (m/sec)')

    ax3.plot(time[:-1], yDot3, linewidth=0.9, alpha=0.9, label='Raw')
    ax3.axvline(x=_VARS['zoomX1'], color='green', alpha=0.4)
    ax3.axvline(x=_VARS['zoomX2'], color='green', alpha=0.4)
    ax3.axvline(x=_VARS['markerX1'], color='red', alpha=0.4)
    ax3.axvline(x=_VARS['markerX2'], color='red', alpha=0.4)
    sorter = np.argsort(_VARS['expData']['time'])
    xInd = sorter[np.searchsorted(_VARS['expData']['time'], badDataPoints3X, sorter=sorter)]
    ax3.plot(xInd, badDataPoints3Y, 'o', alpha=0.9, label='Raw bad points')
    ax3.hlines([velocityLimit2, -1.0* velocityLimit2], 0, 1, transform=ax3.get_yaxis_transform(), colors='r')
    ax3.legend(loc='right')
    ax3.set(ylabel='Velocity (m/sec)')

    _VARS['pltFig2'].tight_layout()
    _VARS['pltFig2Canvas'].draw()

    #_VARS['fig_agg2'] = draw_figure(
    #    _VARS['window']['figCanvas2'].TKCanvas, _VARS['pltFig2'])

    if len(xInd) > 0:
        _VARS['nextBadpoint'] = xInd[0]

def updateChart2():
    #_VARS['fig_agg'].get_tk_widget().forget()

    # plt.cla()
    #plt.clf()
    #plt.close(_VARS['pltFig'])
    #_VARS['pltFig'].clear()

    time = _VARS['x']
    y = _VARS['y']

    yMod=modifyY(y)
    _VARS['yToWrite'] = yMod

    #ax1 =  _VARS['pltFig'].get_subplot(211)
    #ax2 =  _VARS['pltFig'].get_subplot(212)

    #_VARS['pltFig'] = plt.figure(figsize=[10,6])
    #ax1 = _VARS['pltFig'].add_subplot(211)
    #ax2 = _VARS['pltFig'].add_subplot(212)
    ax1 = _VARS['pltFigAx'][0]
    ax2 = _VARS['pltFigAx'][1]
    ax1.cla()
    ax2.cla()

    ax1.plot(time, yMod, linewidth=0.9, alpha=0.9, label='Raw')

    ax2.plot(time[_VARS['zoomX1']:_VARS['zoomX2']], yMod[_VARS['zoomX1']:_VARS['zoomX2']], linewidth=0.9, alpha=0.9, label='Raw')
    #collection = collections.BrokenBarHCollection.span_where(
    #    time,
    #    ymin=ax2.get_ylim()[0],
    #    ymax=ax2.get_ylim()[1],
    #    where=badDataPointsVis,
    #    facecolor='green',
    #    alpha=0.7)
    #ax2.add_collection(collection)
    #ax2.plot(badDataPointsX, badDataPointsY, 'o', alpha=0.9, label='Raw bad points')
    #ax2.hlines([velocityLimit1, -1.0* velocityLimit1], 0, 1, transform=ax2.get_yaxis_transform(), colors='r')

    ax1.axvline(x=_VARS['zoomX1'], color='green', alpha=0.4)
    ax1.axvline(x=_VARS['zoomX2'], color='green', alpha=0.4)
    ax1.axvline(x=_VARS['markerX1'], color='red', alpha=0.4)
    ax1.axvline(x=_VARS['markerX2'], color='red', alpha=0.4)
    ax2.axvline(x=_VARS['markerX1'], color='red', alpha=0.4)
    ax2.axvline(x=_VARS['markerX2'], color='red', alpha=0.4)
    
    if (_VARS['isHorizontalTool']):
        ax2.hlines([_VARS['horizontalToolOffset']], 0, 1, transform=ax2.get_yaxis_transform(), colors='g')
        if (_VARS['markerY1'] != 0.0):
            yModified = deepcopy(y)
            if (_VARS['markerY1'] > 0):
                inds = np.where(yModified[_VARS['markerX1']:_VARS['markerX2']] < _VARS['horizontalToolOffset'])[0]
                yModified[_VARS['markerX1'] + inds] = yModified[_VARS['markerX1'] + inds] + _VARS['markerY1']
            else:
                inds = np.where(yModified[_VARS['markerX1']:_VARS['markerX2']] > _VARS['horizontalToolOffset'])[0]
                yModified[_VARS['markerX1'] + inds] = yModified[_VARS['markerX1'] + inds] + _VARS['markerY1']
            if len(inds) > 0:
                avg = 0.0
                avgi = 0
                for i in inds:
                    if (not (i - 1) in inds) and (i - 1) > 0:
                        avgi = avgi + 1
                        avg = avg + yModified[_VARS['markerX1'] + i - 1]
                    if (not (i + 1) in inds) and (_VARS['markerX1'] + i + 2) <= len(y):
                        avgi = avgi + 1
                        avg = avg + yModified[_VARS['markerX1'] + i + 1]
                if (avgi > 0):
                    avg = avg / avgi
                    avg2 = np.sum(y[_VARS['markerX1'] + inds])/len(inds)
                    #print("{} {} {}".format(avg, avg2, avg - avg2))
                    _VARS['horizontalToolAutoAdd'] = avg - avg2
                    ax2.plot(time[_VARS['markerX1'] + inds], y[_VARS['markerX1'] + inds] + (avg - avg2), linewidth=0.9, alpha=0.9, label='Raw')
    

    arr = yMod[ min(_VARS['zoomX1'], _VARS['markerX1']) : max(_VARS['zoomX2'], _VARS['markerX2'])]
    if len(arr) == 0:
        arr = [0]
    minY = min(arr)
    maxY = max(arr)
    ax2.set_ylim( bottom = minY, top = maxY )
    ax2.set_xlim( left = _VARS['zoomX1'], right = _VARS['zoomX2'] )
    #ax2.legend(loc='right')
    #ax2.set(ylabel='Velocity (m/sec)')

    _VARS['pltFigCanvas'].draw()
    #_VARS['fig_agg'] = draw_figure(
    #    _VARS['window']['figCanvas'].TKCanvas, _VARS['pltFig'])

def writeCSV():
    name = 'marker/1_'+str(_VARS['expMarker'])+'/' + _VARS['expAxis']
    #_VARS['expData'][name] = _VARS['yToWrite']
    #data = pd.DataFrame(_VARS['expData'])
    #data.to_csv( _VARS['dataInputDir'] + prevTrial + '_2.csv', index=False,  na_rep='NaN')
    #fileContents = pd.read_csv(_VARS['dataInputDir'] + prevTrial + '.csv', skiprows = 0)
    _VARS['loadedFileContents'][name] = _VARS['yToWrite']
    _VARS['loadedFileContents'].to_csv( _VARS['dataInputDir'] + prevTrial + '.csv', index=False,  na_rep='NaN')

def saveToMemory():
    name = 'marker/1_'+str(_VARS['expMarker'])+'/' + _VARS['expAxis']
    _VARS['loadedFileContents'][name] = _VARS['yToWrite']

def moveToNextBadPoint():
    if (_VARS['nextBadpoint'] > 0.0):
        _VARS['markerX1'] = _VARS['nextBadpoint'] - 5
        _VARS['zoomX1'] = _VARS['markerX1'] - 5
        _VARS['window']['-SliderZoom1-'].Update(value=_VARS['zoomX1'])
        updateMarkerSlider1(_VARS['markerX1'])
        _VARS['markerX2'] = _VARS['nextBadpoint'] + 5
        _VARS['zoomX2'] = _VARS['markerX2'] + 5
        _VARS['window']['-SliderZoom2-'].Update(value=_VARS['zoomX2'])
        updateMarkerSlider2(_VARS['markerX2'])
        updateVars()

def updateHToolSlider(v):
    _VARS['window']['-SliderHTool-'].Update(value=v)
    _VARS['window']['-SliderHTool-'].Update(range=(v + 0.5, v - 0.5))

def updateMarkerSlider1(v):
    _VARS['window']['-T-Slider1-'].Update(value=v)
    _VARS['window']['-Slider1-'].Update(value=v)

def updateMarkerSlider2(v):
    _VARS['window']['-T-Slider2-'].Update(value=v)
    _VARS['window']['-Slider2-'].Update(value=v)

def updateVars():
    if (_VARS['markerX1'] >= _VARS['markerX2']):
        _VARS['markerX1'] = _VARS['markerX2'] - 5
        updateMarkerSlider1(_VARS['markerX1'])
    if (_VARS['markerX1'] <= _VARS['zoomX1']-1):
        _VARS['zoomX1'] = _VARS['markerX1'] - 5
        _VARS['window']['-SliderZoom1-'].Update(value=_VARS['zoomX1'])
    
    if (_VARS['markerX2'] <= _VARS['markerX1']):
        _VARS['markerX2'] = _VARS['markerX1'] + 5
        updateMarkerSlider2(_VARS['markerX2'])
    if (_VARS['markerX2'] >= _VARS['zoomX2']):
        _VARS['zoomX2'] = _VARS['markerX2'] + 5
        _VARS['window']['-SliderZoom2-'].Update(value=_VARS['zoomX2'])

        
    if (_VARS['zoomX1'] > _VARS['markerX1']):
        _VARS['markerX1'] = _VARS['zoomX1'] + 5
        updateMarkerSlider1(_VARS['markerX1'])
    if (_VARS['zoomX1'] >= _VARS['zoomX2']):
        _VARS['zoomX1'] = _VARS['zoomX2'] - 5
        _VARS['window']['-SliderZoom1-'].Update(value=_VARS['zoomX1'])

    if (_VARS['zoomX2'] < _VARS['markerX2']):
        _VARS['markerX2'] = _VARS['zoomX2'] - 5
        updateMarkerSlider2(_VARS['markerX2'])
    if (_VARS['zoomX2'] <= _VARS['zoomX1']):
        _VARS['zoomX2'] = _VARS['zoomX1'] + 5
        _VARS['window']['-SliderZoom2-'].Update(value=_VARS['zoomX2'])
    updateSliders()

def reload():
    reloadData()
    loadData2()
    _VARS['window']['-MARK-LIST-'].Update(set_to_index=[i for i in range(1,13)].index(int(_VARS['expMarker'])))
    _VARS['window']['-AXIS-LIST-'].Update(set_to_index=['x', 'y', 'z'].index(_VARS['expAxis']))

drawChart()

prevCAD = ''
prevTrial = ''
prevMark = ''
prevAxis = ''
prevSlider1 = 0
prevSlider2 = 0
isDownToggleHTool = True
prevHToolMid = 0.0
# MAIN LOOP
while True:
    event, values = _VARS['window'].read(timeout=200)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    elif event == 'Reload':
        reload()
    elif event == '-ToggleHTool-':
        isDownToggleHTool = not isDownToggleHTool
        _VARS['window'].Element('-ToggleHTool-').Update(('Off','On')[isDownToggleHTool], button_color=(('white', ('red', 'green')[isDownToggleHTool])))
        _VARS['isHorizontalTool'] = isDownToggleHTool
        updateChart()
    elif event == 'Update':
        updateChart()
    elif event == 'Check':
        updateChart22()
    elif event == 'Save To Memory':
        saveToMemory()
        reload()
    elif event == 'Save To Disk':
        writeCSV()
        reload()
    elif event == 'Check Next':
        if 'x' in _VARS['expAxis']:
            _VARS['expAxis'] = 'y'
        elif 'y' in _VARS['expAxis']:
            _VARS['expAxis'] = 'z'
        elif 'z' in _VARS['expAxis']:
            _VARS['expAxis'] = 'x'
            if int(_VARS['expMarker']) == _VARS['modelInfo']['numMarkers']:
                
                _VARS['expMarker'] = str(1)
                if int(_VARS['dataFiles'].index(prevTrial)) + 1 >= len(_VARS['dataFiles']):
                    indCAD = experiments.index(prevCAD)+1
                    if indCAD >= len(experiments):
                        indCAD = 0
                    prevCAD = experiments[indCAD]
                    updateExpList(prevCAD)
                    _VARS['window']['-CADA-LIST-'].update(set_to_index=[indCAD], scroll_to_index=indCAD)
                    prevTrial = _VARS['dataFiles'][0]
                    _VARS['window']['-TRIAL-LIST-'].update(set_to_index=[0], scroll_to_index=0)
                else:
                    prevTrial = _VARS['dataFiles'][_VARS['dataFiles'].index(prevTrial) + 1]
                indTrial = _VARS['dataFiles'].index(prevTrial)
                _VARS['window']['-TRIAL-LIST-'].update(set_to_index=[indTrial], scroll_to_index=indTrial)
                loadData(prevTrial)
            else:
                _VARS['expMarker'] = str(int(_VARS['expMarker']) + 1)
            indMarker = int(_VARS['expMarker']) - 1
            _VARS['window']['-MARK-LIST-'].update(set_to_index=[indMarker], scroll_to_index=indMarker)
            
        indAxis = ['x', 'y', 'z'].index(_VARS['expAxis'])
        _VARS['window']['-AXIS-LIST-'].update(set_to_index=[indAxis], scroll_to_index=indAxis)
            
        loadData2()
        updateChart22()
    elif event == 'To Next Bad Point':
        moveToNextBadPoint()
    elif event == 'Horizontal Auto Add':
        if (_VARS['isHorizontalTool']):
            _VARS['markerY1'] = _VARS['horizontalToolAutoAdd']
            _VARS['window']['-Slider3-'].Update(value=_VARS['markerY1'])
            updateChart()
    elif event == 'Zero':
        _VARS['window']['-Slider3-'].Update(value=0.0)
        _VARS['markerY1'] = 0.0
        updateChart()
    elif event == '-Slider1-':
        _VARS['markerX1'] = int(values['-Slider1-'])
        updateVars()
        _VARS['window']['-T-Slider1-'].Update(value=_VARS['markerX1'])
        updateChart()
    elif event == '-SliderHTool-':
        _VARS['horizontalToolOffset'] = values['-SliderHTool-']
        if (_VARS['isHorizontalTool']):
            updateChart()
    elif event == '-T-Slider1-':
        _VARS['markerX1'] = int(values['-T-Slider1-'])
        updateMarkerSlider2(_VARS['markerX1'])
        updateChart()
    elif event == '-Slider2-':
        _VARS['markerX2'] = int(values['-Slider2-'])
        updateVars()
        _VARS['window']['-T-Slider2-'].Update(value=_VARS['markerX2'])
        updateChart()
    elif event == '-T-Slider2-':
        _VARS['markerX2'] = int(values['-T-Slider2-'])
        updateMarkerSlider2(_VARS['markerX2'])
        updateChart()
    elif event == '-Slider3-':
        _VARS['markerY1'] = values['-Slider3-']
        updateChart()
        # print(int(values['-Slider3-']))
    elif event == '-SliderZoom1-':
        _VARS['zoomX1'] = int(values['-SliderZoom1-'])
        updateVars()
        updateChart()
    elif event == '-SliderZoom2-':
        _VARS['zoomX2'] = int(values['-SliderZoom2-'])
        updateVars()
        updateChart()
    else:
        if (float(values['-SliderHToolMid-']) != prevHToolMid):
            prevHToolMid = float(values['-SliderHToolMid-'])
            updateHToolSlider(prevHToolMid)
        if (values['-T-Slider1-'] != prevSlider1):
            prevSlider1 = int(values['-T-Slider1-'])
            _VARS['markerX1'] = prevSlider1
            updateVars()
            updateMarkerSlider1(_VARS['markerX1'])

        if (values['-T-Slider2-'] != prevSlider2):
            prevSlider2 = int(values['-T-Slider2-'])
            _VARS['markerX2'] = prevSlider2
            updateVars()
            updateMarkerSlider2(_VARS['markerX2'])
            
        if (len(values['-CADA-LIST-']) > 0 ) and (values['-CADA-LIST-'][0] != prevCAD):
            prevCAD = values['-CADA-LIST-'][0]
            updateExpList(prevCAD)
            if (prevAxis != ''):
                loadData2()
        if (len(values['-TRIAL-LIST-']) > 0 ) and (values['-TRIAL-LIST-'][0] != prevTrial):
            prevTrial = values['-TRIAL-LIST-'][0]
            loadData(prevTrial)
            if (prevAxis != ''):
                loadData2()
        if (len(values['-MARK-LIST-']) > 0 ) and (values['-MARK-LIST-'][0] != prevMark):
            prevMark = values['-MARK-LIST-'][0]
            _VARS['expMarker'] = prevMark
            if (prevAxis != ''):
                loadData2()
        if (len(values['-AXIS-LIST-']) > 0 ) and (values['-AXIS-LIST-'][0] != prevAxis):
            prevAxis = values['-AXIS-LIST-'][0]
            _VARS['expAxis'] = prevAxis
            loadData2()
    #print(values)
    #if (event != '__TIMEOUT__'):
    #    print(event)
_VARS['window'].close()
