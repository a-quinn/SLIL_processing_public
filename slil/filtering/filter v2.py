import sys
import threading
import concurrent.futures
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import slil.common.filtering as ft
import slil.common.c3d
import copy

frames = []
count = 0
mocapFPS = 0
with open('take4.c3d', 'rb') as handle:
    reader = c3d.Reader(handle)
    lastFrame = reader.last_frame()
    mocapFPS = reader.header.frame_rate
    for i, (points, analog, labels) in enumerate(reader.read_frames()):
        #print('Frame {}: {} {}'.format(i, points, labels)) 
        frames.append(analog)
        count = count + 1
        if (count > lastFrame - 1):
            break
print('Loaded ' +str(count)+ ' frames.')

frame1 = frames[0][0:2]

# only use markers 1 to 12
for i, frame in enumerate(frames):
    frames[i] = frame[0:13]

# Use time between
endTime = 21 #seconds 482
startTime = 10 #seconds
endFrame = int(endTime*mocapFPS)
startFrame = int(startTime*mocapFPS)
if (endTime != 0):
    frames = frames[startFrame:endFrame]
    count = (endFrame - startFrame)
else:
    frames = frames[startFrame:]
    count -= startFrame

# group into markers
markers = [[],[], [], []]
for i, frame in enumerate(frames):
    frames[i] = frame[0:13]
    markers[0].append(frame[0:3])
    markers[1].append(frame[3:6])
    markers[2].append(frame[6:9])
    markers[3].append(frame[9:12])

# Check valid frames
#markerPoints[marker][frame][point][x,y,z]
newMarkers = np.zeros(shape=(4,count,3,3))
for i, marker in enumerate(markers):
    newMarker = np.array([[[]]])
    for ii, frame in enumerate(marker):
        newFrame = np.array([[0,0,0]])
        for iii, point in enumerate(frame):
            if (point[4] == -1): # bad point
                newMarkers[i][ii][iii] = [0,0,0]
            else:
                newMarkers[i][ii][iii] = markers[i][ii][iii][0:3]

aX = []
aY = []
aZ = []
#print('Frame {}'.format(markers[0][0:4]))
for i in range(count):
    isGood = markers[0][i][0][4]
    x = markers[0][i][0][0]
    y = markers[0][i][0][1]
    z = markers[0][i][0][2]
    if (isGood == -1):
        aX.append(0)
        aY.append(0)
        aZ.append(0)
    else:
        aX.append(x)
        aY.append(y)
        aZ.append(z)


# Remove begging of zeroes
num = 0
for i in range(count -1):
    if (aX[i]!= 0):
        num = i
        break
aX = aX[num:]
aY = aY[num:]
aZ = aZ[num:]
count -= num

newMarkers = newMarkers[:,num:]

aXOriginal = copy.deepcopy(aX)
aYOriginal = copy.deepcopy(aY)
aZOriginal = copy.deepcopy(aZ)

for i in range(count-1, 0, -1):
    if (aYOriginal[i]!=0):
        c = i
        break;
endFew = aYOriginal[c - 10:]

# Interpolation bad data
print('Interpolating data.')
#ft.interpolateMoCap(aX, 0.5)
#ft.interpolateMoCap(aY, 0.5)
#ft.interpolateMoCap(aZ, 0.5)

threads = [] # I don't think threading helps
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:        
    for i in range(len(newMarkers)):
        print('Interpolating - Marker:' + str(i))
        for ii in range(len(newMarkers[i,0])):
            for iii in range(len(newMarkers[i,0,0])):
                #ft.interpolateMoCap(newMarkers[i,:,ii,iii], 0.5)
                executor.map(ft.interpolateMoCap, (newMarkers[i,:,ii,iii], 0.5))
                #x = threading.Thread(target=ft.interpolateMoCap, args=(newMarkers[i,:,ii,iii], 0.5) )
                #threads.append(x)
                #try:
                #    x.start()
                #except:
                #    print("Error: unable to start thread")

#for i in range(len(threads)):
#    print('Interpolating - Marker:' + str(i))
#    try:
#        threads.index(-1).start()
#    except:
#        print("Error: unable to start thread")
#for i in range(len(threads)):
#    print('Waiting for thread '+str(i)+' to finish.')
#    threads[i].join()


# Filtering
print('Filtering data.')
xn = aY
NyquistFreq = mocapFPS/2
cutoffFreq = 8
#b, a = signal.butter(8, cutoffFreq/NyquistFreq, 'low')
b, a = signal.butter(3, cutoffFreq/NyquistFreq, 'lowpass', fs = mocapFPS)
zi = signal.lfilter_zi(b, a)
z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
y1 = signal.filtfilt(b, a, xn)

cutoffFreq = 30
b, a = signal.butter(3, cutoffFreq/NyquistFreq, 'lowpass', fs = mocapFPS)
y2 = signal.filtfilt(b, a, xn)

# Graphing
print('Graphing data.')
xAxis = np.arange(0,count/mocapFPS,1/mocapFPS)

fig, ax = plt.subplots()
#plt.plot(xAxis, aXOriginal, label = 'XOriginal')
#plt.plot(xAxis, aYOriginal, label = 'YOriginal')
#plt.plot(xAxis, aZOriginal, label = 'ZOriginal')
#plt.plot(xAxis, aX, label = 'X')
plt.plot(xAxis, aY, label = 'Y')
#plt.plot(xAxis, aZ, label = 'Z')
#plt.plot(xAxis, z, 'r--', xAxis, z2, 'g--')
plt.plot(xAxis, y1, 'k', label = 'Filtered 8Hz')
plt.plot(xAxis, y2, 'r', label = 'Filtered 30Hz')
plt.xlabel('Time (Seconds)')
plt.ylabel('Distance (Millimeters)')
plt.legend()
plt.title('Test plot')

#f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
#ax1.plot(xAxis, aX, label = 'X')
#ax2.plot(xAxis, aY, label = 'Y')
#plt.ylabel('Distance (Millimeters)')
#ax3.plot(xAxis, aZ, label = 'Z')
#
#plt.xlabel('Time (Seconds)')
#plt.legend()
#ax1.set_title('Test plot')


aX = newMarkers[0,:,0,0]
aY = newMarkers[0,:,0,1]
aZ = newMarkers[0,:,0,2]
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(xAxis, aX, label = 'X')
ax2.plot(xAxis, aY, label = 'Y')
plt.ylabel('Distance (Millimeters)')
ax3.plot(xAxis, aZ, label = 'Z')

plt.xlabel('Time (Seconds)')
plt.legend()
ax1.set_title('Test plot')

plt.show()



sys.exit(0)
print('Creating GoFigure.')
# Create figure
fig3 = go.Figure()

# Add traces, one for each slider step
print('Adding traces.')
ii =0
trace_list= []
for t in np.arange(0,len(aX), 1):
    newList = list()
    for marks in range(len(newMarkers)):
        for i in range(len(newMarkers[0,t])):
            newList.append(go.Scatter3d(
                visible=True,
                #line=dict(color="#"+str(i)+"0CED1", width=6),
                marker=dict(color="#"+str(i*4)+"0"+str(i*4)+"ED1"),
                name="𝜈 = "+str(i),
                mode='markers',
                x=[newMarkers[marks,t,i,0]],
                y=[newMarkers[marks,t,i,1]],
                z=[newMarkers[marks,t,i,2]]
            ))
    trace_list.extend(newList)

data=[]
for idx,item in enumerate(trace_list):  # I believe the key is that the format for data should be as though list1+list2
    data.extend(trace_list[idx])

fig2 = go.Figure(data=trace_list, layout=go.Layout(
                title= 'Virus spread in a Network',
                titlefont_size=16,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                scene = dict(
                    xaxis=go.layout.scene.XAxis(),
                    yaxis=go.layout.scene.YAxis(),
                    zaxis=go.layout.scene.ZAxis(),
                )))


fig2.data[0].visible = True
## Create and add slider
print('Adding slider.')
steps = []
# Number of previosu points shown
numberOPPS = 0
for i in range(len(aX)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(aX)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    if (i+numberOPPS < len(aX)):
        for ii in range(i,i+numberOPPS):
            step["args"][0]["visible"][ii] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Time: "},
    #pad={"t": 50},
    steps=steps
)]

fig2.update_layout(
    sliders=sliders
)

print('Show figure.')
fig2.show()