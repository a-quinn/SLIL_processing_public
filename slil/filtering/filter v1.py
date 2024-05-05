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
endTime = 482 #seconds
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

aX = []
aY = []
aZ = []
# Check valid frames
#for i, marker in enumerate(markers):
#    for ii, frame in enumerate(marker):
#        for iii in frame:
#            if (iii[4] == -1):
#                a.append(0)
#            else:
#                a.append(markers[0][iii][0][4])

print('Frame {}'.format(markers[0][0:4]))

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

aXOriginal = copy.deepcopy(aX)
aYOriginal = copy.deepcopy(aY)
aZOriginal = copy.deepcopy(aZ)

for i in range(count-1, 0, -1):
    if (aYOriginal[i]!=0):
        c = i
        break
endFew = aYOriginal[c - 10:]

# Interpolation bad data
ft.interpolateMoCap(aX, 0.5)
ft.interpolateMoCap(aY, 0.5)
ft.interpolateMoCap(aZ, 0.5)

# Filtering
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

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(xAxis, aX, label = 'X')
ax2.plot(xAxis, aY, label = 'Y')
plt.ylabel('Distance (Millimeters)')
ax3.plot(xAxis, aZ, label = 'Z')

plt.xlabel('Time (Seconds)')
plt.legend()
ax1.set_title('Test plot')
plt.show()