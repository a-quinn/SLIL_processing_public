
from . import c3d_modified as c3d
import numpy as np

def readC3D(fileToImport):
    frames = [] # should be np array?

    with open(fileToImport, 'rb') as handle:
        reader = c3d.Reader(handle)
        frames = reader.read_frames_nonGenerator()
        #for i, (points, analog, labels) in enumerate(reader.read_frames()):
        #    #print('Frame {}: {} {} {}'.format(i, points, analog, labels))
        #    frames.append(analog)
    return frames # output is in millimeters

# isPureMarker meaning no other data in arrays
def writeC3D(data, markersUsed = 12, outputFile = 'random_data.c3d', isPureMarker = True, verbose = True):
    writer = c3d.Writer(point_scale=-1.0, point_rate=250)
    if verbose:
        print("Writing file: " + str(outputFile))

    scale = 1000.0 # from m to mm
    if (isPureMarker):
        pointNum = len(data)
        x = np.empty(( markersUsed, pointNum))
        y = np.empty(( markersUsed, pointNum))
        z = np.empty(( markersUsed, pointNum))
        for ii in range( markersUsed ):
            x[ii] = data[:,ii,0] * scale
            y[ii] = data[:,ii,1] * scale
            z[ii] = data[:,ii,2] * scale * -1.0
    else:
        pointNum = len(data['marker/1_2/x'])
        x = np.empty(( markersUsed, pointNum))
        y = np.empty(( markersUsed, pointNum))
        z = np.empty(( markersUsed, pointNum))
        for ii in range( markersUsed ):
            x[ii] = data['marker/1_'+str(ii+1)+'/x'] * scale
            y[ii] = data['marker/1_'+str(ii+1)+'/y'] * scale
            z[ii] = data['marker/1_'+str(ii+1)+'/z'] * scale * -1.0

    frames = []
#    frames = np.empty(( 2, pointNum, markersUsed, 5))
    for i in range(pointNum):
        points = np.empty((markersUsed, 5))
        for ii in range( markersUsed ):
            #points[ii] = [x[ii][i], -1.0 * z[ii][i], y[ii][i], ii+1, 0]
            #points[ii] = [ y[ii][i], -1.0 * z[ii][i], -1.0 * (x[ii][i]-100), ii+1, 0] # works with opensim for RE9
            points[ii] = [ y[ii][i], z[ii][i], x[ii][i], ii+1, 0] # RE10
            #points[ii] = [ y[ii][i], -1.0 * z[ii][i], (x[ii][i]-100), ii+1, 0]
            #points[ii] = [ x[ii][i] * scale, y[ii][i] * scale, z[ii][i] * scale, ii+1, 0]
#            frames[0, i, ii, :] = [ y[ii][i], z[ii][i], x[ii][i], ii+1, 0]
#            frames[1, i, :, :] = np.array([[],[]]) 
        frames.append(( points, np.array([[],[]]) ))
    writer.add_frames(frames)
    
    lables = []
    for i in range( markersUsed ):
        lables.append('marker'+str(i))

    with open(outputFile, 'wb') as h:
        writer.write(h, lables)
