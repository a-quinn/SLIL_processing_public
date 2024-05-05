
#%% Script for manipulating markers in 3-Matic
# Author: Alastair Quinn 2022
# This script works with the 3-Matic API trimatic
import slil.mesh.interface as smi
mi = smi.MeshInterface(1)
import slil.common.math as fm
import numpy as np
import operator

# defaults
lunateName = 'lunate'
scaphoidName = 'scaphoid'
radiusName = 'ulnar and radius_reduced'
metacarp3Name = 'all but s l r u_reduced'
boneplugLunateName = 'lunate boneplug'
boneplugScaphoidName = 'scaphoid boneplug'
plateAssignment = {
    lunateName: [0, 1, 8],
    scaphoidName: [6, 7, 3],
    radiusName: [10, 11, 2],
    metacarp3Name: [4, 5, 9],
}
#%%
def check_project_open(modelInfo):
    current = mi.get_project_name()
    if len(current) == 0 or modelInfo['experimentID'] not in current:
        mi.open_project(modelInfo['3_matic_file'])

def boneNames():
    return ['lunate', 'scaphoid', 'radius', 'metacarp3']

def boneModelNames(modelInfo):
    lunateName = modelInfo['names']['lunate']
    scaphoidName = modelInfo['names']['scaphoid']
    radiusName = modelInfo['names']['radius']
    metacarp3Name = modelInfo['names']['metacarp3']
    return [lunateName, scaphoidName, radiusName, metacarp3Name]

def getPoints():
    if (not mi.find_point('marker0')):
        return
    points = []
    for i in range(12):
        points.append(mi.find_point('marker' + str(i)))
    return points

def colourPoints(p, modelInfo):
    colours = {
        'red': (1,0,0),
        'blue': (0,0,1),
        'green': (0,1,0),
        'yellow': (1,1,0),
        'grey': (0.9,0.9,0.9)
    }
    plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
    [lunateName, scaphoidName, radiusName, metacarp3Name] = boneModelNames(modelInfo)
    if (mi.find_point('marker0')):
        numPoints = 0
        for i, p in enumerate(modelInfo['plateAssignment_' + modelInfo['currentModel']]):
            for j in range(len(modelInfo['plateAssignment_' + modelInfo['currentModel']][p])):
                numPoints += 1
        points = []
        for i in range(numPoints):
            points.append(mi.find_point('marker' + str(i)))
        for i in plateAssignment[lunateName]:
            points[i].color = colours['grey']
        for i in plateAssignment[scaphoidName]:
            points[i].color = colours['yellow']
        for i in plateAssignment[radiusName]:
            points[i].color = colours['red']
        for i in plateAssignment[metacarp3Name]:
            points[i].color = colours['blue']

    else:
        print('Colouring unlabled points!')
        p[0].color = colours['grey']
        p[1].color = colours['grey']
        p[2].color = colours['blue']
        p[3].color = colours['blue']
        p[4].color = colours['grey']
        p[5].color = colours['red']
        p[6].color = colours['blue']
        p[7].color = colours['yellow']
        p[8].color = colours['red']
        p[9].color = colours['red']
        p[10].color = colours['yellow']
        p[11].color = colours['yellow']

def namePoints(p):
    prefix = 'marker'
    i = 0
    for n in p:
        n.name = prefix + str(i)
        i+=1
        
def checkMarkersValid(points):
    for p in points:
        for i in p.coordinates:
            if (abs(i) > 100000.0):
                return False
    return True

def convertPointsToSphereGroup(modelInfo, sphereGroupName = 'tempMarkerSpheres'):
    if (not mi.find_point('marker0')):
        return
    points = []
    numPoints = 0
    for i, p in enumerate(modelInfo['plateAssignment_' + modelInfo['currentModel']]):
        for j in range(len(modelInfo['plateAssignment_' + modelInfo['currentModel']][p])):
            numPoints += 1
    for i in range(numPoints):
        points.append(mi.find_point('marker' + str(i)))
        
    spheres = []
    for point in points:
        sphere = mi.create_sphere_part(point,2)
        sphere.color = point.color
        spheres.append(sphere)
    mi.delete(points)
    #spheres = trimatic.get_parts()
    #latestSpheres = spheres[int(-1*numPoints):]
    groupedSpheres = mi.data_merge(spheres)
    groupedSpheres.name = sphereGroupName

    colours = {
        'red': (1,0,0),
        'blue': (0,0,1),
        'green': (0,1,0),
        'yellow': (1,1,0),
        'grey': (0.9,0.9,0.9)
    }
    points = groupedSpheres.get_surfaces()
    plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
    [lunateName, scaphoidName, radiusName, metacarp3Name] = boneModelNames(modelInfo)
    for i in plateAssignment[lunateName]:
        points[i].color = colours['grey']
    for i in plateAssignment[scaphoidName]:
        points[i].color = colours['yellow']
    for i in plateAssignment[radiusName]:
        points[i].color = colours['red']
    for i in plateAssignment[metacarp3Name]:
        points[i].color = colours['blue']

def convertSphereGroupToPoints(modelInfo, sphereGroupName = 'tempMarkerSpheres'):
    groupedSpheres = mi.find_part(sphereGroupName)
    if (not groupedSpheres):
        print('Error: no sphere group {} found'.format(sphereGroupName))
        return
    latestSpheres = mi.surfaces_to_parts(groupedSpheres)
    latestPoints = []
    for sphere in latestSpheres:
        # “Based on mesh” or “Based on points” or “Based on volume”
        com = mi.compute_center_of_gravity(sphere, method ='Based on mesh')
        latestPoints.append(mi.create_point(com))
    mi.delete(latestSpheres)
    namePoints(latestPoints)
    colourPoints(latestPoints, modelInfo)

def getPointsFromSphereGroup(mi, sphereGroupName = 'tempMarkerSpheres'):
    groupedSpheres = mi.find_part(sphereGroupName)
    groupedSpheres2 = mi.duplicate(groupedSpheres)
    spheres = mi.surfaces_to_parts(groupedSpheres2)

    points = []
    for sphere in spheres:
        com = mi.compute_center_of_gravity(sphere, method ='Based on mesh')
        points.append(com)
    mi.delete(spheres)
    return np.array(points)

def deleteMarkers():
    points = []
    for i in range(12):
        points.append(mi.find_point('marker' + str(i)))
    mi.delete(points)

def deleteSphereGroupToPoints(sphereGroupName):
    groupedSpheres = mi.find_part(sphereGroupName)
    if (not groupedSpheres):
        print('Error delete: no sphere group {} found'.format(sphereGroupName))
        return
    mi.delete(groupedSpheres)

# Measure distance between points
# there's probably a cleaner way to code this...
def printDistancesFromMarkers(modelInfo):
    if (not mi.find_point('marker0')):
        print('Error: no marker {} found'.format('marker0'))
        return
    plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
    [lunateName, scaphoidName, radiusName, metacarp3Name] = boneModelNames(modelInfo)
    lunateCOM = mi.compute_center_of_gravity(mi.find_part(lunateName))
    scaphoidCOM = mi.compute_center_of_gravity(mi.find_part(scaphoidName))
    radiusCOM = mi.compute_center_of_gravity(mi.find_part(radiusName))
    metacarp3COM = mi.compute_center_of_gravity(mi.find_part(metacarp3Name))
    
    otherHandBones = modelInfo['otherHandBones']
    if (otherHandBones != []):
        a = mi.find_part(modelInfo['names']['metacarp3'])
        b = mi.duplicate(a)
        for i in range(len(otherHandBones)):
            c = mi.find_part(otherHandBones[i])
            c = mi.duplicate(c)
            b = mi.merge([b, c])
        metacarp3COM = mi.compute_center_of_gravity(b)
        mi.delete(b)

    def getPointsCoord(points):
        coord = []
        for point in points:
            coord.append(mi.find_point('marker' + str(point)).coordinates)
        return coord

    distances = []
    def findDistances(aPoint, bPoints):
        for i in getPointsCoord(bPoints):
            distances.append(tuple(np.subtract(i, aPoint)))
    findDistances(lunateCOM, plateAssignment[lunateName])
    findDistances(scaphoidCOM, plateAssignment[scaphoidName])
    findDistances(radiusCOM, plateAssignment[radiusName])
    findDistances(metacarp3COM, plateAssignment[metacarp3Name])
    #print('Distances in 3-Matic coordinates')
    #print(distances)

    openSimDistances = fm.convert3MaticToOpenSimCoords(distances)
    print('Distances in OpenSim coordinates')
    print('Order: ' + str(plateAssignment.keys()))
    #print(openSimDistances)
    m = []
    m.extend(plateAssignment[lunateName])
    m.extend(plateAssignment[scaphoidName])
    m.extend(plateAssignment[radiusName])
    m.extend(plateAssignment[metacarp3Name])
    for i in range(len(openSimDistances)):
        print('marker' + str(m[i]) + ' = ' \
        + str(openSimDistances[i][0]) + ' '\
        + str(openSimDistances[i][1]) + ' '\
        + str(openSimDistances[i][2]))

# Measure distance between bodies
def calcBodyDistancesFromGround(modelInfo):
    if (not bonesExist(modelInfo)):
        return
    [lunateName, scaphoidName, radiusName, metacarp3Name] = boneModelNames(modelInfo)
    lunateCOM = mi.compute_center_of_gravity(mi.find_part(lunateName))
    scaphoidCOM = mi.compute_center_of_gravity(mi.find_part(scaphoidName))
    radiusCOM = mi.compute_center_of_gravity(mi.find_part(radiusName))
    metacarp3COM = mi.compute_center_of_gravity(mi.find_part(metacarp3Name))

    otherHandBones = modelInfo['otherHandBones']
    if (otherHandBones != []):
        a = mi.find_part(modelInfo['names']['metacarp3'])
        b = mi.duplicate(a)
        for i in range(len(otherHandBones)):
            c = mi.find_part(otherHandBones[i])
            c = mi.duplicate(c)
            b = mi.merge([b, c])
        metacarp3COM = mi.compute_center_of_gravity(b)
        mi.delete(b)

    distances = []
    distances.append(tuple(np.subtract(scaphoidCOM, radiusCOM)))
    distances.append(tuple(np.subtract(lunateCOM, radiusCOM)))
    distances.append(tuple(np.subtract(metacarp3COM, radiusCOM)))

    #print('Distances in 3-Matic coordinates')
    #print(distances)

    openSimDistances = fm.convert3MaticToOpenSimCoords(distances)
    print('Distances in OpenSim coordinates')
    print(openSimDistances)

# Print implant bone plug points relative to scaphoid and lunate
def printBoneplugsToBonesDistances(modelInfo):
    if (not bonesExist(modelInfo)):
        return
    lunateName = modelInfo['names']['lunate']
    scaphoidName = modelInfo['names']['scaphoid']
    boneplugScaphoidName = modelInfo['names']['boneplugScaphoid']
    boneplugLunateName = modelInfo['names']['boneplugLunate']
    if (not mi.find_point(boneplugScaphoidName) or not mi.find_point(boneplugLunateName)):
        print('Error, could not find boneplug point' + str(boneplugScaphoidName) + 'or ' /
            str(boneplugLunateName) + '.')
        return
    lunateCOM = mi.compute_center_of_gravity(mi.find_part(lunateName))
    scaphoidCOM = mi.compute_center_of_gravity(mi.find_part(scaphoidName))
    lunateBoneplug = mi.find_point(boneplugLunateName).coordinates
    scaphoidBoneplug = mi.find_point(boneplugScaphoidName).coordinates

    distances = []
    distances.append(tuple(np.subtract(lunateBoneplug, lunateCOM)))
    distances.append(tuple(np.subtract(scaphoidBoneplug, scaphoidCOM)))

    print('Distances in 3-Matic coordinates (1. lunate, 2. scaphoid)')
    print(distances)
    openSimDistances = fm.convert3MaticToOpenSimCoords(distances)
    print('Distances in OpenSim coordinates (1. lunate, 2. scaphoid)')
    print(openSimDistances)

def generateBoneplugsToBonesDistances(modelInfo):
    if (not bonesExist(modelInfo)):
        return
    lunateName = modelInfo['names']['lunate']
    scaphoidName = modelInfo['names']['scaphoid']
    boneplugScaphoidName = modelInfo['names']['boneplugScaphoid']
    boneplugLunateName = modelInfo['names']['boneplugLunate']
    if (not mi.find_point(boneplugScaphoidName) or not mi.find_point(boneplugLunateName)):
        print('Error, could not find boneplug point' + str(boneplugScaphoidName) + 'or ' /
            str(boneplugLunateName) + '.')
        return
    lunateCOM = mi.compute_center_of_gravity(mi.find_part(lunateName))
    scaphoidCOM = mi.compute_center_of_gravity(mi.find_part(scaphoidName))
    lunateBoneplug = mi.find_point(boneplugLunateName).coordinates
    scaphoidBoneplug = mi.find_point(boneplugScaphoidName).coordinates

    distances = []
    distances.append(tuple(np.subtract(lunateBoneplug, lunateCOM)))
    distances.append(tuple(np.subtract(scaphoidBoneplug, scaphoidCOM)))

    openSimDistances = fm.convert3MaticToOpenSimCoords(distances)
    modelInfo['boneplug'] = {
        'lunate': openSimDistances[0],
        'scaphoid': openSimDistances[1]
    }
    print('Distances in OpenSim coordinates (1. lunate, 2. scaphoid)')
    print(openSimDistances)

def printDistancesFromMarkersForSphereGroup(plateAssignment, sphereGroupName):
    convertSphereGroupToPoints(plateAssignment, sphereGroupName)
    printDistancesFromMarkers()
    convertPointsToSphereGroup(sphereGroupName)

def generatePointOnMarkerPlates(modelInfo, plateName, boneName):
    plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
    if (not mi.find_part(plateName)):
        print('Error no part named \'' + plateName + '\' found.')
        return
    plate = mi.find_part(plateName)
    oPoint = plate.object_coordinate_system.origin
    xAxis = plate.object_coordinate_system.x_axis
    yAxis = plate.object_coordinate_system.y_axis
    zAxis = plate.object_coordinate_system.z_axis

    dist1 = 28 # from bone surface to center of middle marker along Y axis
    dist2 = 6.06 + 2 # from center of rod to face of marker LEDs. 2mm for thickness of LED top
    dist3 = 15.85

    a = list(map(operator.mul, yAxis, (dist1, dist1, dist1)))
    a = list(map(operator.add, a, oPoint))
    b = list(map(operator.mul, zAxis, (dist2, dist2, dist2)))
    a = list(map(operator.add, a, b))
    mi.create_point(a)
    mi.get_points()[-1].name = 'marker' + str(plateAssignment[boneName][2])

    # place second marker top right
    b = list(map(operator.mul, xAxis, (dist3, dist3, dist3)))
    a = list(map(operator.add, a, b))
    b = list(map(operator.mul, yAxis, (dist3, dist3, dist3)))
    a = list(map(operator.add, a, b))
    mi.create_point(a)
    mi.get_points()[-1].name = 'marker' + str(plateAssignment[boneName][1])

    # place second marker top left
    b = list(map(operator.mul, xAxis, (dist3 * -2, dist3 * -2, dist3 * -2)))
    a = list(map(operator.add, a, b))
    mi.create_point(a)
    mi.get_points()[-1].name = 'marker' + str(plateAssignment[boneName][0])

def generateMarkerPinsFromMarkers(modelInfo):
    if (not mi.find_point('marker0')):
        return
    plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
    [lunateName, scaphoidName, radiusName, metacarp3Name] = boneModelNames(modelInfo)
    
    points = plateAssignment[lunateName]
    p0 = mi.find_point('marker' + str(points[0]))
    p1 = mi.find_point('marker' + str(points[1]))
    p2 = mi.find_point('marker' + str(points[2])) # middle
    mi.create_distance_measurement(p0, p1)
    #for point in points:
    #    coord.append(mi.find_point('marker' + str(point)).coordinates)
    #return coord
    #for i in range()
    #mi.find_point()
    #mi.create_distance_measurement()

#%% Place marker near surface for each marker group
#
#lun = [0, 1, 4] # 4
#sca = [7, 10, 11] # 11
#rad = [5, 8, 9] # 9
#metacarp3 = [2, 3, 6] # 6
#lunMidMarker = 4
#scaMidMarker = 11
#radMidMarker = 9
#metacarp3MidMarker = 6
#
#def getCoord(p):
#    return mi.find_point('marker' + str(p)).coordinates
#
#getCoord(lunMidMarker)
## find point on plane where there is equal distance from either side
## create point 28mm from the center point away from the top two

#%%

# only basic marker pin, meaning through middle marker and directly in between left and right markers
def generateMarkerPinsFromMarkerPoints(modelInfo):
    if (not mi.find_point('marker0')):
        return
    plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
    wireLength = 15.85 + 45

    #for i, markerGroupName in enumerate(plateAssignment):
    #    points = plateAssignment[markerGroupName]
    #    p0 = mi.find_point('marker' + str(points[0]))
    #    p1 = mi.find_point('marker' + str(points[1]))
    #    p2 = mi.find_point('marker' + str(points[2])) # middle
    #    
    #    line = trimatic.create_line(p0, p1)
    #    d = trimatic.create_distance_measurement(p0, p1)
    #    m = trimatic.create_line_direction_and_length(p0, line.direction, d.value/2)
    #    mi.delete(d)
    #    mi.delete(line)
    #    midPoint = mi.create_point(m.point2)
    #    mi.delete(m)
#
    #    line = trimatic.create_line(midPoint, p2)
    #    #d = trimatic.create_distance_measurement(midPoint, p2)
    #    m2 = trimatic.create_line_direction_and_length(midPoint, line.direction, wireLength)
    #    #mi.delete(d)
    #    mi.delete(midPoint)
    #    mi.delete(line)
    #    m2.name = 'wire_'+markerGroupName
        
    for i, markerGroupName in enumerate(plateAssignment):
        points = plateAssignment[markerGroupName]
        p0 = mi.find_point('marker' + str(points[0]))
        p1 = mi.find_point('marker' + str(points[1]))
        p2 = mi.find_point('marker' + str(points[2])) # middle
        pm = (np.array(p0) - p1)/2 + p1 # between p0 and p1
        vec1 = np.array(p2) - pm # vector where wire is
        vec1 = (fm.normalizeVector(vec1)).reshape(3)
        m2 = mi.create_line_direction_and_length(pm, vec1, wireLength)
        m2.name = 'wire_'+markerGroupName

def generatePin(modelInfo):
    # Using the marker tool select a triangle on the inside of a marker guide pin hole.
    from copy import deepcopy
    a = mi.get_selection()
    if len(a) == 0:
        print("Nothing selected!")
        print("Using the marker tool select a triangle on the inside of a marker guide pin hole.")
        return
    part = a[0].get_parent()
    partAllTriangles = np.array(deepcopy(part.get_triangles()), dtype=object)
    b = deepcopy(a[0].get_triangles())
    partPoints = b[0]
    partPointIndexes = b[1]

    angleDeviationLimit = np.deg2rad(24)
    # if you think this value needs to be larger, do not.
    # Modify the mesh instead to include more triangles. (subdivide)
    maxDistance = 7.0

    firstSelectedTriangle = np.where((np.array(partAllTriangles[1]) == np.array(partPointIndexes[0])).all(axis=1))[0][0]

    def calcDist(a, b):
        return np.sqrt(
            np.power(a[0] - b[0], 2) + 
            np.power(a[1] - b[1], 2) + 
            np.power(a[2] - b[2], 2))
    def calcNormal(p0, p1, p2):
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
        vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

        u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

        normal = np.array(u_cross_v)

        # This edit is specifically for the parent function
        if (normal[0] < 0):
            normal = normal * -1.0
        return normal

    toCheckStartingTri = np.empty(shape=(0), dtype=int)
    toCheckStartingTri = np.append(toCheckStartingTri, firstSelectedTriangle)

    tempArray = np.array(partAllTriangles[1] )

    alreadyDone = np.empty(shape=(0), dtype=int)
    # check if individual point has already been created
    def checkIsAlreadyDone(indexOfThree):
        nonlocal alreadyDone
        for i in range(3):
            if (not np.any(alreadyDone[:] == indexOfThree[i])):
                alreadyDone = np.append(alreadyDone, indexOfThree[i])
            #    mi.create_point(partAllTriangles[0][indexOfThree[i]])

    # this function could be cleaner...
    def isWithinDistance(p1,p2):
        #return True
        if (maxDistance < calcDist(
            partAllTriangles[0][p1[0]],
            partAllTriangles[0][p2[0]])):
            return False
        if (maxDistance < calcDist(
            partAllTriangles[0][p1[0]],
            partAllTriangles[0][p2[1]])):
            return False
        if (maxDistance < calcDist(
            partAllTriangles[0][p1[0]],
            partAllTriangles[0][p2[2]])):
            return False
        if (maxDistance < calcDist(
            partAllTriangles[0][p1[1]],
            partAllTriangles[0][p2[1]])):
            return False
        if (maxDistance < calcDist(
            partAllTriangles[0][p1[1]],
            partAllTriangles[0][p2[2]])):
            return False
        if (maxDistance < calcDist(
            partAllTriangles[0][p1[2]],
            partAllTriangles[0][p2[2]])):
            return False
        return True
    i=0
    stop = False
    stopProblem = False
    startingTriIndex = toCheckStartingTri[0]
    alreadyDoneTriIndex = np.empty(shape=(0), dtype=int)
    while(True):
        alreadyDoneTriIndex = np.append(alreadyDoneTriIndex, startingTriIndex)
        normalInitial = calcNormal(
            partAllTriangles[0][partAllTriangles[1][startingTriIndex][0]],
            partAllTriangles[0][partAllTriangles[1][startingTriIndex][1]],
            partAllTriangles[0][partAllTriangles[1][startingTriIndex][2]]
            )
        for tri3 in partAllTriangles[1][startingTriIndex]:
            itemindex = np.where(tempArray==tri3)[0]
            for triIndex in itemindex:
                if (triIndex != startingTriIndex):
                    normal = calcNormal(
                        partAllTriangles[0][partAllTriangles[1][triIndex][0]],
                        partAllTriangles[0][partAllTriangles[1][triIndex][1]],
                        partAllTriangles[0][partAllTriangles[1][triIndex][2]]
                        )
                    angle = fm.angleBetweenVectors(normal,normalInitial)
                    dBool = isWithinDistance(partAllTriangles[1][startingTriIndex], partAllTriangles[1][triIndex])
                    if (abs(angle) < angleDeviationLimit) and dBool:
                        #print(angle)
                        checkIsAlreadyDone(partAllTriangles[1][triIndex])
                        #partAllTriangles[1][triIndex]
                        if (not np.any(alreadyDoneTriIndex[:] == triIndex)):
                            if (not np.any(toCheckStartingTri[:] == triIndex)):
                                toCheckStartingTri = np.append(toCheckStartingTri, triIndex)
                                
                        i = i + 1
                        # Shouldn't take this long so something's wrong and searching in worng area
                        if (i>100000):
                            stopProblem = True
                            stop = True
                        if (len(alreadyDone) > 600):
                            stop = True
        if len(toCheckStartingTri) == 0:
            break
        while (np.any(alreadyDoneTriIndex[:] == toCheckStartingTri[0])):
            toCheckStartingTri = np.delete(toCheckStartingTri, 0)
            if len(toCheckStartingTri) == 0:
                break
        if len(toCheckStartingTri) == 0:
            break
        if (stop):
            break
        startingTriIndex = toCheckStartingTri[0]
        toCheckStartingTri = np.delete(toCheckStartingTri, 0)
        #print("1: {} 2: {}", len(toCheckStartingTri), len(alreadyDoneTriIndex))
    print("Found {} verticies.".format(len(alreadyDone)))
    if (stopProblem or len(alreadyDone) == 0):
        print("Something went wrong!")
        return
    
    xyz = np.empty(shape=(len(alreadyDone),3), dtype=float)
    for ind, i in enumerate(alreadyDone):
        xyz[ind] = partAllTriangles[0][i]

    # Filter point cloud to remove outliers
    print("Before filter: ", len(xyz))
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    cl, ind = pcd.remove_radius_outlier(nb_points=2, radius=0.10)
    pcd = pcd.select_by_index(ind)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.1)
    xyz = np.asarray(pcd.select_by_index(ind).points)
    print("After filter: ", len(xyz))

    #for i in xyz:
    #    mi.create_point(i)

    # Fit cylinder to point cloud
    from cylinder_fitting import fit
    w_fit, C_fit, r_fit, fit_err = fit(xyz)
    print("Estimated Parameters: ")
    print(w_fit, C_fit, r_fit, fit_err)
    p1xyz = (float(C_fit[0]), float(C_fit[1]), float(C_fit[2]))
    p1vec = (float(w_fit[0]), float(w_fit[1]), float(w_fit[2]))

    #pin1Prim = trimatic.create_cylinder_axis(
    #    p1xyz, 
    #    p1vec,
    #    float(0.1), 10.0)
    #
    #pin2Prim = trimatic.create_cylinder_axis(
    #    p1xyz, 
    #    (-1.0 * p1vec[0], -1.0 * p1vec[1], -1.0 * p1vec[2]),
    #    float(0.1), 10.0)
    #
    #pin1 = trimatic.convert_analytical_primitive_to_part(pin1Prim)
    #pin2 = trimatic.convert_analytical_primitive_to_part(pin2Prim)
    #mi.delete([pin1Prim, pin2Prim])
    #
    #def getBonesRef(modelInfo):
    #    if (not bonesExist(modelInfo)):
    #        return
    #    [lunateName, scaphoidName, radiusName, metacarp3Name] = boneModelNames(modelInfo)
    #    return [mi.find_part(lunateName), mi.find_part(scaphoidName), mi.find_part(radiusName), mi.find_part(metacarp3Name)]
    #
    #bonesRefs = fn.getBonesRef(modelInfo)
    #bonesRefs.append(pin1)
    #bonesRefs.append(pin2)
    #partsColliding = trimatic.collision_detection(entities = bonesRefs, clearance_check=0.0)


    # Find where ray intersects part
    scene = o3d.t.geometry.RaycastingScene()
    #[lunateName, scaphoidName, radiusName, metacarp3Name]
    for name in boneModelNames(modelInfo):
        bone = mi.find_part(name)
        tri = np.asarray(deepcopy(bone.get_triangles()[1]))
        vert = np.asarray(deepcopy(bone.get_triangles()[0]))
        tBone = o3d.geometry.TriangleMesh()
        tBone.vertices = o3d.utility.Vector3dVector(vert)
        tBone.triangles = o3d.utility.Vector3iVector(tri)
        #o3d.visualization.draw_geometries([tBone])

        t = o3d.t.geometry.TriangleMesh.from_legacy(tBone)
        tBone_id = scene.add_triangles(t)

    raysIn = [
        [p1xyz[0], p1xyz[1], p1xyz[2], p1vec[0], p1vec[1], p1vec[2]],
        [p1xyz[0], p1xyz[1], p1xyz[2], -1.0 * p1vec[0], -1.0 * p1vec[1], -1.0 * p1vec[2]]
        ]
    rays = o3d.core.Tensor(raysIn, dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)
    print(ans['t_hit'], ans['geometry_ids'], ans['primitive_ids'], ans['primitive_normals'], ans['primitive_uvs'])

    if (ans['t_hit'][0].isfinite() or ans['t_hit'][1].isfinite()):
        if (ans['t_hit'][0] < ans['t_hit'][1]):
            d = ans['t_hit'][0].numpy()
            p = raysIn[0]
        else:
            d = ans['t_hit'][1].numpy()
            p = raysIn[1]
    else:
        print("Error: No collision found!")
        return

    #if (ans['t_hit'][0].isfinite()):
    #    d = ans['t_hit'][0].numpy()
    #    p = raysIn[0]
    #else:
    #    if (ans['t_hit'][1].isfinite()):
    #        d = ans['t_hit'][1].numpy()
    #        p = raysIn[1]
    #    else:
    #        print("Error: No collision found!")
    #        return
    c = [
        p[0] + (p[3] * d),
        p[1] + (p[4] * d),
        p[2] + (p[5] * d)]
    #mi.create_point(c)

    vecAway = [
        (p[3] * -1.0),
        (p[4] * -1.0),
        (p[5] * -1.0)]

    # TODO: Check this is correct!
    pinLength = 45 - 5 # 5 mm for length pin is in bone

    m2 = mi.create_line_direction_and_length(c, vecAway, pinLength)
    m2.name = 'pin_temp'

def makeAxis(c, v1, v2, v3=None):
    # 1.0e-10 good enough!
    if (np.abs(np.dot(v1, v2)) > 1.0e-10):
        print("vectors are not perpendicular!")
        return
    if v3 is None:
        v3 = fm.calcNormalVec(c, c + v1, c + v2)
    # 1.0e-10 good enough!
    if (np.abs(np.dot(v1, v3)) > 1.0e-10):
        print("vectors are not perpendicular!")
        return
    mi.create_line(c, c + v1)
    mi.create_line(c, c + v2)
    mi.create_line(c, c + v3)

def makeAxisR(c, r):
    makeAxis(c, r[:,0], r[:,1], r[:,2])

def generateSLILGap(modelInfo):
    '''
    Generated from scaffold placement.
        1. Finds line between insertion points.
        2. Finds intersection of line and lunate and scaphoid surfaces.
        3. Respective lunate and scaphoid intersection points are output.
    '''
    pL = np.array(mi.find_point(modelInfo['names']['boneplugLunate']).coordinates)
    pS = np.array(mi.find_point(modelInfo['names']['boneplugScaphoid']).coordinates)
    direction = np.array(fm.normalizeVector(pS-pL)) # direction vector L to S

    from copy import deepcopy
    l = mi.find_part(modelInfo['names']['lunate'])
    tri = l.get_triangles()
    points = deepcopy(tri[0])
    conns = deepcopy(tri[1])
    foundPoints = fm.findIntersectingPoints(pL, direction, conns, points)
    if len(foundPoints) > 1:
        print('Found more than one intersection for Lunate!')
    lunateEdge = foundPoints[0]

    l = mi.find_part(modelInfo['names']['scaphoid'])
    tri = l.get_triangles()
    points = deepcopy(tri[0])
    conns = deepcopy(tri[1])
    foundPoints = fm.findIntersectingPoints(pS, (-1.0 * direction), conns, points)
    if len(foundPoints) > 1:
        print('Found more than one intersection for Scaphoid.')
    scaphoidEdge = foundPoints[0]
    return [lunateEdge, scaphoidEdge]

def convertLinesToCylinders(modelInfo):
    plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
    if (not mi.find_line('wire_' + list(plateAssignment.keys())[0])):
        print('Error: no wire {} found'.format('wire_' + list(plateAssignment.keys())[0]))
        return
    #[lunateName, scaphoidName, radiusName, metacarp3Name] = fn.boneModelNames(modelInfo)
    for i, markerGroupName in enumerate(plateAssignment):
        wire = mi.find_line('wire_' + markerGroupName)
        cylinder = mi.create_cylinder_part(wire.point1, wire.point2, 0.2)
        cylinder.name = 'wire_' + markerGroupName
        mi.delete(wire)
  
def addCylindersToSphereGroup(modelInfo, spheregroup):
    plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
    if (not mi.find_part('wire_' + list(plateAssignment.keys())[0])):
        print('Error: no wire {} found'.format('wire_' + list(plateAssignment.keys())[0]))
        return
    groupedSpheres = mi.find_part(spheregroup)
    if (not groupedSpheres):
        print('Error: no sphere group {} found'.format(spheregroup))
        return
    #[lunateName, scaphoidName, radiusName, metacarp3Name] = fn.boneModelNames(modelInfo)
    for i, markerGroupName in enumerate(plateAssignment):
        cylinder = mi.find_part('wire_' + markerGroupName)
        #trimatic.copy_to_part(groupedSpheres.get_surface_sets()[0], cylinder)
        a = cylinder.find_surface('Mantle')
        a.name = 'wire_' + markerGroupName +'_Mantle'
        b = cylinder.find_surface('Bottom')
        b.name = 'wire_' + markerGroupName +'_Bottom'
        c = cylinder.find_surface('Top')
        c.name = 'wire_' + markerGroupName +'_Top'
        #d = trimatic.copy_to_part([a, b, c])
        #d = trimatic.merge([a, b, c])
        mi.move_to_part([a, b, c], groupedSpheres)
        mi.delete(cylinder)

def removeCylindersFromSphereGroup(modelInfo, spheregroup):
    plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
    
    groupedSpheres = mi.find_part(spheregroup)
    if (not groupedSpheres):
        print('Error: no sphere group {} found'.format(spheregroup))
        return
    if (not groupedSpheres.find_surface('wire_' + list(plateAssignment.keys())[0] + '_Mantle')):
        print('Error: no wire {} found'.format('wire_' + list(plateAssignment.keys())[0] + '_Mantle'))
        return
    
    for i, markerGroupName in enumerate(plateAssignment):
        a = groupedSpheres.find_surface('wire_' + markerGroupName +'_Mantle')
        b = groupedSpheres.find_surface('wire_' + markerGroupName +'_Bottom')
        c = groupedSpheres.find_surface('wire_' + markerGroupName +'_Top')

        cylinder = mi.move_to_part([a, b, c])
        cylinder.name = 'wire_' + markerGroupName
        a = cylinder.find_surface('wire_' + markerGroupName +'_Mantle')
        a.name = 'Mantle'
        b = cylinder.find_surface('wire_' + markerGroupName +'_Bottom')
        b.name = 'Bottom'
        c = cylinder.find_surface('wire_' + markerGroupName +'_Top')
        c.name = 'Top'

def moveToCloseToPart(markerGroupName, part):
    a = mi.find_part(markerGroupName)
    if (not a):
        print('Error: no sphere group {} found'.format(markerGroupName))
        return
    b = mi.find_part(part)
    if (not b):
        print('Error: no part {} found'.format(part))
        return
    
    # Started failing randomly
    #trimatic.global_registration(fixed_entity=b, moving_entity=a, distance_threshold=3000)
    
    xyzCentreA = [
        (a.dimension_max[0] - a.dimension_min[0])/2 + a.dimension_min[0],
        (a.dimension_max[1] - a.dimension_min[1])/2 + a.dimension_min[1],
        (a.dimension_max[2] - a.dimension_min[2])/2 + a.dimension_min[2]]
    xyzCentreB = [
        (b.dimension_max[0] - b.dimension_min[0])/2 + b.dimension_min[0],
        (b.dimension_max[1] - b.dimension_min[1])/2 + b.dimension_min[1],
        (b.dimension_max[2] - b.dimension_min[2])/2 + b.dimension_min[2]]
    translation_vector = list(map(operator.sub, xyzCentreB, xyzCentreA))
    mi.translate(a, translation_vector)
    
def generateBoneplugPoints(modelInfo, flipDirection=False, line='scaffold_edge_l_to_s', part='lunate_boneplug_bottom'):
    # this function will not work in public release as scaffold dimensions have been redacted
    redacted_1 = 0.0
    
    s = mi.find_part(part)
    len = redacted_1 - redacted_1/2 #mm
    s1= s.get_triangles()
    p1 = s1[0][0]
    p2 = s1[0][1]
    p3 = s1[0][3]
    surf = s.get_surfaces()[0]
    d1 = mi.create_plane_2_points_perpendicular_1_plane(p1,p2,surf)
    d2 = mi.create_plane_2_points_perpendicular_1_plane(p1,p3,surf)
    if (flipDirection):
        line1 = mi.create_line_plane_intersection(d2, d1)
    else:
        line1 = mi.create_line_plane_intersection(d1, d2)
    c = mi.compute_center_of_gravity(s, method='Based on mesh')
    point = mi.create_point(c)
    line2 = mi.create_line_direction_and_length(point, line1.direction, len)
    pointBonePlug1 = mi.create_point(line2.point2)
    mi.delete([d1, d2, line1, point, line2])

    edge = mi.find_line(line)
    line2 = mi.create_line_direction_and_length(pointBonePlug1, edge.direction, redacted_1)
    pointBonePlug2 = mi.create_point(line2.point2)
    mi.delete([line2])

    pointBonePlug1.name = modelInfo['names']['boneplugLunate']
    pointBonePlug2.name = modelInfo['names']['boneplugScaphoid']

def runIK(modelInfo, openSimModel, trial, visualise=False, threads = 2, timeout_sec = None):
    import subprocess
    import os
    import signal
    lunatePos = modelInfo['boneplug']['lunate']
    scaphoidPos = modelInfo['boneplug']['scaphoid']
    cadNum = modelInfo['experimentID'] # eg. 11535
    
    from pathlib import Path
    cwd = Path(modelInfo['dataOutputDir']).parent.absolute() #modelInfo['rootFolder'] + r'\Data processed'
    
    #from slil.common.opensim import getModelPath
    #modelPath = getModelPath(modelInfo)

    # command to run
    # '6DFOS_mainIKFromFile.exe --model ..\models\11535\wrist.osim --c3d 11535\cut_fe_40_40\log2.c3d -j 2 -bpLx 0.0027808198232904713 -bpLy 0.002301050569413772 -bpLz -0.005457571314230678 -bpSx 0.004565478863538909 -bpSy -0.0021595049948197166 -bpSz -0.011029887773751626'

    cmd = "..\\6DFOS_mainIKFromFile.exe"
    #cmd += " --model ..\\models\\" + str(cadNum) + "\\wrist_" + modelInfo['currentModel'] + ".osim"
    #cmd += " --model \'" + modelPath + "\'" # doesn't like spaces...
    cmd += " --model " + str(cadNum) + "\\wrist_" + openSimModel + ".osim"
    cmd += " --c3d " + str(cadNum) + trial # "\\normal_fe_40_40\\log1.c3d"
    cmd += " -j " + str(threads)
    cmd += " -bpLx " + str(lunatePos[0])
    cmd += " -bpLy " + str(lunatePos[1])
    cmd += " -bpLz " + str(lunatePos[2])
    cmd += " -bpSx " + str(scaphoidPos[0])
    cmd += " -bpSy " + str(scaphoidPos[1])
    cmd += " -bpSz " + str(scaphoidPos[2])
    if (visualise):
        cmd += " -v"

    print("Running command: {}".format(cmd))
    process = subprocess.Popen(cmd, cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE)

    if timeout_sec != None:
        wasKilled = False
        try:
            stdout, stderr = process.communicate(timeout=timeout_sec)
            print("Command stdout:\n{}".format(stdout.decode()))
            print("Command stderr:\n{}".format(stderr.decode()))
        except subprocess.TimeoutExpired:
            print("Process timeout, killing.")
            #process.kill()
            subprocess.Popen("TASKKILL /F /PID {pid} /T".format(pid=process.pid)) # fix for killing processon Windows
            stdout, stderr = process.communicate()
            print("stdout:\n{}".format(stdout.decode()))
            print("stderr:\n{}".format(stderr.decode()))
            wasKilled = True
        if wasKilled:
            print("Process was terminated.")
    else:
        stdout, stderr = process.communicate()
        print("Command output:\n{}".format(stdout.decode()))
    returnCode = process.returncode
    print(f'Process return code: {returnCode}')
    return returnCode

def setBonesInModel(modelInfo, coms):
    #if (not bonesExist(modelInfo)):
    #    return
    #[lunateName, scaphoidName, radiusName, metacarp3Name] = boneModelNames(modelInfo)
    #    
    #lunateCOM = mi.compute_center_of_gravity(mi.find_part(lunateName))
    #scaphoidCOM = mi.compute_center_of_gravity(mi.find_part(scaphoidName))
    #radiusCOM = mi.compute_center_of_gravity(mi.find_part(radiusName))
    #metacarp3COM = mi.compute_center_of_gravity(mi.find_part(metacarp3Name))
    #
    #otherHandBones = modelInfo['otherHandBones']
    #if (otherHandBones != []):
    #    a = mi.find_part(modelInfo['names']['metacarp3'])
    #    b = mi.duplicate(a)
    #    for i in range(len(otherHandBones)):
    #        c = mi.find_part(otherHandBones[i])
    #        c = mi.duplicate(c)
    #        b = mi.merge([b, c])
    #    metacarp3COM = mi.compute_center_of_gravity(b)
    #    mi.delete(b)
    [lunateCOM, scaphoidCOM, radiusCOM, metacarp3COM] = [coms[x] for x in coms]

    distances = []
    distances.append(tuple(np.subtract(scaphoidCOM, radiusCOM)))
    distances.append(tuple(np.subtract(lunateCOM, scaphoidCOM)))
    distances.append(tuple(np.subtract(metacarp3COM, scaphoidCOM)))

    #print('Distances in 3-Matic coordinates')
    #print(distances)

    openSimDistances = fm.convert3MaticToOpenSimCoords(distances)
    print('Distances in OpenSim coordinates')
    print(openSimDistances)

    import slil.common.opensim as fo
    jointsToDistances = [
        'radial_scaphoid',
        'scaphoid_lunate',
        'hand_wrist'
    ]
    fo.setBonesInModel(modelInfo, jointsToDistances, openSimDistances)

def setMarkersInModel(modelInfo, coms, markerPoints):
    #if (not mi.find_point('marker0')):
    #    print('Error, no markers points in 3-matic!')
    #    return
    plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
    [lunateName, scaphoidName, radiusName, metacarp3Name] = boneModelNames(modelInfo)
    #
    #lunateCOM = mi.compute_center_of_gravity(mi.find_part(lunateName))
    #scaphoidCOM = mi.compute_center_of_gravity(mi.find_part(scaphoidName))
    #radiusCOM = mi.compute_center_of_gravity(mi.find_part(radiusName))
    #metacarp3COM = mi.compute_center_of_gravity(mi.find_part(metacarp3Name))
    #
    #otherHandBones = modelInfo['otherHandBones']
    #if (otherHandBones != []):
    #    a = mi.find_part(modelInfo['names']['metacarp3'])
    #    b = mi.duplicate(a)
    #    for i in range(len(otherHandBones)):
    #        c = mi.find_part(otherHandBones[i])
    #        c = mi.duplicate(c)
    #        b = mi.merge([b, c])
    #    metacarp3COM = mi.compute_center_of_gravity(b)
    #    mi.delete(b)
    [lunateCOM, scaphoidCOM, radiusCOM, metacarp3COM] = [coms[x] for x in coms]

    def getPointsCoord(points):
        coord = []
        for point in points:
            coord.append(markerPoints[point])
            #coord.append(mi.find_point('marker' + str(point)).coordinates)
        return coord

    distances = []
    def findDistances(aPoint, bPoints):
        for i in getPointsCoord(bPoints):
            distances.append(tuple(np.subtract(i, aPoint)))
    findDistances(lunateCOM, plateAssignment[lunateName])
    findDistances(scaphoidCOM, plateAssignment[scaphoidName])
    findDistances(radiusCOM, plateAssignment[radiusName])
    findDistances(metacarp3COM, plateAssignment[metacarp3Name])

    openSimDistances = fm.convert3MaticToOpenSimCoords(distances)
    import slil.common.opensim as fo
    fo.setMarkersInModel(modelInfo, openSimDistances)

def setMarkersInModel_old(modelInfo):
    if (not mi.find_point('marker0')):
        print('Error, no markers points in 3-matic!')
        return
    plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
    [lunateName, scaphoidName, radiusName, metacarp3Name] = boneModelNames(modelInfo)
    
    lunateCOM = mi.compute_center_of_gravity(mi.find_part(lunateName))
    scaphoidCOM = mi.compute_center_of_gravity(mi.find_part(scaphoidName))
    radiusCOM = mi.compute_center_of_gravity(mi.find_part(radiusName))
    metacarp3COM = mi.compute_center_of_gravity(mi.find_part(metacarp3Name))
    
    otherHandBones = modelInfo['otherHandBones']
    if (otherHandBones != []):
        a = mi.find_part(modelInfo['names']['metacarp3'])
        b = mi.duplicate(a)
        for i in range(len(otherHandBones)):
            c = mi.find_part(otherHandBones[i])
            c = mi.duplicate(c)
            b = mi.merge([b, c])
        metacarp3COM = mi.compute_center_of_gravity(b)
        mi.delete(b)
    
    def getPointsCoord(points):
        coord = []
        for point in points:
            coord.append(mi.find_point('marker' + str(point)).coordinates)
        return coord

    distances = []
    def findDistances(aPoint, bPoints):
        for i in getPointsCoord(bPoints):
            distances.append(tuple(np.subtract(i, aPoint)))
    findDistances(lunateCOM, plateAssignment[lunateName])
    findDistances(scaphoidCOM, plateAssignment[scaphoidName])
    findDistances(radiusCOM, plateAssignment[radiusName])
    findDistances(metacarp3COM, plateAssignment[metacarp3Name])

    openSimDistances = fm.convert3MaticToOpenSimCoords(distances)
    import slil.common.opensim as fo
    fo.setMarkersInModel(modelInfo, openSimDistances)

def exportBones(modelInfo, scaleFactor = 0.001):
    if (not bonesExist(modelInfo)):
        return

    origin = (0,0,0)
    otherHandBones = modelInfo['otherHandBones']
    exportDir = modelInfo['dataOutputDir'] + r'\Geometry'

    names = boneModelNames(modelInfo)
    for i in range(len(names)):
        a = mi.find_part(names[i])
        b = mi.duplicate(a)
        b = mi.scale_factor(b, (scaleFactor, scaleFactor, scaleFactor))
        com = mi.compute_center_of_gravity(b)
        moveVector = tuple(np.subtract(origin, com))
        b = mi.translate(b, moveVector)
        b = mi.reduce(b, geometrical_error=0.01, number_of_iterations=2)
        b.name = names[i]
        mi.export_stl_binary(b, exportDir)
        mi.delete(b)
    
    a = mi.find_part(modelInfo['names']['metacarp3'])
    comMet = mi.compute_center_of_gravity(a)

    if (otherHandBones != []):
        a = mi.find_part(modelInfo['names']['metacarp3'])
        b = mi.duplicate(a)
        for i in range(len(otherHandBones)):
            c = mi.find_part(otherHandBones[i])
            c = mi.duplicate(c)
            b = mi.merge([b, c])
        com1 = mi.compute_center_of_gravity(b)
        b = mi.scale_factor(b, (scaleFactor, scaleFactor, scaleFactor))
        com = mi.compute_center_of_gravity(b)
        com = com + (np.array(comMet) - np.array(com1)) / 1000.0 # keep origin at meta carp COM
        moveVector = tuple(np.subtract(origin, com))
        b = mi.translate(b, moveVector)
        b = mi.reduce(b, geometrical_error=0.01, number_of_iterations=2)
        b.name = modelInfo['names']['metacarp3']
        mi.export_stl_binary(b, exportDir)
        mi.delete(b)

def bonesExist(modelInfo):
    [lunateName, scaphoidName, radiusName, metacarp3Name] = boneModelNames(modelInfo)
    if (not mi.find_part(lunateName)):
        print('Error, could not find bone ' + str(lunateName))
        return False
    if (not mi.find_part(scaphoidName)):
        print('Error, could not find bone ' + str(scaphoidName))
        return False
    if (not mi.find_part(radiusName)):
        print('Error, could not find bone ' + str(radiusName))
        return False
    if (not mi.find_part(metacarp3Name)):
        print('Error, could not find bone ' + str(metacarp3Name))
        return False
    return True

def getMarkerGroupTranformationMat(modelInfo, boneName, newPointsRaw, flip = False):
    plateAssignment = modelInfo['plateAssignment_' + modelInfo['currentModel']]
    p0 = np.array(newPointsRaw[plateAssignment[boneName][0]])
    p1 = np.array(newPointsRaw[plateAssignment[boneName][1]])
    p2 = np.array(newPointsRaw[plateAssignment[boneName][2]]) # middle marker
    #p0 = np.array(mi.find_point('marker' + str(plateAssignment[boneName][0])).coordinates)
    #p1 = np.array(mi.find_point('marker' + str(plateAssignment[boneName][1])).coordinates)
    #p2 = np.array(mi.find_point('marker' + str(plateAssignment[boneName][2])).coordinates) # middle marker
    pm = (p0 - p1)/2 + p1 # between p0 and p1
    vecAligned = fm.normalizeVector(p2 - pm) # vector where wire is
    posAligned = p2
    vecAlignedNorm = fm.normalizeVector(fm.calcNormalVec(p0, p1, p2))
    if flip:
        vecAlignedNorm = vecAlignedNorm * -1.0
    rotAligned = np.array(fm.create3DAxis(vecAligned, vecAlignedNorm)).T
    MTrans = np.eye(4)
    MTrans[:3, :3] = rotAligned
    MTrans[:3, 3] = posAligned.T
    return MTrans

def getLines(mi, names):
    lines = {}
    for name in names:
        line = mi.find_line(name)
        lines[line.name] = {
            'length': line.length,
            'point1': line.point1,
            'point2': line.point2,
            'direction': line.direction
        }
    return lines

def getCOMs(mi, names):
    coms = {}
    for name in names:
        com = mi.compute_center_of_gravity( mi.find_part(name) , method ='Based on mesh')
        coms[name] = {
            'COM': com,
        }
    return coms

def getFacesAndVertices(boneName):
    from copy import deepcopy
    rad = mi.find_part(boneName)
    if not hasattr(rad, 'get_triangles'):
        print('Error, no part names "{}" found!'.format(boneName))
        return [], []
    tri = rad.get_triangles()
    vertices = deepcopy(tri[0])
    faces = deepcopy(tri[1])
    return faces, vertices

def getGeometry(modelInfo, boneName, useCache = True):
    if useCache:
        fileName = 'geometry\\' + modelInfo['experimentID'] + '_' + boneName
        from slil.cache_results_plot import loadCache, saveCache
        geometry = loadCache(fileName)
        if geometry != None:
            return geometry
    print('No file found for {} so generating...'.format(fileName))

    check_project_open(modelInfo)
    faces, vertices = getFacesAndVertices(boneName)
    geometry = [faces, vertices]
    if useCache:
        saveCache(geometry, fileName)
    return geometry

def saveLoadOptimizationResults(modelInfo, newResult = None, extendedName = ''):
    from slil.common.cache import loadCache, saveCache
    fileName = 'model_' + str(modelInfo['experimentID'] + '_optimizationResults' + extendedName)
    if newResult == None:
        results = loadCache(fileName)
        if results != None:
            return results
        else:
            print('No cache file found for {}'.format(modelInfo['experimentID']))
            return
    else:
        results = loadCache(fileName)
        if results != None:
            print('Found {}'.format(fileName))
        else:
            print('No cache file found for {}. Creating...'.format(fileName))
            results = {}

        if modelInfo['currentModel'] in results:
            print('Overriding results for {} : {}'.format(modelInfo['experimentID'], modelInfo['currentModel']))
        results[modelInfo['currentModel']] = newResult

        saveCache(results, fileName)
        return results

def getOptimizationResults(modelInfo, extendedName = ''):
    res = saveLoadOptimizationResults(modelInfo, newResult = None, extendedName = extendedName)
    x0Return = {}
    for modelType in [ 'normal', 'cut', 'scaffold' ]:
        if modelType in res:
            if type(res[modelType]) == list:
                x0 = res[modelType][0].results['t']
            else:
                x0 = res[modelType]['t']
                if 't' in x0:
                    x0 = x0['t']
            x0Return[modelType] = x0
    return x0Return
    
def getOptimizationResultsAsTransMat(modelInfo, extendedName = ''):
    res = getOptimizationResults(modelInfo, extendedName = extendedName)
    adjustTransMarkers = {}
    for modelType in [ 'normal', 'cut', 'scaffold' ]:
        if not modelType in res:
            continue
        x0 = res[modelType]
        adjustTransMarkers[modelType] = np.eye(4)
        adjustTransMarkers[modelType][:3, :3] = fm.eulerAnglesToRotationMatrix((x0[3], x0[4], x0[5]))
        adjustTransMarkers[modelType][:3, 3] = [x0[0], x0[1], x0[2]]
    return adjustTransMarkers

def modifyModelCache_markerPositions(modelInfo, markers):
    from slil.common.cache import loadCache, saveCache
    fileName = 'model_' + str(modelInfo['experimentID'])
    modelCache = loadCache(fileName)
    if modelCache == None:
        print(f'No cache model file found for {fileName}')
        return
    
    modelCache['markers'] = {}
    for modelType in markers:
        modelCache['markers'][modelType] = markers[modelType]

    saveCache(modelCache, fileName)

def modifyModelCache_initTmatRadiusMarker(modelInfo, t_init):
    from slil.common.cache import loadCache, saveCache
    fileName = 'model_' + str(modelInfo['experimentID'])
    modelCache = loadCache(fileName)
    if modelCache == None:
        print(f'No cache model file found for {fileName}')
        return

    modelCache['initialTmatRadiusMarker'][modelInfo['currentModel']] = \
        t_init

    saveCache(modelCache, fileName)
    
#def modifyModelCache_adjTmatRadiusMarker(modelInfo, t_optimized):
#    from slil.common.cache import loadCache, saveCache
#    try:
#        fileName = 'model_' + str(modelInfo['experimentID'])
#        modelCache = loadCache(fileName)
#    except:
#        print('No cache model file found for {}'.format(fileName))
#        return
#
#    #orig = modelCache['initialTmatRadiusMarker']['normal']
#    #modelCache['adjustedTmatRadiusMarker'][modelInfo['currentModel']] = \
#    #    np.dot(orig, t_optimized)
#    modelCache['adjustmentTmatRadiusMarker'][modelInfo['currentModel']] = \
#        t_optimized
#
#    saveCache(modelCache, fileName)

def getModelCache(modelInfo, return_false_if_not_found = False):
    from slil.common.cache import loadCache, saveCache
    from copy import deepcopy
    fileName = 'model_' + str(modelInfo['experimentID'])
    arr = loadCache(fileName)
    if arr != None:
        return arr

    if return_false_if_not_found:
        return False
    
    print('No cache model file found for {} so generating...'.format(fileName))

    mi.open_project(modelInfo['3_matic_file'])

    pLunate, pScaphoid = generateSLILGap(modelInfo)
    bpL = np.array(mi.find_point(modelInfo['names']['boneplugLunate']).coordinates)
    bpS = np.array(mi.find_point(modelInfo['names']['boneplugScaphoid']).coordinates)
    
    [lunateName, scaphoidName, radiusName, metacarp3Name] = boneModelNames(modelInfo)
    lunateCOM = mi.compute_center_of_gravity(mi.find_part(lunateName))
    scaphoidCOM = mi.compute_center_of_gravity(mi.find_part(scaphoidName))
    radiusCOM = mi.compute_center_of_gravity(mi.find_part(radiusName))
    metacarp3COM = mi.compute_center_of_gravity(mi.find_part(metacarp3Name))


    generateBoneplugsToBonesDistances(modelInfo)
    lunateCOM2bp = modelInfo['boneplug']['lunate']
    scaphoidCOM2bp = modelInfo['boneplug']['scaphoid']

    startingModel = deepcopy(modelInfo['currentModel'])
    # find rough rotation and translation from whereever mocap was to origin for each model type
    initialAlignment = {}
    #radiusName = modelInfo['names']['radius']
    for model in [ 'normal', 'cut', 'scaffold' ]:
        #modelInfo['currentModel'] = model
        #markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static'
        ##markerGroupName = 'tempMarkerSpheres_' + modelInfo['currentModel'] + '_static_after'
        #points = getPointsFromSphereGroup(mi, markerGroupName)
        #t_WOP1 = getMarkerGroupTranformationMat(modelInfo, radiusName, points)
        #initialAlignment[model] = t_WOP1
        initialAlignment[model] = np.eye(4)
    modelInfo['currentModel'] = startingModel

    # so each bone is in a pickle
    for bone in boneModelNames(modelInfo) + modelInfo['otherHandBones']:
        getGeometry(modelInfo, bone)

    modelCache = {
        'SLILpointL': pLunate,
        'SLILpointS': pScaphoid,
        'boneplugL': bpL,
        'boneplugS': bpS,
        'lunateCOM2bpOS': lunateCOM2bp, # in OpenSim coords
        'scaphoidCOM2bpOS': scaphoidCOM2bp, # in OpenSim coords
        'initialTmatRadiusMarker': initialAlignment,
        'lunateCOM': lunateCOM,
        'scaphoidCOM': scaphoidCOM,
        'radiusCOM': radiusCOM,
        'metacarp3COM': metacarp3COM,
    }
    saveCache(modelCache, fileName)
    return modelCache

def getModelCacheExtra(modelInfo, return_false_if_not_found = False):
    from slil.common.cache import loadCache, saveCache
    fileName = 'model_' + str(modelInfo['experimentID']) + '_extra'
    arr = loadCache(fileName)
    if arr != None:
        return arr
        
    if return_false_if_not_found:
        return False
    
    print('No extra cache model file found for {} so generating...'.format(fileName))

    mi.open_project(modelInfo['3_matic_file'])

    def getGeo(geo):
        faces, vertices = getFacesAndVertices(geo)
        geometry = [faces, vertices]
        return geometry

    sensorGuidePins = getLines(mi, modelInfo['moCapPins'])

    modelCache = {
        'sensorGuide': getGeo(modelInfo['sensorGuideName']),
        'placedScaffold': getGeo(modelInfo['placedScaffoldName']),
        'surgicalGuideSca': getGeo(modelInfo['surgicalGuideSca']),
        'surgicalGuideLun': getGeo(modelInfo['surgicalGuideLun']),
        'sensorGuidePins': sensorGuidePins,
    }
    saveCache(modelCache, fileName)
    return modelCache

##%%
#trimatic.message_box(message="Select points" , title= "Point Select")
#points_selected=trimatic.get_selection()
##trimatic.align.rotate()
#print("Done.")
