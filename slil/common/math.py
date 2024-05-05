# Author: Alastair Quinn 2022
from copy import deepcopy
import numpy as np
import math

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
        to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

# From MATLAB
def rotationMatrixToEulerAngles(R, seq = 'XYZ'):
    if R.shape != (3, 3):
        raise ValueError("Rotation Matrix must be 3x3")
    #if isRotationMatrix(R):
        #print("Rotation Matrix not valid")
        #raise ValueError("Rotation Matrix not valid")

    eul = np.zeros((1, 3))
    nextAxis = [1, 2, 0, 1]
    # Pre-populate settings for different axis orderings
    # Each setting has 4 values:
    #   1. firstAxis : The right-most axis of the rotation order. Here, X=1,
    #      Y=2, and Z=3.
    #   2. repetition : If the first axis and the last axis are equal in
    #      the sequence, then repetition = 1 otherwise repetition = 0.
    #   3. parity : Parity is 0 if the right two axes in the sequence are
    #      YX, ZY, or XZ. Otherwise, parity is 1.
    #   4. movingFrame : movingFrame = 1 if the rotations are with
    #      reference to a moving frame. Otherwise (in the case of a static
    #      frame), movingFrame = 0.
    seqSettings= {
        'ZYX': [0, 0, 0, 1],
        'ZYZ': [2, 1, 1, 1],
        'XYZ': [2, 0, 1, 1]
    }

    # Retrieve the settings for a particular axis sequence
    setting = seqSettings[seq]
    firstAxis = setting[0]
    repetition = setting[1]
    parity = setting[2]
    movingFrame = setting[3]

    # Calculate indices for accessing rotation matrix
    i = firstAxis
    j = nextAxis[i+parity]
    k = nextAxis[i-parity+1]

    if repetition:
        # Find special cases of rotation matrix values that correspond to Euler
        # angle singularities.
        sy = np.sqrt(R[i,j]*R[i,j] + R[i,k]*R[i,k])
        singular = sy < 10 * np.finfo(np.float64).eps
        
        # Calculate Euler angles
        eul = [math.atan2(R[i,j], R[i,k]), math.atan2(sy, R[i,i]), math.atan2(R[j,i], -R[k,i])]
        
        if singular:
            eul = [math.atan2(-R[j,k], R[j,j]), math.atan2(sy, R[i,i]), 0.0]

    else:
        # Find special cases of rotation matrix values that correspond to Euler
        # angle singularities.  
        sy = np.sqrt(R[i,i]*R[i,i] + R[j,i]*R[j,i])
        singular = sy < 10 * np.finfo(np.float64).eps
        
        # Calculate Euler angles
        eul = [math.atan2(R[k,j], R[k,k]), math.atan2(-R[k,i], sy), math.atan2(R[j,i], R[i,i])]
        
        if singular:
            eul = [math.atan2(-R[j,k], R[j,j]), math.atan2(-R[k,i], sy), 0.0]

    if parity:
        # Invert the result
        eul = np.multiply(eul, -1.0)

    if movingFrame:
        # Swap the X and Z columns
        eul[0], eul[2] = eul[2], eul[0]
    return eul

# From MATLAB
def eulerAnglesToRotationMatrix(eul, seq = 'XYZ'):
        
    R = np.zeros((3,3))
    ct = np.cos(eul)
    st = np.sin(eul)

    # The parsed sequence will be in all upper-case letters and validated
    if seq == 'ZYX':
            #     The rotation matrix R can be constructed as follows by
            #     ct = [cz cy cx] and st = [sy sy sx]
            #
            #     R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
            #            cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
            #              -sy            cy*sx             cy*cx]
            #       = Rz(tz) * Ry(ty) * Rx(tx)
            
            R[0,0] = ct[1]*ct[0]
            R[0,1] = st[2]*st[1]*ct[0] - ct[2]*st[0]
            R[0,2] = ct[2]*st[1]*ct[0] + st[2]*st[0]
            R[1,0] = ct[1]*st[0]
            R[1,1] = st[2]*st[1]*st[0] + ct[2]*ct[0]
            R[1,2] = ct[2]*st[1]*st[0] - st[2]*ct[0]
            R[2,0] = -st[1]
            R[2,1] = st[2]*ct[1]
            R[2,2] = ct[2]*ct[1]
            
    if seq == 'ZYZ':
            #     The rotation matrix R can be constructed as follows by
            #     ct = [cz cy cz2] and st = [sz sy sz2]
            #
            #     R = [  cz2*cy*cz-sz2*sz   -sz2*cy*cz-cz2*sz    sy*cz
            #            cz2*cy*sz+sz2*cz   -sz2*cy*sz+cz2*cz    sy*sz
            #                     -cz2*sy              sz2*sy       cy]
            #       = Rz(tz) * Ry(ty) * Rz(tz2)
            
            R[0,0] = ct[0]*ct[2]*ct[1] - st[0]*st[2]
            R[0,1] = -ct[0]*ct[1]*st[2] - st[0]*ct[2]
            R[0,2] = ct[0]*st[1]
            R[1,0] = st[0]*ct[2]*ct[1] + ct[0]*st[2]
            R[1,1] = -st[0]*ct[1]*st[2] + ct[0]*ct[2]
            R[1,2] = st[0]*st[1]
            R[2,0] = -st[1]*ct[2]
            R[2,1] = st[1]*st[2]
            R[2,2] = ct[1]
            
    if seq == 'XYZ':
            #     The rotation matrix R can be constructed as follows by
            #     ct = [cx cy cz] and st = [sx sy sz]
            #
            #     R = [            cy*cz,           -cy*sz,     sy]
            #         [ cx*sz + cz*sx*sy, cx*cz - sx*sy*sz, -cy*sx]
            #         [ sx*sz - cx*cz*sy, cz*sx + cx*sy*sz,  cx*cy]
            #       = Rx(tx) * Ry(ty) * Rz(tz)
            
            R[0,0] = ct[1]*ct[2]
            R[0,1] = -ct[1]*st[2]
            R[0,2] = st[1]
            R[1,0] = ct[0]*st[2] + ct[2]*st[0]*st[1]
            R[1,1] = ct[0]*ct[2] - st[0]*st[1]*st[2]
            R[1,2] = -ct[1]*st[0]
            R[2,0] = st[0]*st[2] - ct[0]*ct[2]*st[1]
            R[2,1] = ct[2]*st[0] + ct[0]*st[1]*st[2]
            R[2,2] = ct[0]*ct[1]
    return R

def rotMat2AxisAngle(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    thetaX = np.arctan2(R[2, 1], R[2, 2])
    thetaY = np.arctan2(R[2, 0], np.sqrt(np.power(R[2, 1], 2) + np.power(R[2, 2], 2)))
    thetaZ = np.arctan2(R[1, 0], R[0, 0])
    return np.array([thetaX, thetaY, thetaZ]), np.arccos((tr-1)/2)

def rotMat2AxisAngle_onlyAngle(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    return np.arccos((tr-1)/2)

def createTransformationMatFromPosAndEuler(x, y, z, r1, r2, r3, seq = 'XYZ'):
    R = eulerAnglesToRotationMatrix((r1, r2, r3), seq = seq)
    t = np.eye(4)
    t[:3, :3] = R
    t[:3, 3] = np.array((x, y, z)).T
    return t

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (normalizeVector(vec1)).reshape(3), (normalizeVector(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def rotation_matrix_from_vectors2(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions

def isRotationMatrix(R):
    # square matrix test
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    should_be_identity = np.allclose(R.dot(R.T), np.identity(R.shape[0], np.float))
    should_be_one = np.allclose(np.linalg.det(R), 1)
    return should_be_identity and should_be_one

def rotation_matrix_around_axis(t, vec):
    """
    theta is the angle, and ux, uy, and uz are the x, y, and z components of the normalized axis vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    [ux, uy, uz] = [vec[0], vec[1], vec[2]]
    r = np.array([
        [
            np.cos(t) + np.power(ux, 2) * (1 - np.cos(t)),
            ux * uy * (1 - np.cos(t)) - uz * np.sin(t),
            ux * uz * (1 - np.cos(t)) + uy * np.sin(t)
        ],
        [
            uy * ux * (1 - np.cos(t)) + uz * np.sin(t),
            np.cos(t) + np.power(uy, 2) * (1 - np.cos(t)),
            uy * uz * (1 - np.cos(t)) - ux * np.sin(t)
        ],
        [
            uz * ux * (1 - np.cos(t)) - uy * np.sin(t),
            uz * uy * (1 - np.cos(t)) + ux * np.sin(t),
            np.cos(t) + np.power(uz, 2) * (1 - np.cos(t))
        ]
    ])
    return r

def angle(v1, v2, acute):
# v1 is your firsr vector
# v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if (acute == True):
        return angle
    else:
        return 2 * np.pi - angle

def normalizeVector(v):
    return v / np.linalg.norm(v)

def calcDist(a, b):
    return np.sqrt(
        np.power(a[0] - b[0], 2) + 
        np.power(a[1] - b[1], 2) + 
        np.power(a[2] - b[2], 2))

def isRotationMatrix(M):
    tag = False
    I = np.identity(M.shape[0])
    if np.all((np.matmul(M, M.T)) == I) and (np.linalg.det(M)==1): tag = True
    return tag  

def rotateVector(vec, rot):
    """
    vec: 1x3 point in space
    rot: 3x3 rotation matrix
    """
    return np.dot(rot, np.array(vec))
    
def calcMag(vec):
    return np.sqrt(
        np.power(vec[0], 2) + 
        np.power(vec[1], 2) + 
        np.power(vec[2], 2))
    return calcDist(np.zeros(shape=(3)), vec)

def createMatrixI():
    return np.identity(4)

def angleBetweenVectors(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = normalizeVector(v1)
    v2_u = normalizeVector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def create3DAxis(vec1, vec2):
    """ Returns 3 vectors which are all perpandicular
    (or close enough with e-15 kind of range)

    return: [v1, v2, v3]
    v1 is vec1
    v2 is perpandicular to vec1 and vec2
    v3 is perpandicular to vec1 and v2
    """
    v2 = np.cross(vec1, vec2)
    v3 = np.cross(vec1, v2)
    vec1 = normalizeVector(vec1)
    v2 = normalizeVector(v2)
    v3 = normalizeVector(v3)
    return [vec1, v2, v3]

def radToDeg(degArr):
    return [math.degrees(rot) for rot in degArr]

def degToRad(radArr):
    return [math.radians(rot) for rot in radArr]

def addPosToRotationMatrix(rotMat, pos):
    mat = rotMat
    mat[0,3] = pos[0]
    mat[1,3] = pos[1]
    mat[2,3] = pos[2]
    return mat

def calcNormalVec(p0, p1, p2):
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
    vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

    u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

    normal = np.array(u_cross_v)
    return normal

def transposeRotationMatrixT(rotMat):
    rotMat = np.array(rotMat)
    R = np.matrix(rotMat[:3,:3])
    Ttemp = rotMat[:3,3]
    R = np.linalg.inv(R)
    Tsub = np.dot(-R, Ttemp)
    T = np.append(R,Tsub.T,axis=1)
    T = np.append(T,[[0, 0, 0, 1]],axis=0)
    #T = np.matrix([
    #    [R[0,:3], Tsub[0,0]],
    #    [R[1,:3], Tsub[0,1]],
    #    [R[2,:3], Tsub[0,2]],
    #    [0, 0, 0, 1]])
    return T
    
def transposeRotationMatrix(rotMat):
    rotMat = np.array(rotMat)
    R = np.matrix(rotMat[:3,:3])
    Ttemp = rotMat[:3,3]
    R = np.linalg.inv(R)
    Tsub = np.dot(-R, Ttemp)
    T = np.append(R,Tsub.T,axis=1)
    T = np.append(T,[[0, 0, 0, 1]],axis=0)
    #T = np.matrix([
    #    [R[0,:3], Tsub[0,0]],
    #    [R[1,:3], Tsub[0,1]],
    #    [R[2,:3], Tsub[0,2]],
    #    [0, 0, 0, 1]])
    return T

def inverseTransformationMat(mat):
    m = np.eye(4)
    m[:3, :3] = np.linalg.inv(mat[:3, :3])
    m[:3, 3] = mat[:3, 3] * -1.0
    m[:3, 3] = np.dot(m[:3, :3], m[:3, 3])
    return m

def inverseRotationMat(mat):
    mat[:3, :3] = np.linalg.inv(mat[:3, :3])
    return mat

def transformPoints_1(points, tm):
    pointsTemp = np.ones((points.shape[0], points.shape[1] + 1))
    pointsTemp[:, :3] = deepcopy(points)
    pointsTempReturn = deepcopy(points)
    for i, vector in enumerate(pointsTemp):
        pointsTempReturn[i] = np.dot(tm, vector)[:3]
    return pointsTempReturn

def transformPoints(points, tmStart, tmEnd):
    M2 = inverseTransformationMat(tmStart)
    pointsTemp = np.ones((points.shape[0], points.shape[1] + 1))
    pointsTemp[:, :3] = deepcopy(points)
    pointsTempReturn = deepcopy(points)
    for i, vector in enumerate(pointsTemp):
        coord = np.dot(M2, vector) # first move to origin
        pointsTempReturn[i] = np.dot(tmEnd, coord)[:3] # then move to new pos
    return pointsTempReturn
    points = deepcopy(points)
    for i, point in enumerate(points):
        #pm = np.array(point)
        #coord = np.dot(M2[:3, :3], pm + (M2[:3, 3].T)) # first move to origin
        #points[i] = np.dot(tmEnd[:3, :3], coord) + tmEnd[:3, 3].T # then move to new pos
        # now using 4-vectors (appending a 1 to the array)
        pm = np.array((point[0], point[1], point[2], 1))
        coord = np.dot(M2, pm) # first move to origin
        points[i] = np.dot(tmEnd, coord)[:3] # then move to new pos
    return points
    
def transformPoints_byRef(points, tmStart, tmEnd):
    M2 = inverseTransformationMat(tmStart)
    #for i, vector in enumerate(vectors):
    #    pm = np.array((vector[0], vector[1], vector[2], 1))
    #    coord = np.dot(M2, pm) # first move to origin
    #    vectors[i] = np.dot(T, coord)[:3] # then move to new pos
    vecTemp = np.ones((points.shape[0], points.shape[1] + 1))
    vecTemp[:, :3] = deepcopy(points)
    for i, vector in enumerate(vecTemp):
        coord = np.dot(M2, vector) # first move to origin
        points[i] = np.dot(tmEnd, coord)[:3] # then move to new pos

def calcRadius(p1, p2, p3):
    # Calcualte radius given three points along circle edge.
    A = np.array(p1)
    B = np.array(p2)
    C = np.array(p3)
    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(B - A)
    s = (a + b + c) / 2
    R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
    return R

    #arc = trimatic.create_arc_3_points(p1, p2, p3)
    #r = arc[0].radius
    #trimatic.delete(arc[0])
    #return r

def isInside(p, a, b):
    # Is p inside cube defined by points opposite eachother
    if min(a[0], b[0]) < p[0] and p[0] > min(a[0], b[0]):
        if min(a[1], b[1]) < p[1] and p[1] > min(a[1], b[1]):
            if min(a[2], b[2]) < p[2] and p[2] > min(a[2], b[2]):
                return True
    return False

def isBetween3(P, v0, v1, v2):
    # Is P inside triangle
    # Method: inside-outside test
    edge0 = v1 - v0
    edge1 = v2 - v1
    edge2 = v0 - v2
    C0 = P - v0
    C1 = P - v1
    C2 = P - v2
    N = np.cross(edge0, edge1)
    if (np.dot(N, np.cross(edge0, C0)) > 0 and \
        np.dot(N, np.cross(edge1, C1)) > 0 and \
        np.dot(N, np.cross(edge2, C2)) > 0):
        return True
    return False

    # Method: using Barycentric Coordinates, check triangle areas
    def areaS(V0, V1, V2):
        return 1/2 * np.abs(np.cross((V0-V1),(V0-V2)))
    s = areaS(V0, V1, V2)
    a = areaS(V1, P, V2)/s
    b = areaS(P, V0, V2)/s
    c = areaS(P, V0, V1)/s
    t = np.array([a, b, c])
    if np.all(t > 0.0) and np.all(t < 1.0):
        return True
    return False

def orthogVector(v):
    # orthog Abhishek method
    u = np.array([1.0, 0.0, 0.0])
    u_dot_v = np.dot(u, v)
    if(abs(u_dot_v) != 1.0):
        return normalizeVector(u + (v * -u_dot_v))
    else:
        return np.array([0.0, 1.0, 0.0])

def findLinePlaneIntersectPoint(rayOrig, rayVector, planeNormal, pointOnPlane):
    # stolen from: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
    
    # check if ray and plane are parallel.
    NdotRayDirection = np.dot(planeNormal, rayVector)
    if (np.abs(NdotRayDirection) < 0.000000001): # almost 0 
        return [False, np.array([0,0,0])]  #they are parallel so they don't intersect ! 
 
    d = -1.0 * np.dot(planeNormal, pointOnPlane) # compute d parameter using equation 2
    t = -(np.dot(planeNormal, rayOrig) + d) / NdotRayDirection # compute t (equation 3)

    if (t < 0): # check if the triangle is in behind the ray
        return [False, np.array([0,0,0])]  #the triangle is behind 
 
    # compute the intersection point using equation 1
    P = rayOrig + t * rayVector
    return [True, P]

def findIntersectPoint(orig, dir, v0, v1, v2):
    # v0, v1, v2 are three corners of a triangle
    # stolen from: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
    edge0 = v1 - v0
    edge1 = v2 - v1
    # no need to normalize
    N = np.cross(edge0, edge1) # compute plane's normal
 
    # Step 1: finding P
 
    # check if ray and plane are parallel.
    NdotRayDirection = np.dot(N, dir)
    #if (np.abs(NdotRayDirection) < 0.000000001): # almost 0 
    #    return [False, np.array([0,0,0])]  #they are parallel so they don't intersect ! 
 
    d = -1.0 * np.dot(N,v0) # compute d parameter using equation 2
    t = -(np.dot(N,orig) + d) / NdotRayDirection # compute t (equation 3)

    if (t < 0): # check if the triangle is in behind the ray
        return [False, np.array([0,0,0])]  #the triangle is behind 
 
    # compute the intersection point using equation 1
    P = orig + t * dir

    # Step 2: inside-outside test

    # edge 0
    vp0 = P - v0
    C = np.cross(edge0, vp0) #vector perpendicular to triangle's plane
    if (np.dot(N,C) < 0):
        return [False, np.array([0,0,0])] #P is on the right side 
 
    # edge 1
    vp1 = P - v1
    C = np.cross(edge1, vp1)
    if (np.dot(N,C) < 0):
        return [False, np.array([0,0,0])] #P is on the right side 
 
    # edge 2
    edge2 = v0 - v2
    vp2 = P - v2
    C = np.cross(edge2, vp2)
    if (np.dot(N,C) < 0):
        return [False, np.array([0,0,0])] #P is on the right side; 
    return [True, P]

def findIntersectingPoints(p, dir, conns, points):
    foundPoints = []
    for surfI in range(len(conns)):
        P1 = np.array(points[conns[surfI][0]])
        P2 = np.array(points[conns[surfI][1]])
        P3 = np.array(points[conns[surfI][2]])
        found, P = findIntersectPoint(p, dir, P1, P2, P3)
        if found:
            foundPoints.append(P)
    return foundPoints

def getTransformationMatrix(kinematics, bone, coordSys = 'opensim'):
    bonesMap = {
        'lunate': [
            'lunate_flexion',
            'lunate_deviation',
            'lunate_rotation',
            'lunate_xtrans',
            'lunate_ytrans',
            'lunate_ztrans'
            ],
        'scaphoid': [
            'sca_xrot',
            'sca_yrot',
            'sca_zrot',
            'sca_xtran',
            'sca_ytran',
            'sca_ztran'
            ],
        'radius': [
            "uln_xrot",
            "uln_yrot",
            "uln_zrot",
            "uln_xtran",
            "uln_ytran",
            "uln_ztran"
            ],
        'metacarp3': [
            "hand_flexion",
            "hand_deviation",
            "hand_rotation",
            "hand_xtrans",
            "hand_ytrans",
            "hand_ztrans"
            ]
    }
    rx = kinematics[bonesMap[bone][0]].to_numpy()
    ry = kinematics[bonesMap[bone][1]].to_numpy()
    rz = kinematics[bonesMap[bone][2]].to_numpy()
    x = kinematics[bonesMap[bone][3]].to_numpy()
    y = kinematics[bonesMap[bone][4]].to_numpy()
    z = kinematics[bonesMap[bone][5]].to_numpy()
    if coordSys == '3-matic': # convert OpenSim coords to 3-Matic
        osimX = deepcopy(x)
        osimY = deepcopy(y)
        osimZ = deepcopy(z)
        x = osimZ * 1000.0
        y = osimX * 1000.0
        z = osimY * 1000.0
        osimRX = deepcopy(rx)
        osimRY = deepcopy(ry)
        osimRZ = deepcopy(rz)
        rx = osimRZ
        ry = osimRX
        rz = osimRY
    RP = np.empty((len(rx), 4, 4)) # rotation & position
    R = np.empty((len(rx), 3, 3)) # rotation
    P = np.array([x, y, z]).T
    for i in range(len(rx)):
        RI = createMatrixI()
        R[i,:,:] = eulerAnglesToRotationMatrix([rx[i], ry[i], rz[i]])
        RI[:3,:3] = R[i,:,:]
        RP[i,:,:] = addPosToRotationMatrix(RI, [x[i], y[i], z[i]])
    return RP, R, P

def calcCOM(vectors, triangles, method = 'points'):
    # This can be based either on points
    # (triangle corners), mesh (triangles) or the volume in general
    #triangles = faces.reshape(-1, 4) # to get from pyvista faces

    if method == 'points':
        return np.mean(vectors, axis=0) # based on points

    if type(triangles) == np.ndarray:
        if triangles.shape[1] > 3:
            triangles = np.delete(triangles, 0, 1) # this could screw things up for pyvista
    else:
        if len(triangles[0]) > 3:
            print('Error: COM calc bad array shape!')
            return [0.0, 0.0, 0.0]
    
    # centroid of the surface area of the mesh
    if method == 'mesh':
        area_sum=0.0
        centroid=(0.0,0.0,0.0)
        for t in triangles:
            center = (vectors[t[0]]+vectors[t[1]]+vectors[t[2]]) /3
            area = 0.5 * calcMag(np.cross(vectors[t[1]]-vectors[t[0]], vectors[t[2]]-vectors[t[0]]))
            centroid += area*center
            area_sum += area
        
        centroid /= area_sum
        return centroid

    # center of the mesh volume
    if method == 'volume':
        meshVolume = 0
        centroid = (0,0,0)
        for triangle in triangles:
            t = triangle
            center = (vectors[t[0]]+vectors[t[1]]+vectors[t[2]]) / 4 # center of tetrahedron
            volume = np.dot(vectors[t[0]], np.cross(vectors[t[1]], vectors[t[2]])) / 6 # signed volume of tetrahedron
            meshVolume += volume
            centroid += center * volume
        meshCenter = centroid / meshVolume
        return meshCenter

    # based on triangle centroids
    # I don't think this has a use!
    #centers = []
    #for triangle in triangles:
    #    t = triangle
    #    centers.append((vectors[t[1]]+vectors[t[2]]+vectors[t[3]]) / 3 ) # center of triangle
    #    area= 1/2 * np.abs( np.cross(vectors[t[2]]-vectors[t[1]], vectors[t[3]]-vectors[t[1]]) )
    #return np.mean(centers, axis=0)


def convert3MaticToOpenSimCoords(coords):
    newCoords = []
    for coord in coords:
        newCoords.append((coord[1]/1000.0, coord[2]/1000.0, coord[0]/1000.0))
    return newCoords
    
def convertOpenSimTo3MaticCoords(coords):
    newCoords = []
    for coord in coords:
        newCoords.append((coord[2]*1000.0, coord[0]*1000.0, coord[1]*1000.0))
    return newCoords