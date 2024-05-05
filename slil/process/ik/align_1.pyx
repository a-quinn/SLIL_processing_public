#cython: language_level=3

from numpy.math cimport INFINITY
import cython
#from scipy.optimize import OptimizeResult
import numpy as np
cimport numpy as np

from libc.math cimport sin, cos, sqrt

np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef void align(np.ndarray[DTYPE_t, ndim=3] framesArg, long[:] plateAssignment, np.ndarray[DTYPE_t, ndim=2] t_adjustment):
    
    cdef int fN
    cdef int i, j
    cdef DTYPE_t [:, :, :] frames = framesArg
    for fN in range(0, frames.shape[0]):
        p0 = np.array(frames[fN][plateAssignment[0]])
        p1 = np.array(frames[fN][plateAssignment[1]])
        p2 = np.array(frames[fN][plateAssignment[2]]) # middle marker
        pm = (p0 - p1)/2 + p1 # between p0 and p1
        vecAligned = p2 - pm
        vecAligned = vecAligned / np.linalg.norm(vecAligned) # vector where wire is
        posAligned = p2
        vecAlignedNorm = calcNormalVec(p0, p1, p2)
        vecAlignedNorm = vecAlignedNorm / np.linalg.norm(vecAlignedNorm)
        
        rotAligned = np.array(create3DAxis(vecAligned, vecAlignedNorm)).T
        t_WOP1 = np.eye(4)
        t_WOP1[:3, :3] = rotAligned
        t_WOP1[:3, 3] = posAligned.T

        M2 = np.eye(4)
        M2[:3, :3] = np.linalg.inv(t_WOP1[:3, :3])
        M2[0, 3] = t_WOP1[0, 3] * -1.0
        M2[1, 3] = t_WOP1[1, 3] * -1.0
        M2[2, 3] = t_WOP1[2, 3] * -1.0
        M2[:3, 3] = np.dot(M2[:3, :3], M2[:3, 3])
        pointsTemp = np.ones((frames[fN].shape[0], frames[fN].shape[1] + 1))

        for i in range(frames[fN].shape[0]):
            for j in range(frames[fN].shape[1]):
                pointsTemp[i, j] = frames[fN, i, j]
        #pointsTemp[:, :3] = p1[:,:]
        #cdef DTYPE_t[:, :] p1Temp = np.zeros((p1.shape[0], p1.shape[1]))
        #p1Temp = p1[:,:]
        for i in range(pointsTemp.shape[0]):
            #p1[i] = np.dot(t_adjustment, np.dot(M2, pointsTemp[i]))[:3]
            r = np.dot(t_adjustment, np.dot(M2, pointsTemp[i]))
            frames[fN, i, 0] = r[0]
            frames[fN, i, 1] = r[1]
            frames[fN, i, 2] = r[2]


cdef DTYPE_t[:,:] create3DAxis(DTYPE_t[:] vec1, DTYPE_t[:] vec2):
    """ Returns 3 vectors which are all perpandicular
    (or close enough with e-15 kind of range)

    return: [v1, v2, v3]
    v1 is vec1
    v2 is perpandicular to vec1 and vec2
    v3 is perpandicular to vec1 and v2
    """
    cdef DTYPE_t out[3][3]
    cdef DTYPE_t [:, :] out_view = out
    v2 = np.cross(vec1, vec2)
    v3 = np.cross(vec1, v2)

    out_view[0,:3] = normalizeVector(vec1)
    out_view[1,:3] = normalizeVector(v2)
    out_view[2,:3] = normalizeVector(v3)
    return out_view

cdef DTYPE_t[:] normalizeVector(DTYPE_t[:] v):
    return v / np.linalg.norm(v)

cdef DTYPE_t[:] calcNormalVec(DTYPE_t[:] p0, DTYPE_t[:] p1, DTYPE_t[:] p2):
    cdef DTYPE_t u_cross_v[3]
    cdef DTYPE_t [:] u_cross_v_view = u_cross_v

    ux = p1[0]-p0[0]
    uy = p1[1]-p0[1]
    uz = p1[2]-p0[2]
    vx = p2[0]-p0[0]
    vy = p2[1]-p0[1]
    vz = p2[2]-p0[2]

    u_cross_v_view[0] = uy*vz-uz*vy
    u_cross_v_view[1] = uz*vx-ux*vz
    u_cross_v_view[2] = ux*vy-uy*vx
    return u_cross_v_view