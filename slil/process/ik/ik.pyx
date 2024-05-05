#cython: language_level=3

import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos, sqrt

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

cdef inline square(x): return x * x

#cpdef double cost_fast(np.float64_t x, np.float64_t y, np.float64_t z, np.float64_t rx, np.float64_t ry, np.float64_t rz, np.ndarray[np.float64_t] p0, np.ndarray[np.float64_t] p1, np.ndarray[np.float64_t] t_WOP1):   
cpdef DTYPE_t cost_fast(DTYPE_t x, DTYPE_t y, DTYPE_t z, DTYPE_t rx, DTYPE_t ry, DTYPE_t rz, DTYPE_t[:,:] p0_view, DTYPE_t[:,:] p1_view, DTYPE_t[:,:] t_WOP1_view):   

    #R = fm.eulerAnglesToRotationMatrix((rx, ry, rz))
    eul = np.array([rx, ry, rz])
    R = np.zeros((3, 3))
    cdef DTYPE_t[:, :] R_view = R
    cdef DTYPE_t[3] ct
    ct[0] = cos(rx)
    ct[1] = cos(ry)
    ct[2] = cos(rz)
    cdef DTYPE_t[3] st
    st[0] = sin(rx)
    st[1] = sin(ry)
    st[2] = sin(rz)
    #seq == 'ZYX':
    R_view[0,0] = ct[1]*ct[0]
    R_view[0,1] = st[2]*st[1]*ct[0] - ct[2]*st[0]
    R_view[0,2] = ct[2]*st[1]*ct[0] + st[2]*st[0]
    R_view[1,0] = ct[1]*st[0]
    R_view[1,1] = st[2]*st[1]*st[0] + ct[2]*ct[0]
    R_view[1,2] = ct[2]*st[1]*st[0] - st[2]*ct[0]
    R_view[2,0] = -st[1]
    R_view[2,1] = st[2]*ct[1]
    R_view[2,2] = ct[2]*ct[1]

    cdef cnp.ndarray[DTYPE_t, ndim=2] t_adjustment
    t_adjustment = np.eye(4)
    t_adjustment[:3, :3] = R_view
    t_adjustment[:3, 3] = np.array((x, y, z)).T

    t_adjustment = np.dot(t_adjustment, t_WOP1_view)

    #p1 = fm.transformPoints(p1, t_WOP1, t_adjustment)
    #M2 = inverseTransformationMat(tmStart)
    cdef cnp.ndarray[DTYPE_t, ndim=2] M2
    M2 = np.eye(4)
    M2[:3, :3] = np.linalg.inv(t_WOP1_view[:3, :3])
    #M2[:3, 3] = t_WOP1[:3, 3] * -1.0
    M2[0, 3] = t_WOP1_view[0, 3] * -1.0
    M2[1, 3] = t_WOP1_view[1, 3] * -1.0
    M2[2, 3] = t_WOP1_view[2, 3] * -1.0
    M2[:3, 3] = np.dot(M2[:3, :3], M2[:3, 3])

    #cdef DTYPE_t[:,:] p1_view = p1
    pointsTemp = np.ones((p1_view.shape[0], p1_view.shape[1] + 1))
    cdef DTYPE_t[:,:] pointsTemp_view = pointsTemp
    pointsTemp_view[:, :3] = p1_view[:,:]
    p1Temp = np.ones((p1_view.shape[0], p1_view.shape[1]))
    cdef DTYPE_t[:,:] p1Temp_view = p1Temp
    #p1Temp = p1[:,:]
    cdef int i
    r = np.ones((p1_view.shape[1]))
    cdef DTYPE_t[:] r_view = r
    for i in range(p1_view.shape[0]):
        #p1Temp_view[i] = np.dot(t_adjustment, np.dot(M2, pointsTemp[i]))[:3]
        #r_view = np.dot(t_adjustment, np.dot(M2, pointsTemp_view[i]))
        r_view = np.dot(t_adjustment, np.dot(M2, pointsTemp_view[i]))
        p1Temp_view[i, 0] = r_view[0]
        p1Temp_view[i, 1] = r_view[1]
        p1Temp_view[i, 2] = r_view[2]

    #error = np.sqrt(
    #        np.power(np.sum(p0_view[0,:] - p1[0,:]), 2) + 
    #        np.power(np.sum(p0_view[1,:] - p1[1,:]), 2) + 
    #        np.power(np.sum(p0_view[2,:] - p1[2,:]), 2))
    #cdef DTYPE_t error = sqrt(
    #    square(
    #        p0_view[0, 0] - p1Temp_view[0, 0] + 
    #        p0_view[0, 1] - p1Temp_view[0, 1] + 
    #        p0_view[0, 2] - p1Temp_view[0, 2]
    #        ) + 
    #    square(
    #        p0_view[1, 0] - p1Temp_view[1, 0] + 
    #        p0_view[1, 1] - p1Temp_view[1, 1] + 
    #        p0_view[1, 2] - p1Temp_view[1, 2]
    #        ) + 
    #    square(
    #        p0_view[2, 0] - p1Temp_view[2, 0] + 
    #        p0_view[2, 1] - p1Temp_view[2, 1] + 
    #        p0_view[2, 2] - p1Temp_view[2, 2]
    #        )
    #    )
    cdef DTYPE_t error = (
        square(p1Temp_view[0, 0] - p0_view[0, 0]) + 
        square(p1Temp_view[0, 1] - p0_view[0, 1]) + 
        square(p1Temp_view[0, 2] - p0_view[0, 2]) +
        square(p1Temp_view[1, 0] - p0_view[1, 0]) + 
        square(p1Temp_view[1, 1] - p0_view[1, 1]) + 
        square(p1Temp_view[1, 2] - p0_view[1, 2]) +
        square(p1Temp_view[2, 0] - p0_view[2, 0]) + 
        square(p1Temp_view[2, 1] - p0_view[2, 1]) + 
        square(p1Temp_view[2, 2] - p0_view[2, 2])) + 1.0

    #error = np.sqrt(np.sum((p0_view[0] - p1[0]) + (p0_view[1] - p1[1]) + (p0_view[2] - p1[2])))
    return error