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


cdef inline square(x): return x * x

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef DTYPE_t objective(DTYPE_t[:] x0, DTYPE_t[:,:] p0_view, DTYPE_t[:,:] p1_view, DTYPE_t[:,:] t_WOP1_view):   
    cdef DTYPE_t x = x0[0]
    cdef DTYPE_t y = x0[1]
    cdef DTYPE_t z = x0[2]
    cdef DTYPE_t rx = x0[3]
    cdef DTYPE_t ry = x0[4]
    cdef DTYPE_t rz = x0[5]

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

    cdef np.ndarray[DTYPE_t, ndim=2] t_adjustment
    t_adjustment = np.eye(4)
    t_adjustment[:3, :3] = R_view
    t_adjustment[:3, 3] = np.array((x, y, z)).T

    t_adjustment = np.dot(t_adjustment, t_WOP1_view)

    #p1 = fm.transformPoints(p1, t_WOP1, t_adjustment)
    #M2 = inverseTransformationMat(tmStart)
    cdef np.ndarray[DTYPE_t, ndim=2] M2
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

#cpdef DTYPE_t objective(DTYPE_t[:] x0, DTYPE_t[:,:] p0, DTYPE_t[:,:] p1, DTYPE_t[:,:] t_WOP1):
#    cdef DTYPE_t x = x0[0]
#    cdef DTYPE_t y = x0[1]
#    cdef DTYPE_t z = x0[2]
#    cdef DTYPE_t rx = x0[3]
#    cdef DTYPE_t ry = x0[4]
#    cdef DTYPE_t rz = x0[5]
#
#    #R = fm.eulerAnglesToRotationMatrix((rx, ry, rz))
#    eul = np.array([rx, ry, rz])
#    R = np.zeros((3,3))
#    cdef DTYPE_t[:, :] R_view = R
#    cdef DTYPE_t[3] ct
#    ct[0] = cos(rx)
#    ct[1] = cos(ry)
#    ct[2] = cos(rz)
#    cdef DTYPE_t[3] st
#    ct[0] = sin(rx)
#    ct[1] = sin(ry)
#    ct[2] = sin(rz)
#    #seq == 'ZYX':
#    R_view[0,0] = ct[1]*ct[0]
#    R_view[0,1] = st[2]*st[1]*ct[0] - ct[2]*st[0]
#    R_view[0,2] = ct[2]*st[1]*ct[0] + st[2]*st[0]
#    R_view[1,0] = ct[1]*st[0]
#    R_view[1,1] = st[2]*st[1]*st[0] + ct[2]*ct[0]
#    R_view[1,2] = ct[2]*st[1]*st[0] - st[2]*ct[0]
#    R_view[2,0] = -st[1]
#    R_view[2,1] = st[2]*ct[1]
#    R_view[2,2] = ct[2]*ct[1]
#
#    t_adjustment = np.eye(4)
#    t_adjustment[:3, :3] = R_view
#    t_adjustment[:3, 3] = np.array((x, y, z)).T
#
#    #p1 = fm.transformPoints(p1, t_WOP1, t_adjustment)
#    #M2 = inverseTransformationMat(tmStart)
#    M2 = np.eye(4)
#    M2[:3, :3] = np.linalg.inv(t_WOP1[:3, :3])
#    #M2[:3, 3] = t_WOP1[:3, 3] * -1.0
#    M2[0, 3] = t_WOP1[0, 3] * -1.0
#    M2[1, 3] = t_WOP1[1, 3] * -1.0
#    M2[2, 3] = t_WOP1[2, 3] * -1.0
#    M2[:3, 3] = np.dot(M2[:3, :3], M2[:3, 3])
#    pointsTemp = np.ones((p1.shape[0], p1.shape[1] + 1))
#    cdef int i, j
#    for i in range(p1.shape[0]):
#        for j in range(p1.shape[1]):
#            pointsTemp[i, j] = p1[i, j]
#    #pointsTemp[:, :3] = p1[:,:]
#    cdef DTYPE_t[:, :] p1Temp = np.zeros((p1.shape[0], p1.shape[1]))
#    #p1Temp = p1[:,:]
#    cdef int ii
#    for ii in range(pointsTemp.shape[0]):
#        #p1[i] = np.dot(t_adjustment, np.dot(M2, pointsTemp[i]))[:3]
#        r = np.dot(t_adjustment, np.dot(M2, pointsTemp[ii]))
#        p1Temp[ii, 0] = r[0]
#        p1Temp[ii, 1] = r[1]
#        p1Temp[ii, 2] = r[2]
#
#    #error = np.sqrt(
#    #        np.power(np.sum(p0[0,:] - p1[0,:]), 2) + 
#    #        np.power(np.sum(p0[1,:] - p1[1,:]), 2) + 
#    #        np.power(np.sum(p0[2,:] - p1[2,:]), 2))
#    cdef DTYPE_t error = sqrt(
#            square(p0[0, 0] - p1Temp[0, 0]) + 
#            square(p0[0, 1] - p1Temp[0, 1]) + 
#            square(p0[0, 2] - p1Temp[0, 2]) +
#            square(p0[1, 0] - p1Temp[1, 0]) + 
#            square(p0[1, 1] - p1Temp[1, 1]) + 
#            square(p0[1, 2] - p1Temp[1, 2]) +
#            square(p0[2, 0] - p1Temp[2, 0]) + 
#            square(p0[2, 1] - p1Temp[2, 1]) + 
#            square(p0[2, 2] - p1Temp[2, 2]))
#    #error = np.sqrt(np.sum((p0[0] - p1[0]) + (p0[1] - p1[1]) + (p0[2] - p1[2])))
#    return error

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def nelder_mead(DTYPE_t[:,:] p0, DTYPE_t[:,:] p1, DTYPE_t[:,:] t_WOP1,
                maxevals=int(1e3)):
#def nelder_mead(DTYPE_t[:,:] p0, DTYPE_t[:,:] p1, DTYPE_t[:,:] t_WOP1,
#                np.ndarray[DTYPE_t, ndim=1] x0,
#                np.ndarray[DTYPE_t, ndim=1] xmin,
#                np.ndarray[DTYPE_t, ndim=1] xmax,
#                np.ndarray[DTYPE_t, ndim=1] simplex_scale,
#                np.ndarray[DTYPE_t, ndim=1] xtol,
#                ftol=1e-7, maxevals=int(1e3)):
    '''
    Cythonized Nelder-mead optimization adapted from
    scipy.optimize.fmin. This function is not optimal for a large
    number of parameters beacause it uses an insertion sort
    and does not utilize parallel array operations.
    Arguments
    ---------
    objective : callable
        Scalar objective function to be minimized
    x0 : np.ndarray
        [N] Initial guess for solution in space of N parameters
    xmin : np.ndarray
        [N] Lower bounds for parameters. These should be 
        far lower than the values the simplex explores
        and is only meant to catch the simplex if it runs
        far off from the solution
    xmax : np.ndarray
        [N] Upper bounds for parameters. See xmin documentation
        for usage
    simplex_scale : np.ndarray or float
        [N] Scale factor for each parameter in generating an 
        initial simplex.
    xtol : np.ndarray or float
        [N] Tolerance in each parameter for convergence. The
        algorithm stops when all values in the simplex are 
        within xtol of each other
    Keywords
    --------
    ftol : float
        Tolerance in objective function for convergence. The
        algorithm stops when all function values in simplex are 
        within ftol of each other.
    maxevals : int
        Max number of function evaluations before function quits
    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Contains information about solution, best function value,
        number of function evaluations and iterations, reason for
        termination, and success of the fit. See scipy's documentation.
    '''
    cdef np.ndarray[DTYPE_t, ndim=1] x0 = np.zeros([6])
    cdef int N = x0.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] xmin = np.empty(N)
    xmin[:] = -100.0
    cdef np.ndarray[DTYPE_t, ndim=1] xmax = np.empty(N)
    xmax[:] = 100.0
    cdef np.ndarray[DTYPE_t, ndim=1] simplex_scale = np.ones([6])
    cdef np.ndarray[DTYPE_t, ndim=1] xtol = np.empty(N)
    xtol[:] = 1e-4
    cdef float ftol = 1e-7

    # Int and float values for number of parameters
    cdef DTYPE_t Nf = DTYPE(N)
    # Indexing
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = x0.shape[0]
    # Temporary variables for algorithm
    cdef DTYPE_t tempf, diff
    cdef int doshrink
    # Initialize simplex
    cdef np.ndarray[DTYPE_t, ndim=2] simplex = np.empty([N+1, N])
    cdef np.ndarray[DTYPE_t, ndim=2] temp2d = np.empty([N+1, N])
    cdef np.ndarray[DTYPE_t, ndim=1] temp1d = np.empty(N)
    for i in range(N):
        simplex[0, i] = x0[i]
        for j in range(N):
            simplex[i+1, j] = x0[j]
        simplex[i+1, i] += simplex_scale[i]
    # Create cdefs for python keyword arguments
    cdef int max_nfev = maxevals
    cdef DTYPE_t fcntol = ftol
    # Allocate buffers for current function evaluations and
    # index ordering, and ints for # of evaluations and iterations
    cdef int neval = 1
    cdef int niter = 1
    cdef np.ndarray[DTYPE_t, ndim=1] evals = np.empty(N+1)
    cdef np.ndarray[Py_ssize_t, ndim=1] idxs = np.empty(N+1,
                                                        dtype=np.intp)
    # Fill evaluations for initial simplex and sort
    for i in range(N+1):
        for j in range(N):
            temp1d[j] = simplex[i, j]
        minimum(temp1d, xmax)
        maximum(temp1d, xmin)
        for j in range(N):
            simplex[i, j] = temp1d[j]
        evals[i] = objective(temp1d, p0, p1, t_WOP1)
        neval += 1
    sort(evals, idxs)
    take(simplex, idxs, temp2d)

    # Parameters for simplex transformations
    cdef DTYPE_t rho = 1.
    cdef DTYPE_t chi = 2.
    cdef DTYPE_t psi = 0.5
    cdef DTYPE_t sigma = 0.5

    # Buffers and floats for transformations and their evaluations
    cdef np.ndarray[DTYPE_t, ndim=1] xbar = np.empty(N)
    cdef np.ndarray[DTYPE_t, ndim=1] xr = np.empty(N)
    cdef np.ndarray[DTYPE_t, ndim=1] xe = np.empty(N)
    cdef np.ndarray[DTYPE_t, ndim=1] xc = np.empty(N)
    cdef np.ndarray[DTYPE_t, ndim=1] xcc = np.empty(N)
    cdef DTYPE_t fxr, fxe, fxc, fxcc

    # Message and boolean for termination
    cdef str message = 'failure (hit max evals)'
    cdef bint end

    # START FITTING
    while(neval < max_nfev):
        # Test if simplex is small
        end = True
        for i in range(N):
            tempf = simplex[0, i]
            for j in range(1, N+1):
                diff = simplex[j, i] - tempf
                if diff < 0.:
                    diff = diff * -1.
                if diff > xtol[i]:
                    end = False
                    break
            if end is False:
                break
        if end:
            message = 'convergence (simplex small)'
            break
        # Test if function values are similar
        tempf = -INFINITY
        for i in range(1, N+1):
            diff = evals[0] - evals[i]
            if diff < 0.:
                diff = diff * -1.
            if diff > tempf:
                tempf = diff
        if tempf <= fcntol:
            message = 'convergence (fvals similar)'
            break
        # Test if simplex hits edge of parameter space
        end = False
        for j in range(N):
            for i in range(N+1):
                if xmax[j] == simplex[i, j] or xmin[j] == simplex[i, j]:
                    end = True
        if end:
            message = 'failure (stuck to boundary)'
            break
        # Reflection
        for j in range(N):
            tempf = 0.
            for i in range(N):
                tempf += simplex[i, j]
            xbar[j] = tempf / Nf
        for j in range(N):
            xr[j] = (1 + rho) * xbar[j] - rho * simplex[n, j]
        minimum(xr, xmax)
        maximum(xr, xmin)
        fxr = objective(xr, p0, p1, t_WOP1)
        neval += 1
        doshrink = 0
        # Check if reflection is better than best estimate
        if fxr < evals[0]:
            # If so, reflect double and see if that's even better
            for j in range(N):
                xe[j] = (1. + rho * chi) * xbar[j]\
                    - rho * chi * simplex[n, j]
            minimum(xe, xmax)
            maximum(xe, xmin)
            fxe = objective(xe, p0, p1, t_WOP1)
            neval += 1
            if fxe < fxr:
                for j in range(N):
                    simplex[n, j] = xe[j]
                evals[n] = fxe
            else:
                for j in range(N):
                    simplex[n, j] = xr[j]
                evals[n] = fxr
        else:
            if fxr < evals[n-1]:
                for j in range(N):
                    simplex[n, j] = xr[j]
                evals[n] = fxr
            else:
                # If reflection is not better, contract.
                if fxr < evals[n]:
                    for j in range(N):
                        xc[j] = (1. + psi * rho) * xbar[j] \
                            - psi * rho * simplex[n, j]
                    minimum(xc, xmax)
                    maximum(xc, xmin)
                    fxc = objective(xc, p0, p1, t_WOP1)
                    neval += 1
                    if fxc <= fxr:
                        for j in range(N):
                            simplex[n, j] = xc[j]
                        evals[n] = fxc
                    else:
                        doshrink = 1
                else:
                    # Do 'inside' contraction
                    for j in range(N):
                        xcc[j] = (1 - psi) * xbar[j] \
                            + psi * simplex[n, j]
                    minimum(xcc, xmax)
                    maximum(xcc, xmin)
                    fxcc = objective(xcc, p0, p1, t_WOP1)
                    neval += 1
                    if fxcc < evals[n]:
                        for j in range(N):
                            simplex[n, j] = xcc[j]
                        evals[n] = fxcc
                    else:
                        doshrink = 1
                if doshrink:
                    for i in range(1, N+1):
                        for j in range(N):
                            simplex[i, j] = simplex[0, j] + sigma * \
                                (simplex[i, j] - simplex[0, j])
                            temp1d[j] = simplex[i, j]
                        minimum(temp1d, xmax)
                        maximum(temp1d, xmin)
                        for j in range(N):
                            simplex[i, j] = temp1d[j]
                        evals[i] = objective(temp1d, p0, p1, t_WOP1)
                        neval += 1
        sort(evals, idxs)
        take(simplex, idxs, temp2d)
        niter += 1
    success = False if 'failure' in message else True
    return simplex[0]
    #result = OptimizeResult(x=simplex[0],
    #                        success=success,
    #                        message=message,
    #                        nit=niter,
    #                        nfev=neval,
    #                        fun=evals[0])
    #return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void take(np.ndarray[DTYPE_t, ndim=2] a,
               np.ndarray[Py_ssize_t, ndim=1] idxs,
               np.ndarray[DTYPE_t, ndim=2] buff):
    '''
    Replaces order of a's axis 0 according to idxs
    '''
    cdef int N1, N2
    N1 = a.shape[0]
    N2 = a.shape[1]
    cdef Py_ssize_t i, j
    for i in range(N1):
        for j in range(N2):
            buff[i, j] = a[idxs[i], j]
    for i in range(N1):
        for j in range(N2):
            a[i, j] = buff[i, j]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void minimum(np.ndarray[DTYPE_t, ndim=1] buff,
                  np.ndarray[DTYPE_t, ndim=1] bounds):
    '''
    Takes elementwise minimum of buff and bounds, storing
    the result in buff
    '''
    cdef int N = buff.shape[0]
    cdef Py_ssize_t i
    for i in range(N):
        if buff[i] > bounds[i]:
            buff[i] = bounds[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void maximum(np.ndarray[DTYPE_t, ndim=1] buff,
                  np.ndarray[DTYPE_t, ndim=1] bounds):
    '''
    Takes elementwise maximum of buff and bounds, storing
    the result in buff
    '''
    cdef int N = buff.shape[0]
    cdef Py_ssize_t i
    for i in range(N):
        if buff[i] < bounds[i]:
            buff[i] = bounds[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sort(np.ndarray[DTYPE_t, ndim=1] a,
               np.ndarray[Py_ssize_t, ndim=1] order):
    '''
    Inplace insertion sort that modifies order to reflect
    ordering of the sorted a
    '''
    cdef int N = a.shape[0]
    cdef Py_ssize_t i, j
    cdef DTYPE_t key

    for i in range(N):
        order[i] = i

    for i in range(1, N):
        key = a[i]
        j = i-1
        while j >= 0 and key < a[j]:
            a[j+1] = a[j]
            order[j+1] = order[j]
            j -= 1
        a[j+1] = key
        order[j+1] = i