# Author: Alastair Quinn 2023

import numpy as np

# add plane
# ray cast vectors to plane
# plot all points on plane
# find outside points
# measure angles
def get_plane_corners(normal, origin, size):
    # Get the perpendicular vectors to the plane normal
    vec1 = np.array([-normal[1], normal[0], 0])
    vec2 = np.cross(normal, vec1)
    
    # Normalize the vectors
    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)
    
    # Calculate the four corners of the plane
    corner1 = origin + vec1 * size + vec2 * size
    corner2 = origin + vec1 * size - vec2 * size
    corner3 = origin - vec1 * size + vec2 * size
    corner4 = origin - vec1 * size - vec2 * size
    
    return corner1, corner2, corner3, corner4

def fitPlaneNormal(data):
    # from: https://stackoverflow.com/questions/38754668/plane-fitting-in-a-3d-point-cloud
    mean = np.mean(data, axis=0)
    data_adjust = data - mean
    
    matrix = np.cov(data_adjust.T) 

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    #: sort eigenvalues and eigenvectors
    sort = eigenvalues.argsort()[::-1]
    #eigenvalues = eigenvalues[sort]
    eigenvectors = eigenvectors[:,sort]

    return eigenvectors[:,2], mean # normal, and center

def findPlaneIntersections(planeOrigin, planeNormal, planeSize, bonePoints):
    from pyvista import Plane
    planeNormal = np.array(planeNormal)
    plane = Plane(center = planeOrigin, direction = planeNormal,
                    i_size=planeSize, j_size=planeSize,
                    i_resolution=1, j_resolution=1)
    plane.triangulate(inplace=True)

    start = bonePoints + (planeNormal * planeSize)
    direction = np.full((start.shape[0], 3), -1.0 * planeNormal)

    # requres 'pip install trimesh rtree pyembree'
    points, rays, cells = plane.multi_ray_trace(start, direction, retry=True)
    return points

def project_points_onto_plane3D(points, plane_model):
    """
    Projects points onto a plane.

    Parameters
    ----------
    points : numpy.ndarray
        A numpy array of shape (n, 3) containing the 3D points.
    plane_model : tuple
        A tuple containing the plane model coefficients (a, b, c, d), where
        the plane equation is ax + by + cz + d = 0.

    Returns
    -------
    numpy.ndarray
        A numpy array of shape (n, 3) containing the projected points.
    """
    # Extract the plane normal and distance from origin
    normal = plane_model[:3]
    dist = plane_model[3]

    # Project the points onto the plane
    projected_points = points - np.dot(points, normal)[:, np.newaxis] * normal
    projected_points -= dist * normal

    return projected_points

def project_points_onto_plane2D(points, plane_normal, plane_origin, upward_vector):
    # Normalize the plane normal and upward vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    upward_vector = upward_vector / np.linalg.norm(upward_vector)

    # Calculate the right vector by taking the cross product of the plane normal and the upward vector
    right_vector = np.cross(plane_normal, upward_vector)
    right_vector = right_vector / np.linalg.norm(right_vector)

    # Calculate the projection matrix
    projection_matrix = np.zeros((3, 3))
    projection_matrix[:, 0] = right_vector
    projection_matrix[:, 1] = upward_vector
    projection_matrix[:, 2] = plane_normal

    # Calculate the projection of the points onto the plane
    projected_points = np.dot(points - plane_origin, projection_matrix)

    # Return the 2D representation of the points
    return projected_points[:, :2]

def transform_points_from_plane(points, plane_model):
    """
    Transforms points from a plane back to the original 3D space.

    Parameters
    ----------
    points : numpy.ndarray
        A numpy array of shape (n, 3) containing the points in the plane.
    plane_model : tuple
        A tuple containing the plane model coefficients (a, b, c, d), where
        the plane equation is ax + by + cz + d = 0.

    Returns
    -------
    numpy.ndarray
        A numpy array of shape (n, 3) containing the transformed points.
    """
    # Extract the plane normal and distance from origin
    normal = plane_model[:3]
    dist = plane_model[3]

    # Compute the inverse of the plane equation
    inv_denom = 1.0 / np.linalg.norm(normal)
    inv_normal = normal * inv_denom
    inv_dist = -dist * inv_denom

    # Transform the points from the plane to the 3D space
    transformed_points = points / inv_denom
    transformed_points -= inv_dist * inv_normal

    return transformed_points

def shortest_distance_to_origin(point_on_plane, plane_normal):
    """
    Calculates the shortest distance from a plane to the world origin.

    Parameters
    ----------
    point_on_plane : numpy.ndarray
        A numpy array of shape (3,) containing a point on the plane.
    plane_normal : numpy.ndarray
        A numpy array of shape (3,) containing the normal vector of the plane.

    Returns
    -------
    float
        The shortest distance from the plane to the world origin.
    """
    # Compute the distance from the plane to the world origin
    d = np.abs(np.dot(plane_normal, point_on_plane))

    a, b, c = plane_normal
    d = -(a * point_on_plane[0] + b * point_on_plane[1] + c * point_on_plane[2])

    return d

def orthogonal_matrix(normal):
    """
    Computes a 3x3 matrix that is orthogonal to a plane defined by its normal vector and a point on the plane.
    
    Args:
    normal (tuple): A tuple containing the x,y,z components of the plane's normal vector

    Returns:
    numpy.ndarray: A 3x3 numpy array that is orthogonal to the plane
    """
    # Create a vector v that is perpendicular to the plane's normal vector
    v = np.array(normal)
    if v[0] != 0:
        v = [-v[1], v[0], 0]
    else:
        v = [0, -v[2], v[1]]
    v = v / np.linalg.norm(v)
    
    # Create a second vector u that is perpendicular to both the normal vector and v
    u = np.cross(normal, v)
    u = u / np.linalg.norm(u)
    
    # Construct the orthogonal matrix using the three vectors
    return np.array([v, u, normal])

def convert_points_to_2d(points, plane_normal):
    # homogeneous coordinates
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    
    orthog_plane = np.vstack((orthogonal_matrix(plane_normal), np.zeros((1,3))))
    projected_points = np.dot(points_h, orthog_plane)

    # Extract the 2D coordinates of the projected points
    x = projected_points[:, 0]
    y = projected_points[:, 1]

    # Create a 2D numpy array from the coordinates
    points_2d = np.stack((x, y), axis=1)

    return points_2d

def convert_points_to_3d(points_2d, plane_origin, plane_normal):
    """
    Converts 2D points in a plane to 3D points on the plane.

    Parameters
    ----------
    points_2d : numpy.ndarray
        A numpy array of shape (n, 2) containing the 2D points in the plane.
    plane_origin : numpy.ndarray
        A numpy array of shape (3,) containing a point on the plane.
    plane_normal : numpy.ndarray
        A numpy array of shape (3,) containing the normal vector of the plane.

    Returns
    -------
    numpy.ndarray
        A numpy array of shape (n, 3) containing the 3D points on the plane.
    """
    # Check input shapes
    if points_2d.shape[1] != 2:
        raise ValueError("points_2d must have shape (n, 2)")
    if plane_origin.shape != (3,):
        raise ValueError("plane_origin must have shape (3,)")
    if plane_normal.shape != (3,):
        raise ValueError("plane_normal must have shape (3,)")

    # Compute the projection matrix for the plane
    p = np.eye(3) - np.outer(plane_normal, plane_normal)
    proj_mat = np.dot(p, np.transpose(p))

    # Compute the translation vector
    t = plane_origin

    # Add a column of ones to the points_2d array
    points_2d_hom = np.hstack((points_2d, np.ones((points_2d.shape[0], 1))))

    # Compute the 3D points on the plane
    points_3d = np.dot(points_2d_hom, proj_mat.T) + t.reshape(1, -1)

    return points_3d

#import cv2
#def outer_contour(points, resolution = 512):
#    """
#    Given a list of points in 2D, returns a list of points representing the outer contour.
#    
#    Args:
#    points (list): A list of tuples representing the x,y coordinates of each point
#    
#    Returns:
#    list: A list of tuples representing the x,y coordinates of each point on the outer contour
#    """
#    # Create a black image with a white polygon representing the given points
#    image = np.zeros((resolution, resolution, 3), np.uint8)
#
#    points_array = np.array(points, dtype=np.int32)
#    cv2.fillPoly(image, [points_array], (255, 255, 255))
#    #cv2.fillConvexPoly(image, points_array, (255, 255, 255))
#    #cv2.Can
#    #cv2.imwrite('test.png', image)
#    
#    # Convert the image to grayscale and apply a threshold
#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    ret, thresh = cv2.threshold(gray, 127, 255, 0)
#    
#    # Find the contours of the thresholded image
#    contours, hierarchy = cv2.findContours(thresh,
#                                           cv2.RETR_EXTERNAL,
#                                           cv2.CHAIN_APPROX_TC89_L1)
#    
#    # Get the points of the outer contour
#    outer_contour_points = []
#    for contour in contours:
#        for point in contour:
#            outer_contour_points.append(tuple(point[0]))
#    
#    # Return the points on the outer contour
#    return np.array(outer_contour_points)



def line_segments1(points):
    segments = []
    for i in range(len(points) - 1):
        segment = (points[i], points[i+1])
        segments.append(segment)
    print(segments)
    return np.array(segments)

def points_to_line_segments(points):
    segments = np.array([points[0], points[1]],ndmin=2)
    for i in range(1, len(points) - 1):
        segments = np.append(segments, [points[i]],axis=0)
        segments = np.append(segments, [points[i+1]],axis=0)
    return np.array(segments)

def linear_regression(data):
    """Find the line of best fit for a set of 2D data.

    Args:
        data (list or numpy array): The x&y-coordinates of the data points.

    Returns:
        tuple: A tuple containing the slope and y-intercept of the line of best fit.
    """
    if len(data[0]) != len(data[1]):
        raise ValueError("Input arrays must have the same length.")

    x_data = np.array(data[0])
    y_data = np.array(data[1])

    x_mean = np.mean(x_data)
    y_mean = np.mean(y_data)

    xy_mean = np.mean(x_data * y_data)
    x_squared_mean = np.mean(x_data**2)

    slope = (x_mean * y_mean - xy_mean) / (x_mean**2 - x_squared_mean)
    y_intercept = y_mean - slope * x_mean

    return slope, y_intercept

def findLine(data):
    """
    Uses x axis.
    """
    sliceAxis = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    coefficientsY = np.polyfit(sliceAxis, y, 1)
    coefficientsZ = np.polyfit(sliceAxis, z, 1)
    sFitted = np.arange(
        min(sliceAxis),
        max(sliceAxis),
        (max(sliceAxis) - min(sliceAxis))/(sliceAxis.shape[0])
        )
    
    yFitted = np.poly1d(coefficientsY)(sFitted)
    zFitted = np.poly1d(coefficientsZ)(sFitted)
    return np.array([sFitted, yFitted, zFitted]).T

def find_closest_point(points, p):
    """
    Find the closest point to a given point from an array of points in 3D space.

    Parameters:
    points (ndarray): Array of points in 3D space, with shape (N, 3).
    p (ndarray): The point to find the closest point to, with shape (3,).

    Returns:
    closest_point (ndarray): The closest point to the given point, with shape (3,).
    idx (int): The index of the closest point in the input array.
    """
    distances = np.linalg.norm(points - p, axis=1)
    closest_idx = np.argmin(distances)
    closest_point = points[closest_idx]
    return closest_point, closest_idx


def findPlane(points):
    # from: https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    # do fit
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)

    # Manual solution
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)

    #print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    #print("errors: \n", errors)
    #print("residual:", residual)
    return fit