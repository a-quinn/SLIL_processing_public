# Author: Alastair Quinn 2022
try:
    import trimatic
    # if pyqt is used then trimatic needs to be replaced with threaded_trimatic
except:
    print('Failed to import trimatic module.')

class Mixin:
    def create_point(self, coords, name = 'Point'):
        if self.usingSlicer:
            #pointListNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
            #pointListNode.AddControlPoint(vtk.vtkVector3d(1,0,5))
            #pointListNode.SetName('Seed Point')
            #return pointListNode
            a = self.client.fun('slicer.mrmlScene.AddNewNodeByClass', "vtkMRMLMarkupsFiducialNode")
            vec = self.client.fun('vtk.vtkVector3d', coords[0], coords[1], coords[2])
            self.client.funMember(a, 'AddControlPoint', vec)
            self.client.funMember(a, 'SetName', name)
            return a

        return trimatic.create_point(coords)

    def create_line(self, point1, point2, name = 'Line'):
        if self.usingSlicer:
            a = self.client.fun('slicer.mrmlScene.AddNewNodeByClass', "vtkMRMLMarkupsLineNode")
            self.client.funMember(a, 'SetName', name)
            #self.client.funMember(a, 'SetPosition1', point1[0], point1[1], point1[2])
            #self.client.funMember(a, 'SetPosition2', point2[0], point2[1], point2[2])
            vec = self.client.fun('vtk.vtkVector3d', point1[0], point1[1], point1[2])
            self.client.funMember(a, 'AddControlPoint', vec)
            vec = self.client.fun('vtk.vtkVector3d', point2[0], point2[1], point2[2])
            self.client.funMember(a, 'AddControlPoint', vec)
            return a

        return trimatic.create_line(point1, point2)

    def create_line_direction_and_length(self, point, vector, wireLength):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.create_line_direction_and_length(point, vector, wireLength)

    def create_arc_3_points(self, a, b, c):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.create_arc_3_points(a, b, c)

    def create_plane_3_points(self, a, b, c):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.create_plane_3_points(a, b, c)

    def create_sphere_part(self, point, size):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.create_sphere_part(point, size)

    def create_cylinder_part(self, point_1, point_2, radius, tolerance = 0.01):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.create_cylinder_part(point_1, point_2, radius, tolerance)

    def create_line_plane_intersection(self, d1, d2):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.create_line_plane_intersection(d1, d2)

    def create_plane_2_points_perpendicular_1_plane(self, point1, point2, perpendicular_plane):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.create_plane_2_points_perpendicular_1_plane(point1, point2, perpendicular_plane)

    def create_distance_measurement(self, p0, p1):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.create_distance_measurement(p0, p1)
