# Author: Alastair Quinn 2022
try:
    import trimatic
    # if pyqt is used then trimatic needs to be replaced with threaded_trimatic
except:
    print('Failed to import trimatic module.')
from .find import *
from slil.mesh.slicerConn.client import load_remote_functions

class Mixin:
    def move_to_part(self, entities, destination = None):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.move_to_part(entities, destination)

    def rotate(self, entities, angle_deg, axis_origin, axis_direction, number_of_copies = 0):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.rotate(entities, angle_deg, axis_origin, axis_direction, number_of_copies)

    def rotate_around_axes(self, entities, angle_axes, rotation_origin, number_of_copies = 0):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.rotate_around_axes(entities, angle_axes, rotation_origin, number_of_copies)

    def translate(self, entities, translation_vector, number_of_copies = 0):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.translate(entities, translation_vector, number_of_copies)

    def scale_factor(self, entities, factor, center_object = True, number_of_copies = 0):
        if self.usingSlicer:
            
            load_remote_functions(globals())
            SurfaceToolbox = slicer.util.getModuleWidget('SurfaceToolbox')
            SurfaceToolbox.initializeParameterNode()
            SurfaceToolboxLogic = SurfaceToolbox.logic
            
            if type(entities) != list:
                entities = [entities]
            for entity in entities:
                inputModel = self.find_part(entity)

                SurfaceToolboxLogic.transform(inputModel, inputModel,
                scaleX = factor,
                scaleY = factor,
                scaleZ = factor)
            return
        return trimatic.scale_factor(entities, factor, center_object, number_of_copies)

    def reduce(self, entities, flip_threshold_angle = 15.0, geometrical_error = 0.0500, number_of_iterations = 5, preserve_surfaces = False):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.reduce(entities, flip_threshold_angle, geometrical_error, number_of_iterations, preserve_surfaces)

    def get_selection(self):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.get_selection()

    def get_points(self):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.get_points()

    def compute_center_of_gravity(self, part, method='Based on mesh'):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.compute_center_of_gravity(part, method)
    
    def create_intersection_curve(self, entity_set1, entity_set2, intersection_curve_in=2):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.create_intersection_curve(entity_set1, entity_set2, intersection_curve_in)

    def merge(self, entities):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.merge(entities)

    def data_merge(self, entities):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.data.merge(entities)

    def duplicate(self, entities):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.duplicate(entities)

    def surfaces_to_parts(self, entities):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.surfaces_to_parts(entities)

    def n_points_registration(self, fixed_points, moving_points, moving_entities):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.n_points_registration(self, fixed_points, moving_points, moving_entities)

    def import_volume(self, filename):
        if self.usingSlicer:
            #return self.client.fun('slicer.util.loadSegmentation', filename, properties, True)
            #return self.client.fun('slicer.util.loadVolume', filename, properties, True)
            return self.client.fun('slicer.util.loadModel', filename)
        raise AttributeError('No 3-Matic wrapper.')

    def export_stl_binary(self, entities, output_directory, include_color = False):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.export_stl_binary(entities, output_directory, include_color)
    
    def export_stl(self, entities, output_directory, include_color = False):
        if self.usingSlicer:
            fin = []
            if type(entities) != list:
                entities = [entities]
            for entity in entities:
                a = self.find_part(entity)
                fin.append(self.client.fun('slicer.util.saveNode', a, output_directory))
            return fin
        raise AttributeError('No 3-Matic wrapper.')

    def open_project(self, projectFile):
        if self.usingSlicer:
            return self.client.fun('slicer.util.loadScene', projectFile)
        return trimatic.open_project(projectFile)

    def save_project(self, projectFile):
        if self.usingSlicer:
            return self.client.fun('slicer.util.saveScene', projectFile)
        return trimatic.save_project(projectFile)

    def close_project(self):
        if self.usingSlicer:
            return self.client.fun('slicer.mrmlScene.Clear', 0)
        raise AttributeError('No 3-matic wrapper.')
    
    def get_project_name(self):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.get_project_filename()

    def delete(self, i):
        if self.usingSlicer:
            raise AttributeError('No 3D Slicer wrapper.')
        return trimatic.delete(i)