# Author: Alastair Quinn 2022
try:
    import trimatic
    # if pyqt is used then trimatic needs to be replaced with threaded_trimatic
except:
    print('Failed to import trimatic module.')

class Mixin:
    def find_part(self, name):
        if self.usingSlicer:
            try:
                return self.client.fun('slicer.util.getNode', name)
            except Exception as e:
                print('Node not found.')
                return None
        return trimatic.find_part(name)
        
    def find_parts(self, names):
        if self.usingSlicer:
            list = []
            for name in names:
                try:
                    node = self.client.fun('slicer.util.getNode', name)
                except Exception as e:
                    print('Node not found.')
                    node = None
                list.append(node)
            return list
        return trimatic.find_parts(names)

    def find_point(self, name):
        if self.usingSlicer:
            return self.client.fun('getNode', name)
        return trimatic.find_point(name)

    def find_line(self, name):
        if self.usingSlicer:
            return self.client.fun('getNode', name)
        return trimatic.find_line(name)
