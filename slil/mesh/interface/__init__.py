from sys import path
from os.path import isdir
materialise_dir = "C:\\Program Files\\Materialise\\3-matic 18.0 (x64)"
if not isdir(materialise_dir):
    print(f"Materialise 3-matic not found at '{materialise_dir}'. Please install it or change the path in slil/mesh/interface/__init__.py")
else:
    path.extend([materialise_dir])

# you need to enable pickleing by editing the 3-Matic\ScriptingUtils\toggle_script_listener.py
# Around line 100: args = {'protocol_config': {"allow_public_attrs" : True, "allow_setattr" : True, 'allow_pickle': True},

from . import create
from . import find
from . import misc
from . import util

using_slicer = False

class MeshInterface(create.Mixin, find.Mixin, misc.Mixin):
    
    def __init__(self, sessionID):
        self.sessionID = sessionID
        self.RPyCport = 15004 + self.sessionID
        self.usingSlicer = using_slicer
        if self.usingSlicer:
            from slil.mesh.slicerConn.client import client
            self.client = client(self.RPyCport)
        self.launchSession()
    
    def launchSession(self):
        if self.usingSlicer:
            if not self.isSessionAlive():
                print('3D Slicer not connected.')
                import slil.mesh.slicerConn.launcher as launch
                launch.launchSlicer(self.client, self.RPyCport)

                self.client.check_connection()

    def load_remote_functions(self, glob):
        from slil.mesh.slicerConn.client import load_remote_functions
        load_remote_functions(glob)

    def isSessionAlive(self):
        import slil.mesh.slicerConn.launcher as launch
        return launch.is_already_running(self.RPyCport) and self.client.check_connection()

    def closeSession(self):
        self.client.close_connection()
        import slil.mesh.slicerConn.launcher as launch
        launch.closeSlicer(self.RPyCport)

