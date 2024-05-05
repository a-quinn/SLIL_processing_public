# Author: Alastair Quinn 2022-2024
import rpyc
global fun_execute_remote

class client:
    def __init__(self, port = 15002):
        self.conn_3DSlicer = None
        self.portnumber = port

    def create_connection(self):
        global fun_execute_remote
        try:
            #self.conn_3DSlicer = rpyc.connect("100.78.75.5",self.portnumber, \
            # config = {"allow_public_attrs" : True, 'sync_request_timeout': 200})
            self.conn_3DSlicer = rpyc.connect("localhost", self.portnumber, \
                config = {"allow_public_attrs" : True, 'sync_request_timeout': 200})
            print("Connected to 3D Slicer listener.")
        except:
            raise RuntimeError("Could Not Connect, first start 3D Slicer \
                external IDE scripting service")
        fun_execute_remote = self.fun
        return True

    def close_connection(self):
        if self.conn_3DSlicer != None:
            self.conn_3DSlicer.close()
            self.conn_3DSlicer = None

    def check_connection(self):
        if self.conn_3DSlicer == None:
            return self.create_connection()
        elif self.conn_3DSlicer.closed == True:
            return self.create_connection()
        elif self.conn_3DSlicer.ping() == False:
            return self.create_connection()
        return True

    def send_command(self, name, mem, params):
        self.check_connection()
        try:
            return self.conn_3DSlicer.root.request_sent(name, mem, params)
        except EOFError: #try to reestablish the connection in case the listener restarted for some reason
            print("Connection to 3D Slicer lost, trying to reestablish connection.")
            self.create_connection()
        return self.conn_3DSlicer.root.request_sent(name, mem, params)

    def fun(self, method, *args):
        result = self.send_command(method, None, args)
        return result

    def funMember(self, member, method, *args):
        result = self.send_command(method, member, args)
        return result

def load_remote_functions(glob):
    global fun_execute_remote
    try:
        fun_execute_remote
    except NameError:
        print("well, it WASN'T defined after all!")
        return
    from slil.mesh.slicerConn import _getSlicerInstallPath
    import ast
    from os import listdir
    from pathlib import Path

    path = _getSlicerInstallPath()
    path = path.joinpath('bin', 'Python')

    #modules = [x for x in listdir(path) if Path(path.joinpath(x)).is_dir() and x != '__pycache__']
    # only use slicer for now. Other modules are not yet implemented as they're difficult.
    module_1 = 'slicer'

    new_functions = ''
    new_functions += f'global {module_1}\n\n'
    new_functions += f'class {module_1}:\n\n'
    path_2 = path.joinpath(module_1)
    files = [x for x in listdir(path_2) if Path(path_2.joinpath(x)).is_file() and x != '__pycache__']
    for file in files:
        #file = 'util.py'
        file_path = path.joinpath(module_1, file)

        file_name = file.split('.')[0]

        # read contents of a python file
        # find all functions
        # take the functions and replace the body with a template for remote call
        # execute all new functions

        with open(file_path, 'r') as file:
            file_contents = file.read()

        # Find all functions
        tree = ast.parse(file_contents)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

        other = [node for node in tree.body if not isinstance(node, ast.FunctionDef)]

        # Remove comments
        other_2 = [node for node in other if not isinstance(node, ast.Expr )]

        gs = [node for node in other_2 if isinstance(node, ast.Assign  )]
        # Create a new Python file with function templates

        new_functions += f'    class {file_name}:\n\n'
        for g in gs:
            text = ast.unparse(g)
            if '_Internal' in text or 'standalone_python' in text:
                continue
            new_functions += '        ' + text + '\n'
        for function in functions:
                
            #function = functions[7]
            #function = functions[14]
            #print(function.name)

            args = [x.arg for x in function.args.args]
            default_values = []
            for x in function.args.defaults:
                default_values.append(ast.unparse(x))
            
            #function.args.defaults[0]._fields
            #function.args.defaults[0].id

            for ind, arg in enumerate(args):
                #print(ind)
                if ind < len(default_values):
                    args[-ind - 1] = args[-ind - 1] + '=' + str(default_values[-ind - 1]).replace('\'', '\"')
                else:
                    args[-ind - 1] = args[-ind - 1]

            args_str = ''
            if len(args) > 0:
                args_str = ', '.join(args)
                args_str = args_str.replace('=,', '=\"\",')
                if args_str[-1] == '=':
                    args_str = args_str[:-1] + '=\"\"'

            args_wo_defaults = ', '.join([x.arg for x in function.args.args])

            function_template = f'''
        def {function.name}({args_str}):
            return fun_execute_remote('{module_1}.{file_name}.{function.name}', {args_wo_defaults})
'''
            new_functions += function_template
        new_functions += '        pass\n\n'

    #print(new_file_contents)
    #glob['fun_execute_remote'] = self.fun
    glob['fun_execute_remote'] = fun_execute_remote
    try:
        exec(new_functions, glob)
    except Exception as e:
        print('Failed to convert remote functions.')
        print(new_functions)
        print(e)
