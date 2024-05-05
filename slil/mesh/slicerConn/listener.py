# Author: Alastair Quinn 2022-2024

# Used by 3D Slicer's python interpreter
from concurrent.futures import thread
import threading
from .. import slicerConn

import sys
import time
import slicer
import vtk
import queue

has_required_packages = True
defaultPortNumber = 15002

try:
  slicer.util.pip_install('rpyc')
  import rpyc
  from rpyc.utils.server import ThreadedServer, OneShotServer
except:
  has_required_packages = False
  pass
  
# Evaluated every ms to retrieve python commands send to server
def execute_command(send_queue, receive_queue, wait_for_ms = None, exception_queue =None):
    # block the calling thread for 0.005 seconds, this will
    # allow rpyc server thread to process the commands.
    # with out this delay command execution will be slow(strange!!!)
    # the number is a magic number(found out by trial and error
    # method)
    # decreasing or increasing delay will impact the performance.
    if wait_for_ms is not None:
       time.sleep(wait_for_ms)
    while not send_queue.empty():
        try:
            command_dict = send_queue.get_nowait()
            name = command_dict['name']
            mem = command_dict['mem']
            args = command_dict['args']
            if 'event' in command_dict.keys(): # 'event' key is valid only for command queued by other threads
                event = command_dict['event']
                kwargs = command_dict['kwargs']
                try:
                   result = name(*args, **kwargs) # issued by other threads
                   receive_queue.put(result)
                except Exception as e :
                       if exception_queue is not None:
                          exception_queue.put(e)
                       pass
                event.set()
            else:
                if mem is not None:
                    method = getattr(mem, name)
                    returnvalue = method(*args)
                else:
                    #print('Events not implimented2!')
                    if name[:4] == 'vtk.':
                        sub_name = name[4:]
                        if '.' in sub_name:
                            parts = sub_name.split('.')
                            method1 = getattr(vtk, parts[0])
                            method = getattr(method1, parts[1])

                            #method1 = getattr(vtk, name[name.index('.')+1:])
                            #method = getattr(method1, name[name.index('.')+1:])
                        else:
                            method = getattr(vtk, sub_name)
                    elif name[:7] == 'slicer.':
                        sub_name = name[7:]
                        if '.' in sub_name:
                            print(name)
                            parts = sub_name.split('.')
                            print(parts)
                            
                            method1 = getattr(slicer, parts[0])
                            method = getattr(method1, parts[1])
                        else:
                            method = getattr(slicer, sub_name)
                    else:
                        if '.' in name:
                            parts = name.split('.')
                            method = getattr(parts[0], parts[1])
                        else:
                            method = getattr(globals, name)
                    try:
                        if '__name__' in name:
                            returnvalue = method
                        else:
                            returnvalue = method(*args)
                    except Exception as e:
                        print('Error in executing command: ', name, args)
                        print(e)
                        returnvalue = ('exception', e)
                receive_queue.put(returnvalue)
        except queue.Empty:
            pass
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            receive_queue.put(e)

slicerConn.script_listener_send_queue = queue.Queue()
slicerConn.script_listener_receive_queue = queue.Queue()
slicerConn.listener_server_thread = None

def command_fetch_timer_callback():
    #print('execute1')
    execute_command(slicerConn.script_listener_send_queue, slicerConn.script_listener_receive_queue,0.005)
    # This will execute methods queued by other threads
    #execute_command(main_thread_send_queue, 
    #                main_thread_receive_queue,
    #                None, 
    #                main_thread_execption_queue)

if has_required_packages:
    # create_slicerConn_listener_service is a wrapper enabling to pass two queues
    # to the Service class of rpyc.
    def create_slicerConn_listener_service(send_queue, receive_queue):
        class ListenerService(rpyc.Service):
            # code that runs when a connection is created
            # Notify the user that the server connected
            def on_connect(self, conn):
                print("Connected to Server")
 
            def exposed_request_sent(self, name, mem, args):
                command_dict = {'name': name, 'mem': mem, 'args': args}
                                
                send_queue.put(command_dict) # Put the command on the queue
                answer = receive_queue.get()  # Blocking wait for for answer
                if type(answer) == RuntimeError: # Catch Exceptions
                    raise answer
                elif type(answer) == AttributeError:
                    raise answer
                elif type(answer) == ValueError:
                    raise answer
                elif type(answer) == TypeError:
                    raise answer
                elif type(answer) == EOFError: # happens when the connection to the listener is lost
                    raise answer
                elif type(answer) == tuple and answer[0] == 'exception':
                    raise answer[1]
                else:
                    return answer
        return ListenerService
 
    class ServerThread(threading.Thread):
       def __init__(self, send_queue, receive_queue, portnumber):
            threading.Thread.__init__(self)
            slicerConn_listener_service = create_slicerConn_listener_service(send_queue, receive_queue)
            args = {'protocol_config': {"allow_public_attrs" : True, "allow_setattr" : True},
                    'port': portnumber } #Set arguments of server, allow to retrieve attributes for debugging.
            self.processing_server = ThreadedServer(slicerConn_listener_service, **args)
 
       def run(self):
            self.processing_server.start()
 
    def reset_listener_thread():
       slicerConn.listener_server_thread.processing_server.close()
       #closing the processing server will cause the server to stop
       #and the thread to end
       slicerConn.listener_server_thread.join()
       slicerConn.listener_server_thread = None
 
    def toggle_listener(portnumber = defaultPortNumber):
        try:
            if slicerConn.listener_server_thread is None:   
                # There can be exception when socket is not bound, which will make to fail the initailization of ThreadedServer
                slicerConn.listener_server_thread = ServerThread(slicerConn.script_listener_send_queue, slicerConn.script_listener_receive_queue, portnumber)
        except:
                print("Failed to start 3D Slicer script listener: Already in use in another instance")
                return False
             
        if not slicerConn.listener_server_thread.is_alive():
            try:
                slicerConn.listener_server_thread.start()
                print("3D Slicer script listener is active")
                return True
            except:
                ## The threaded servers can't be intialized, because port
                ## #15000 is not available.
                ## in this case release all the resources which are aquired.
                print("Failed to start 3D Slicer script listener")
                if slicerConn.listener_server_thread.processing_server.active:
                    reset_listener_thread()
                return False
        else:
            try:
                reset_listener_thread()
                print("3D Slicer script listener is deactivated")
                return False
            except:
                print("Failed to deactivate 3D Slicer script listener")
                return True

class ExecServerThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    # Initialisation method
    def run(self):
        self.running = True
        while self.running:
            time.sleep(0.5)
            print('trying...')
            command_fetch_timer_callback()
    
    def stop(self):
        self.running = False

from qt import QTimer

slicerConn.timer = QTimer()
slicerConn.timer.timeout.connect(command_fetch_timer_callback)

slicerConn.executioner_server_thread = None

def reset_executioner_thread():
    slicerConn.executioner_server_thread.stop()
    #closing the processing server will cause the server to stop
    #and the thread to end
    slicerConn.executioner_server_thread.join()
    slicerConn.executioner_server_thread = None

def toggle_executioner():
    
    if slicerConn.timer.isActive():
        slicerConn.timer.stop()
        return False
    else:
        slicerConn.timer.start(10)
    return True