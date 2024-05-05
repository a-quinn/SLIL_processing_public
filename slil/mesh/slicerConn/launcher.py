# Author: Alastair Quinn 2022-2024
import psutil
import slil.mesh.slicerConn.client as client
from slil.mesh.slicerConn import _getSlicerInstallPath

procsSlicer = {}

def launchSlicer(client, port = 15002, visualise=True, closeAlreadyRunning = False, slicer_install_path = None):
    global procsSlicer
    import subprocess
    from time import sleep
    from pathlib import Path
    cwd = Path.cwd()

    if slicer_install_path == None:
        slicer_install_path = _getSlicerInstallPath()

    cmd = f"powershell -command \"& '{slicer_install_path}\\Slicer.exe'"
    if not visualise:
        cmd += " --no-main-window --show-python-interactor"
    cmd += " --python-script \'" + str(cwd) + "\\slil\\mesh\\slicerConn\\startup.py\'"
    cmd += " --port " + str(port)
    cmd += "\"; Start-Sleep -Seconds 4.5"

    #powershell -command "& '%USERPROFILE%\AppData\Local\NA-MIC\Slicer 5.0.3\Slicer.exe' --python-script 'C:\Users\Griffith\OneDrive - Griffith University\Projects\MTP - SLIL project\cadaver experiements\data processing vscode\slil\mesh\slicerConn\startup.py' -p 15004"

    check_running()
    if str(port) in procsSlicer.keys():
        if closeAlreadyRunning:
            print('Was already running, restarting.')
            procsSlicer[str(port)].terminate()
            del(procsSlicer[str(port)])
            if client.conn_3DSlicer != None:
                client.close_connection()
        else:
            print('Was already running, not restarting.')
            return False

    print('Starting 3D Slicer with RPyC on port {}'.format(port))
    #print("Running command: {}".format(cmd))
    process = subprocess.Popen(cmd, cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE)
    sleep(1.5)

    current_process = psutil.Process(process.pid)
    children = current_process.children(recursive=True)
    for i, proc in enumerate(children):
        if 'SlicerApp-real.exe' in proc.name():
            process = proc

    procsSlicer[str(port)] = process
    sleep(5.5) # wait for 3D Slicer to initialise

    #stdout, stderr = process.communicate()
    #print("Command output:\n{}".format(stdout.decode()))
    #returnCode = process.returncode
    #return returnCode

def check_running():
    global procsSlicer
    procsFound = []
    for proc in psutil.process_iter():
        if proc.name() == 'SlicerApp-real.exe':
            procsFound.append(proc.pid)
    #print(procsFound)
    pidsKnown = [procsSlicer[portID].pid for i, portID in enumerate(procsSlicer)]
    for i, pid in enumerate(procsFound):
        if not pid in pidsKnown:
            try:
                #print('found: {}'.format(pid))
                # get port from pid
                proc = psutil.Process(pid)
                if '--port' in proc.cmdline():
                    port = int(proc.cmdline()[proc.cmdline().index('--port')+1])
                    procsSlicer[str(port)] = proc
                else:
                    print('Found 3D Slicer running (pid: {}) with unknown RPyC port!'.format(pid))
            except psutil.AccessDenied:
                #print('Access denied to process with pid: {}'.format(pid))
                pass

    procsToRemove = []
    for i, portID in enumerate(procsSlicer):
        if not procsSlicer[portID].is_running(): # process has closed
            procsToRemove.append(portID)
    for i, portID in enumerate(procsToRemove):
        del(procsSlicer[portID])

def is_already_running(port):
    global procsSlicer
    try:
        for i, portID in enumerate(procsSlicer):
            for i, arg in enumerate(procsSlicer[portID].cmdline()):
                if '--port' in arg:
                    if port == int(procsSlicer[portID].cmdline()[i+1]):
                        return True
        return False
    except psutil.NoSuchProcess:
        return False

def closeAllSlicers(includingUnknownPorts = False):
    global procsSlicer
    portIDs= [portID for i, portID in enumerate(procsSlicer)]
    for portID in portIDs:
        closeSlicer(portID)
    if includingUnknownPorts:
        for proc in psutil.process_iter():
            if proc.name() == 'SlicerApp-real.exe':
                proc.terminate()

def closeSlicer(port = 15002):
    global procsSlicer
    if str(port) in procsSlicer.keys():
        print('Stopping 3D Slicer on port: {}'.format(port))
        procsSlicer[str(port)].terminate()
        del(procsSlicer[str(port)])
    else:
        print('No 3D Slicer with port {} was running.'.format(port))