# Author: Alastair Quinn 2022
# set this file as the application startup script in 3D Slicer settings
import slil.mesh.slicerConn.listener as listener
import sys
import getopt

def startRPyCServer(argv):
    arg_port = ""
    arg_help = "Bad inputs. Format:\n{0} --port <port number>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hp:", ["help", "port="])
    except:
        print(arg_help)
        #sys.exit(2)
        return
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            #sys.exit(2)
            return
        elif opt in ("-p", "--port"):
            arg_port = int(arg)
    
    print('Trying to setup RPyC server on port {}'.format(arg_port))
    listener.toggle_listener(arg_port)
    listener.toggle_executioner()

if __name__ == "__main__":
    startRPyCServer(sys.argv)