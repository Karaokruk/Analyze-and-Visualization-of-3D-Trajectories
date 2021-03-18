from trajectory_clustering import Trajectories
from trajectory_clustering import createCSVTrajectories
from k_mean import kmean

import socket
import sys

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip_address = sys.argv[1]
port = int(sys.argv[2])

print(f"Connecting to {ip_address}, port {port}")
s.connect((ip_address, port))

### PYTHON TO UNITY FUNCTIONS ###
# Gets the message from Unity and sends back a message to Unity
def getMessageFromUnity(message, boolean=False):
    val = s.recv(999999)
    if boolean:
        val = bool(val.decode("utf8"))
    else:
        val = int(val.decode("utf8"))
    message += str(val)
    s.send(message.encode())
    return val

# Sends a message to Unity and waits for Unity to send back a message
def sendMessageToUnity(message):
    s.send(str(message).encode())
    r = s.recv(9999999)
    print(r.decode("utf8"))

### GETTING PARAMETERS ###
# File writing method
write_method = getMessageFromUnity("Python : Writing method selected : ")
# Number of files
trajLen = getMessageFromUnity("Python : Number of files : ")
# Number of kmeans
kmeans = getMessageFromUnity("Python : Number of clusters : ")
# Number of kmean method
method = getMessageFromUnity("Python : k-mean method selected : ")
# Number of kmean method
soft = getMessageFromUnity("Python : Using soft k-mean : ", boolean=True)
# Soft k-mean beta argument
beta = getMessageFromUnity("Python : Soft k-mean beta value : ")

### LOADING FILES ###

traj = Trajectories()
for i in range(trajLen):
    trajDir = s.recv(9999999)
    trajDir = trajDir.decode("utf8")

    traj.addTrajectoriesFromCsv(trajDir, groupBy = "None", groupID = "2", verbose = True)

    message = "Python : loaded file " + str(trajDir)
    s.send(message.encode())
    print(message)

### K-MEAN CLUSTERING ###

traj.attuneTrajectories(0.98, 0)
a, t = kmean(traj, k = kmeans, method = method, soft = soft, Beta = beta, verbose = False)
nb_files, file_names = traj.trajectoriesToCsv(write_method = write_method, k = kmeans, a = a)

# Sending assignment array
sendMessageToUnity(a)
# Sending file names created
sendMessageToUnity(file_names)
