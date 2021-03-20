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
def getMessageFromUnity(message, boolean=False, floating = False):
    val = s.recv(999999)
    val = (val.decode("utf8") == "True") if boolean else float(val.decode("utf8").replace('f','').replace(',','.')) if floating else int(val.decode("utf8"))
    message += str(val)
    s.send(message.encode())
    return val

# Sends a message to Unity and waits for Unity to send back a message
def sendMessageToUnity(message, size):
    begin_i = 0
    str_message = str(message)
    while True:
        end_i = begin_i + size
        if end_i > len(str_message):
            end_i = len(str_message)
        s.send(str_message[begin_i:end_i].encode())
        begin_i += size
        r = s.recv(9999999)
        print(r.decode("utf8"))
        if begin_i > len(str_message):
            break

### GETTING PARAMETERS ###
# File writing method
write_method = getMessageFromUnity("Python : Writing method selected : ")
# Number of files
trajLen = getMessageFromUnity("Python : Number of files : ")
# Attuning parameters
limit = getMessageFromUnity("Python : Attuning goal : ")
ratio = getMessageFromUnity("Python : Attuning ratio : ", floating=True)
# Number of iterations
nb_iter = getMessageFromUnity("Python : Number of iterations : ")
# Type of selection
per_layout = getMessageFromUnity("Python : Type of selection : ") == 1
per_layout_randomization = getMessageFromUnity("Python : Per layout randomization : ", boolean=True)
# Number of kmeans
kmeans = getMessageFromUnity("Python : Number of clusters : ")
# Number of kmean method
method = getMessageFromUnity("Python : k-mean method selected : ")
# Number of kmean method
soft = getMessageFromUnity("Python : Using soft k-mean : ", boolean=True)
# Soft k-mean beta argument
beta = getMessageFromUnity("Python : Soft k-mean beta value : ", floating=True)

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

traj.attuneTrajectories(ratio, limit)
a, t = kmean(traj, k = kmeans, nb_iter = nb_iter, method = method, soft = soft, Beta = beta, per_layout = per_layout, per_layout_randomization = per_layout_randomization, verbose = False)
nb_files, file_names = traj.trajectoriesToCsv(write_method = write_method, k = kmeans, a = a)

# Sending assignment array
new_a = []
for i in a:
    new_a.append([i])

sendMessageToUnity(new_a, 1024)
# Sending file names created
sendMessageToUnity(file_names, 1024)
