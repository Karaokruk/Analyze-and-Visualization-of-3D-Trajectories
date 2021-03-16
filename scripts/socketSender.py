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

### GETTING PARAMETERS ###
# File writing method
write_method = s.recv(9999999)
write_method = int(write_method.decode("utf8"))
message = "Python : Writing method " + str(write_method) + " selected"
s.send(message.encode())
# Number of files
trajLen = s.recv(9999999)
trajLen = int(trajLen.decode("utf8"))
message = "Python : counting " + str(trajLen) + " files"
s.send(message.encode())
# Number of kmeans
kmeans = s.recv(9999999)
kmeans = int(kmeans.decode("utf8"))
message = "Python : counting " + str(kmeans) + " clusters"
s.send(message.encode())
# Number of kmean method
method = s.recv(9999999)
method = int(method.decode("utf8"))
message = "Python : method " + str(method) + " selected"
s.send(message.encode())
# Number of kmean method
soft = s.recv(9999999)
soft = bool(soft.decode("utf8"))
if soft:
    message = "Python : Soft k-mean selected"
else:
    message = "Python : Regular k-mean selected"
s.send(message.encode())
# Soft k-mean beta argument
beta = s.recv(9999999)
beta = int(beta.decode("utf8"))
message = "Python : value of argument beta : " + str(beta)
s.send(message.encode())

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
traj.trajectoriesToCsv(write_method = write_method, k = kmeans, a = a)

s.send(str(a).encode())
r = s.recv(9999999)
print(r.decode("utf8"))
