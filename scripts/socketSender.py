from trajectory_clustering import Trajectories
from trajectory_clustering import createCSVTrajectories
from k_mean import kmean

import socket
import sys

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#hostname = socket.gethostname()
#ip_address = socket.gethostbyname(hostname)
ip_address = sys.argv[1]
port = int(sys.argv[2])
print(f"Connecting to {ip_address}, port {port}")
s.connect((ip_address, port))

trajLen = s.recv(9999999)
trajLen = int(trajLen.decode("utf8"))
message = "Python : counting " + str(trajLen) + " files"
s.send(message.encode())

traj = Trajectories()
for i in range(trajLen):
    trajDir = s.recv(9999999)
    trajDir = trajDir.decode("utf8")

    traj.addTrajectoriesFromCsv(trajDir, groupBy = "Trial", groupID = 2)

    message = "Python : loaded file " + str(trajDir)
    s.send(message.encode())
    print(message)


traj.attuneTrajectories(0.98, 0)
a, t = kmean(traj, k = 3, method = 1, verbose = False)

s.send(str(a).encode())
r = s.recv(9999999)
print(r.decode("utf8"))
