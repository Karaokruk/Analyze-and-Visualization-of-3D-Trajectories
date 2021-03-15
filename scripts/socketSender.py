from trajectory_clustering import Trajectories
from trajectory_clustering import createCSVTrajectories
from k_mean import kmean

import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Connecting...")
s.connect(("192.168.1.21", 5000))

traj = createCSVTrajectories("../datapoints/SmallMultipleVR_Study/Study 2/Participant_7_HeadPositionLog.csv", verbose = False)
a = kmean(traj, k = 3, method = 1, verbose = False)

s.send(str(a).encode())
r = s.recv(9999999)
print(r.decode("utf8"))