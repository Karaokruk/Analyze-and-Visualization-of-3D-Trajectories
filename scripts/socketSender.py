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

trajDir = s.recv(9999999)
s.send("Python : directory file received".encode())

traj = createCSVTrajectories(trajDir.decode("utf8"), verbose = False)
a = kmean(traj, k = 3, method = 1, verbose = False)

s.send(str(a).encode())
r = s.recv(9999999)
print(r.decode("utf8"))
