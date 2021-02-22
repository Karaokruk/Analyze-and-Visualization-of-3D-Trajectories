from trajectory_clustering import Trajectories
from trajectory_clustering import createRandomTrajectories
from trajectory_clustering import trajDistance

import numpy as np
import matplotlib.pyplot as plt
import random

# Get random trajectories
def initializationFromTrajectories(size, traj):
    # TODO
    # Make sure that the kmeans are not similar
    return random.sample(traj.trajectories, size)

# At each step, assign each trajectories to a cluster
def assignment(kmeans, traj):
    assign = []
    for t in traj.trajectories:
        y_hat = []
        for k in kmeans:
            y_hat.append(trajDistance(t, k, True))
        assign.append(np.argmin(y_hat))

    return assign

# Update each k-mean trajectories depending on 
def update(kmeans, traj, assign):
    return None

def kmean(k, traj, nb_iter = 10):
    print("\n-- K-MEAN CLUSTERING --\n")
    print("Initializing kmeans...")
    m = initializationFromTrajectories(k, traj)
    print("Done.\n")
    for _ in range(nb_iter):
        a = assignment(m, traj)
        print(a)
        update(m, traj, a)

traj = createRandomTrajectories()
kmean(3, traj)