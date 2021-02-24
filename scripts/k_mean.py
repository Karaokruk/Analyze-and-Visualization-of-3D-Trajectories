from trajectory_clustering import Trajectories
from trajectory_clustering import createRandomTrajectories

import numpy as np
import matplotlib.pyplot as plt
import random

# Get random trajectories
def initializationFromTrajectories(size, traj):
    # TODO
    # Make sure that the kmeans are not similar
    sample =  random.sample(traj.trajectories, size)
    kmeans = Trajectories()
    for t in sample:
        kmeans.addTrajectory(t)
    return kmeans

# At each step, assign each trajectories to a cluster
def assignment(kmeans, traj):
    assign = []
    for t in traj.trajectories:
        y_hat = []
        for k in kmeans.trajectories:
            y_hat.append(Trajectories.trajectoryDistance(traj, t, k))
        assign.append(np.argmin(y_hat))
    return assign

# Update each k-mean trajectories depending on 
def update(kmeans, traj, assign):
    sums = np.zeros_like(kmeans.trajectories)
    cpts = np.zeros(len(sums))

    for i in range(len(assign)):
        sums[assign[i]] += traj.trajectories[i]
        cpts[assign[i]] +=1
    for i in range(len(sums)):
        sums[i] /= cpts[i]
    print(kmeans.trajectories - sums)
    kmeans.trajectories = sums

def kmean(k, traj, nb_iter = 10):
    print("\n-- K-MEAN CLUSTERING --\n")
    print("Initializing kmeans...")
    m = initializationFromTrajectories(k, traj)
    translatedTraj = Trajectories()
    for t in traj.trajectories:
        translatedTraj.addTrajectory(t-t[0])
    #translatedTraj.completeDisplay()
    print("Done.\n")
    for _ in range(nb_iter):
        a = assignment(m, translatedTraj)
        print(a)
        #traj.showTrajectories(a)
        update(m, translatedTraj, a)
        m.showTrajectories()



traj = createRandomTrajectories(nb_trajectories=30)
kmean(4, traj)