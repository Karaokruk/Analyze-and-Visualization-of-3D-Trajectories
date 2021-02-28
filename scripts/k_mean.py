from trajectory_clustering import Trajectories
from trajectory_clustering import createRandomTrajectories
from trajectory_clustering import createCSVTrajectories

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
        cpts[assign[i]] += 1
    for i in range(len(sums)):
        if cpts[i] != 0:
            sums[i] /= cpts[i]
        else:
            sums[i] = kmeans.trajectories[i]
    kmeans.trajectories = sums
    return kmeans


def kmean(k, traj, nb_iter=10, translation=True):
    print("\n-- K-MEAN CLUSTERING --\n")
    print("Initializing kmeans...")

    workingTraj = None
    if translation:
        workingTraj = Trajectories()
        for t in traj.trajectories:
            workingTraj.addTrajectory(t - t[0])
    else:
        workingTraj = traj

    m = initializationFromTrajectories(k, workingTraj)
    print("Done.\n")
    for _ in range(nb_iter):
        a = assignment(m, workingTraj)
        update(m, workingTraj, a)
        #m.showTrajectories()
        workingTraj.showTrajectories(clusters = a)

traj = createCSVTrajectories("../datapoints/Participant_7_HeadPositionLog.csv")
traj.show2DTrajectoriesSeparately(verbose = True)
#kmean(4, traj, nb_iter=20)
