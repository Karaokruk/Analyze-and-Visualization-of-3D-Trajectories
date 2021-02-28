from trajectory_clustering import Trajectories
from trajectory_clustering import createRandomTrajectories
from trajectory_clustering import createCSVTrajectories

import numpy as np
import matplotlib.pyplot as plt
import random

# Get random trajectories
def initializationFromTrajectories(size, traj):
    print("Initializing kmeans...")
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
            y_hat.append(Trajectories.trajectoryDistance(traj, t, k, heuristic = 0))
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
    diff = np.sum(np.absolute(kmeans.trajectories - sums))
    return sums, diff

def kmean(k, traj, nb_iter = 10, method = 2):
    print("\n-- K-MEAN CLUSTERING --\n")

    workingTraj = None
    if method == 2:
        workingTraj = traj.vectorizedTrajectories()
    elif method == 1:
        workingTraj = traj.translatedTrajectories()
    else:
        workingTraj = traj

    m = initializationFromTrajectories(k, workingTraj)
    print("Done.\n")
    for i in range(nb_iter):
        print(f"Iteration {i}.")
        a = assignment(m, workingTraj)
        m.trajectories, diff = update(m, workingTraj, a)
        if diff == 0:
            print(f"K-mean algorithm converged at iteration {i}. Stopped.")
            break
        else:
            print(f"The update made a total difference of {diff} on this iteration.")
    if method == 2:
        m = m.trajectoriesFromVectors()
    m.showTrajectories()
    traj.showTrajectories(a)

traj = createCSVTrajectories("../datapoints/Participant_7_HeadPositionLog.csv")
kmean(round(len(traj.trajectories)/3), traj, nb_iter=20, method = 2)