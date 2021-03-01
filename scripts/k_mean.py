from trajectory_clustering import Trajectories
from trajectory_clustering import createRandomTrajectories
from trajectory_clustering import createCSVTrajectories

import numpy as np
import matplotlib.pyplot as plt
import random

# Get random trajectories
def initializationFromTrajectories(size, traj, per_layouts = False, per_layouts_randomization = False):
    print("Initializing kmeans...")
    kmeans = Trajectories()
    # Initializing our kmeans randomly
    if not per_layouts:
        sample = random.sample(traj.trajectories, size)
        for t in sample:
            kmeans.addTrajectory(t)
    # Initialization using one random trajectory per layouts
    # NOT RANDOM
    else:
        layouts_got = []
        if per_layouts_randomization:
            indexes = list(range(0, len(traj.layouts)))
            random.shuffle(indexes)
            for i in indexes:
                if traj.layouts[i] not in layouts_got:
                    kmeans.addTrajectory(traj.trajectories[i])
                    layouts_got.append(traj.layouts[i])
                    print(f"Layout type : {traj.layouts[i]}")
        else:
            for i in range(len(traj.layouts)):
                print(f"Length : {len(traj.trajectories)}, layouts : {len(traj.layouts)}, i : {i}")
                if traj.layouts[i] not in layouts_got:
                    kmeans.addTrajectory(traj.trajectories[i])
                    layouts_got.append(traj.layouts[i])
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
    workingTraj.layouts = traj.layouts

    m = initializationFromTrajectories(k, workingTraj, per_layouts=True, per_layouts_randomization=True)
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
        workingTraj.show2DTrajectoriesSeparately(verbose = False, clusters = a)
    if method == 2:
        m = m.trajectoriesFromVectors()
    #m.showTrajectories()
    #traj.show2DTrajectoriesSeparately(clusters = a)
    workingTraj.show2DTrajectoriesSeparately(verbose = False, clusters = a)

traj = createCSVTrajectories("../datapoints/Participant_7_HeadPositionLog.csv")
#kmean(round(len(traj.trajectories)/3), traj, nb_iter=20, method = 2)
kmean(3, traj, method=0)