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

def getAssignmentFromResponsabilty(responsability):
    assign = []
    for t in responsability.T:
        assign.append(np.argmax(t))
    return assign

# At each step, assign each trajectories to a cluster
def assignment(kmeans, traj):
    assign = []
    for t in traj.trajectories:
        y_hat = []
        for k in kmeans.trajectories:
            y_hat.append(Trajectories.trajectoryDistance(traj, t, k, heuristic = 0))
        assign.append(np.argmin(y_hat))
    responsability = np.zeros((len(kmeans.trajectories), len(traj.trajectories)))
    for i in range(len(kmeans.trajectories)):
        for j in range(len(traj.trajectories)):
            if assign[j] == i:
                responsability[i][j] = 1
    return responsability

def softAssignment(kmeans, traj, beta):
    responsability = np.zeros((len(kmeans.trajectories), len(traj.trajectories)))
    for i in range(len(kmeans.trajectories)):
        total = 0
        for j in range(len(traj.trajectories)):
            dist = np.exp(-beta * Trajectories.trajectoryDistance(traj, traj.trajectories[j], kmeans.trajectories[i], heuristic = 0))
            responsability[i][j] = dist
            total += dist
        responsability[i] /= total
    return responsability

# Update each k-mean trajectories depending on
def update(kmeans, traj, responsability):
    sums = np.zeros_like(kmeans.trajectories)
    cpts = np.zeros(len(sums))

    for i in range(len(responsability)):
        for j in range(len(responsability[i])):
            sums[i] += traj.trajectories[j] * responsability[i][j]
            cpts[i] += responsability[i][j]
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

    m = initializationFromTrajectories(k, workingTraj, per_layouts = False, per_layouts_randomization=True)
    print("Done.\n")
    for i in range(nb_iter):
        print(f"Iteration {i}.")
        r = softAssignment(m, workingTraj,1000)
        a = getAssignmentFromResponsabilty(r)
        m.trajectories, diff = update(m, workingTraj, r)
        print(a)
        input(r)
        if diff == 0:
            print(f"K-mean algorithm converged at iteration {i}. Stopped.")
            break
        else:
            print(f"The update made a total difference of {diff} on this iteration.")
    print("End of k-mean algorithm. Displaying clusters and clustered trajectories.")
    if method == 2:
        m = m.trajectoriesFromVectors()
    nbPerCluster = np.zeros(len(m.trajectories))
    for i in a:
        nbPerCluster[i] += 1
    print(nbPerCluster)
    m.showTrajectories()
    traj.showTrajectories(a)

traj = createCSVTrajectories("../datapoints/Participant_7_HeadPositionLog.csv")
kmean(round(len(traj.trajectories)/4), traj, nb_iter=20, method = 2)