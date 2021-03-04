from trajectory_clustering import Trajectories
from trajectory_clustering import createRandomTrajectories
from trajectory_clustering import createCSVTrajectories

import numpy as np
import matplotlib.pyplot as plt
import random

# Get random trajectories
def initializationFromTrajectories(size, traj, per_layout=False, per_layout_randomization=False, verbose=False):
    print("Initializing kmeans...")
    kmeans = Trajectories()
    # Initializing our kmeans randomly
    if not per_layout:
        sample = random.sample(traj.trajectories, size)
        def offsetTrajectory(t, min_offset=-0.1, max_offset=0.1):
            new_t = []
            for p in t:
                new_t.append(p + (np.random.uniform(min_offset, max_offset)))
            return new_t
        for t in sample:
            kmeans.addTrajectory(offsetTrajectory(t))
    # Initialization using one random trajectory per layout
    # NOT RANDOM
    else:
        layouts_got = []
        if per_layout_randomization:
            indexes = list(range(0, len(traj.layouts)))
            random.shuffle(indexes)
            for i in indexes:
                if traj.layouts[i] not in layouts_got:
                    kmeans.addTrajectory(traj.trajectories[i])
                    layouts_got.append(traj.layouts[i])
                    if verbose:
                        print(f"Layout type : {traj.layouts[i]}")
        else:
            for i in range(len(traj.layouts)):
                if verbose:
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
            y_hat.append(Trajectories.trajectoryDistance(traj, t, k, heuristic=0))
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

    #traj.showTrajectories()
    workingTraj = None
    if method == 2:
        workingTraj = traj.vectorizedTrajectories()
    elif method == 1:
        workingTraj = traj.translatedTrajectories()
    else:
        workingTraj = traj
    workingTraj.layouts = traj.layouts

    m = initializationFromTrajectories(k, workingTraj, per_layout = False, per_layout_randomization = False, verbose = True)
    print("Initialization done.\n")
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

def kmean_opencv(traj, k = 2, nb_iter = 10):
    print("\n-- OPENCV K-MEAN CLUSTERING --\n")
    import cv2 as cv

    Z = np.float32(traj.trajectories)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(Z, k, None, criteria, nb_iter, cv.KMEANS_RANDOM_CENTERS)

    a = []
    for l in label:
        a.append(l[0])

    traj.show2DTrajectoriesSeparately(clusters = a)

traj = createCSVTrajectories("../datapoints/SmallMultipleVR_Study/Study 2/Participant_7_HeadPositionLog.csv", verbose=False)
#kmean(round(len(traj.trajectories)/3), traj, nb_iter = 20, method = 2)
kmean(3, traj, method = 1)
#kmean_opencv(traj, k = 3, nb_iter = 10)
