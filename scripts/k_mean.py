from trajectory_clustering import Trajectories
from trajectory_clustering import createRandomTrajectories
from trajectory_clustering import createCSVTrajectories

import numpy as np
import matplotlib.pyplot as plt
import random

# Get random trajectories
def initializationFromTrajectories(size, traj, per_layout = False, per_layout_randomization = True, verbose = False):
    print("Initializing kmeans...")
    kmeans = Trajectories()
    # Initializing clusters randomly
    if not per_layout:
        sample = random.sample(traj.trajectories, size)
        def offsetTrajectory(t, min_offset = -0.1, max_offset = 0.1):
            new_t = []
            for p in t:
                new_t.append(p + (np.random.uniform(min_offset, max_offset)))
            return new_t
        for t in sample:
            kmeans.addTrajectory(offsetTrajectory(t))
    # Initialization using one random trajectory per layout
    else:
        layout_types = []
        # Randomly generate the clusters depending on the layouts
        if per_layout_randomization:
            indexes = list(range(0, len(traj.layouts)))
            random.shuffle(indexes)
            for i in indexes:
                if traj.layouts[i] not in layout_types:
                    kmeans.addTrajectory(traj.trajectories[i])
                    layout_types.append(traj.layouts[i])
                    if verbose:
                        print(f"Layout type : {traj.layouts[i]}")
        # Getting the first occurence of each layouts
        else:
            for i in range(len(traj.layouts)):
                if traj.layouts[i] not in layout_types:
                    kmeans.addTrajectory(traj.trajectories[i])
                    layout_types.append(traj.layouts[i])
    return kmeans

# Returns the assigned cluster for each trajectory
def getAssignmentFromResponsibility(responsibility):
    assign = []
    for t in responsibility.T:
        assign.append(np.argmax(t))
    return assign


# At each step, assign each trajectories to a cluster
def assignment(kmeans, traj):
    assign = []
    # Find the closest cluster for each trajectory
    for t in traj.trajectories:
        y_hat = []
        for k in kmeans.trajectories:
            y_hat.append(Trajectories.trajectoryDistance(traj, t, k, heuristic = 0))
        assign.append(np.argmin(y_hat))

    # Put the responsability value to 1 for the closest, 0 for the others
    responsibility = np.zeros((len(kmeans.trajectories), len(traj.trajectories)))
    for i in range(len(kmeans.trajectories)):
        for j in range(len(traj.trajectories)):
            if assign[j] == i:
                responsibility[i][j] = 1
    return responsibility


# Alternative version of the assignment, where each trajectory takes a ratio of the responsablity for a cluster
# Parameter beta is the stiffness, in [0, inf]. The higher it goes, the closer it will be to "hard" k-mean.
def softAssignment(kmeans, traj, beta):
    responsibility = np.zeros((len(kmeans.trajectories), len(traj.trajectories)))
    for i in range(len(kmeans.trajectories)):
        total = 0
        for j in range(len(traj.trajectories)):
            dist = np.exp(-beta * Trajectories.trajectoryDistance(traj, traj.trajectories[j], kmeans.trajectories[i], heuristic = 0))
            responsibility[i][j] = dist
            total += dist
        responsibility[i] /= total
    return responsibility

# Update each cluster trajectories to match the trajectories they are responsible of
def update(kmeans, traj, responsibility):
    sums = np.zeros_like(kmeans.trajectories)
    cpts = np.zeros(len(sums))

    for i in range(len(responsibility)):
        for j in range(len(responsibility[i])):
            sums[i] += traj.trajectories[j] * responsibility[i][j]
            cpts[i] += responsibility[i][j]
    for i in range(len(sums)):
        if cpts[i] != 0:
            sums[i] /= cpts[i]
        else:
            sums[i] = kmeans.trajectories[i]
    diff = np.sum(np.absolute(kmeans.trajectories - sums))
    return sums, diff

# K-mean algorithm. Clusters trajectories that are more alike depending on different criterias.
# Parameter k is the desired number of clusters.
# Parameter method : in [0,2]. 0 is for the point by point euclidian distance, same for 1, but their share a common origin, and 2 is the vector by vector distance.
# Parameter soft : wheter the soft alternative should be used or not.
def kmean(traj, k = 3, nb_iter = 10, method = 2, soft = True, Beta = 1000, verbose = False):
    if verbose:
        print("\n-- Starting K-mean clustering --\n")

    # Initialize the working set of trajectories
    workingTraj = None
    if method == 0:
        workingTraj = traj
    elif method == 1:
        workingTraj = traj.translatedTrajectories()
    elif method == 2:
        workingTraj = traj.vectorizedTrajectories()
    workingTraj.layouts = traj.layouts

    # K-mean Initialization
    m = initializationFromTrajectories(k, workingTraj, per_layout = True, per_layout_randomization = False, verbose = verbose)
    if verbose:
        print("\n-- Initialization done. --\n")

    # K-mean Assigment & Update
    for i in range(nb_iter):
        if verbose:
            print(f"Iteration {i}")
        r = None
        if soft:
            r = softAssignment(m, workingTraj, Beta)
        else:
            r = assignment(m, workingTraj)
        a = getAssignmentFromResponsibility(r)
        m.trajectories, diff = update(m, workingTraj, r)
        if diff == 0:
            if verbose:
                print(f"K-mean algorithm converged at iteration {i}. Stopped.")
            break
        else:
            if verbose:
                print(f"The update made a total difference of {diff} on this iteration.")

    if verbose:
        print("\n-- Ending K-mean clustering. --\n")

    if verbose:
        if method == 2:
            m = m.trajectoriesFromVectors()
        print("\n-- Displaying clusters and clustered trajectories. --\n")
        nbPerCluster = np.zeros(len(m.trajectories), dtype = int)
        for i in a:
            nbPerCluster[i] += 1
        print(f"Number of trajectories per clusters : {nbPerCluster}")
        m.showTrajectories()
        traj.showTrajectories(a)
        workingTraj.showTrajectoriesSeparately(verbose = True, clusters = a)
    return a, workingTraj


# K-mean algorithm imported from opencv
def kmean_opencv(traj, k = 3, nb_iter = 10):
    print("\n-- OPENCV K-MEAN CLUSTERING --\n")
    import cv2 as cv

    Z = np.float32(traj.trajectories)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(Z, k, None, criteria, nb_iter, cv.KMEANS_RANDOM_CENTERS)

    a = []
    for l in label:
        a.append(l[0])

    traj.show2DTrajectoriesSeparately(clusters = a)

    return a
