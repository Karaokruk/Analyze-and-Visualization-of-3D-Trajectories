import numpy as np
import csv

np.random.seed()

class Trajectories:

    def __init__(self):
        self.trajectories = []

    def addTrajectory(self, trajectory):
        self.trajectories.append(np.array(trajectory))

    def addRandomTrajectory(self, maxValue=10, nbPoints=10, nbCoordinates=2):
        self.trajectories.append(maxValue * np.random.rand(nbPoints, nbCoordinates))

    def addTrajectoriesFromCsv(self, file):
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            line_count = 0
            t = []
            x = y = z = state_id = trial_id = -1
            prev_trial_id = -1
            for row in csv_reader:
                if line_count == 0:
                    print(f"Column names are {', '.join(row)}")
                    # Automatic version
                    for i in range(len(row)):
                        if row[i] == "CameraPosition.x":
                            x = i
                        elif row[i] == "CameraPosition.y":
                            y = i
                        elif row[i] == "CameraPosition.z":
                            z = i
                        elif row[i] == "TrialState":
                            state_id = i
                        elif row[i] == "TrialID":
                            trial_id = i
                    print(f"before : x = {x}, y = {y}, z = {z}")
                    # If the col isn't found
                    if x == -1:
                        x = int(input("Camera position X index not found. Enter the column (index - 1) : "))
                    if y == -1:
                        y = int(input("Camera position Y index not found. Enter the column (index - 1) : "))
                    if z == -1:
                        z = int(input("Camera position Z index not found. Enter the column (index - 1) : "))
                    if state_id == -1:
                        state_id = int(input("Trial state index not found. Enter the column (index - 1) : "))
                    if trial_id == -1:
                        trial_id = int(input("Trial ID index not found. Enter the column (index - 1) : "))
                    print(f"after : x = {x}, y = {y}, z = {z}, with state_id = {state_id} and trial_id = {trial_id}")
                else:
                    if row[trial_id] != prev_trial_id and t != [] and prev_trial_id != -1:
                        self.addTrajectory(t)
                        t = []
                    if row[state_id] == "OnTask":
                        prev_trial_id = row[trial_id]
                        t.append([float(row[x]), float(row[y]), float(row[z])])
                line_count += 1
            if t != []:
                self.addTrajectory(t)
            print(f"Processed {line_count} lines.")


    def getTrajectory(self, id):
        return self.trajectories[id]

    # Make all trajectories the same length
    def attuneTrajectories(self,ratio,limit):

        def minimizedTrajectory0(trajectory):
            minimized = []

            # init last_point & last_direction
            nb_dimensions = trajectory.shape[1]
            last_point = last_direction = np.zeros(nb_dimensions)

            def updateDirection(p1, p2):
                direction = np.zeros(nb_dimensions)
                for i in range(nb_dimensions):
                    direction[i] = 1 if (p1[i] > p2[i]) else -1
                return direction

            for i in range(len(trajectory)):
                point = trajectory[i]
                new_direction = updateDirection(point, last_point)
                if i > 0 and not (last_direction == new_direction).all():
                    minimized.append(last_point)
                last_direction = new_direction
                last_point = point
            minimized.append(trajectory[len(trajectory) - 1])
            return np.array(minimized)
        # for i in range(len(self.trajectories)):
        #     self.trajectories[i] = minimizedTrajectory1(self.trajectories[i],0.98,limit)

        def minimizedTrajectory1(trajectory, ratio,limit):
            tmpTraj = trajectory
            i = 0
            cpt = 0
            while i < len(tmpTraj)-2 and len(tmpTraj) > limit:
                if self.pointDistance(tmpTraj[i],tmpTraj[i+2]) >= ratio*(self.pointDistance(tmpTraj[i],tmpTraj[i+1])+self.pointDistance(tmpTraj[i+1],tmpTraj[i+2])):
                    tmpTraj = np.delete(tmpTraj,i+1,0)
                    cpt+=1
                i+=1
            if cpt == 0 and len(tmpTraj) > limit:
                limit = len(tmpTraj)
            print(f"{cpt} points have been removed")
            print(f"{len(trajectory)} points remaining")
            return tmpTraj,limit,cpt

        removed = -1
        while removed != 0:
            removed = 0
            tmpTrajectories = []
            for i in range(len(self.trajectories)):
                tmp,limit,cpt = minimizedTrajectory1(self.trajectories[i],ratio,limit)
                tmpTrajectories.append(tmp)
                removed+=cpt
            for i in range(len(self.trajectories)):
                if len(tmpTrajectories[i]) < limit:
                    tmpTrajectories[i] = self.trajectories[i]
            self.trajectories = tmpTrajectories

    def printTrajectories(self):
        print("-- Trajectories --")
        for i in range(len(self.trajectories)):
            print(f"Trajectory #{i} :\n{self.trajectories[i]}")

    def showTrajectories(self, clusters = None):
        import matplotlib.pyplot as plt
        nb_dimensions = self.trajectories[0].shape[1]
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        # 2D visualization
        if nb_dimensions == 2:
            if clusters is not None:
                for i in range(len(self.trajectories)):
                    plt.plot(self.trajectories[i][:, 0], self.trajectories[i][:, 1], color = colors[clusters[i]])
            else:
                for trajectory in self.trajectories:
                    plt.plot(trajectory[:, 0], trajectory[:, 1]) #x, y
        # 3D visualization
        elif nb_dimensions == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            if clusters is not None:
                for i in range(len(self.trajectories)):
                    ax.plot(self.trajectories[i][:, 0], self.trajectories[i][:, 1], self.trajectories[i][:, 2], color = colors[clusters[i]])
            else:
                for trajectory in self.trajectories:
                    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
        if clusters is not None:

            print(colors)
            print(clusters)
            for i in range(len(self.trajectories)):
                print(f"plotting line {i} with color {colors[clusters[i]]}")
                 #x, y

        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def completeDisplay(self):
        self.getTrajectoriesDistances(verbose=True)
        self.printTrajectories()
        self.showTrajectories()

    ## Different heuristics to compute distance between two trajectories
    # Distance between the points p1 and p2
    def pointDistance(self, p1, p2):
        return np.sqrt(np.sum(np.square(p1 - p2)))

    # Means the squares of the distance between points of the same the same index in the two trajectories
    def heuristic0(self, t1, t2, translation=False, verbose=False):
        size = min(t1.shape[0], t2.shape[0])

        deltat2 = None
        if translation:
            deltat2 = t2 - t1[0]
        else:
            deltat2 = t2

        distance = 0
        for i in range(size):
            distance += self.pointDistance(t1[i], deltat2[i]) ** 2
        distance = np.sqrt(distance / size)
        if (verbose):
            print(f"Distance between trajectories : {distance}")
        return distance

    # Adds a translation to heuristic0 by pasting the 2nd trajectory origin onto the 1st
    def heuristic1(self, t1, t2, verbose=False):
        return self.heuristic0(t1, t2, translation=True, verbose=verbose)

    # Distance between two trajectories of the same class using indexes
    def indexedTrajectoriesDistance(self, index1, index2, heuristic=1, verbose=False):
        t1 = self.getTrajectory(index1)
        t2 = self.getTrajectory(index2)
        return self.trajectoryDistance(self, t1, t2, heuristic=1, verbose=False)

    # Distance between two trajectories.
    @staticmethod
    def trajectoryDistance(self, t1, t2, heuristic=1, verbose=False):
        if (heuristic == 0):
            return self.heuristic0(t1, t2, verbose=verbose)
        if (heuristic == 1):
            return self.heuristic1(t1, t2, verbose=verbose)

    def getTrajectoriesDistances(self, verbose=False):
        if (verbose):
            print("-- Distances between all trajectories --")
        distances = []
        nb_trajectories = len(self.trajectories)
        for i in range(nb_trajectories - 1):
            i_distances = []
            for j in range(i + 1, nb_trajectories):
                i_distances.append(self.indexedTrajectoriesDistance(i, j, verbose=verbose))
            distances.append(i_distances)
        return distances


## Different kind of Trajectories call

# Create two same trajectories but spaced out by 4 units
def createSimilarTrajectories(minimize=False):
    similar_trajectories = Trajectories()

    t0 = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]
    t1 = [[0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9], [6, 10], [7, 11], [8, 12], [9, 13]]
    similar_trajectories.addTrajectory(t0)
    similar_trajectories.addTrajectory(t1)
    similar_trajectories.attuneTrajectories(8,0.8)

    similar_trajectories.completeDisplay()
    if (minimize):
        similar_trajectories.attuneTrajectories(8,0.8)
        similar_trajectories.completeDisplay()
    return similar_trajectories

# Create nb_trajectories random trajectories
def createRandomTrajectories(nb_trajectories=10, nb_points=10, minimize=False):
    random_trajectories = Trajectories()

    #t0 = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]
    #random_trajectories.addTrajectory(t0)
    for _ in range(nb_trajectories - 1):
        random_trajectories.addRandomTrajectory(nbPoints=nb_points)

    random_trajectories.completeDisplay()
    if (minimize):
        random_trajectories.attuneTrajectories(5,0.8)
        random_trajectories.completeDisplay()
    return random_trajectories

# Create one trajectory from a file
def createCSVTrajectories(file):
    csv_trajectories = Trajectories()

    csv_trajectories.addTrajectoriesFromCsv(file)
    #csv_trajectories.printTrajectories()
    csv_trajectories.showTrajectories()
    csv_trajectories.attuneTrajectories(0.99, 700)
    csv_trajectories.showTrajectories()
    return csv_trajectories

#createSimilarTrajectories(minimize=True)
#createRandomTrajectories(nb_trajectories=2, minimize=True)
#createCSVTrajectories("../datapoints/participant7trial1-ontask.csv")
