import numpy as np
import matplotlib.pyplot as plt
import csv

np.random.seed()

class Trajectories:

    def __init__(self):
        self.trajectories = []

    def addTrajectory(self, trajectory):
        self.trajectories.append(np.array(trajectory))

    def addRandomTrajectory(self, maxValue=10, nbPoints=10, nbCoordinates=2):
        self.trajectories.append(maxValue * np.random.random((nbPoints, nbCoordinates)))

    def addTrajectoryFromCsv(self, dir):
        with open(dir) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            line_count = 0
            t = []
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                else:
                    #print(f'x = {row[3]}, y = {row[3]}, z = {row[5]}')
                    t.append([float(row[3]), float(row[4]), float(row[5])])
                line_count += 1
            self.addTrajectory(t)
            print(f'Processed {line_count} lines.')
    
    def getTrajectory(self, id):
        return self.trajectories[id]

    def printTrajectories(self):
        for trajectory in self.trajectories:
            print(trajectory)

    def showTrajectories(self):
        import matplotlib.pyplot as plt
        tr = np.asarray(self.trajectories)
        size = tr.shape[2]
        # 2D visualisation
        if size == 2:
            for trajectory in self.trajectories:
                plt.plot(trajectory[:, 0], trajectory[:, 1]) #x, y
        # 3D visualisation
        elif size == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            for trajectory in self.trajectories:
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])

        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


    ## Different heuristics to compute distance between two trajectories

    # Means the squares of the distance between points of the same the same index in the two trajectories
    def heuristic0(self, index1, index2, translation=False, verbose=False):
        t1 = self.getTrajectory(index1)
        t2 = self.getTrajectory(index2)
        size = min(t1.shape[0], t2.shape[0])

        origin_difference = ((t1[0][0] - t2[0][0]), (t1[0][1] - t2[0][1])) if (translation) else 0

        def distanceBetweenTwoPoints(p1, p2):
            return np.sqrt(np.sum(np.square(p1 - p2)))

        distance = 0
        for i in range(size):
            distance += distanceBetweenTwoPoints(t1[i], origin_difference + t2[i]) ** 2
        distance = np.sqrt(distance / size)
        if (verbose):
            print("Distance between trajectory {} & {} : {}".format(index1, index2, distance))
        return distance

    # Adds a translation to heuristic0 by pasting the 2nd trajectory origin onto the 1st
    def heuristic1(self, index1, index2, verbose=False):
        return self.heuristic0(index1, index2, translation=True, verbose=verbose)

    def distanceBetweenTwoTrajectories(self, index1, index2, heuristic=1, verbose=False):
        if (heuristic == 0):
            return self.heuristic0(index1, index2, verbose=verbose)
        if (heuristic == 1):
            return self.heuristic1(index1, index2, verbose=verbose)

    def getTrajectoriesDistances(self, verbose=False):
        if (verbose):
            print("-- Distances between all trajectories --")
        distances = []
        nb_trajectories = len(self.trajectories)
        for i in range(nb_trajectories - 1):
            i_distances = []
            for j in range(i + 1, nb_trajectories):
                i_distances.append(self.distanceBetweenTwoTrajectories(i, j, verbose=verbose))
            distances.append(i_distances)
        return distances


## Create similar trajectories
similar_trajectories = Trajectories()

t0 = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]
t1 = [[0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9], [6, 10], [7, 11], [8, 12], [9, 13]]
similar_trajectories.addTrajectory(t0)
similar_trajectories.addTrajectory(t1)

#similar_trajectories.distanceBetweenTwoTrajectories(0, 1, verbose=True)
similar_trajectories.showTrajectories()

print() # \n

## Create random trajectories
random_trajectories = Trajectories()
random_trajectories.addTrajectory(t0)
nb_trajectories = 10
for i in range(nb_trajectories - 1):
    random_trajectories.addRandomTrajectory()

#distances = random_trajectories.getTrajectoriesDistances(verbose=True)
#print(distances)
#random_trajectories.printTrajectories()
#random_trajectories.showTrajectories()

csv_traj = Trajectories()
csv_traj.addTrajectoryFromCsv("datapoints/participant7trial2-ontask-100.csv")
csv_traj.showTrajectories()