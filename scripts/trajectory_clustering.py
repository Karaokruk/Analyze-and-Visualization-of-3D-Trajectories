import numpy as np
import csv

np.random.seed()

class Trajectories:

    def __init__(self, id = 0):
        self.trajectories = []
        self.layouts = []
        self.user = []

    def copy(self):
        newTraj = Trajectories()
        for t in self.trajectories:
            newTraj.addTrajectory(t.copy())
        for l in self.layouts:
            newTraj.addLayout(l)
        return newTraj

    def translatedTrajectories(self):
        newTraj = Trajectories()
        for t in self.trajectories:
            newTraj.addTrajectory(t - t[0])
        return newTraj

    def vectorizedTrajectories(self):
        newTraj = self.copy()
        for i in range(len(newTraj.trajectories)):
            for j in range(len(newTraj.trajectories[i]) - 1):
                newTraj.trajectories[i][j] = newTraj.trajectories[i][j+1] - newTraj.trajectories[i][j]
            newTraj.trajectories[i] = np.delete(newTraj.trajectories[i], len(newTraj.trajectories[i]) - 1, 0)
        return newTraj

    def trajectoriesFromVectors(self):
        newTraj = Trajectories()
        for t in self.trajectories:
            traj = np.zeros_like(t)
            traj = np.append(traj, [traj[0]], axis = 0)
            pos = np.zeros_like(t[0])
            for i in range(len(t)):
                pos += t[i]
                traj[i+1] = pos
            newTraj.addTrajectory(traj)
        return newTraj

    def addTrajectory(self, trajectory):
        self.trajectories.append(np.array(trajectory))

    def addRandomTrajectory(self, maxValue = 10, nbPoints = 10, nbCoordinates = 2):
        self.trajectories.append(maxValue * np.random.rand(nbPoints, nbCoordinates))

    def addLayout(self, layout_type):
        self.layouts.append(layout_type)

    def addUser(self, user):
        self.user.append(user)

    def addTrajectoriesFromCsv(self, file, groupBy = "None", groupID = 0, verbose = False):
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            line_count = 0
            t = []
            layout_type = []
            user = []
            user_id = x = y = z = state_id = trial_id = task_id = prev_trial_id = layout = -1
            group = None
            for row in csv_reader:
                if line_count == 0:
                    if verbose:
                        print(f"Column names are {', '.join(row)}")
                    # Automatic version
                    for i in range(len(row)):
                        if row[i] == "UserID":
                            user_id = i
                        elif row[i] == "CameraPosition.x":
                            x = i
                        elif row[i] == "CameraPosition.y":
                            y = i
                        elif row[i] == "CameraPosition.z":
                            z = i
                        elif row[i] == "TrialState":
                            state_id = i
                        elif row[i] == "TrialID":
                            trial_id = i
                        elif row[i] == "TaskID":
                            task_id = i
                        elif row[i] == "Layout":
                            layout = i
                    if verbose:
                        print(f"Before input auto-detection : x = {x}, y = {y}, z = {z}, with state_id = {state_id} and trial_id = {trial_id} with layout {layout}")
                    # If the colomn isn't found
                    if user_id == -1:
                        user_id = int(input("User ID index not found. Enter the column (index - 1) : "))
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
                    if task_id == -1:
                        task_id = int(input("Task ID index not found. Enter the column (index - 1) : "))
                    if layout == -1:
                        layout = int(input("Layout index not found. Enter the column (index - 1) : "))
                    if verbose:
                        print(f"After input auto-detection : x = {x}, y = {y}, z = {z}, with state_id = {state_id}, task_id = {task_id} and trial_id = {trial_id} with layout {layout}")
                    
                    if groupBy == "Task":
                        group = task_id
                    elif groupBy == "Trial":
                        group = trial_id
                    elif groupBy == "Layout":
                        group = layout

                else:
                    if row[trial_id] != prev_trial_id and t != [] and prev_trial_id != -1:
                        self.addTrajectory(t)
                        t = []
                        if layout_type != []:
                            self.addLayout(layout_type[0])
                            layout_type = []
                        if user != []:
                            self.addUser(user[0])
                            user = []

                    if row[state_id] == "OnTask" and (groupBy == "None" or row[group] == groupID):
                        if layout_type == []:
                            layout_type.append(row[layout])
                        if user == []:
                            user.append(row[user_id])
                        prev_trial_id = row[trial_id]
                        t.append([float(row[x]), float(row[y]), float(row[z])]) # 3d
                        #t.append([float(row[x]), float(row[z])]) # 2d
                line_count += 1
            if t != []:
                self.addTrajectory(t)
                if layout_type != []:
                    self.addLayout(layout_type[0])

            if verbose:
                print(f"Processed {line_count} lines.")

    def trajectoriesToCsv(self, write_method = 0, k = None, a = None):
        nb_files = 0

        # Handling errors
        if write_method < 0 and write_method > 2:
            print("Warning in trajectoriesToCsv : method number does not correspond to an actuel method. Getting back to method 0.")
            write_method = 0
        if write_method == 1 and a == None:
            print("Error in trajectoriesToCsv : method 1 selected but no assignment list given.")
            return -1
        if write_method == 1 and k == None:
            print("Warning in trajectoriesToCsv : method 1 selected but no k given.")
            k = input("Please enter the number of k : >")

        # One file per trajectories
        if write_method == 0:
            for i in range(len(self.trajectories)):
                filename = 'traj' + str(nb_files) + 'method' + str(write_method) + '.csv'
                with open(filename, 'w') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                    writer.writerow(['TrajectoryID', 'x', 'y', 'z'])
                    for j in range(len(self.trajectories[i])):
                        writer.writerow([i, self.trajectories[i][j][0], self.trajectories[i][j][1], self.trajectories[i][j][2]])
                nb_files += 1
        # One file per cluster
        if write_method == 1:
            for i in range(k):
                for j in range(len(self.trajectories)):
                    if a[j] == i:
                        filename = 'minimizedTrajectories/traj' + str(i) + 'method' + str(write_method) + '.csv'
                        with open(filename, 'w') as csvfile:
                            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                            writer.writerow(['TrajectoryID', 'x', 'y', 'z'])
                            for l in range(len(self.trajectories[j])):
                                writer.writerow([j, self.trajectories[l][0], self.trajectories[l][1], self.trajectories[l][2]])
                        nb_files += 1
        # One file
        if write_method == 2:
            filename = 'trajmethod' + str(write_method) + '.csv'
            with open(filename, 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                writer.writerow(["TrajectoryID", "x", "y", "z"])
                for i in range(len(self.trajectories)):
                    for j in range(len(self.trajectories[i])):
                        writer.writerow([i, self.trajectories[i][j][0], self.trajectories[i][j][1], self.trajectories[i][j][2]])

        return nb_files

    def getTrajectory(self, id):
        return self.trajectories[id]

    # Make all trajectories the same length
    def attuneTrajectories(self, ratio, limit, verbose = False):

        # Method to put away the points that don't change the xyz direction of the trajectory
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

        # Method to put away the points that don't change the global distance of the trajectory very much.
        # Can be (and should be) called multiple times.
        def minimizedTrajectory1(trajectory, ratio, limit):
            tmpTraj = trajectory
            i = 0
            cpt = 0
            while i < len(tmpTraj) - 2 and len(tmpTraj) > limit:
                if self.pointDistance(tmpTraj[i], tmpTraj[i + 2]) >= ratio * (self.pointDistance(tmpTraj[i], tmpTraj[i + 1]) + self.pointDistance(tmpTraj[i + 1], tmpTraj[i + 2])):
                    tmpTraj = np.delete(tmpTraj, i + 1, 0)
                    cpt += 1
                i += 1
            if cpt == 0 and len(tmpTraj) > limit:
                limit = len(tmpTraj)
                if verbose:
                    print(f"Limit is now {limit}")
            if verbose:
                print(f"{cpt} points have been removed")
                print(f"{len(trajectory)} points remaining")
            return tmpTraj, limit, cpt

        # Allows to find the trajectory that has the largest number of points, among those that are not equal to the limit.
        def findTrajectoryToIter(trajectories, limit):
            id = 0
            size = len(trajectories[0])
            for i in range(len(trajectories)):
                if (len(trajectories[i]) > size and len(trajectories[i]) != limit) or size == limit:
                    size = len(trajectories[i])
                    id = i
            return id, size

        toMinimize = self.trajectories.copy()
        while True:
            id, size = findTrajectoryToIter(toMinimize, limit)
            if size == limit: # each trajectory has a size equal to the limit, the minimization is over.
                break

            toMinimize[id], limit, removed = minimizedTrajectory1(toMinimize[id], ratio, limit)

            if removed == 0:
                if len(toMinimize[id]) < limit:
                    if len(self.trajectories[id]) < limit:
                        print("Unable to attune trajectories because of ratio too high or limit wrongly placed")
                        quit()
                    if verbose:
                        print(f"Resetting trajectory {id} because length {len(toMinimize[id])} is lower than limit {limit}")
                    toMinimize[id] = self.trajectories[id].copy()
        print(f"Trajectories have been attuned to {limit} points.")
        self.trajectories = toMinimize

    def printTrajectories(self):
        print("-- Trajectories --")
        for i in range(len(self.trajectories)):
            print(f"Trajectory #{i} :\n{self.trajectories[i]}")

    def printLayouts(self):
        print("-- Layout types --")
        for i in range(len(self.layouts)):
            print(f"Layout type for trajectory #{i+1}: {self.layouts[i]}")
        print(f"For a total of {len(self.trajectories)} trajectories.")

    def showTrajectories(self, clusters = None, verbose = False):
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
            #ax = fig.gca(projection="3d")
            ax = plt.axes(projection="3d")
            if clusters is not None:
                for i in range(len(self.trajectories)):
                    ax.plot3D(self.trajectories[i][:, 0], self.trajectories[i][:, 1], self.trajectories[i][:, 2], color = colors[clusters[i]])
            else:
                for trajectory in self.trajectories:
                    ax.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
        if clusters is not None and verbose:
            print(colors)
            print(clusters)
            for i in range(len(self.trajectories)):
                print(f"plotting line {i} with color {colors[clusters[i]]}")

        plt.show()

    def showTrajectoriesSeparately(self, clusters = None, verbose = False):
        # Utilities
        nb_trajectories = len(self.trajectories)
        nb_dimensions = self.trajectories[0].shape[1]
        #nb_dimensions = 2

        # Initialize the plot object and its color set
        import matplotlib.pyplot as plt
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        # Find the different types of layouts and their number of occurences
        layout_types = list(set(self.layouts)) # erase all duplicates from self.layouts
        from collections import Counter
        layout_types_counter = Counter(self.layouts)
        if verbose:
            # Print the number of each layouts
            for i in range(len(layout_types)):
                print(f"Layout \"{layout_types[i]}\" found {layout_types_counter[layout_types[i]]} times")
        nb_columns = max(layout_types_counter.values())
        nb_rows = len(layout_types)

        def findLayoutTypeIndex(trajectory_index):
            for i in range(len(layout_types)):
                if self.layouts[trajectory_index] == layout_types[i]:
                    return i
            return 0

        # Display the trajectories
        fig = plt.figure()
        reference_subplot = None
        if clusters is None:
            clusters = np.zeros(nb_trajectories, dtype=int) # same color for every trajectory
        for i in range(nb_trajectories):
            layout_type_index = findLayoutTypeIndex(i)
            index = layout_type_index * nb_columns + (nb_columns - layout_types_counter[layout_types[layout_type_index]]) + 1
            layout_types_counter[layout_types[layout_type_index]] -= 1
            if nb_dimensions == 2:
                subplot = fig.add_subplot(nb_rows, nb_columns, index, sharex = reference_subplot, sharey = reference_subplot)
                subplot.plot(self.trajectories[i][:, 0], self.trajectories[i][:, 1], color = colors[clusters[i]])
                subplot.set_title(f"#{i + 1}, {self.layouts[i]}")
            elif nb_dimensions == 3:
                subplot = fig.add_subplot(nb_rows, nb_columns, index, projection = "3d", sharex = reference_subplot, sharey = reference_subplot, sharez = reference_subplot)
                subplot.plot(self.trajectories[i][:, 0], self.trajectories[i][:, 1], self.trajectories[i][:, 2], color = colors[clusters[i]])
                subplot.set_title(f"#{i + 1}, {self.layouts[i]}")
                if i == 0:
                    reference_subplot = subplot

        plt.show()

    def completeDisplay(self, clusters = None):
        self.getTrajectoriesDistances(verbose = True)
        self.printTrajectories()
        self.showTrajectories()
        self.showTrajectoriesSeparately(clusters = clusters, verbose = True)

    ## Different heuristics to compute distance between two trajectories
    # Distance between the points p1 and p2
    def pointDistance(self, p1, p2):
        return np.sqrt(np.sum(np.square(p1 - p2)))

    # Means the squares of the distance between points of the same the same index in the two trajectories
    def heuristic0(self, t1, t2, translation = False, verbose = False):
        size = min(t1.shape[0], t2.shape[0])

        deltat1 = None
        deltat2 = None
        if translation:
            deltat2 = t2 - t2[0]
            deltat1 = t1 - t1[0]
        else:
            deltat2 = t2
            deltat1 = t1

        distance = 0
        for i in range(size):
            distance += self.pointDistance(deltat1[i], deltat2[i]) ** 2
        distance = np.sqrt(distance / size)
        if (verbose):
            print(f"Distance between trajectories : {distance}")
        return distance

    # Adds a translation to heuristic0 by pasting the 2nd trajectory origin onto the 1st
    def heuristic1(self, t1, t2, verbose = False):
        return self.heuristic0(t1, t2, translation = True, verbose = verbose)

    # Distance between two trajectories of the same class using indexes
    def indexedTrajectoriesDistance(self, index1, index2, heuristic = 1, verbose = False):
        t1 = self.getTrajectory(index1)
        t2 = self.getTrajectory(index2)
        return self.trajectoryDistance(self, t1, t2, heuristic = heuristic, verbose = verbose)

    # Distance between two trajectories.
    @staticmethod
    def trajectoryDistance(self, t1, t2, heuristic = 1, verbose = False):
        if heuristic == 0:
            return self.heuristic0(t1, t2, verbose = verbose)
        if heuristic == 1:
            return self.heuristic1(t1, t2, verbose = verbose)

    def getTrajectoriesDistances(self, verbose = False):
        if (verbose):
            print("-- Distances between all trajectories --")
        distances = []
        nb_trajectories = len(self.trajectories)
        for i in range(nb_trajectories - 1):
            i_distances = []
            for j in range(i + 1, nb_trajectories):
                i_distances.append(self.indexedTrajectoriesDistance(i, j, verbose = verbose))
            distances.append(i_distances)
        return distances


## Different kind of Trajectories call

# Create two same trajectories but spaced out by 4 units
def createSimilarTrajectories(minimize = False):
    similar_trajectories = Trajectories()

    t0 = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]
    t1 = [[0, 4], [1, 5], [2, 6], [3, 7], [4, 8], [5, 9], [6, 10], [7, 11], [8, 12], [9, 13]]
    similar_trajectories.addTrajectory(t0)
    similar_trajectories.addTrajectory(t1)
    similar_trajectories.attuneTrajectories(0.98,0)

    similar_trajectories.completeDisplay()
    if (minimize):
        similar_trajectories.attuneTrajectories(0.98,0)
        similar_trajectories.completeDisplay()
    return similar_trajectories

# Create nb_trajectories random trajectories
def createRandomTrajectories(nb_trajectories = 10, nb_points = 10, minimize = False):
    random_trajectories = Trajectories()

    #t0 = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]
    #random_trajectories.addTrajectory(t0)
    for _ in range(nb_trajectories - 1):
        random_trajectories.addRandomTrajectory(nbPoints = nb_points)

    random_trajectories.completeDisplay()
    if (minimize):
        random_trajectories.attuneTrajectories(0.98, 0)
        random_trajectories.completeDisplay()
    return random_trajectories

# Create one trajectory from a file
def createCSVTrajectories(file, verbose = False):
    csv_trajectories = Trajectories()

    csv_trajectories.addTrajectoriesFromCsv(file, verbose = True)
    if verbose:
        csv_trajectories.printTrajectories()
        csv_trajectories.showTrajectories()
    csv_trajectories.attuneTrajectories(0.98, 0)
    if verbose:
        csv_trajectories.printTrajectories()
        csv_trajectories.showTrajectories()
    return csv_trajectories
