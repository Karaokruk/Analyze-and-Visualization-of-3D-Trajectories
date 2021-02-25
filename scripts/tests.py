import numpy as np
from trajectory_clustering import Trajectories

class TrajectoriesTest:

    def printTestMessage(self, condition):
        print("PASSED" if condition else "FAILED")

    def passAllTests(self):
        self.printTestMessage(self.compare2dStraightTrajectoriesTest(verbose=True))
        self.printTestMessage(self.trajectoriesMinimizationTest(verbose=True))
        self.printTestMessage(self.trajectoriesAttuningTest(verbose=True))
        self.printTestMessage(self.kmeanTest(verbose=True))

    def compare2dStraightTrajectoriesTest(self, offset=2, nb_points=10, verbose=False):
        if verbose:
            print("-- 2D straight trajectories distance test --")
        trajectories = Trajectories()

        t0 = []
        t1 = []
        for x in range(nb_points):
            t0.append((x, x))
            t1.append((x + offset, x))
        trajectories.addTrajectory(t0)
        trajectories.addTrajectory(t1)

        return trajectories.indexedTrajectoriesDistance(0, 1, heuristic=1) == 0

    def trajectoriesMinimizationTest(self, nb_points=10, verbose=False):
        if verbose:
            print("-- Trajectories minimization test --")

        straightTrajectory = Trajectories()
        t0 = []
        for x in range(nb_points):
            t0.append((x, x, x))
        straightTrajectory.addTrajectory(t0)
        straightTrajectory.attuneTrajectories(ratio=0.98, limit=0)
        return straightTrajectory.getTrajectory(0).shape[0] == 2

    def trajectoriesAttuningTest(self, nb_points=10, verbose=False):
        if verbose:
            print("-- Trajectories attuning test --")

        trajectories = Trajectories()
        t0 = []
        t1 = []
        for i in range(nb_points):
            x = i / nb_points
            t0.append((x, x))
            t1.append((x - 2, x * x))
        trajectories.addTrajectory(t0)
        trajectories.addTrajectory(t1)
        trajectories.attuneTrajectories(ratio=0.995, limit=0)
        return trajectories.getTrajectory(0).shape[0] > 2

    def kmeanTest(self, verbose=False):
        if verbose:
            print("-- k-mean test --")
        return False

tt = TrajectoriesTest()
tt.passAllTests()
