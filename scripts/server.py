from k_mean import kmean
from trajectory_clustering import Trajectories
from trajectory_clustering import createCSVTrajectories
from flask import Flask

app = Flask(__name__)

@app.route('/')
def test():
    traj = createCSVTrajectories("../datapoints/SmallMultipleVR_Study/Study 2/Participant_7_HeadPositionLog.csv", verbose = False)
    a = kmean(traj, k = 3, method = 1, verbose = False)
    return str(a)

if __name__ == "__main__":
    app.run()