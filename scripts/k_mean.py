import trajectory_clustering

import numpy as np
import matplotlib.pyplot as plt


def initialization(size):
    return np.random.randn(1, size)[0]

m = initialization(10)
print("\n-- K-MEAN CLUSTERING --\n")
print(m)