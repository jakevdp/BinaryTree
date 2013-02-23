import numpy as np
from ball_tree_v1 import BallTree as BallTree1
from ball_tree_v2 import BallTree as BallTree2

np.random.seed(0)
X = np.random.random((10, 3))

bt1 = BallTree1(X, leaf_size=2)
bt2 = BallTree2(X, leaf_size=2)

print bt1.query(X[0], 5)
print bt2.query(X[0], 5)
