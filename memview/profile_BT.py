#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile
import numpy as np

from binary_tree import BallTree, _BinaryTree

X = np.random.random((1000, 3)).astype(float)
bt = BallTree(X)

Y = np.random.random((2000, 3))

cProfile.runctx("bt.query(Y, 3)", globals(), locals(), "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

