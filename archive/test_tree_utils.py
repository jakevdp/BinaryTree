import numpy as np
from tree_utils import MaxHeap, sort_dist_idx

x = np.random.random(20)

val = np.zeros(10)
val.fill(np.inf)
idx = np.zeros(10, dtype=int)

m = MaxHeap(val, idx)

for i, xi in enumerate(x):
    m.insert(xi, i)
    print np.sort(x[:i+1])[:10][-1], m.largest()

x = np.random.random(50)
i = np.arange(50)

x2 = x.copy()
i2 = i.copy()

sort_dist_idx(x, i)

ind = np.argsort(x2)
print np.all(x2[ind] == x)
print np.all(i2[ind] == i)
