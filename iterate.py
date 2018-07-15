import numpy as np

maxnum = 0
minnum = 1
for _ in range(int(1E7)):
    n = np.matmul(np.random.rand(4), np.random.rand(4))
    if n > maxnum:
        maxnum = n
    elif n < minnum:
        minnum = n

print("Max", maxnum)
print("Min", minnum)