import os
import numpy as np
################################################################################
# Program that creates the splits for the data
################################################################################
os.chdir('C:\\Users\\Guill\\Documents\\NIRPSML\\three_cols\\splits')

array_0 = np.arange(24,4071).tolist()
array_1 = np.arange(24,4071).tolist()
array_2 = np.arange(24,4071).tolist()
array_3 = np.arange(24,4071).tolist()
array_4 = np.arange(24,4071).tolist()
array_5 = np.arange(24,4071).tolist()
arrays = [array_0, array_1, array_2, array_3, array_4, array_5]

counts_1 = [0,0,0,0,0,0]
counts_2 = np.zeros((4047)).tolist()

for i in range(47):
    print(i)
    indexes_1 = []
    indexes_2 = []
    for j in range(500):
        n1 = np.random.randint(0,6)
        counts_1[n1] += 1
        n2 = np.random.choice(arrays[n1])
        counts_2[n2-24] += 1
        arrays[n1].remove(n2)
        indexes_1.append(n1)
        indexes_2.append(n2)
        np.save('splits_1_{}'.format(i), np.array(indexes_1))
        np.save('splits_2_{}'.format(i), np.array(indexes_2))

print(counts_1)
print(counts_2)
