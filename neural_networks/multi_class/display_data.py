import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat	
import random

data = loadmat('data.mat')

random_numbers = random.sample(range(0, 5000), 100)
matrix = None
index = 0

for i in range(0, 10):
	row = None
	for j in range(0, 10):
		number = (data['X'][random_numbers[index]]).reshape((20, 20)).transpose()
		row = number if row == None else np.concatenate((row, number), axis=1)
		index += 1
			
	matrix = row if matrix == None else np.concatenate((matrix, row), axis=0) 


im = plt.imshow(matrix, cmap='gray')
plt.show()