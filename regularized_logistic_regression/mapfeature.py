import numpy as np

def map_feature(x1, x2):

	degree = 6
	feature_mapping = np.ones(len(x1))
	for i in range(1, degree + 1):
 	    for j in range(i + 1):   	
	        new_column = np.power(x1, i - j) * np.power(x2, j)
	        feature_mapping = np.column_stack((feature_mapping, new_column))

	return feature_mapping    