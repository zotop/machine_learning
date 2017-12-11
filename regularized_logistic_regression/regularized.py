import matplotlib.pyplot as plt
import numpy as np

#### MAIN ####

data = np.loadtxt('data.txt', delimiter=',') 
test_1 = data[:, 0:1]
test_2 = data[:, 1:2]
results = data[:, 2:3]

accepted = np.where(results == 1)[0]
accepted_test_1_result = test_1[accepted]
accepted_test_2_result = test_2[accepted]

rejected = np.where(results == 0)[0]
rejected_test_1_result = test_1[rejected]
rejected_test_2_result = test_2[rejected]


plt.title('Microchip QA')
plt.xlabel('Test 1')
plt.ylabel('Test 2')
plt.scatter(accepted_test_1_result, accepted_test_2_result, marker='*', color='b', label='Accepted')
plt.scatter(rejected_test_1_result, rejected_test_2_result, marker='x', color='r', label='Rejected')
plt.legend(loc='upper right');
plt.show()    