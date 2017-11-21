import matplotlib.pyplot as plt

class Data(object):
  def __init__(self, x, y):
  	self.x = x
  	self.y = y	


def get_data(file_name):
  	x = []
	y = []
	with open(file_name, "r") as lines:
		for line in lines:
			split_line = line.strip().split(',')
			x.append(float(split_line[0]))
			y.append(float(split_line[1]))	
	return Data(x,y)		

def O0(o0, o1, x, y):
	sum_all_terms = sum([(o0 + o1 * x[i] - y[i]) for i in range(n)])
	return (float(1)/n) * sum_all_terms


def O1(o0, o1, x, y):
	sum_all_terms = sum([(o0 + o1 * x[i] - y[i]) * x[i] for i in range(n)])
	return (float(1)/n) * sum_all_terms

def cost(o0,o1,x,y):
	return (float(1)/(2 * n)) * sum([(o0 + o1 * x[i] - y[i])**2 for i in range(n)])	


data = get_data("data.txt")
x = data.x
y = data.y

n = len(data.x)
o0 = 0
o1 = 0
min_o0 = o0
min_o1 = o1
learning_rate = 0.01

min_cost = cost(o0, o1, x, y)	
costs = []
for i in range(0, 1000):
	grad0 = O0(o0, o1, x, y)
	grad1 = O1(o0, o1, x, y)
	temp0 = o0 - learning_rate * grad0
	temp1 = o1 - learning_rate * grad1
	o0 = temp0
	o1 = temp1

	new_cost = cost(o0, o1, x, y)
	if new_cost <= min_cost:
		min_cost = new_cost
		min_o0 = o0
		min_o1 = o1			   


regression = [ min_o0 + min_o1 * x[i] for i in range(n)]

print 'Theta 0: {}'.format(min_o0)
print 'Theta 1: {}'.format(min_o1)
print 'Minimum cost: {}'.format(min_cost)

plt.title('Gradient Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(data.x, data.y, marker='x', color='r', label='data')
plt.plot(x, regression, 'b')
plt.show()    



