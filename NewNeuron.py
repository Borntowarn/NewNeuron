import numpy as np
from numpy.random import seed
from numpy.random import uniform
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class neuron:
	x = np.array([[]])
	y = np.array([])
	eta = 0.5
	iter = 0
	errors_ = []
	w = np.array([])
	random_s = 0
	llambda = 0.0001
	start_w = 0.0

	def __init__(self, iter, data, answer, eta, start_w, random_s = None):
		self.eta = eta
		self.iter = iter
		self.x = data
		self.y = answer
		self.w = np.full(1 + self.x.shape[1], start_w)
		self.random_s = seed(random_s)

	def clear_out(self, x):
		return (np.dot(x, self.w[1:]) + self.w[0])
	
	def activation(self, x):
		return (1.0/(1.0 + np.exp(-self.clear_out(x))))
	   
	def learning(self):
		for i in range(self.iter):
			self.x, self.y = self.shuffle(self.x, self.y)
			e = 0.0
			for q in range(len(self.y)):
				error = self.y[q] - self.activation(self.x[q])
				self.w[1:] += self.eta * (self.x[q].T.dot(error) - self.w[1:]*self.llambda)
				self.w[0] += self.eta * (error - self.w[0]*self.llambda)
				dop = -np.log(self.activation(self.x[q, :]))
				dop1 = -np.log((1 - self.activation(self.x[q, :])))
				if self.y[q] == 1 : e += dop
				else: e += dop1
			self.errors_.append(e + (self.llambda/2.0)*self.w.T.dot(self.w))
		return self

	def shuffle(self, x, y):
		r = np.random.permutation(len(y))
		return x[r], y[r]

	def predict(self,x):
		return (np.where(self.activation(x) >= 0.5, 1, 0))

	def _predict(self,x):
		return (self.activation(x))

	def proverka(self, x, y):
		return (np.where(self.predict(x) == y, 1, 0).sum())


x1 = []
y1 = []
with open("C:\\Users\\kozlo\\source\\repos\\VSCODE\\Neuron\\NewNeuron\\data.txt", "r") as f:
	for a in f:
		a = a.strip().split()
		x1.append(a[0:4])
		if a[4] == "Iris-setosa":
			y1.append(0)
		elif a[4] == "Iris-versicolor":
			y1.append(1)
		else: y1.append(2)

x = np.array(x1, dtype=float)
y = np.array(y1, dtype=float)
eta = 0.5
iter = 150

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

classific = []
x = sc.transform(x)

while (True):
	start_w = -30.0
	################# FOR SETOSA ################
	y_train_1 = np.where(y_train == 0, 1, 0)
	y_test_1 = np.where(y_test == 0, 1, 0)
	first = neuron(iter, x_train_std, y_train_1, eta, start_w)
	first.learning()
	print(first.proverka(x_test_std, y_test_1))
	classific.append(first)
	#############################################


	############# FOR VERSICOLOR ################
	y_train_2 = np.where(y_train == 1, 1, 0)
	y_test_2 = np.where(y_test == 1, 1, 0)
	second = neuron(iter, x_train_std, y_train_2, eta, start_w)
	second.learning()
	print(second.proverka(x_test_std, y_test_2))
	classific.append(second)
	#############################################


	############## FOR VIRGINICA ################
	y_train_3 = np.where(y_train == 2, 1, 0)
	y_test_3 = np.where(y_test == 2, 1, 0)
	third = neuron(iter, x_train_std, y_train_3, eta, start_w)
	third.learning()
	print(third.proverka(x_test_std, y_test_3))
	classific.append(third)
	#############################################

	q = np.array([i._predict(x) for i in classific]).T.tolist()
	result = []
	for i in q:
		result.append(i.index(max(i)))
	print ("Число ошибок:", np.where(y != result, 1, 0).sum(), sep=" ")
	print ("Точность:", round(accuracy_score(y, result) * 10000) / 100, "%", sep=" ")
	if ((np.where(y != result, 1, 0).sum()) <= 2 or (start_w >= 30)) : break
	else: 
		classific.pop()
		classific.pop()
		classific.pop()
		start_w+=1.0

figure, ax = plt.subplots()
colors = ['red', 'green', 'blue']
x_ax = np.linspace(-3,3,100)

for i in np.unique(y1):
	ax.plot(x[y==i,0], x[y==i,2], 'o', label="класс {}".format(i), color = colors[i])
	y_ax = ((-classific[i].w[0]-classific[i].w[1]*x_ax)/classific[i].w[3])
	ax.plot(x_ax,y_ax, color = colors[i])

plt.show()

a = [float(s) for s in input().split()]

while(a[0] != 0):
	a = np.atleast_2d(a)
	a = sc.transform(a)
	print([i._predict(a) for i in classific])
	a = [float(s) for s in input().split()]

