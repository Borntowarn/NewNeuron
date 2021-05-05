import numpy as np
from numpy.random import seed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class neuron:
	temp = 0.001
	x = np.array([[]])
	y = np.array([])
	eta = 0.5
	cost = []
	iter = 0
	errors_ = []
	errors_1 = []
	w = np.array([])
	random_s = 0
	llambda = 0.0001

	def __init__(self, iter, data, answer, eta, random_s = None):
		self.eta = eta
		self.iter = iter
		self.x = data
		self.y = answer
		self.w = np.zeros(1 + self.x.shape[1])
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
			print (self.errors_[i])
			print (self.proverka(self.x, self.y))
		return self

	def shuffle(self, x, y):
		r = np.random.permutation(len(y))
		return x[r], y[r]

	def predict(self,x):
		return (np.where(self.clear_out(x) >= 0.5, 1, 0))

	def _predict(self,x):
		return (self.activation(x), np.where(self.activation(x) >= 0.5, 1, 0))

	def proverka(self, x, y):
		return (np.where(self.predict(x) == y, 1, 0).sum())

x1 = []
y1 = []
with open("C:\\Users\\kozlo\\source\\repos\\VSCODE\\Neuron\\NewNeuron\\data.txt", "r") as f:
	for a in f:
		a = a.strip().split()
		x1.append(a[0:4])
		if a[4] == "Iris-versicolor":
			y1.append(1)
		else:
			y1.append(0)

x_train, x_test, y_train, y_test = train_test_split(np.array(x1),np.array(y1), test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

n = neuron(20, x_train_std, y_train, 0.5)
n.learning()
print(n.proverka(x_test_std, y_test))


a = [float(s) for s in input().split()]

while(a[0] != 0):
	a = np.atleast_2d(a)
	a = sc.transform(a)
	odd, _class = n._predict(a)
	if (_class == 1) : print("Iris-versicolor c вероятностью ", round(odd[0]*10000)/100, "%")
	else : print ("Iris-virginica c вероятностью ", round((1-odd[0])*10000)/100, "%")
	a = [float(s) for s in input().split()]

