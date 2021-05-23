from re import X
from typing import Generator
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch import nn
from scipy.spatial.distance import cdist

class lin(nn.Module):
	def __init__(self, output = 1):
		super().__init__()
		self.layers = nn.Sequential()
		self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
		self.layers.add_module('layer', nn.Linear(1, 32))
		self.layers.add_module('activation', nn.ReLU())
		self.layers.add_module('layer1', nn.Linear(32, 32))
		self.layers.add_module('activation1', nn.ReLU())
		self.layers.add_module('lin', nn.Linear(32, output))
		self.to(self.device)

	def forward(self, data):
		return self.layers(data)


def training(model, data, answer, loss_f, optim, epochs):
	for epoch in range(epochs):
		r = random.permutation(len(data))
		data, answer = data[r], answer[r]
		for data11, target in zip(data, answer):
			data1 = torch.from_numpy(np.array(data11).reshape(1)).to(model.device)
			answer1 = torch.from_numpy(np.array(target).reshape(1)).to(model.device)

			optim.zero_grad()
			out = model.forward(data1)
			loss = loss_f(out, answer1)
			loss.backward()
			optim.step()

def testing(model, datas):
	o = []
	for data in datas:
		data1 = torch.from_numpy(np.array(data).reshape(1)).to(model.device)
		out = model.forward(data1).cpu().detach().numpy()
		o.append(out)
	return (o)

def K(dist, h= 0.01):
	dist /= h
	return (np.exp(-2*dist**2))

class Nd:
	def __init__(self, kernel, h):
		self.kernel = kernel
		self.h = h
	def fit(self, x_train, y_train):
		self.x_train, self.y_train = np.array(x_train), np.array(y_train)
	def predict(self, data):
		a = cdist(np.reshape(data,[-1,1]), np.reshape(self.x_train,[-1,1]), 'euclidean')
		b = self.kernel(a, self.h)
		val = np.sum(self.y_train * b, -1)
		val_wtht_y = np.sum(b, -1)
		return (val / val_wtht_y)
	

random.seed(0)

x = np.linspace(-5, 5, 300)
w = [0.3, 1, 2.3, 0.7]
a = random.randn(300)*0.4
y = w[0]*x*x -(w[1]**2)*x + w[2]*x+ w[3] + a 
y1 = w[0]*x*x -(w[1]**2)*x + w[2]*x+ w[3] 
fig, ax = plt.subplots()

#pl = PolynomialFeatures(2)
#x = pl.fit_transform(np.array(x).reshape(300,1))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.3, random_state = 0)

model = lin(output=1)
print(model)
training(model, np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32), nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001), 30)
c = testing(model, np.array(x, dtype=np.float32))

"""modelsvc = SVR(C = 1000, kernel='linear')
x_train = np.array(x_train).reshape(210,3)
x_test = np.array(x_test).reshape(90,3)
modelsvc.fit(x_train, y_train)
modelsvc.predict(np.array(x_test).reshape(90,3))"""

modelND = Nd(K, 0.5)
modelND.fit(x_train, y_train)
ND_pred = modelND.predict(x)

ax.scatter(x, y, color = 'r')
ax.plot(x, y1, color = 'b')
ax.plot(x, ND_pred, color='pink')
ax.plot(x, c, color='g')
#ax.scatter(x_test[:, 1], c, color='g')
#ax.scatter(x_test[:, 1], modelsvc.predict(x_test), color='pink')
plt.show()

"""x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.3, random_state = 0)

modelsvc = SVR(C = 1000, kernel='rbf', gamma = 'auto')
x_train = np.array(x_train).reshape(210,1)
x_test = np.array(x_test).reshape(90,1)
modelsvc.fit(x_train, y_train)
modelsvc.predict(np.array(x_test).reshape(90,1))
print(modelsvc.dual_coef_)

ax.scatter(x, y, color = 'r')
ax.plot(x, y1, color = 'b')
ax.scatter(x_test, modelsvc.predict(x_test), color='g')
plt.show()"""