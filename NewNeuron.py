import numpy as np
from numpy.random import seed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def classifier(data, xtrain, ytrain, h, n_neighbours = 100):
	ytrain = np.array(ytrain)
	answers_data =[]
	distance = cdist(data, xtrain, 'euclidean')
	indexes = np.array([i.argsort() for i in distance[:]])
	distance /= h # h - размер окна Парзена
	weights_ker = np.exp(-2*distance**2) # ядро
	v = []
	for i, x in enumerate(data):
		s = [np.where(ytrain[indexes[i, 0:n_neighbours]] == j, weights_ker[i, indexes[i, 0:n_neighbours]], 0).sum() for j in np.unique(ytrain)] # считает веса соседей разных классов вблизи объекта
		v.append(s.index(max(s))) # выбирает класс, вес объектов которого больше воздействует на объект
	return (v)


def weight(dist, h= 0.01):
	dist /= h
	return (np.exp(-2*dist**2))


x1 = []
y1 = []
with open("C:\\Users\\kozlo\\source\\repos\\VSCODE PYTHON\\Neuron\\NewNeuron\\data.txt", "r") as f:
	for a in f:
		a = a.strip().split()
		x1.append(a[0:4])
		if a[4] == "Iris-setosa":
			y1.append(0)
		elif a[4] == "Iris-versicolor":
			y1.append(1)
		else: y1.append(2)

x_train, x_test, y_train, y_test = train_test_split(np.array(x1)[:,0:3:2],np.array(y1), test_size=0.3, random_state=0)

x_train, x_test = [np.array(i, dtype=float) for i in[x_train, x_test]]
y_train, y_test = [np.array(i, dtype=int) for i in[y_train, y_test]]

model = KNeighborsClassifier(n_neighbors = 105, weights=weight)
model.fit(x_train, y_train)

print(np.where(model.predict(x_test) == y_test, 1, 0).sum())
print(np.where(classifier(x_test, x_train, y_train, h=1,  n_neighbours = 105) == y_test, 1, 0).sum())

fig, axes = plt.subplots(1,2)

plot_decision_regions(x_train, y_train, model, ax=axes[0])
plot_decision_regions(x_test, y_test, model, ax=axes[1])
plt.show()