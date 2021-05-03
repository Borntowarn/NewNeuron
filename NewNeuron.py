import numpy as np
from numpy.random import seed

class neuron:
    temp = 0.001
    x = np.array([[]])
    y = np.array([])
    eta = 0.001
    cost = []
    iter = 0
    errors_ = []
    w = np.array([])
    prov1 = 0.0
    prov2 = 0.0
    prov3 = 0.0
    prov4 = 0.0
    random_s = 0

    def __init__(self, iter, data, answer, eta, random_s = None):
        self.eta = eta
        self.iter = iter
        self.x = data[0: ,0 : 3 : 2]
        self.y = answer
        self.prov1 = self.x[:, 0].mean()
        self.prov2 = self.x[:, 1].mean()
        self.prov3 = self.x[:, 0].std()
        self.prov4 = self.x[:, 1].std()
        self.w = np.zeros(1 + self.x.shape[1])
        self.x[:, 0] = (self.x[:,0] - self.prov1) / self.prov3
        self.x[:, 1] = (self.x[:,1] - self.prov2) / self.prov4
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
                error = self.y[q] - self.activation(self.x[q, :])
                self.w[1:] += self.eta * self.x[q].T.dot(error)
                self.w[0] += self.eta * error
                dop = -np.log(self.activation(self.x[q, :]))
                dop1 = -np.log((1 - self.activation(self.x[q, :])))
                if self.y[q] == 1 : e += dop
                else: e += dop1
            self.errors_.append(e)
            print (self.errors_[i])
            print (self.proverka())
        return self

    def shuffle(self, x, y):
        r = np.random.permutation(len(y))
        return x[r], y[r]

    def predict(self,x):
        return (np.where(self.clear_out(x) >= 0.5, 1, 0))

    def _predict(self,x):
        x = np.array(x)
        if x.ndim == 1:
            x[0] = (x[0] - self.prov1) / self.prov3
            x[1] = (x[1] - self.prov2) / self.prov4
        else:
            x[:, 0] = (x[:,0] - self.prov1) / self.prov3
            x[:, 1] = (x[:,1] - self.prov2) / self.prov4
        return (self.activation(x), np.where(self.activation(x) >= 0.5, 1, 0))

    def proverka(self):
        return (np.where(self.predict(self.x) == self.y, 1, 0).sum())

x1 = []
y1 = []
with open("C:\\Users\\kozlo\\source\\repos\\VSCODE\\Neuron\\NewNeuron\\data.txt", "r") as f:
    for a in f:
        a = a.strip().split()
        x1.append(a[0:4])
        if a[4] == "setosa":
            y1.append(1)
        else:
            y1.append(0)

n = neuron(20, np.array(x1, dtype = float), np.array(y1), 0.02)
n.learning()
print(n.proverka())


a = [float(s) for s in input().split()]

while(a[0] != 0):
    odd, _class = n._predict(a[0:])
    if (_class == 1) : print("setosa c вероятностью ", round(odd*10000)/100, "%")
    else : print ("versicolor c вероятностью ", round((1-odd)*10000)/100, "%")
    a = [float(s) for s in input().split()]

