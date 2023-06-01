import numpy as np

def linear(z, derivative=False):
    a = z
    if derivative:
        da = np.ones(z.shape())
        return a, da
    return a

def logistic(z, derivative=False):
    a = 1 / (1 + np.exp(-z))
    if derivative:
        da = a * (1 - a)
        return a, da
    return a

def tanh(z, derivative=False):
    a = np.tanh(z)

    if derivative:
        da = (1+a) * (1-a)
        return a, da
    return a

def relu(z, derivative=False):
    a = z * (z >= 0)
    if derivative:
        da = np.array(z >= 0, dtype=float) #casteo a numeros reales
        return a, da 
    return a

class Neuron:
    def __init__(self, n_inputs,activation_function, learning_rate=0.1):
        self.w = 1 - 2 *np.random.rand(n_inputs)
        self.b = 1 - 2 *np.random.rand()
        self.eta = learning_rate
        self.f = activation_function

    def predict(self, X):
        Z = np.dot(self.w, X) + self.b

        if self.f == logistic:
            return 1.0*(self.f(Z) >= 0.5)
        return self.f(Z)

    def fit(self, X, Y, epochs=500):
        p = X.shape[1]  #Toma el numero de elementos en el arreglo de datos

        for _ in range(epochs):
            ####### PROPAGATE ##########
            Z = np.dot(self.w, X) + self.b
            Yest, dy = self.f(Z, derivative=True)   #Toma la derivada de la funcion generalizada con la regla delta

            ####### TRAINING ########
            #Calculate local gradient
            local_grad = (Y -Yest) * dy

            ####### UPDATING PARAMETERS #########
            self.w += (self.eta/p) * np.dot(local_grad, X.T).ravel()
            self.b += (self.eta/p) * np.sum(local_grad)

def draw_2d(net):
    w1, w2, b = net.w[0], net.w[1], net.b
    plt.plot([-2,2], [(1/w2)*(-w1*(-2)-b), (1/w2)*(-w1*2-b)],'--k')

#Ejemploooooooo

import matplotlib.pyplot as plt 

X = np.array([[0,0,1,1],[0,1,0,1]])

Y = np.array([0,0,0,1])

model = Neuron(2, logistic, 1)
model.fit(X, Y)
Y_est = model.predict(X)
print(Y_est)

#Plot
draw_2d(model)
for i in range(X.shape[1]):
    if Y_est[i] == 1:
        plt.plot(X[0, i], X[1, i], 'ob')
    else:
        plt.plot(X[0, i], X[1, i], '*r')

plt.title('Imc Perceptron')
plt.xlim([-1,2])
plt.ylim([-1,2])
plt.xlabel(r'$masa [40,200]$')
plt.ylabel(r'$altura [1.0,2.4]$')
plt.show()