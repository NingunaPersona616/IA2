import numpy as np#Activation fucntions for output layer
def logistic(z, derivative=False):
    a = 1/(1 + np.exp(-z))
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a


def softmax(z, derivative=False):
    e_z = np.exp(z - np.max(z, axis=0))
    a = e_z / np.sum(e_z, axis=0)
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a


def linear(z, derivative=False):
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a


#Activation functions for hiden layers
def relu(z, derivative=False):
    a = z * (z>=0)
    if derivative:
        da = np.array(z>=0, dtype=float)
        return a, da
    return a


def tanh(z, derivative=False):
    a = np.tanh(z)
    if derivative:
        da = (1-a) * (1+a)
        return a, da
    return a


def logistic_hidden(z, derivative=False):
    a = 1 / (1 + np.exp(-z))
    if derivative:
        da = a * (1 - a)
        return a, da
    return a


class DenseNetwork:
    def __init__(self, layers_dim, hidden_activation=tanh, output_activation=logistic):
        #Atributes
        self.L = len(layers_dim) - 1
        self.w = [None] * (self.L+1)
        self.b = [None] * (self.L+1)
        self.f = [None] * (self.L+1)
       
        #initialize
        for l in range(1, self.L+1):
            self.w[l] = -1 + 2* np.random.rand(layers_dim[l], layers_dim[l-1])
            self.b[l] = -1 + 2* np.random.rand(layers_dim[l],1)
           
            if l == self.L:
                self.f[l] = output_activation
            else:
                self.f[l] = hidden_activation


    def predict(self, X):
        a = X
        for l in range(1, self.L+1):
            z = self.w[l] @ a + self.b[l] #Propaga la salida de la capa a la entrada de la sig
            a = self.f[l](z)
        return a


    def fit(self, X, Y, epochs=500, lr=0.1):
        p = X.shape[1]
       
        for _ in range(epochs):
            #Initializing activations, derivatives and local gradient
            a = [None] * (self.L+1)
            da = [None] * (self.L+1)
            lg = [None] * (self.L+1)
           
            a[0] = X #Capa de entrada o capa 0
           
            #Propagation
            for l in range(1, self.L+1):
                z = self.w[l] @ a[l-1] + self.b[l]
                a[l], da[l] = self.f[l](z, derivative=True)
               
            #Backpropagation
            for l in range(self.L, 0, -1):#Ciclo hacia atras
                if l == self.L:
                    lg[l] = -(Y-a[l]*da[l])
                else:
                    lg[l] = (self.w[l+1].T @ lg[l+1]) * da[l]
           
            for l in range(1, self.L+1):
                self.w[l] -= (lr/p) * (lg[l]@a[l-1].T)
                self.b[l] -= (lr/p) * np.sum(lg[l])