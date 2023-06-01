import numpy as np

def linear(z, derivative=False):
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

def logistic(z, derivative=False):
    a = 1 / (1 + np.exp(-z))
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

class OLN:
    def __init__(self, n_inputs, n_outputs, activation_function=linear):
        self.w = -1 + 2 * np.random.rand(n_outputs, n_inputs)
        self.b = -1 + 2 * np.random.rand(n_outputs, 1)
        self.f = activation_function

    def predict(self, X):
        Z = self.w @ X + self.b
        return self.f(Z)
    
    def fit(self, X, Y, epochs = 1000, lr = 0.3):
        p = X.shape[1]
        for _ in range(epochs):
            # Propagar la red
            Z = self.w @ X + self.b
            Yest, dY = self.f(Z, derivative=True)
            
            # Calcular el gradiente
            lg = (Y - Yest) * dY

            # Actualización de parámetros
            self.w += (lr/p) * lg @ X.T
            self.b += (lr/p) * np.sum(lg)


# Ejemplo del profe
xmin, xmax = -5, 5

classes = 8
p_c = 2
ruido = 0.15


X = np.zeros((2,classes*p_c))
Y = np.zeros((classes, classes * p_c))

for i in range(classes):
    seed = xmin + (xmax - xmin) * np.random.rand(2,1)
    X[:,i*p_c:(i+1)*p_c] = seed + ruido * np.random.rand(2, p_c)
    Y[:,i*p_c:(i+1)*p_c] = np.ones((1, p_c))

print(Y)
# Dos entradas, 8 salidas, softmax porque es multiclase
net = OLN(2, 8, softmax)
net.fit(X,Y)

# Dibujo
import matplotlib.pyplot as plt

cm = [[0,0,0],
      [1,0,0],
      [0,1,0],
      [1,1,0],
      [0,0,1],
      [1,0,1],
      [0,1,1],
      [1,1,1]]

ax1 = plt.subplot(1,2,1)
y_c = np.argmax(Y, axis=0)
print(y_c)
for i in range(X.shape[1]):
    ax1.plot(X[0,i], X[1, i], '*', c=cm[y_c[i]])
ax1.axis([-5.5,5.5,-5.5,5.5])
ax1.grid()
ax1.set_title('Problema original')

ax2 = plt.subplot(1,2,2)
y_c = np.argmax(net.predict(X), axis=0)
print(y_c)
for i in range(X.shape[1]):
    ax2.plot(X[0,i], X[1, i], '*', c=cm[y_c[i]])
ax2.axis([-5.5,5.5,-5.5,5.5])
ax2.grid()
ax2.set_title('Predicción de la red')

plt.show()