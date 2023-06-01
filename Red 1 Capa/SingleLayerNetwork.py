import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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


xmin, xmax = -1, 1
ymin, ymax = -1, 1

###### Exporting Data ###########
file = "Dataset_A03.csv"
data = pd.read_csv(file)
n_inputs = 2
n_outputs = 4

input_columns = list(data.columns[0:2].values)    #Takes the name of the input rows
output_columns = list(data.columns[2:].values)    #Takes the name of the output row            

X = np.array(data[input_columns]).T   #Takes the values of the inputs in format n x p
Y = np.array(data[output_columns].values).T   #Takes the values of the output in format m x p

"""p = X.shape[1]
p_train = (p/100) * 70
p_test = (p/100) * 30
print(p_train, p_test)"""

#Training
net = OLN(n_inputs, n_outputs, softmax)
net.fit(X,Y)

# Prediction
prediction = net.predict(X)

# Plot
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
for i in range(X.shape[1]):
    ax1.plot(X[0,i], X[1, i], '*', c=cm[y_c[i]])
ax1.axis([xmin,xmax,ymin,ymax])
ax1.grid()
ax1.set_title('Problema original')

ax2 = plt.subplot(1,2,2)
y_c = np.argmax(prediction, axis=0)
for i in range(X.shape[1]):
    ax2.plot(X[0,i], X[1, i], '.', c=cm[y_c[i]])
ax2.axis([xmin,xmax,ymin,ymax])
ax2.grid()
ax2.set_title('Predicción de la red')

plt.show()