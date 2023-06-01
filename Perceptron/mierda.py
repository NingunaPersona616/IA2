import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

class Perceptron:
    def __init__(self, n_input, learning_rate):
        self.w = -1 + 2 * np.random.rand(n_input)
        self.b = -1 + 2 * np.random.rand()
        self.eta = learning_rate

    def Predict(self, X):
        _, p = X.shape
        y_est = np.zeros(p)
        for i in range(p):
            y_est[i] = np.dot(self.w, X[:, i]) + self.b
            if y_est[i] >= 0:
                y_est[i] = 1
            if y_est[i] < 0:
                y_est[i] = 0
        return y_est

    def fit(self, X, Y, epochs=50):
        _, p = X.shape
        for _ in range(epochs):
            for i in range(p):
                y_est = self.Predict(X[:, i].reshape(-1,1))
                self.w += self.eta * (Y[i] - y_est) * X[:, i]
                self.b += self.eta * (Y[i] - y_est)

def draw_2d(net):
    w1, w2, b = net.w[0], net.w[1], net.b
    plt.plot([-2,2], [(1/w2)*(-w1*(-2)-b), (1/w2)*(-w1*2-b)],'--k')

#Ejemplo
def Compuertas_Perceptron(Compuerta):
    X = np.array([[0,0,1,1],[0,1,0,1]])

    if Compuerta == "AND":
        Y = np.array([0,0,0,1])

    elif Compuerta == "OR":
        Y = np.array([0,1,1,1])

    elif Compuerta == "XOR":
        Y = np.array([0,1,1,0])

    model = Perceptron(2, 0.1)

    print(model.Predict(X))
    model.fit(X, Y)
    print(model.Predict(X))

    _, p = X.shape


    draw_2d(model)
    for i in range(p):
        if Y[i] == 1:
            plt.plot(X[0, i], X[1, i], 'ob')
        else:
            plt.plot(X[0, i], X[1, i], '*r')

    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.show()

def IMC_Normalization(x, p, csv_file):
    x[0, :] = csv_file['weight'].values
    x[1, :] = csv_file['height'].values

    #Normalizacion de datos
    min_w = np.amin(x[0,:])
    max_w = np.amax(x[0,:])

    min_h = np.amin(x[1,:])
    max_h = np.amax(x[1,:])

    for i in range(p):
        x[0,i] = (x[0,i] - min_w)/(max_w-min_w)
        x[1,i] = (x[1,i] - min_h)/(max_h-min_h)

def IMC_Perceptron(Dimension):
    #Creacion y almacenamiento de datos de prueba y datos para prediccion
    """
    p = 50 #Poblacion
    d = 2 #Dimension
    X = np.zeros([d,p])
    Y = np.zeros(p)
    for i in range(p):
        X[0,i] = round(200+(40-200)*random.random(), 2) #Peso   
        X[1,i] = round(2.40+(1.0-2.40)*random.random(), 2) #Altura
        #imc = round(X[0,i]/X[1,i]**2,2)
        #if imc > 25:
        #    Y[i] = 1
        #else:
        #    Y[i] = 0


    raw_data = {'weight': X[0,:],
                'height':  X[1,:],
                #'IMC': Y
                }

    #df = pd.DataFrame(raw_data, columns = ['weight', 'height', 'IMC'])
    df = pd.DataFrame(raw_data, columns = ['weight', 'height'])
    df.to_csv('raw_data.csv', index=False)
    """

    #Read TEST Data from a csv file
    data = pd.read_csv("TEST_data.csv")
    Y = data['IMC'].values
    p = len(Y)
    X = np.zeros([Dimension,p])
    IMC_Normalization(X, p, data)

    #Entrenamiento
    model = Perceptron(Dimension, 0.1)
    print("training ...\n\n")
    model.fit(X, Y)

    #Predicciones
    data = pd.read_csv("raw_data.csv")
    p = data.shape[0]
    X = np.zeros([Dimension,p])
    IMC_Normalization(X, p, data)

    print("Predict once trained:")
    Y_est = model.Predict(X)
    print(Y_est)

    #Plot
    draw_2d(model)
    for i in range(p):
        if Y_est[i] == 1:
            plt.plot(X[0, i], X[1, i], 'ob')
        else:
            plt.plot(X[0, i], X[1, i], '*r')

    plt.title('Imc Perceptron')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel(r'$masa [40,200]$')
    plt.ylabel(r'$altura [1.0,2.4]$')
    plt.show()


if __name__ == "__main__":
    #Compuertas_Perceptron("XOR")
    IMC_Perceptron(Dimension=2)