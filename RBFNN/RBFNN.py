import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class RBFNN:
    def __init__(self, h_hidden=15):
        self.nh = h_hidden
        
    def predict(self, X):
        G=np.exp(-(distance.cdist(X, self.C))**2 / self.sigma**2)
        return G @ self.w
    
    def predict_class(self, X):
        G=np.exp(-(distance.cdist(X, self.C))**2 / self.sigma**2)
        return np.argmax(G @ self.w)
    
    def train(self, X, Y):
        #Entrenamiento primer capa
        self.ni , self.no = X.shape[1], Y.shape[1]
        km = KMeans(n_clusters=self.nh).fit(X) #Encuentra los centros de los clusters
        self.C = km.cluster_centers_
        
        #Encontrar la sigma para entrenar la capa salida
        self.sigma = (self.C.max() - self.C.min()) / (np.sqrt(2*self.nh))
        G=np.exp(-(distance.cdist(X, self.C))**2 / self.sigma**2) #Se calcula G, las salidas de las funciones gausianas
        self.w = np.linalg.pinv(G) @ Y #crea las w con la pseudoinversa

def RBFNN_BinaryClasification2D(X, net):
    plt.figure()
    centroids = net.C
    ynew = net.predict(X)
    ynew = np.round(ynew)
    colores=['red', 'blue', 'magenta', 'yellow','fuchsia']
    asignar=[]
    for row in ynew:
        i = int(row)
        asignar.append(colores[i])
    plt.scatter(X[:,0], X[:,1], c=asignar, s=12, marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='black', s=160, label="Centroids") # Marco centroides.
    plt.legend()
    plt.show()

if __name__ == "__main__":
    problem='classification'
    if problem == 'regression':
        p = 500 
        xl, xu = 0, 10

        x = np.linspace(xl, xu, p).reshape(-1, 1)

        y = (0.1 * x -0.5) * np.cos(1/ (0.1*x -0.5))

        #crear y entrenar la red
        neurons = 480
        net = RBFNN(neurons)
        net.train(x,y)
        centroids = net.C

        xnew = np.linspace(xl, xu, p).reshape(-1,1)
        ynew = net.predict(xnew)
        plt.plot(xnew, ynew, '--k')
        plt.plot(x, y, '-c')
        plt.legend(['prediction', 'original'], loc='lower right')
        plt.show()
    
    elif problem == 'classification':
        #Defining magnitudes
        n_inputs = 2
        m_outputs = 1
        break_point = n_inputs #index when the output columns start

        ###### Exporting Data ###########
        dataset = 'moons.csv'
        data = pd.read_csv(dataset)
        input_columns = list(data.columns[0:break_point].values)    #Takes the name of the input rows
        output_columns = data.columns[break_point:].values    #Takes the name of the output row            
        X = np.array(data[input_columns])  #Takes the values of the inputs in format n x p
        Y = np.array(data[output_columns].values)   #Takes the values of the output in format m x p

        ###### Standarization of data ##########
        sd = StandardScaler()
        X = sd.fit_transform(X)

        ###### Training network  #########
        neurons = 16
        net = RBFNN(neurons)
        net.train(X,Y)
        RBFNN_BinaryClasification2D(X, net)