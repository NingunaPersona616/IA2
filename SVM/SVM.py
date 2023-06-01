import numpy as np
import pandas as pd
from plotting import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import cvxopt

class SVM():
    def __init__(self, kernel='linear', C=0.001, gamma=0.001, degree=3):
        #parameters
        self.C = float(C)
        self.gamma = float(gamma)
        self.d = int(degree)
        
        if kernel == 'linear':
            self.kernel = self.linear
        elif kernel == 'polynomial':
            self.kernel = self.polynomial
        elif kernel == 'gaussian':
            self.kernel = self.gaussian
        else:
            raise NameError('Unknown')
            
    def linear(self, x1, x2):
        return(np.dot(x1, x2))
        
    def polynomial(self, x1, x2):
        return (np.dot(x1,x2)+1) ** self.d
    
    def gaussian(self, x1, x2):
        return np.exp(-self.gamma*np.linalg.norm(x1-x2) ** 2)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape 
        
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])
        
        #------Solve cuadratic problem
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)
        
        if self.C == 0:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.identity(n_samples) * -1
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
            
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        #-------------------------------------------------------------------------------------------------
        
        #extrct lagrange multipliers
        lamb = np.ravel(solution['x'])
        mask = lamb > 1e-5
        ind = np.arange(len(lamb))[mask]
        
        self.lamb = lamb[mask]
        
        #extrac support vectors
        self.sv = X[mask]
        self.sv_y = y[mask]
        
        #calculate b
        self.b = 0
        for i in range(len(self.lamb)):
            self.b += self.sv_y[i]
            self.b -= np.sum(self.lamb * self.sv_y * K[ind[i], mask])
            
        self.b = self.b / len(self.lamb)
        
    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.lamb, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv)
            y_predict[i] = s
                
        return y_predict + self.b
    
    def predict(self, X):
        return np.sign(self.project(X))
    
if __name__ == '__main__':
    #defining magnitudes
    n_inputs = 2
    m_outputs = 1
    break_point = n_inputs #index when the output columns start

    ###### Exporting Data ###########
    dataset = 'moons.csv'
    data = pd.read_csv(dataset)
    input_columns = list(data.columns[0:break_point].values)    #Takes the name of the input rows
    output_columns = data.columns[break_point:].values    #Takes the name of the output row            
    x = np.array(data[input_columns])  #Takes the values of the inputs in format n x p
    y = np.array(data[output_columns].values).reshape(-1,)   #Takes the values of the output in format m x p
    y = np.asanyarray(y, dtype=np.float64)
    y += (y == 0) * -1.0

    #Training SVM
    model = SVM(kernel='polynomial', C=0.5)
    model.fit(x, y)

    #PLotting
    plot_svm(x, y, model)