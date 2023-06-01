import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from MLP import *

# Delay block
class delayBlock:
    def __init__(self, n_inputs, n_delays):
        self.memory = np.zeros((n_inputs, n_delays)) #creates memory buffer

    def add(self, x):
       #creates a copy of the original signal
       x_new = x.copy().reshape(1,-1)

       #retard implementation
       self.memory[:,1:] = self.memory[:,:-1]
       self.memory[:,0] = x_new

    def get(self):
        return self.memory.reshape(-1, 1, order='F')

    def add_and_get(self, x):
        self.add(x)
        return self.memory.reshape(-1, 1, order='F')

#test code
'''A = [np.random.rand(2,1) for i in range(50)]
db = db = delayBlock(2,5)
for i in range(6):
    db.add(A[i])
    print(db.get())'''

class NARX:
    def __init__(self, n_inputs, n_outputs, n_delays,
                    dense_hidden_layers=(100,),
                    learning_rate=0.01,
                    n_repeat_train=5):
        self.net = DenseNetwork(((n_inputs+n_outputs)*n_delays,
                                *dense_hidden_layers, n_outputs),
                                output_activation=linear)
        self.dbx = delayBlock(n_inputs, n_delays)
        self.dby = delayBlock(n_outputs, n_delays)
        self.learning_rate = learning_rate
        self.n_repeat_train = n_repeat_train

    def predict(self, x):
        '''
        NARX prediction 
        x its a vector of shape (n_inputs, 1)
        '''

        #prepare an entry extended trough time
        X_block = self.dbx.add_and_get(x)
        Y_est_block = self.dby.get()
        net_input = np.vstack((X_block, Y_est_block))

        #neural network prediction
        y_est = self.net.predict(net_input)

        #store prediction in the recurrent bock
        self.dby.add(y_est)

        return y_est #return prediction

    def predict_and_train(self, x, y):
        '''
        NARX prediction and train in line 
        x is the input vector of shape (n_inputs, 1)
        y is the output vector of shape(n_outputs, 1)

        gets prediction before before training, but is stored after the training
        '''

        #prepare an input extended trough time
        X_block = self.dbx.add_and_get(x)
        Y_est_block = self.dby.get()
        net_input = np.vstack((X_block, Y_est_block))

        #neural network prediction
        y_est = self.net.predict(net_input)

        #training neural network
        self.net.fit(net_input, y,
                    epochs=self.n_repeat_train,
                    lr=self.learning_rate)
        
        #store prediction in the recurrent bock
        self.dby.add(y_est)

        return y_est #return prediction


if __name__ == '__main__':
    #getting the data from first year
    dataset = 'daily-min-temperatures.csv'
    data = pd.read_csv(dataset, parse_dates=True)
    columns = list(data.columns[0:])
    X = np.array(data[columns])
    X = X[0:365]
    print(X.shape[0])

    #configuring NARX
    narx = NARX(1, 1, 10,
            dense_hidden_layers=(100,),
            learning_rate=0.01, n_repeat_train=1)
    y_est = np.zeros((1, X.shape[0]))

    #Training and predicting NARX
    for i in range(X.shape[0]-1):
        x = np.array([X[i,1]]).reshape(-1,1)
        y = np.array([X[i+1,1]]).reshape(-1,1)
        #print(y)
        y_est[:,i] = narx.predict_and_train(x, y).ravel()

    #Predicting one last step in "future"
    #take the last data used to train, to predict a step
    future_temperature = narx.predict(np.array([X[i,1]]).reshape(-1,1)) 
    print(X[i+1,1], future_temperature)

    '''
    fig, ax = plt.subplots(1,1)
    ax.plot(X[0:,0], X[0:,1], '-b')
    #plt.tick_params(axis='x', labelrotation=90)
    for i, label in enumerate(ax.get_xticklabels()):
        if i > 0 and i < len(ax.get_xticklabels()) - 1:
            label.set_visible(False)
    '''

    plt.plot(X[0:,0], X[0:,1], '-b')
    plt.plot(X[0:i+1,0], y_est[0,0:i+1], '--k')
    plt.plot(X[i+1,0], future_temperature, '.r')
    plt.legend(['original', 'train-prediction', 'predict'], loc='lower right')
    plt.xticks([])
    plt.show()