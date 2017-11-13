import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(z))

def dsig(y):
    return y*(1.0-y)

def softmax(output):
    exp = np.exp(output)
    return exp/np.sum(exp,axis=1)

class Network:
    def __init__(self,layersizes, learning_rate=0.1):


class Layer:
    def __init__(self, number_in, number_out):
        self.size = number_in
        self.weights = np.random.randn(n_in,n_out) * 0.05
        self.biases = np.random.uniform(0.0, 0.1, number_in)
        self.output=np.zeros(number_in)

    def computeoutput(self,input):
        for x in range(0,self.size):
            self.output[x] = self.weights[x] * input + self.biases[x]

    def get_gradient(self, X, output )
