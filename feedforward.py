import numpy as np
import pickle
import random

#Network object creates a feed forward neural Network
#Initialize the network by passing a list of layersizes
class Network:
    def __init__(self,layersizes, learning_rate=0.1):
        self.K = len(layersizes) #K stores the number of layers
        self.learning_rate = learning_rate
        self.Layers =[] #Layers is an array of Layer objects of length layersizes
        l = Layer(layersizes[0],0,inputlayer=True)
        self.Layers.append(l)
        for i in range(1, self.K-1):
            l = Layer(layersizes[i],layersizes[i-1])
            self.Layers.append(l)

        l = Layer(layersizes[self.K-1],layersizes[self.K-2],outputlayer=True)
        self.Layers.append(l)



    #Train takes in X and its labels and trains for a number of steps
    def train(self,X, y, steps):
        #Train for a given number of steps
        for i in range(steps):
            error = 0.0
            for x in range(y.shape[0]):
                self.forwardpass(X[x])
                error = self.backpropogation(y[x], 82)
            #print out the error (MSE) every 50 iterations
            if i % 50:
                print(error)

    #make a prediction
    #returns a list of predictions for your test set
    def predict(self, x_test):
        pred = np.zeros(x_test.shape[0])
        for i in range(x_test.shape[0]):
            self.forwardpass(x_test[i])
            pred[i] = np.argmax(self.Layers[self.K-1].Output)

        return pred

    ####cross_validate returns a prediction accuracy on your validation set
    ####It automatically splits your training set into a training and validation
    ####size of each is specified by user (default = 0.25)
    def crossvalidate(self,X,y, steps, test_size=0.25):
        #train the network
        self.train(X[(test_size*X.shape[0]):,:], y[(test_size*X.shape[0]):], steps)
        #make predictions
        predictions = self.predict([X[0:(test_size*X.shape[0]),:]])

        #evaluate your predictions
        y_val = y[0:(test_size*X.shape[0])]
        count = 0 #count is the number of correct predictions
        for i in range(predictions.shape[0]):
            if predictions[i] == y_val[i]:
                count+=1

        return count/predictions.shape[0];

    #forward pass through the network
    #compute the prediction given an input
    def forwardpass(self,inp):
        self.Layers[0].output = inp

        ###hidden layer
        for i in range(1,self.K):
            self.Layers[i].computeoutput(self.Layers[i-1].output)

    def backpropogation(self, y_labels, classes):
        self.output_error(y_labels,classes)
        return self.BackPropogateError(y_labels,82)

    #compute the error on the output
    def output_error(self, y_labels, classes):
        #compare the softmax values to the actual values of y which are one hot encoded
        soft=softmax(self.Layers[self.K-1].output)
        one = one_hot(y_labels, classes)

        for i in range(classes):
            #calcualte the delta on the output later
            err = soft[i] - one[i]
            self.Layers[self.K-1].delta[i] = sigmoidder(self.Layers[self.K-1].output[i])*err

    def BackPropogateError(self,y_labels, classes):
        k = self.K-2 #start at the second to last layer

        #temrinate before we get to the input layer because the input layer has no weights
        #first we are going to calculate the deltas
        while k >= 1:
            #i iterates through all nodes in a given layer
            for i in range(self.Layers[k].size):
                error = 0.0 #j iterates through all the nodes which we need to backprogate the error along
                for j in range(self.Layers[k+1].size):
                    error += self.Layers[k+1].delta[j] *self.Layers[k+1].weights[i][j]

                self.Layers[k].delta[i] = sigmoidder(self.Layers[k].output[i]) * error
            k-=1

        k = self.K-2
        ###update the weights
        while k >= 1:
            for i in range(self.Layers[k].size):
                for j in range(self.Layers[k+1].size):
                    updates = self.Layers[k+1].delta[j] *self.Layers[k].output[j]
                    self.Layers[k+1].weights[i][j] -= self.learning_rate*updates #update the weights based on the elarning rate

            k-=1

        #calculate error
        error = 0.0
        one = one_hot(y_labels, classes)
        soft=softmax(self.Layers[self.K-1].output)
        for k in range(classes):
            error += 0.5 * (one[k] - soft[k]) ** 2 #MSE is returned so we can return some user feedback
        return error



####LAYER class ia a layer of the neural Net
#####It stores an np array for the nodes, weights, biases and output#####
class Layer:
    def __init__(self,layersize,inputsize=0,outputlayer=False,inputlayer=False):
        self.size = layersize
        self.nodes = np.zeros(layersize+1)
        #for the input layer we don't want to declare weights, biases or delta
        #weights are defined as the weights coming into the layer
        if inputlayer==False:
            #weights is a 2d nparray which is[size of input by size of this layer]
            #weights are initialized as a random normal
            self.weights = np.random.normal(0,0.1,(inputsize,layersize))
            self.delta = np.zeros((layersize), dtype=np.float32)
            self.biases = np.zeros((layersize), dtype=np.float32)
        if outputlayer == True:
            self.output = np.zeros((layersize),dtype=np.float32)
        else:
            self.output = np.zeros(layersize, dtype=np.float32)

    #given a layer this computes the output for that layer
    #note that this function doesn't work for the input layer because it has no weights or biases defined
    def computeoutput(self,inp):
        for x in range(0,self.size):
            self.output[x] = sigmoid(dotproduct(self.weights[:,x],inp) + self.biases[x])

##############IMPORTANT MATH FUCNTIONS USED THROUGHOUT###############
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoidder(output):
    return (1-output)*output

def dotproduct(w, x):
    return np.dot(w,x)

def softmax(output):
    exp = np.exp(output)
    return exp/np.sum(exp,axis=0)

def one_hot(y,classes):
    one_hot = np.zeros((classes),dtype=np.float32)
    one_hot[y] = 1.0
    return one_hot

def tnormal(mean=0, sd=.1, low=-.3, upp=0.3):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
##################END OF MATH FUNCTIONS#############################


if __name__ =='__main__':
    nn = Network([4096,4000, 4000, 82])
    x_in = open('x_train_data.pkl','rb')

    x = pickle.load(x_in) # load from text
    print('done loading x data')

    print('done loading y data')


    y_in = open('y_train_data.pkl', 'rb')
    y = pickle.load(y_in)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    x= x.reshape(-1,4096)
    x_in.close()
    y_in.close()
    #define a network
    #our example is 4 layers
    nn = Network([4096, 4000, 2000, 82])
    #train the network
    print('starting training')
    nn.train(x,y, 1000)
    print('done_training')

    x_test_in = open('x_test_data.pkl','rb')
    #y_in = open('y_train_data.pkl', 'rb')

    print('loading picked data...')
    x_test = pickle.load(x_test_in) # load from text

    print('done loading data!')
    x_test= binarynormalization(x_test,0.71)
    x_test = np.asarray(x_test, dtype=np.float32)
    #y = np.asarray(y, dtype=np.int32)
    x_test_in.close()
