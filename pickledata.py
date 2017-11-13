import numpy as np
import pickle

###picke the x_training data
x_train = np.loadtxt('train_x.csv',delimiter=",")
x_out = open('x_train_data.pkl', 'wb')
x_train= x_train.reshape(-1,64,64)
pickle.dump(x_train, x_out)
x_out.close()

###pickle the labels for the y data
y_train = np.loadtxt('train_y.csv',delimiter=",")
y_out = open('y_train_data.pkl', 'wb')
pickle.dump(y_train, y_out)
y_out.close()

#pickle the test data
x_test = np.loadtxt('test_x.csv',delimiter=",")
x_out = open('x_test_data.pkl', 'wb')
x_test= x_test.reshape(-1,64,64)
pickle.dump(x_test, x_out)
x_out.close()
