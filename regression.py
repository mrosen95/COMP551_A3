import pickle
import io
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import scipy.ndimage as img

x_in = open('x_train_data.pkl','rb')

y_in = open('y_train_data.pkl', 'rb')
x = pickle.load(x_in) # load from text
print('done loading x data')
y = pickle.load(y_in)
print('done loading y data')
x = np.asarray(x, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)
x_in.close()
y_in.close()

###binary normalization of the data###
##threshold value from [0,1]
##if noramlized pixel >= threshold, pixel = 1 else pixel = 0
def binarynormalization(arr, threshold):
    arr = arr.astype('float')

    for i in range(arr.shape[0]):
        minval = arr[i,...].min()
        maxval = arr[i,...].max()
        if minval!= maxval:
            arr[i,...] -= minval
            arr[i,...] /= (maxval-minval)

        for x in range(arr.shape[1]):
            for y in range(arr.shape[2]):
                if arr[i,x,y] >= threshold:
                    arr[i,x,y] = 1.0
                else:
                    arr[i,x,y] = 0.0
    return arr


print('remodeling data...')
x = binarynormalization(x, 0.76)
x = x.reshape(-1, 4096) #each entry was in a 2d array, reshape to 1-d
print('done remodeling data!')

#create a logistic regression classifier
regr = linear_model.LogisticRegression()

# Fit the model to our training set
print('fitting the model...')
regr.fit(x, y)
print('model fit!')

print('loading the test data...')
x_test_in = open('x_test_data.pkl','rb')
x_test = pickle.load(x_test_in)
print('done reading test data')

print('making prediction....')
x_test = binarynormalization(x_test, 0.76)
x_test = x_test.reshape(-1, 4096)

x_test = np.asarray(x_test, dtype=np.float32)

# Make predictions using the testing set
y_pred = regr.predict(x_test)


output = io.open('lin_regression.csv', 'w', encoding='utf-8')
count = 1
output.write(u'Id,Label\n')
for x in np.nditer(y_pred):
    output.write(str(count) + u',' + str(x) + u'\n')
    count += 1
output.close()
