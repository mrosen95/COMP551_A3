import numpy as np
import scipy.ndimage as img
import scipy.misc
import scipy.signal as md
import matplotlib.pyplot as plt

x = np.genfromtxt('test_head.csv',delimiter=",")
x = np.asarray(x, dtype=np.float32)
x = x.reshape(-1, 64,64)


#min of every picture = 0
#max of every picture = 255
def normalize(arr):
    ##linear normalization
    arr = arr.astype('float')
    for i in range(arr.shape[0]):
        minval = arr[i,...].min()
        maxval = arr[i,...].max()
        if minval != maxval:
            arr[i,...] -= minval
            arr[i,...] *= (255.0/(maxval-minval))
    return arr



#converts everything to a 0 or a 1
#binary threshold (i haven't tested but one of my friends said use >0.90)
#said his best results were with 0.96
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


def binaryinversion(arr, threshold):
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
                    arr[i,x,y] = 0.0
                else:
                    arr[i,x,y] = 1.0
    return arr

def binarywithdilation(arr,threshold):
    for i in range(arr.shape[0]):
        minval = arr[i,...].min()
        maxval = arr[i,...].max()
        if minval!= maxval:
            arr[i,...] -= minval
            arr[i,...] /= (maxval-minval)

        for j in range(arr.shape[1]):
            for y in range(arr.shape[2]):
                if arr[i,j,y] >= threshold:
                    arr[i,j,y] = 1.0
                else:
                    arr[i,j,y] = 0.0

        plt.subplot(211)
        plt.imshow(x[i,...])
        #md.medfilt(arr[i])
        plt.subplot(212)
        plt.imshow(arr[i,...])
        plt.show()
    return arr

z = np.array(x)
binarywithdilation(z,0.81)





def inversedilation(arr, threshold):
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
        img.binary_dilation(arr[i])

        for x in range(arr.shape[1]):
            for y in range(arr.shape[2]):
                if arr[i,x,y] >= threshold:
                    arr[i,x,y] = 0.0
                else:
                    arr[i,x,y] = 1.0
    return arr
