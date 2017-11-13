import numpy as np
import scipy.ndimage as img
import scipy.misc
import scipy.signal as md
import matplotlib.pyplot as plt



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
#said his best results were with 0.71
def binarynormalization(arr, threshold):
    arr = arr.astype('float')

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
        plt.figure(facecolor="white")
        plt.axis('off')
        plt.imshow(arr[i,...],cmap='gray')
        plt.show

    return arr

####flips the values of binary normalization
####all values less than the threshold are turned to a 1, greater than the threshold = 0
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


####Adds dilation which changes the pixel of interest based on the surrounding pixels
####Used in attempt to eliminate some of the back ground noise
####Best threshold value for binary dilation was 0.81
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
        ###dilation taken from the scipy.ndimage
        img.binary_dilation(arr[i])

    return arr

####Same as binary_dilation but flips the values
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
