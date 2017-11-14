from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import pickle
import scipy.ndimage as img
import io

from sklearn.metrics import confusion_matrix
from IPython.display import Image
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

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




def concatlayer1(input_layer):
    ### This is the first inception module. The strucutre of this is best
    ### visualized in our writeup.
    conv2a=tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[1, 1],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.05),
      activation=tf.nn.relu)

    conv1b=tf.layers.conv2d(
      inputs=input_layer,
      filters=96,
      kernel_size=[1, 1],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.05),
      activation=tf.nn.relu)

    conv1c=tf.layers.conv2d(
      inputs=input_layer,
      filters=16,
      kernel_size=[1, 1],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.05),
      activation=tf.nn.relu)

    pool1d=tf.layers.max_pooling2d(inputs=input_layer, pool_size=[3, 3], strides=1,padding='same')

    conv2b=tf.layers.conv2d(
      inputs=conv1b,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.05),
      activation=tf.nn.relu)

    conv2c=tf.layers.conv2d(
      inputs=conv1c,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.05),
      activation=tf.nn.relu)

    conv2d=tf.layers.conv2d(
      inputs=pool1d,
      filters=32,
      kernel_size=[1, 1],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.05),
      activation=tf.nn.relu)


    return tf.concat([conv2a,conv2b,conv2c,conv2d],3),256


def concatlayer2(input_layer):
    ### This is the second inception module. The strucutre of this is best
    ### visualized in our writeup.
    conv2a=tf.layers.conv2d(
      inputs=input_layer,
      filters=160,
      kernel_size=[1, 1],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.05),
      activation=tf.nn.relu)

    conv1b=tf.layers.conv2d(
      inputs=input_layer,
      filters=112,
      kernel_size=[1, 1],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.05),
      activation=tf.nn.relu)

    conv1c=tf.layers.conv2d(
      inputs=input_layer,
      filters=24,
      kernel_size=[1, 1],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.05),
      activation=tf.nn.relu)

    pool1d=tf.layers.max_pooling2d(inputs=input_layer, pool_size=[3, 3], strides=1,padding='same')

    conv2b=tf.layers.conv2d(
      inputs=conv1b,
      filters=224,
      kernel_size=[3, 3],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.05),
      activation=tf.nn.relu)

    conv2c=tf.layers.conv2d(
      inputs=conv1c,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.05),
      activation=tf.nn.relu)

    conv2d=tf.layers.conv2d(
      inputs=pool1d,
      filters=64,
      kernel_size=[1, 1],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.05),
      activation=tf.nn.relu)


    return tf.concat([conv2a,conv2b,conv2c,conv2d],3),512



def cnn_model_fn(features, labels, mode):
    # Input Layer
    # Need to reshape our images into a square structure for our CNN:
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 1])
    # Batch normalization has proven to be effective at reducing our training
    # time requirements and accuracy on the validation set.
    batch = tf.layers.batch_normalization(
       inputs = input_layer)

    # Convolutional Layer #1
    # Perform convolution with ReLU activation,
    # and don't normalize because we have back to back convolutions.
    conv1 = tf.layers.conv2d(
      inputs=batch,
      filters=64,
      kernel_size=[7, 7],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.05),
      activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

    normd1 = tf.nn.local_response_normalization(
          input = pool1,
          name="norm1")



    # Convolutional Layer #2
    # Start with batch normalization, perform convolution with ReLU activation,
    # and don't normalize because we have back to back convolutions.
    batch2 = tf.layers.batch_normalization(
       inputs = normd1)
    conv2 = tf.layers.conv2d(
      inputs=batch2,
      filters=64,
      kernel_size=[1, 1],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.05),
      activation=tf.nn.relu)


    # Convolutional Layer #3
    # Start with batch normalization, perform convolution with ReLU activation,
    # and use local response normalization before sending the results into
    # the inception modules.
    batch3 = tf.layers.batch_normalization(
       inputs = conv2)

    conv3 = tf.layers.conv2d(
        inputs=batch3,
        filters=192,
        kernel_size=[3, 3],
        padding="same",
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        bias_initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.05),
        activation=tf.nn.relu)


    normd2 = tf.nn.local_response_normalization(
          input = conv3,
          name="norm1")

    batch4 = tf.layers.batch_normalization(
       inputs = normd2)

    # This sends the input through the first inception module
    gnet3,size=concatlayer1(batch4)

    # The result of the inception module concatenation ends up here for pooling
    pool2 = tf.layers.max_pooling2d(inputs=gnet3, pool_size=[3, 3], strides=2)

    # The second inception module:
    gnet4,size4 = concatlayer2(pool2)

    # The result of the second inception module concatenation ends up here
    # for pooling purposes.
    pool3 = tf.layers.max_pooling2d(inputs=gnet4, pool_size=[5, 5], strides=3)


    # Convolutional Layer #4 (not counting the inception module layers)
    # Start with batch normalization, perform convolution with ReLU activation,
    # and do not use local response normalization.
    batch5 = tf.layers.batch_normalization(
       inputs = pool3)
    conv5 = tf.layers.conv2d(
        inputs=batch5,
        filters=128,
        kernel_size=[1, 1],
        padding="same",
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        bias_initializer=tf.random_uniform_initializer(minval=0.01, maxval=0.05),
        activation=tf.nn.relu)

    # We need to make a flat version of our previous tensors in order to use
    # the fully connected layers that follow.
    pool4_flat = tf.reshape(conv5, [-1, 4 * 4 * 128])

    # This is the large fully connected layer. Nearly 50% of these neurons
    # will be dropped to make our model more robust to variations and
    # potentially reduce overfit.
    dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)

    # This layer performs the droupout operation. We keep .5 of the connections
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.5, training=mode==tf.estimator.ModeKeys.TRAIN)
    dense2 = tf.layers.dense(inputs=dropout, units=1024, activation=tf.nn.relu)

    # This layer performs the droupout operation. We keep .5 of the connections
    dropout2 = tf.layers.dropout(
      inputs=dense2, rate=0.5, training=mode==tf.estimator.ModeKeys.TRAIN)
    # This is the final layer. After this the softmax function will be applied.
    logits = tf.layers.dense(inputs=dropout2, units=82)

    predictions = {
      # This generates predictions by finding the array location with the
      # largest probability.
      "classes": tf.argmax(input=logits, axis=1),
      # Our softmax layer acts on our last fully connected layer.
      # The results are used for preedictions
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculates our loss in training and evaluation mode
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=82)
    loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

    # This will activate only during training mode.
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Our optimizer:
        # We use larger values of epsilon and learning rate during the initial
        # training phases of our bigger architectures.
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0000001, epsilon=0.00000001 )
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        eval_metric_ops = {"TrainAccuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,eval_metric_ops=eval_metric_ops)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)





def plot_confusion_matrix(pred, ytemp,num_classes=40):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
#    cls_true = data.test.cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=ytemp,
                          y_pred=pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()



def main(unused_argv):
#    # Load training and eval data
#    x_in = open('x_train_data.pkl','rb')
#    y_in = open('y_train_data.pkl', 'rb')
#
#    print('loading picked data...')
#    x = pickle.load(x_in) # load from text
#    y = pickle.load(y_in)
#    print('done loading data!')
#
#    y = np.asarray(y, dtype=np.int32)
#
#    # the data set will be preprocessed. Each image is done individually.
#    print('preproccessing data...')
#    #x = binarywithdilation(x, 0.97)
#    #x= binarynormalization(x,0.71)
#    print('donepreprocessing data!')
#    x = np.asarray(x, dtype=np.float32)
#
#    # Split the data up into our training and validation set.
#    train_data, eval_data, train_labels, eval_labels = train_test_split(x, y, test_size=0.25)

    # We build our estimator based on the tensorflow recommendations
    mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/comp551_theGoogleNetded10")

    # We will log the loss rate to keep track of progress as our
    # structure is trained. The labels come from the model method
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=1000)

    # Train the model:
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        # Our training data
        x={"x": train_data},
        # The training labels to match the data
        y=train_labels,
        # Increasing the batch size dramatically increases runtime
        batch_size=125,
        num_epochs=None,
        # Random sampling batches will expose the model to more variations
        # during training.
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        # This is not enough steps. We will run this model several times
        # to make sure the accuracy and loss are stable.
        steps=3000,
        hooks=[logging_hook])


    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)



    #This section will produce the main output file that is
    #ready for submission right away. It gets predictions on
    #the processed version of the test set provided by the
    #Kaggle competition.

    # Test set is loaded and binarization is applied:

    x_test_in = open('x_test_data.pkl','rb')
    #y_in = open('y_train_data.pkl', 'rb')

    print('loading picked data...')
    x_test = pickle.load(x_test_in) # load from text

    print('done loading data!')
    x_test= binarynormalization(x_test,0.71)
    x_test = np.asarray(x_test, dtype=np.float32)
    #y = np.asarray(y, dtype=np.int32)
    x_test_in.close()

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":procdtestset},
        num_epochs=1,
        shuffle=False)
    predictions = list(mnist_classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]
    #Next the predicitons will be printed in the proper order in the
    #exported file.
    output = io.open('google_pred10.csv', 'w', encoding='utf-8')
    count = 1
    output.write(u'Id,Label\n')
    for x in predicted_classes:
        output.write(str(count) + u',' + str(x) + u'\n')
        count += 1
    output.close()


if __name__ == "__main__":
  tf.app.run()
