from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import pickle
import scipy.ndimage as img

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    # Input Layer
    # Reshape our images
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 1])

    # Batch normalization has proven to be effective at reducing our training
    # time requirements and accuracy on the validation set.
    batchN1 = tf.layers.batch_normalization(
       inputs = input_layer)
    # Convolutional Layer #1
    # Perform convolution with ReLU activation,
    # and don't normalize because we have back to back convolutions.
    conv1 = tf.layers.conv2d(
      inputs=batchN1,
      filters=64,
      kernel_size=[9, 9],
      padding="same",
      strides = 2,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
      bias_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.05),
      activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)
    normd1 = tf.nn.local_response_normalization(
          input = pool1,
          name="norm1")


    # Convolutional Layer #2
    # Start with batch normalization, perform convolution with ReLU activation,
    # and then pool the results before normalizing them
    batchN2 = tf.layers.batch_normalization(
       inputs = normd1)
    conv2 = tf.layers.conv2d(
      inputs=batchN2,
      filters=256,
      kernel_size=[5, 5],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
      bias_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.05),
      activation=tf.nn.relu)


    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=1)
    normd2 = tf.nn.local_response_normalization(
          input = pool2,
          name="normd2")


    # Convolutional Layer #3
    # Start with batch normalization, perform convolution with ReLU activation,
    # and then pool the results before normalizing them
    batchN3 = tf.layers.batch_normalization(
       inputs = normd2)
    conv3 = tf.layers.conv2d(
      inputs=batchN3,
      filters=384,
      kernel_size=[3, 3],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
      bias_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.05),
      activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3, 3], strides=1)
    normd3 = tf.nn.local_response_normalization(
          input = pool3,
          name="normd3")


    # Convolutional Layer #4
    # Start with batch normalization, perform convolution with ReLU activation,
    # and then pool the results before normalizing them
    batchN4 = tf.layers.batch_normalization(
       inputs = normd3)
    conv4 = tf.layers.conv2d(
      inputs=batchN4,
      filters=384,
      kernel_size=[3, 3],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
      bias_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.05),
      activation=tf.nn.relu)

    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[3, 3], strides=2)
    normd4 = tf.nn.local_response_normalization(
          input = pool4,
          name="normd4")


    # Convolutional Layer #5 (not counting the inception module layers)
    # Start with batch normalization, perform convolution with ReLU activation,
    # and do not use local response normalization. Only pool the results
    batchN5 = tf.layers.batch_normalization(
       inputs = normd4)
    conv5 = tf.layers.conv2d(
      inputs=batchN5,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
      bias_initializer=tf.random_uniform_initializer(minval=0.0, maxval=0.05),
      activation=tf.nn.relu)

    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)

    #Reshape the convolution results
    pool2_flat = tf.reshape(pool5, [-1, 8 * 8 * 16])

    # This is a large fully connected layer. Nearly 50% of these neurons
    # will be dropped to make our model more robust to variations and
    # potentially reduce overfit
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.5, training=mode==tf.estimator.ModeKeys.TRAIN)

    # This is another large fully connected layer. Nearly 50% of these neurons
    # will be dropped to make our model more robust to variations and
    # potentially reduce overfit.
    dense2 = tf.layers.dense(inputs=dropout, units=512, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode==tf.estimator.ModeKeys.TRAIN)

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

    # This will activate only during training mode.
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=82)
    loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)
    # Our optimizer:
    # We use larger values of epsilon and learning rate during the initial
    # training phases of our bigger architectures.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=0.00000001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # This will give us an accuracy result on our validation set
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    x_in = open('x_train_data.pkl','rb')
    y_in = open('y_train_data.pkl', 'rb')

    print('loading picked data...')
    x = pickle.load(x_in) # load from text
    y = pickle.load(y_in)
    print('done loading data!')

    y = np.asarray(y, dtype=np.int32)

    # the data set will be preprocessed. Each image is done individually.
    print('preproccessing data...')
    #x = binarywithdilation(x, 0.97)
    #x= binarynormalization(x,0.71)
    print('donepreprocessing data!')
    x = np.asarray(x, dtype=np.float32)

    # Split the data up into our training and validation set.
    train_data, eval_data, train_labels, eval_labels = train_test_split(x, y, test_size=0.25)

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
        batch_size=100,
        num_epochs=None,
        # Random sampling batches will expose the model to more variations
        # during training.
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        # This is not enough steps. We will run this model several times
        # to make sure the accuracy and loss are stable.
        steps=1000,
        hooks=[logging_hook])


    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    # Load training and eval data
    x_test_in = open('x_test.pkl','rb')
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
    output = io.open('google_pred9.csv', 'w', encoding='utf-8')
    count = 1
    output.write(u'Id,Label\n')
    for x in predicted_classes:
        output.write(str(count) + u',' + str(x) + u'\n')
        count += 1
    output.close()


if __name__ == "__main__":
  tf.app.run()
