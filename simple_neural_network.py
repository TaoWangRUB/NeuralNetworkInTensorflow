import tensorflow as tf
from tensorflow.python.data import Dataset
import math

from matplotlib import pyplot as plt

from sklearn import metrics

import numpy as np
import uuid

import pandas as pd

# phrase input data functions
def get_features(california_housing_dataframe):
    """
    Prepares input features from California housing data set.
    
    Args:
        california_housing_dataframe: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
        A pandas DataFrame that contains the features to be used for the model, including
        synthetic features.
    """
    selected_features = california_housing_dataframe[
        ["latitude",
        "longitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"]]
    get_features = selected_features.copy()
    # Create a synthetic feature.
    get_features["rooms_per_person"] = (
        california_housing_dataframe["total_rooms"] /
        california_housing_dataframe["population"])
    return get_features

def get_targets(california_housing_dataframe):
    """
    Prepares target features (i.e., labels) from California housing data set.
    
    Args:
        california_housing_dataframe: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
        A pandas DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["median_house_value"] = (
        california_housing_dataframe["median_house_value"] / 1000.0)
    return output_targets

# initialize weight and bias variables
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# define a single neural network layer
def nn_layer(input_tensor, input_dim, output_dim, layer_name, activation = tf.nn.relu):
    """
    Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        
        # initialize weights matrix W
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        
        # initialize bias vector b
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        
        # y = w * x + b
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
    
    # f_activate(y)
    activations = activation(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    
    return activations

# construct neural network based on the given data structure and layer-node structure
def nn_networks(data, input_dim, output_dim, layer_num, nodes_num, activation):
    
    '''!
    build up layout of neural network
    @param data: input data for neural network
    @return outputLayer: output layer is returned
    '''
    
    # input layer + hiden layers + output layer
    activations = []
    
    # following layers
    for ilayer in range(0, layer_num, 1):
        ilayer_name = "layer%1d" % (layer + 1)
        
        # input layer
        if layer == 0:
            act = nn_layer(data, input_dim, nNodes, ilayer_name, activation)
            activations.append(act)
        # hidden layer
        else:
            act = nn_layer(activations[ilayer-1], nNodes, nNodes, ilayer_name, activation)
            activations.append(act)
        
        # output layer
        output_layer = nn_layer(activations[layer_num-1], nNodes, output_dim, "output_layer", None)
        activations.append(output_layer)
            
    return output_layer

# read data
california_housing_dataframe = pd.read_csv("california_housing_train.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

train_features_data = get_features(california_housing_dataframe.head(12000))
train_features_data.describe()
train_targets_data = get_targets(california_housing_dataframe.head(12000))
train_targets_data.describe()

test_features_data = get_features(california_housing_dataframe.tail(5000))
test_features_data.describe()
test_targets_data = get_targets(california_housing_dataframe.tail(5000))
test_targets_data.describe()

# basic parameters for NN
steps   = 2000
epochs = 20
batch_size  = 100
steps_per_epoch = steps / epochs
learning_rate = 0.001


# Using TF Dataset to split data into batches
data_train = (train_features_data.values, train_targets_data.values)
data_test  = (test_features_data.values, test_targets_data.values)
ds = Dataset.from_tensor_slices(data_train).batch(batch_size).repeat()
iter_ds = ds.make_one_shot_iterator().get_next()

# construct the NN
input_dim   = 9
output_dim  = 1
activation  = tf.nn.relu #tf.nn.sigmoid
x = tf.placeholder(shape=[None, input_dim], dtype=tf.float32)
y = tf.placeholder(shape=[None, output_dim], dtype=tf.float32)
nn = tf.layers.dense(x, 10, activation=activation)
nn = tf.layers.dense(nn, 10, activation=activation)
predictions = tf.layers.dense(nn, output_dim, activation=activation)

'''nn = nn_layer(x, 3, 3, 'layer1', activation = tf.nn.sigmoid)
nn = nn_layer(nn, 3, 5, 'layer2', activation = tf.nn.sigmoid)
nn = nn_layer(nn, 5, 2, 'layer3', activation=tf.nn.sigmoid)
nn = nn_layer(nn, 2, 5, 'layer4', activation=tf.nn.sigmoid)
nn = nn_layer(nn, 5, 3, 'output_layer', activation=tf.nn.sigmoid)'''

# define cost
loss = tf.reduce_mean(tf.squared_difference(predictions, y))
loss_rmse = tf.sqrt(loss)
# define optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
opt_min = optimizer.minimize(loss)
tf.summary.scalar("MSE", loss)
merged_summary_op = tf.summary.merge_all()

# Initialize the variables
init = tf.global_variables_initializer()

# Train the model, but do so inside a loop so that we can periodically assess
# loss metrics.
print("Training model...")
print("RMSE (on training data):")
training_rmse = []
validation_rmse = []

with tf.Session() as sess:
    sess.run(init)
    uniq_id = "./" + uuid.uuid1().__str__()[:6]
    summary_writer = tf.summary.FileWriter(uniq_id, graph=tf.get_default_graph())
    for epoch in range (0, epochs):
        # Train the model, starting from the prior state.
        loss_test, rmse_test = 0., 0.
        for step in range(0, steps_per_epoch):
            #get the batch data
            xx, yy = sess.run(iter_ds)

            opt, loss_val, rmse_val, summary = sess.run([opt_min, loss, loss_rmse, merged_summary_op], feed_dict={x: xx, y: yy})
            loss_test += loss_val
            rmse_test += rmse_val
        # Take a break and compute predictions.
        training_rmse.append(rmse_test / steps_per_epoch)
        print("step: {}, MSE: {}, RMSE: {}".format((epoch+1)*steps_per_epoch, loss_test / steps_per_epoch, rmse_test / steps_per_epoch))
        # dealing with validation data
        loss_test, rmse_test = sess.run([loss, loss_rmse], feed_dict = {x : data_test[0], y: data_test[1]})
        print("Test Loss: {:4f}; RMSE {:4f}".format(loss_test, rmse_test))
        validation_rmse.append(rmse_test)
        summary_writer.add_summary(summary, epoch)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, linestyle='none', marker = 'o', fillstyle = 'none', ms=5, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()