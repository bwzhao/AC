from __future__ import print_function
#import Generate_TrainingData
import DataSet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1
# Network Parameters
n_hidden_1 = 100  # 1st layer number of features
n_hidden_2 = 100  # 2nd layer number of features


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.elu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.elu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    out_layer = tf.nn.elu(out_layer)
    return out_layer

def main():
    # Read Data
    Training_Ai = np.load("Training_Ai.npy")
    Training_Gtau = np.load("Training_Gtau.npy")
    Testing_Ai = np.load("Testing_Ai.npy")
    Testing_Gtau = np.load("Testing_Gtau.npy")
    omega_list = np.load("Omega_List.npy")
    tau_list = np.load("Tau_List.npy")

    real_Gtau = np.load("MC_Measurement.npy")

    train_ds = DataSet.DataSet(Training_Gtau, Training_Ai)
    test_ds = DataSet.DataSet(Testing_Gtau, Testing_Ai)

    n_input = Training_Gtau.shape[1]  # Num of tau points
    n_output = Training_Ai.shape[1]  # Num of omega points
    #print(n_input, n_output)

    # Create the model
    x = tf.placeholder(tf.float32, [None, n_input])
    y_ = tf.placeholder(tf.float32, [None, n_output])

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_output]))
    }
    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    regu_rate = 0.01
    regularizer = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['out'])

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_))
    #cost = tf.reduce_mean(tf.square(pred - y_))
    #cost = tf.nn.l2_loss(pred - y_) / train_ds.num_examples
    #cost = tf.reduce_mean(cost + regu_rate * regularizer)
    #
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(train_ds.num_examples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = train_ds.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y_: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        accuracy = sess.run(cost, feed_dict={x: test_ds.images, y_: test_ds.labels})
        print("Accuracy:", accuracy)

        predicted_label = []
        #for row in sess.run(pred, feed_dict={x: test_ds.images}):
        for row in sess.run(pred, feed_dict={x: real_Gtau}):
            predicted_label.append(row)

        predicted_label = np.array(predicted_label)
        #print(test_ds.labels)

        #plt.plot(omega_list, test_ds.labels[0], 'o')
        plt.plot(omega_list, predicted_label[0], 'o')
        plt.show()

if __name__ == '__main__':
    main()