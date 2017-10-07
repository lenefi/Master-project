
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as tflayers
import numpy as np
from numpy import newaxis

seq_length = 5
num_steps = 5
num_labels = 1
num_hidden = 1
num_outputs = 5
training_steps= 1000
disp_step = 100

x = np.matrix('1 2 3 4 5;2 3 4 5 6;3 4 5 6 7;4 5 6 7 8;5 6 7 8 9')
x = x[newaxis, :, :]

test_data = np.random.randn(2,num_steps,seq_length)
test_data = test_data[newaxis, :, :]


y = np.matrix('6,7,8,9,10')

input = tf.placeholder(tf.float32, shape=[None, num_steps, seq_length])
labels = tf.placeholder(tf.float32, [None, num_outputs])

lstm = rnn.BasicLSTMCell(num_hidden, forget_bias=1.)

features = tf.unstack(input, num_steps,1)
output, state = rnn.static_rnn(lstm, features, dtype=tf.float32)
output = output[-1]

weight = tf.Variable(tf.random_normal([num_hidden, num_outputs]))
bias = tf.Variable(tf.random_normal([num_outputs]))
predictions = tf.matmul(output, weight) + bias

loss = tf.losses.mean_squared_error(labels,predictions)
train_op = tflayers.optimize_loss(
         loss=loss,
         global_step=tf.contrib.framework.get_global_step(),
         learning_rate=0.01,
         optimizer="SGD")

#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels))
correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1, training_steps+1):
        sess.run(train_op, feed_dict={input: x,labels: y})
    print("Optimization Finished!")
    print("Loss:", sess.run(loss, feed_dict={input: x, labels: y}))
    print("Training Accuracy:", sess.run(accuracy, feed_dict={input: x, labels: y}))

