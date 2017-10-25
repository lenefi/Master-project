import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as tflayers
import numpy as np
from numpy import newaxis

class basicLSTM:
    training_steps = 1000
    num_outputs = 5
    num_hidden = 1
    num_steps = 5
    seq_length = 5
    loss = None


    weight = tf.Variable(tf.random_normal([num_hidden, num_outputs]))
    bias = tf.Variable(tf.random_normal([num_outputs]))

    input = tf.placeholder(tf.float32, shape=[None, num_steps, seq_length])
    labels = tf.placeholder(tf.float32, [None, num_outputs])
    sess = tf.Session()

    def __init__(self, features, output, train_op,accuracy):
        self.features = features
        self.output = output
        self.train_op = train_op
        self.accuracy = accuracy


    def lstm(self):
        self.lstm = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.)


    def predict(self):
        self.features = tf.unstack(self.input, self.num_steps, 1)
        self.output, _ = rnn.static_rnn(self.lstm, self.features, dtype=tf.float32)
        self.pred = tf.matmul(self.output[-1], self.weight) + self.bias
        return self.pred

    def train(self):
        self.loss = tf.losses.mean_squared_error(self.labels, self.pred)
        self.train_op = tflayers.optimize_loss(
            loss=self.loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.01,
            optimizer="SGD")

        self.sess.run(tf.initialize_all_variables())

        for step in range(1, self.training_steps + 1):
            self.sess.run(self.train_op, feed_dict={self.input: x, self.labels: y})

        print("Optimization Finished!")

    def test(self):
        self.pred = self.sess.run(self.pred, feed_dict={self.input: x_test})
        print("pred:", self.pred)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels, 1)), tf.float32))
        print("Loss:", self.sess.run(self.loss, feed_dict={self.input: x_test, self.labels: y_test}))
        print("Training Accuracy:", self.sess.run(self.accuracy, feed_dict={self.input: x_test, self.labels: y_test}))

seq_length = 5
num_steps = 5
num_outputs = 5
num_labels = 1
num_hidden = 1


x = np.matrix('1 2 3 4 5;2 3 4 5 6;3 4 5 6 7;4 5 6 7 8;5 6 7 8 9')
x = x[newaxis, :, :]


x_test = np.random.randn(2, num_steps, seq_length)
y_test = np.random.rand(1,5)
y = np.matrix('6,7,8,9,10')


model =basicLSTM(features=None, output=None, train_op=None, accuracy=None)
lstm = model.lstm()
pred = model.predict()
model.train()
model.test()




