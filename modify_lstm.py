import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import os


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_predictions(df, folder):
    file_path = os.path.join(folder, 'prediction.csv')
    with open(file_path, 'a') as f:
        df.to_csv(f, header=False, index=False)

def write_validation(df, folder):
    file_path = os.path.join(folder, 'validation.csv')
    with open(file_path, 'a') as f:
        df.to_csv(f, header=False, index=False)


def convert_pandas_column_to_sequence(data_frame, column_name, lag, output_length, number_previous_values=0):
    """
    Convert a numeric column of a pandas data frame to a tensorflow compatible senquece of
    labels and features.
    :param data_frame: pandas data frame
    :param column_name: name of the numeric column to be converted into a sequence
    :param lag: number of previous values used as features to predict the current value
    :param number_previous_values: add a number of zeros in the beginning of the numeric column
    :return: dictionary containing labels and features as numpy arrays
    """
    time_series = [0] * number_previous_values + data_frame[column_name].tolist()
    assert len(time_series) == lag + output_length
    features = time_series[0:lag]
    labels = time_series[lag:(lag + output_length)]
    return dict(labels=np.array([labels]), features=np.array([features]))


def simple_rnn(input_size, features, number_lstm_hidden_layers, number_outputs, model_name):
    """
    LSTM model
    :param input_size:
    :param features:
    :param number_lstm_hidden_layers:
    :param number_outputs: The number of points in the end of the sequence to be treated as labels
    :return:
    """
    feature_sequence = tf.split(features, input_size, 1)

    with tf.variable_scope(model_name):
        # 1. configure the RNN
        lstm_cell = rnn.BasicLSTMCell(number_lstm_hidden_layers, forget_bias=1.0)
        outputs, _ = rnn.static_rnn(lstm_cell, feature_sequence, dtype=tf.float32)

        # slice to keep only the last cell of the RNN
        outputs = outputs[-1]

        # output is result of linear activation of last layer of RNN
        weight = tf.Variable(tf.random_normal([number_lstm_hidden_layers, number_outputs]))
        bias = tf.Variable(tf.random_normal([number_outputs]))

    predictions = tf.matmul(outputs, weight) + bias

    return predictions


class ModelCreatorTensorFlowRNN:
    def __init__(self, optimizer_name, optimizer_steps, learning_rate, project_folder, model_name,
                 rnn_input_size, rnn_output_size, number_lstm_hidden_layers, label_name, order_index):

        # model directory
        self.project_folder = project_folder
        self.model_name = model_name
        self.model_folder = os.path.join(self.project_folder, self.model_name)
        create_folder(self.model_folder)
        create_folder(self.model_folder + '_validation')

        # optimizer parameters
        self.optimizer_steps = optimizer_steps
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name

        # model parameters
        self.rnn_input_size = rnn_input_size
        self.rnn_output_size = rnn_output_size
        self.number_lstm_hidden_layers = number_lstm_hidden_layers

        # model placeholders
        self.features = tf.placeholder(tf.float32, [None, rnn_input_size], name='rnn_input')
        self.label = tf.placeholder(tf.float32, [None, rnn_output_size], name='rnn_label')

        # model definition
        self.model = simple_rnn(input_size=self.rnn_input_size,
                                features=self.features,
                                number_lstm_hidden_layers=self.number_lstm_hidden_layers,
                                number_outputs=self.rnn_output_size,
                                model_name=self.model_name)

        # loss function
        self.loss = tf.losses.mean_squared_error(self.label, self.model)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.label, 1),
                                                        tf.argmax(self.model, 1)), tf.float32))

        # tensorflow session and operations
        self.sess = tf.Session()
        self.train_op = self.train_operation()
        self.sess.run(tf.global_variables_initializer())

        # labels, features and other columns names
        self.order_index = order_index
        self.label_name = label_name

        # Keep track of the last n rows of the data frame
        self.keep_last_rows = None
        self.last_train_date = None

    def train_operation(self):
        train_op = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=self.learning_rate,
            optimizer=self.optimizer_name)
        return train_op

    def pre_process_data(self, data):
        """
        Concatenate
        :param data: pandas data frame containing rows used for training
        :return: dictionary with labels and features used by the tensorflow model
        """

        # concatenate last data points
        number_missing_previous_values = 0
        if self.keep_last_rows is None:
            _data = data
            number_missing_previous_values = self.rnn_input_size
        else:
            _data = pd.concat([self.keep_last_rows, data], axis=0)
            if self.keep_last_rows.shape[0] < self.rnn_input_size:
                number_missing_previous_values = self.rnn_input_size - self.keep_last_rows.shape[0]

        return _data, number_missing_previous_values

    def train(self, data):

        # pre-process data
        _data, number_missing_previous_values = self.pre_process_data(data)

        # transform the data into feed_dict
        _dict = convert_pandas_column_to_sequence(data_frame=_data,
                                                  column_name=self.label_name,
                                                  lag=self.rnn_input_size,
                                                  output_length=self.rnn_output_size,
                                                  number_previous_values=number_missing_previous_values)

        # train
        feed_dict = {self.features: _dict['features'],
                     self.label: _dict['labels']}

        for _ in range(self.optimizer_steps):
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)


        # keep track of previous values
        self.keep_last_rows = _data.tail(self.rnn_input_size)

    def predict(self, data, save):

        # pre-process data
        _data, number_missing_previous_values = self.pre_process_data(data)

        # transform the data into feed_dict
        _dict = convert_pandas_column_to_sequence(data_frame=_data,
                                                  column_name=self.label_name,
                                                  lag=self.rnn_input_size,
                                                  output_length=self.rnn_output_size,
                                                  number_previous_values=number_missing_previous_values)

        # feed_dict for placeholders
        feed_dict = {self.features: _dict['features']}
        predictions = self.sess.run([self.model], feed_dict)

        # convert to list
        predictions = predictions[0].reshape([predictions[0].shape[1]]).tolist()

        if save:
            last_obs_date = self.keep_last_rows.tail(1)[self.order_index].values[0]
            prediction_dates = data[self.order_index].values.tolist()
            prediction_values = predictions

            prediction_df = pd.DataFrame({'last_obs_date': last_obs_date,
                                          'prediction_dates': prediction_dates,
                                          'prediction_values': prediction_values})
            write_predictions(df=prediction_df, folder=self.model_folder)

        return predictions

    def validate(self, data, save):

        # pre-process data
        _data, number_missing_previous_values = self.pre_process_data(data)

        # transform the data into feed_dict
        _dict = convert_pandas_column_to_sequence(data_frame=_data,
                                                  column_name=self.label_name,
                                                  lag=self.rnn_input_size,
                                                  output_length=self.rnn_output_size,
                                                  number_previous_values=number_missing_previous_values)

        # feed_dict for placeholders
        feed_dict = {self.features: _dict['features'],
                     self.label: _dict['labels']}

        loss = self.sess.run([self.loss], feed_dict)
        accuracy = self.sess.run([self.accuracy], feed_dict)

        if save:
            validation_df = pd.DataFrame({'loss': loss, 'accuracy': accuracy})
            write_validation(df = validation_df, folder= self.model_folder)

        return loss, accuracy




