import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd

import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils


SEQ_LEN = 10

def create_time_series():
    freq = (np.random.random_sample()*.5) + .1
    ampl = np.random.random_sample() + 0.5
    x = np.sin(np.arange(0, SEQ_LEN)* freq)*ampl
    return x

#for i in range(0,5):
#    print('Sequence nr ',i, ':', create_time_series())
#    sns.tsplot (create_time_series());

def to_csv(filename,N):
    with open(filename, 'w') as ofp:
        for lineno in range(0,N):
            seq = create_time_series()
            line = ",".join(map(str,seq))
            ofp.write(line + '\n')

to_csv('train.csv', 1000)
to_csv('valid.csv', 50)

DEFAULTS = [[0.0] for x in range(0, SEQ_LEN)]
BATCH_SIZE = 20 #can be altered for optimal training
TIMESERIES_COL = 'rawdata'
N_OUTPUTS = 2 # in each sequnece, 1-8 are featured, 9,10 labeled
N_INPUTS = SEQ_LEN - N_OUTPUTS
LSTM_SIZE = 3


def read_dataset(filename, mode=tflearn.ModeKeys.TRAIN):
    def _input_fn():
        num_epochs = 100 if mode==tflearn.ModeKeys.TRAIN else 1

        input_file_names = tf.train.match_filenames_once(filename)

        filename_queue = tf.train.string_input_producer(
            input_file_names, num_epochs=num_epochs, shuffle=True)
        reader = tf.TextLineReader()
        _, value = reader.read_up_to(filename_queue,num_records=BATCH_SIZE)

        value_column = tf.expand_dims(value, -1)
        #print 'readcsv={}'.format(value_column)

        # all_data is a list of tensors
        all_data = tf.decode_csv(value_column,record_defaults=DEFAULTS)
        inputs = all_data[:len(all_data)-N_OUTPUTS] #first values
        label = all_data[len(all_data)-N_OUTPUTS :] #last values

        #from list of tensors to tensor with one dimension
        inputs = tf.concat(inputs, axis=1)
        label = tf.concat(label, axis=1)
        #print 'inputs={}'.format(inputs)

        return {TIMESERIES_COL: inputs}, label #dict of festures, label
    return _input_fn()

# create the inference model
def simple_rnn(features, targets, mode):
    # 0. Reformat input shape to become a sequence
    x = tf.split(features[TIMESERIES_COL], N_INPUTS, 1)
    # print 'x={}'.format(x)

    # 1. configure the RNN
    lstm_cell = rnn.BasicLSTMCell(LSTM_SIZE, forget_bias=1.0)
    outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # slice to keep only the last cell of the RNN
    outputs = outputs[-1]
    # print 'last outputs={}'.format(outputs)

    # output is result of linear activation of last layer of RNN
    weight = tf.Variable(tf.random_normal([LSTM_SIZE, N_OUTPUTS]))
    bias = tf.Variable(tf.random_normal([N_OUTPUTS]))
    predictions = tf.matmul(outputs, weight) + bias

    # 2. loss function, training/eval ops
    if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
        loss = tf.losses.mean_squared_error(targets, predictions)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.01,
            optimizer="SGD")
        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(targets, predictions)
        }
    else:
        loss = None
        train_op = None
        eval_metric_ops = None

    # 3. Create predictions
    predictions_dict = {"predicted": predictions}

    # 4. return ModelFnOps
    return tflearn.ModelFnOps(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def get_train():
    return read_dataset('train.csv', mode=tflearn.ModeKeys.TRAIN)

def get_valid():
    return read_dataset('valid.csv', mode=tflearn.ModeKeys.EVAL)

def serving_input_fn():
    feature_placeholders = {
        TIMESERIES_COL: tf.placeholder(tf.float32, [None, N_INPUTS])
    }

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    features[TIMESERIES_COL] = tf.squeeze(features[TIMESERIES_COL], axis=[2])

    print
    'serving: features={}'.format(features[TIMESERIES_COL])

    return tflearn.utils.input_fn_utils.InputFnOps(
        features,
        None,
        feature_placeholders
    )

def experiment_fn(output_dir):
    # run experiment
    return tflearn.Experiment(
        tflearn.Estimator(model_fn=simple_rnn, model_dir=output_dir),
        train_input_fn=get_train(),
        eval_input_fn=get_valid(),
        eval_metrics={
            'rmse': tflearn.MetricSpec(
                metric_fn=metrics.streaming_root_mean_squared_error
            )
        }
    )

shutil.rmtree('outputdir', ignore_errors=True) # start fresh each time
learn_runner.run(experiment_fn, 'outputdir')
