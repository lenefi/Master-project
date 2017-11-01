import pandas as pd
import csv
import matplotlib.pyplot as plt
from modify_lstm import ModelCreatorTensorFlowRNN
import os


# process data
data = pd.read_excel(io='data/BSE.xlsx')
data['date'] = data['Date'] + pd.to_timedelta(data['Hour'], unit='h')
data = data[['date', 'Load', 'T']]

# build neural network graph
lstm = ModelCreatorTensorFlowRNN(optimizer_name= 'SGD',
                                 optimizer_steps = 15,
                                 learning_rate = 0.5,
                                 project_folder = 'results',
                                 model_name = 'initial_test',
                                 rnn_input_size = 24,
                                 rnn_output_size = 1,
                                 number_lstm_hidden_layers = 3,
                                 label_name = 'Load',
                                 order_index = 'date')

# initialize model training
data_generator = data.iterrows()
index, row = next(data_generator)
data_chunk = row.to_frame().T
lstm.train(data_chunk)

# predict and then train for each hour
i = 0
for index, row in data_generator:
    data_chunk = row.to_frame().T
    if i % 100 == 0:
        #_, accuracy = lstm.validate(data_chunk, save=False)
        #print(accuracy)
        lstm.predict(data_chunk, save=False)
    lstm.train(data_chunk)
    i += 1
print('Successfully trained! Check results')

quit()
loss = pd.read_csv('/home/shomed/l/lenefi/Documents/rnn-lene-msc/results/initial_test_loss/error.csv',
                   names=['Loss'])
loss['index'] = loss.index*100
plt.plot(loss['index'], loss['Loss'])
plt.show()

quit()


