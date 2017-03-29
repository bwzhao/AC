from __future__ import print_function
#import Generate_TrainingData
import DataSet
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras import optimizers
from keras.utils import plot_model

import matplotlib.pyplot as plt

# Define parameters
learning_rate = 0.005
regularizer_rate = 1e-6
training_epochs = 15
batch_size = 500
# Network Parameters
n_hidden_1 = 100  # 1st layer number of features
n_hidden_2 = 100  # 2nd layer number of features
n_hidden_3 = 100  # 3rd layer number of features
n_hidden_4 = 100  # 4th layer number of features


def main():
    # Read Data
    Training_Ai = np.load("./Data/Training_Ai.npy")
    Training_Gtau = np.load("./Data/Training_Gtau.npy")
    Testing_Ai = np.load("./Data/Testing_Ai.npy")
    Testing_Gtau = np.load("./Data/Testing_Gtau.npy")
    omega_list = np.load("./Data/Omega_List.npy")
    tau_list = np.load("./Data/Tau_List.npy")

    #Real Data from QMC
    real_Gtau = np.load("./Data/MC_Measurement.npy")

    train_ds = DataSet.DataSet(Training_Gtau, Training_Ai)
    test_ds = DataSet.DataSet(Testing_Gtau, Testing_Ai)

    n_input = Training_Gtau.shape[1]  # Num of tau points
    n_output = Training_Ai.shape[1]  # Num of omega points

    for learning_rate in np.arange(0.001, 0.01, 0.001):
        for regularizer_rate in np.arange(1e-7, 1e-6, 1e-7):
            # create model
            model = Sequential()
            model.add(Dense(n_hidden_1, input_dim=n_input,
                            init='normal',
                            activation='relu',
                            kernel_regularizer=regularizers.l2(regularizer_rate),
                            bias_regularizer=regularizers.l2(regularizer_rate)
                            ))
            model.add(Dense(n_hidden_2, input_dim=n_input,
                            init='normal',
                            activation='relu',
                            kernel_regularizer=regularizers.l2(regularizer_rate),
                            bias_regularizer=regularizers.l2(regularizer_rate)
                            ))
            model.add(Dense(n_output,
                            init='normal',
                            activation='linear'
                            ))
            # Compile model
            opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            #opt = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='mean_squared_error', optimizer=opt)

            hist = model.fit(train_ds.images, train_ds.labels,
                      epochs=training_epochs,
                      batch_size=batch_size,
                      shuffle=True,
                      validation_data=(test_ds.images, test_ds.labels),
                             verbose=2)

            val_loss = model.evaluate(test_ds.images, test_ds.labels)
            print(learning_rate, regularizer_rate, val_loss)
            with open('./Data/HP.txt', 'a') as the_file:
                the_file.write(str(learning_rate) + '\t')
                the_file.write(str(regularizer_rate) + '\t')
                the_file.write(str(val_loss) + '\n')
                the_file.close()

            #plot_model(model, to_file='model.png')
            #model.save('./Model/Keras_NN')

if __name__ == '__main__':
    main()