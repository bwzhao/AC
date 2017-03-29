from __future__ import division
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

def Read_Data():
    # Real Data from QMC
    real_Ai = np.load("./Data/Real_Ai.npy")
    real_omega_list = np.load("./Data/Real_Omega_List.npy")
    real_Gtau = np.load("./Data/MC_Measurement.npy")
    omega_list = np.load("./Data/Omega_List.npy")
    tau_list = np.load("./Data/Tau_List.npy")

    real_Ai = real_Ai * np.float(len(real_omega_list)) / len(omega_list)

    model = load_model("./Model/Keras_NN")
    predicted_label = model.predict(real_Gtau)
    plt.plot(omega_list, predicted_label[0], 'ro')
    plt.plot(real_omega_list, real_Ai, 'go')
    plt.show()

def Test_set():
    # Read Data
    Testing_Ai_List = np.load("./Data/Testing_Ai.npy")
    Testing_Gtau = np.load("./Data/Testing_Gtau.npy")
    omega_list = np.load("./Data/Omega_List.npy")
    tau_list = np.load("./Data/Tau_List.npy")
    real_Gtau = np.load("./Data/MC_Measurement.npy")

    model = load_model("./Model/Keras_NN")

    # predicted_label = model.predict(test_ds.images)
    predicted_label_list = model.predict(Testing_Gtau)
    for predicted_label, Testing_Ai in zip(predicted_label_list, Testing_Ai_List):
        plt.plot(omega_list, predicted_label, 'ro')
        plt.plot(omega_list, Testing_Ai, 'go')
        plt.show()

def Summary():
    Testing_Ai_List = np.load("./Data/Testing_Ai.npy")
    Testing_Gtau = np.load("./Data/Testing_Gtau.npy")
    model = load_model("./Model/Keras_NN")

    loss = model.evaluate(Testing_Gtau, Testing_Ai_List)
    print(loss)

if __name__ == '__main__':
    #Test_set()
    #Read_Data()
    Summary()
