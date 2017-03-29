from __future__ import division
from __future__ import print_function

from sklearn.kernel_ridge import KernelRidge
import numpy as np
import time
import matplotlib.pyplot as plt

def main():
    # Read Data
    Training_Ai = np.load("./Data/Training_Ai.npy")
    Training_Gtau = np.load("./Data/Training_Gtau.npy")
    Testing_Ai = np.load("./Data/Testing_Ai.npy")
    Testing_Gtau = np.load("./Data/Testing_Gtau.npy")
    omega_list = np.load("./Data/Omega_List.npy")
    tau_list = np.load("./Data/Tau_List.npy")

    # Real Data from QMC
    real_Gtau = np.load("./Data/MC_Measurement.npy")

    t0 = time.time()
    kr = KernelRidge(kernel='rbf', gamma=0.1)
    kr.fit(Training_Gtau, Training_Ai)
    kr_fit = time.time() - t0
    print("KRR complexity and bandwidth selected and model fitted in %.3f s"
          % kr_fit)

    t0 = time.time()
    y_kr = kr.predict(real_Gtau)
    kr_predict = time.time() - t0
    print("KRR prediction for %d inputs in %.3f s"
          % (Testing_Gtau.shape[0], kr_predict))

    plt.plot(omega_list, y_kr[0], 'ro',
             label='KRR (fit: %.3fs, predict: %.3fs)' % (kr_fit, kr_predict))
    #plt.plot(omega_list, Testing_Ai[0], 'go',
             #label='KRR (fit: %.3fs, predict: %.3fs)' % (kr_fit, kr_predict))
    plt.xlabel('data')
    plt.ylabel('target')
    plt.show()

if __name__ == '__main__':
    main()