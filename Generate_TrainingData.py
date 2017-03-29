from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import itertools

def Gaussian_Func(mu, sigma2, x):
    return np.exp(- (x - mu)**2 / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)

class Sample_Generator:
    """Generate Training Set
    """

    def __init__(self, omega_list, tau_list, beta):
        """Generate Training Set

            :param beta: the inverse temperature
            :param omega_list: a np.array list contains the value of omega_i, dim L
            :param tau_list: a np.array list contains the value of tau_j, dim M
        """
        self._omega_list = omega_list
        self._tau_list = tau_list
        self._dimL = len(omega_list)
        self._dimM = len(tau_list)
        self._matK = np.empty([self.dimM, self.dimL], dtype=np.float)

        # Initialize MatK
        for index_omega, omega in enumerate(omega_list):
            for index_tau, tau in enumerate(tau_list):
                self.matK[index_tau, index_omega] = np.exp(- tau * omega) + np.exp((tau - beta) * omega) / \
                                                                            (np.pi * (1 + np.exp(- beta * omega)))

    @property
    def matK(self):
        return self._matK

    @property
    def dimL(self):
        return self._dimL

    @property
    def dimM(self):
        return self._dimM

    @property
    def omega_list(self):
        return self._omega_list

    @property
    def tau_list(self):
        return self._tau_list

    def Get_AiGtau(self):
        """
        We artificially generate some data for the training process
        1. artificially generate some narrow peaks

        :return: a tuple of norm_Ai and Gtau as the training data
        """
        # Add the Gaussian Peaks
        num_peaks = 1
        raw_Ai = np.zeros(self._dimL)
        mu_central = 1.0
        mu_range = 0.4
        mu_list = np.random.uniform(mu_central - mu_range, mu_central + mu_range, num_peaks)

        sigma_central = 0.1
        sigma_range = 0.01
        sigma_list = np.random.uniform(sigma_central - sigma_range, sigma_central + sigma_range, num_peaks)

        for mu, sigma in itertools.izip(mu_list, sigma_list):
            raw_Ai = raw_Ai + Gaussian_Func(mu, sigma**2, self.omega_list)

        # Add random noises
        uniform_height = 0.5
        #raw_Ai = raw_Ai + np.random.uniform(0, uniform_height, self._dimL)

        # Decrease as larger omega
        para_alpha = 1.
        raw_Ai = raw_Ai * np.exp(-para_alpha * self.omega_list)

        # zero small omega
        cuttoff = mu_list[0]
        for index, value in enumerate(self.omega_list):
            if value < cuttoff:
                #raw_Ai[index] *= np.exp(-100.0 * (cuttoff - value))
                raw_Ai[index] = 0


        norm_Ai = raw_Ai / raw_Ai.sum()
        Gtau = np.dot(self.matK, norm_Ai)

        return (norm_Ai, Gtau)

if __name__ == '__main__':
    omega_list = np.linspace(0, 3, 500)
    tau_list = np.arange(0.25, 5.8, 0.25)
    beta = 500

    A = Sample_Generator(omega_list, tau_list, beta)
    Num_Training = 500000
    Num_Testing = 100000
    Training_Ai = []
    Training_Gtau = []
    Testing_Ai = []
    Testing_Gtau = []

    for _ in range(Num_Training):
        _Ai, _Gtau = A.Get_AiGtau()
        Training_Ai.append(_Ai)
        Training_Gtau.append(_Gtau)
        if _ % 10000 == 0:
            print("Step# ", _)

    for _ in range(Num_Testing):
        _Ai, _Gtau = A.Get_AiGtau()
        Testing_Ai.append(_Ai)
        Testing_Gtau.append(_Gtau)

    #print(Testing_Gtau[0])

    np.save("./Data/Training_Ai", np.array(Training_Ai))
    np.save("./Data/Training_Gtau", np.array(Training_Gtau))
    np.save("./Data/Testing_Ai", np.array(Testing_Ai))
    np.save("./Data/Testing_Gtau", np.array(Testing_Gtau))
    np.save("./Data/Omega_List", omega_list)
    np.save("./Data/Tau_List", tau_list)