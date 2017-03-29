from __future__ import print_function
import numpy as np
import pandas as pd

def Gtau():
    data = np.array([[
    0.647798077,
    0.421420419,
    0.275199869,
    0.180357892,
    0.11857653,
    0.078175875,
    0.051691782,
    0.034251267,
    0.02274408,
    0.01513446,
    0.010080688,
    0.006734269,
    0.004504519,
    0.003015451,
    0.002010158,
    0.001362601,
    0.000923763,
    0.00061657,
    0.000410753,
    0.00028356,
    0.000201801,
    0.000147075,
    0.000103389
    ]])

    np.save("MC_Measurement", data)

def Ai():
    df_Ai = pd.read_excel("./Data/Ai.xlsx")
    omega_list = df_Ai['omega'][:1500].values
    Ai_list = df_Ai['Ai'][:1500].values

    # normalization
    Ai_list = Ai_list / Ai_list.sum()

    np.save("./Data/Real_Omega_List", omega_list)
    np.save("./Data/Real_Ai", Ai_list)

if __name__ == '__main__':
    Ai()