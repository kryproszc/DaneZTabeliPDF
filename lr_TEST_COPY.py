import numpy as np
import pandas as pd
import math
from numpy.random import PCG64, Generator
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

def random_stochastic_parameters_Loss_ratio(seed, sigma_j, loss_ratio, sd, dimension):
    rng = Generator(PCG64(seed))  # Inicjalizacja generatora losowego
    stochastic_sigma = pd.DataFrame(data=0, columns=np.arange(0, dimension[3], 1), index=np.arange(0, dimension[2], 1))
    mu_j = pd.DataFrame(data=0, columns=np.arange(0, dimension[3], 1), index=np.arange(0, dimension[2], 1))

    for j in range(0, dimension[3]):
        st_swobody = np.max([1, dimension[0] - j])
        chi_list = rng.chisquare(st_swobody, size=dimension[2])
        stochastic_sigma.iloc[:, j] = [(math.ceil(chi) * sigma_j[j]) / st_swobody for chi in chi_list]
        mu_j.iloc[:, j] = rng.normal(loss_ratio[j], sd[j], size=dimension[2])

    return [mu_j, stochastic_sigma]


def stochastic_triangle_forward_Loss_ratio_szybki(seed, data_paid, sigma_j, dev, sd, sim, eksponsure, ult_cons):
    rng = Generator(PCG64(seed))  # Inicjalizacja generatora losowego
    mm, nn = data_paid.shape[0], data_paid.shape[1]
    dimension = [mm, nn, sim, len(dev)]
    Total_BE = []
    mu, sigma = random_stochastic_parameters_Loss_ratio(seed, sigma_j, dev, sd, dimension)
    data_paid_copy_reRe = data_paid.copy()
    data_paid_copy_reRe[mm + 1] = np.nan
    data_paid_copy = data_paid.copy()
    if len(dev) > mm:
        for k in range(mm + 1, len(dev) + 1):
            data_paid_copy[k] = np.nan
    for row in range(0, sim):
        m_i, sigma_i = mu.iloc[row, :], sigma.iloc[row, :]
        for j in range(len(m_i) - 1):
            max_ind_row = np.max([0, mm - j - 1])
            for i in range(max_ind_row, mm):
                VAR_i_j = (sigma_i[j + 1]) / eksponsure[i]
                lmean_i_j = np.log((m_i[j+ 1] ** 2) / np.sqrt((m_i[j+ 1] ** 2 + VAR_i_j)))
                lstdev_i_j = np.log(1 + (VAR_i_j / (m_i[j+ 1] ** 2)))
                stochastic_LR_i_j = rng.lognormal(lmean_i_j, lstdev_i_j, size=1)
                data_paid_copy.iloc[i, j + 1] = data_paid_copy.iloc[i, j] + eksponsure[i] * stochastic_LR_i_j[0]
        Ultimate = data_paid_copy.iloc[:, data_paid_copy.shape[0] - 1].tolist()
        Ultimate = [x  for x in Ultimate if x<10000000]
        BE = np.sum(Ultimate)-ult_cons
        Total_BE.append(BE)
    return Total_BE


# Dane
eksposure = [20161720.38467477, 26834803.195624165, 27479710.672134166, 28287255.65513652,
             31830000.810523850, 37500122.398117306, 40934722.44187734, 57373121.65739994,
             43694242.72670734, 5818947.98738208, 6917346.91158827, 7538307.27765097,
             82596273.8555047, 93708508.42742883, 102001996.32229000, 12780600.35160262,
             14917882.87290737, 150803789.98920902, 143447375.70219726, 149142181.85000002]

# Odczyt danych z pliku Excel
xl = pd.ExcelFile('Dane_rzeczywiste/ACCID_OTH_21_param.xlsx')
wspolczynniki = xl.parse('wsp_1')
LR = wspolczynniki.iloc[0, 1:].tolist()
sigma_j = wspolczynniki.iloc[1, 1:].tolist()
sd = wspolczynniki.iloc[2, 1:].tolist()

# Przekształcenie danych
sd = [x for x in sd]

# Odczyt danych z pliku CSV
reserv_data = pd.read_csv('reserv_data_ACCID_OTH.csv', sep=',', decimal='.')
reserv_data = reserv_data.iloc[:,1:]
# Uruchomienie funkcji forward_loss_ratio
xxx = stochastic_triangle_forward_Loss_ratio_szybki(202206011, reserv_data, sigma_j, LR, sd, 10000,
                                                    eksposure,353043801.424839)
print(xxx)
# Wydruk wyników
print(np.quantile(xxx, q=0.995))
print(np.quantile(xxx, q=0.995) - np.mean(xxx))


def triangle_forward_loss_ratio_kopia(df_data, LR_j, eksponsure, k_forward):
    # dobra funkcja
    print(eksponsure)
    mm, nn = df_data.shape[0], df_data.shape[1]
    df_t_copy = df_data.copy()
    if (len(LR_j) > mm):
        for k in range(mm + 1, len(LR_j) + 1): df_t_copy[k] = np.nan
    for j in range(k_forward):
        max_ind_row = np.max([0, mm - j-1])
        for i in range(max_ind_row, mm):
            df_t_copy.iloc[i, j + 1] = df_t_copy.iloc[i, j] + eksponsure[i] * LR_j[j+1]
    Ultimate = df_t_copy.iloc[:, df_t_copy.shape[0] - 1].to_list()
    print(df_t_copy.to_string())
    print(Ultimate)
    return (np.sum(Ultimate))-334819879.1251721

from metody_jednoroczne_test import YearHorizont
yh = YearHorizont()
print(reserv_data.to_string())
print(yh.trian_diag(reserv_data))
print(np.sum(yh.trian_diag(reserv_data)))

ddd = triangle_forward_loss_ratio_kopia(reserv_data,LR,eksposure,len(LR)-1)
print(ddd)