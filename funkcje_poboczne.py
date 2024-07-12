import pandas as pd
import numpy as np
df_init = pd.DataFrame(0.1,columns=['a','b','c'],index=['Exponential',"Weibull","Power",'Inverse Power'])
df_init.iloc[0,:] = [0.01,0.01,0.01]
param_list_all = [1,2]
param_list_all[0] = df_init
df_init.iloc[0,:] = [0.01,0.01,0.01]
param_list_all[1] = df_init

sim_factors_list = [0, 8]
Final_factors_list = [0, 8]


class YearHorizont:
    def index_all(self, df_trian):
        index_he = []
        index_ost = []
        index_firs = []
        m = df_trian.shape[0]
        n = df_trian.shape[1]
        for i in range(n - 1):
            ind_row = [x for x in range(m)]
            ind_row_copy = [x for x in range(m)]
            ind_row_copy_2 = [x for x in range(m)]
            ind_col_last = np.where(df_trian.iloc[:, i + 1].isnull())[0].tolist()
            # print(ind_col_last)
            ind_col_before = np.where(df_trian.iloc[:, i].isnull())[0].tolist()
            index_ost.append(self.delete_element_list(ind_row, ind_col_before).pop())
            index_firs.append(np.min(self.delete_element_list(ind_row_copy, ind_col_last)))
            sum_ind = self.Union(ind_col_last, ind_col_before)
            dev_ind = self.delete_element_list(ind_row_copy_2, sum_ind)
            index_he.append(dev_ind)
        # print(index_firs)
        index_ost.append(ind_col_last[0] - 1)
        # print(index_ost)
        return (index_he, index_ost, index_firs)

    def l_i_j(self,dd, indeksy):
        dd_copy = dd.copy()
        n = dd_copy.shape[1]
        l_triangl = pd.DataFrame(0, columns=dd.columns, index=dd.index,dtype='Float64')
        for i in range(n - 1):
            l_triangl.iloc[indeksy[i], i] = [x/y if y!=0 else 1 for x,y in zip(dd_copy.iloc[indeksy[i], i + 1] ,dd_copy.iloc[indeksy[i], i]) ]
        return (l_triangl)


    def create_binary_df(self,ratio_df):
            binary_df = ratio_df.applymap(lambda x: 1 if pd.notna(x) else np.nan)
            return binary_df

    def Dev(self,data_paid, w, l, ind):
        mm = data_paid.shape[0]
        nn = data_paid.shape[1]
        Dev_j = []
        for j in range(nn - 1):
            mianownik = np.sum([data_paid.iloc[i, j] * w.iloc[i, j] for i in ind[j]])
            licznik = np.sum([data_paid.iloc[i, j] * w.iloc[i, j] * l.iloc[i, j] for i in ind[j]])
            if(mianownik==0):
                Dev_j.append(1)
            else:
             Dev_j.append(licznik / mianownik)
        return (Dev_j)
    def Dev_prem(self,data_paid, w, ind):
        mm = data_paid.shape[0]
        nn = data_paid.shape[1]
        Dev_j = []
        for j in range(nn - 1):
            mianownik = np.sum([data_paid.iloc[i, j] * w.iloc[i, j] for i in ind[j]])
            licznik = np.sum([data_paid.iloc[i, j+1] * w.iloc[i, j] for i in ind[j]])
            if(mianownik==0):
                Dev_j.append(1)
            else:
             Dev_j.append(licznik / mianownik)
        return (Dev_j)

    def kwadrat_dzielenia(self,dane, dev, ind):
        dev =dev+[1]
        dane_copy = dane.copy()
        col = 0
        for ind_col in ind:
            if(len(ind_col)>1):
                dane_copy.iloc[ind_col, col] = [(x - dev[col]) ** 2 for x in dane_copy.iloc[ind_col, col]]
            else:
                dane_copy.iloc[0, col] = [(dane_copy.iloc[ind_col[0], col] - dev[col]) ** 2 ]
            col = col + 1
        return (dane_copy)

    def sigma(self,data_paid, w, l,dev, ind):
        mm,nn = data_paid.shape[0],data_paid.shape[1]
        sigma_j = []
        kw_d = self.kwadrat_dzielenia(l,dev,ind)
        for j in range(nn - 1):
            licznik = np.sum([data_paid.iloc[i, j] * w.iloc[i, j] * kw_d.iloc[i,j] for i in ind[j]])
            mianownik = np.sum(w.iloc[i, j] for i in ind[j])-1
            if mianownik == 0:
                sig_cop = sigma_j
                if (sig_cop[len(sig_cop)-2]!=0):
                    s_min = np.min([sig_cop[len(sig_cop)-1],sig_cop[len(sig_cop)-2],((sig_cop[len(sig_cop)-1])**2)/sig_cop[len(sig_cop)-2]])
                else:
                    s_min = 0
                sigma_j.append(s_min)
            else:
                sigma_j.append(licznik / mianownik)
        return (sigma_j)
    def wspolczynnik_sd(self,data_paid, w,wsp_sigma, ind):
        mm = data_paid.shape[0]
        nn = data_paid.shape[1]
        sd_j = []
        for j in range(nn - 1):
            mianownik = np.sum([data_paid.iloc[i, j] * w.iloc[i, j] for i in ind[j]])
            if mianownik == 0:
                sd_j.append(0)
            else:
                sd_j.append(np.sqrt(wsp_sigma[j] / mianownik))
        return (sd_j)

    def check_value(self,data_vector,ind_factor, d_min, d_max):
        k = 0
        x_k_ind = []
        vector_value = []
        for x in ind_factor:
            if (d_min < data_vector[x-1] < d_max):
                vector_value.append(data_vector[x-1])
                x_k_ind.append(x)
            k=k+1
        return ([vector_value, x_k_ind])

    def fit_curve(self, data_input, sd_input, x_k, dopasowanie, n):
        se_factor = [x ** 2 for x in sd_input[:len(data_input)]]
        w_k_sqr = self.wsp_w_k_sqr(data_input, se_factor, dopasowanie, n)
        if (dopasowanie == 'factor_CL'):
            factor_input = [np.log(f - 1) for f in data_input]
        elif (dopasowanie == 'variance_CL'):
            factor_input = [np.log(sigma) for sigma in data_input]
        elif (dopasowanie == 'factor P_to_I'):
            factor_input = [np.log(1 - r_j) for r_j in data_input]
        elif (dopasowanie == 'variance_P_to_I'):
            factor_input = [np.log(var_j) for var_j in data_input]
        elif (dopasowanie == 'factor_LR'):
            factor_input = [np.log(f) for f in data_input]
        elif (dopasowanie == 'variance_LR'):
            factor_input = [np.log(f) for f in data_input]
        A = np.sum(w_k_sqr)
        A_x = np.sum([x * y for x, y in zip(w_k_sqr, x_k)])
        A_x_x = np.sum([x * (y) ** 2 for x, y in zip(w_k_sqr, x_k)])
        A_y = np.sum([x * y for x, y in zip(w_k_sqr, factor_input)])
        A_x_y = np.sum([x * y * z for x, y, z in zip(w_k_sqr, x_k, factor_input)])
        Delta = A * A_x_x - (A_x) ** 2
        a = (A * A_x_y - A_x * A_y) / Delta
        b = (A_x_x * A_y - A_x * A_x_y) / Delta
        return [a, b]

    def wspolczynnik_reg(self, a, b, k_start, k_stop, dopasowanie):
        if (dopasowanie == 'factor_CL'):
            wartosci_reg = [1 + np.exp(a * k + b) for k in range(k_start, k_stop + 1)]
        elif (dopasowanie == 'variance_CL'):
            wartosci_reg = [np.exp(a * k + b) for k in range(k_start, k_stop + 1)]
        elif (dopasowanie == 'variance_CL'):
            wartosci_reg = [np.exp(a * k + b) for k in range(k_start, k_stop + 1)]
        elif (dopasowanie == 'factor P_to_I'):
            wartosci_reg = [1 - np.exp(a * k + b) for k in range(k_start, k_stop + 1)]
        elif (dopasowanie == 'variance_P_to_I'):
            wartosci_reg = [np.exp(a * k + b) for k in range(k_start, k_stop + 1)]
        elif (dopasowanie == 'factor_LR'):
            wartosci_reg = [np.exp(a * k + b) for k in range(k_start, k_stop + 1)]
        elif (dopasowanie == 'variance_LR'):
            wartosci_reg = [np.exp(a * k + b) for k in range(k_start, k_stop + 1)]
        return (wartosci_reg)

#Połączenie metody multyplikatywnej z metodą LOSS RATIO
    def triangle_forward(self, df_data, f,k_forward_start):
        df_t_copy = df_data.copy()
        mm,nn = df_data.shape[0], df_data.shape[1]
        if (len(f) > mm):
            for k in range(mm + 1, len(f) + 2): df_t_copy[k] = np.nan
            #WRÓĆ!!!!
        for j in range(k_forward_start-1,len(f)):
            max_ind_row = np.max([0, mm - j - 1])
            for i in range(max_ind_row, mm):
                df_t_copy.iloc[i, j + 1] = df_t_copy.iloc[i, j] * f[j]
        return (df_t_copy)

    def reverse_list(self, arr):
        left = 0
        right = len(arr) - 1
        while (left < right):
            # Swap
            temp = arr[left]
            arr[left] = arr[right]
            arr[right] = temp
            left += 1
            right -= 1
        return (arr)


