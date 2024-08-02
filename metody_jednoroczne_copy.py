import pandas as pd
import numpy as np
from operator import itemgetter
#from scipy.stats import chi2, lognorm, norm
#from scipy.stats import chi2, lognorm, norm
from numpy.random import lognormal,normal,chisquare,seed
from lmfit import minimize, Parameters, Parameter, report_fit


class YearHorizont:

    def Union(self, lst1, lst2):
        final_list = list(set(lst1) | set(lst2))
        return final_list

    def change_value_less_diagonal(self,data,value_change):
        for j in range(1,data.shape[1]):
            for i in range(data.shape[0]-1,data.shape[0]-j,-1):
                data.iloc[i, j] = value_change
        return data

    def delete_element_list(self, list_1, list_2):
        for item in list_2:
            if item in list_1:
                list_1.remove(item)
        return (list_1)

    def add_el_list(self, a, b):
        pp = []
        for i in range(a, b + 1, 1):
            pp.append(i)
        return (pp)

    def index_all(self, df_trian):
        index_he = []
        index_ost = []
        index_firs = []
        m = df_trian.shape[0]
        n = df_trian.shape[1]
        for i in range(n - 1):
            ind_row = [x for x in range(m)]
            ind_col_last = np.where(df_trian.iloc[:, i + 1].isnull())[0].tolist()
            ind_col_before = np.where(df_trian.iloc[:, i].isnull())[0].tolist()
            index_ost.append(self.delete_element_list(ind_row, ind_col_before).pop())
            index_firs.append(min(self.delete_element_list(ind_row, ind_col_last)))
            sum_ind = self.Union(ind_col_last, ind_col_before)
            dev_ind = self.delete_element_list(ind_row, sum_ind)
            index_he.append(dev_ind)
        index_ost.append(ind_col_last[0] - 1)
        return (index_he, index_ost, index_firs)

    def index_all(self, df_trian):
        index_he = []
        index_ost = []
        index_firs = []
        m = df_trian.shape[0]
        n = df_trian.shape[1]
        for i in range(n-1):
            ind_row = [x for x in range(m)]
            ind_row_copy = [x for x in range(m)]
            ind_row_copy_2 = [x for x in range(m)]
            ind_col_last = np.where(df_trian.iloc[:, i+1].isnull())[0].tolist()
           # print(ind_col_last)
            ind_col_before = np.where(df_trian.iloc[:, i].isnull())[0].tolist()
            index_ost.append(self.delete_element_list(ind_row, ind_col_before).pop())
            index_firs.append(np.min(self.delete_element_list(ind_row_copy, ind_col_last)))
            sum_ind = self.Union(ind_col_last, ind_col_before)
            dev_ind = self.delete_element_list(ind_row_copy_2, sum_ind)
            index_he.append(dev_ind)
        #print(index_firs)
        index_ost.append(ind_col_last[0] - 1)
       # print(index_ost)
        return (index_he, index_ost, index_firs)

    def incremental_triangle(self, df_triangle):
        indeksy_f, ind_d_f, ind_g_f = self.index_all(df_triangle)
        n = df_triangle.shape[1]
        df_trian_copy = df_triangle.copy()
        for i in range(n - 1):
            b = df_triangle.iloc[indeksy_f[i], i + 1] - df_triangle.iloc[indeksy_f[i], i]
            df_trian_copy.iloc[indeksy_f[i], i + 1] = b
        for kk in range(n-1):
            if(ind_g_f[kk]>0 and np.isnan(df_trian_copy.iloc[ind_g_f[kk]-1,kk])):
                df_trian_copy.iloc[ind_g_f[kk + 1], kk + 1] = np.nan
        return (df_trian_copy)

    def calculate_inflation_adjustment(self,past_inflation, future_inflation):
        inf_adj = [1]
        k = 0
        for i in range(len(past_inflation) - 1, 0, -1):
            inf_adj_factor = ((1 + past_inflation[i]) / (1 + future_inflation)) * inf_adj[k]
            inf_adj.append(inf_adj_factor)
            k = k + 1
        return (inf_adj[::-1])

    def incrementa_with_inflation(self,incremental_triangle, infl_adj):
        increme_trian = incremental_triangle.copy()
        mm = increme_trian.shape[0] - 1
        for col in range(incremental_triangle.shape[1]):
            increme_trian.iloc[0:(mm - col), col] = [x * y for x, y in
                                                     zip(incremental_triangle.iloc[0:(mm - col), col], infl_adj[col:])]
        return (increme_trian)
    ###

    def data_for_lines_app(self,df):
        LoBs = np.unique(df['LoB'])
        list_triangle = []
        for LoB in LoBs:
            lies_bis = df.loc[df.iloc[:,0] == LoB]
            AY = np.unique(lies_bis['AY'])
            DY = (lies_bis.loc[df['AY'] == AY[0]].sort_values(by=['DY']))['DY']
            triangle_excel = pd.DataFrame(0, columns=DY,
                                          index=AY)
            for i in range(len(AY)):
                df2 = ((lies_bis.loc[df['AY'] == AY[i]].sort_values(by=['DY']))['Amount']).to_list()
                triangle_excel.iloc[i, 0:((len(df2)))] = df2
            list_triangle.append(triangle_excel)
        return ([LoBs, list_triangle])

    def show_triangle_for_libes_app(self,df_triangles, lob):
        Lobs, triangles = self.data_for_lines_app(df_triangles)
        ind, = np.where(Lobs == lob)
        return (triangles[ind[0]])

    def df_with_incremental(self,df_traingle):
        dr_traingle_cop = pd.DataFrame(0,columns=np.arange(1,df_traingle.shape[1]+1),index= df_traingle.index)
        dr_traingle_cop.iloc[:,0] = df_traingle.iloc[:,0].to_list()
        mm = dr_traingle_cop.shape[0]
        for col in range(1,dr_traingle_cop.shape[1]):
            dr_traingle_cop.iloc[0:(mm-col),col] =[x+y for x,y in zip(dr_traingle_cop.iloc[0:(mm-col),col-1].tolist(),df_traingle.iloc[0:(mm-col),col].tolist())]
        return (dr_traingle_cop)
    ###

    def l_i_j(self,dd, indeksy):
        dd_copy = dd.copy()
        n = dd_copy.shape[1]
        l_triangl = pd.DataFrame(0, columns=dd.columns, index=dd.index)
        for i in range(n - 1):
            l_triangl.iloc[indeksy[i], i] = [x/y if y!=0 else 1 for x,y in zip(dd_copy.iloc[indeksy[i], i + 1] ,dd_copy.iloc[indeksy[i], i]) ]
        return (l_triangl)

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

    def trian_diag(self, df_triangle):
        n = df_triangle.shape[1]
        _, ind_ost, _ = self.index_all(df_triangle)
        el = [df_triangle.iloc[i, j] for i, j in zip(ind_ost, range(n))]
        return (el)

    def iloczn_wstepujacy(self, lista):
        lista_new = []
        lista_new.append(lista[len(lista) - 1])
        for i in range(len(lista) - 1):
            lista_new.append(lista_new[i] * lista[len(lista) - 2 - i])
        return (lista_new)

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

    #forwaed

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

    def diag_1_triangle(self,triangle, nn,mm):
        triangle_diag_1 = []
        for k,l in zip(range(nn-1,-1,-1), range(0,mm)):
            triangle_diag_1.append(triangle.iloc[l,k])
        return triangle_diag_1

    def random_stochastic_parameters(self,sigma_j,dev,sd,dimension):
        stochastic_sigma = pd.DataFrame(0, columns=np.arange(0, dimension[3], 1), index=np.arange(0, dimension[2], 1))
        mu_j = pd.DataFrame(0, columns=np.arange(0, dimension[3], 1), index=np.arange(0, dimension[2], 1))
        for j in range(0,dimension[3]):
            st_swobody = np.max([1, dimension[0] - j - 1])
            stochastic_sigma.iloc[:,j] = (chisquare(st_swobody, size=dimension[2]) * sigma_j[j]) / st_swobody
            mu_j.iloc[:,j] = normal(dev[j],sd[j],size = dimension[2])
        return([mu_j,stochastic_sigma])

    def stochastic_triangle_forward(self,data_paid,sigma_j,dev,sd,sim):
        BE_ForwardProjection_data = pd.DataFrame(0, columns=np.arange(0, sim, 1), index=np.arange(0, data_paid.shape[0], 1))
        seed(10)
        mm,nn = data_paid.shape[0],data_paid.shape[1]
        dimension = [mm,nn,sim,len(dev)]
        Total_BE = []
        diag_1 = pd.DataFrame(0, columns=np.arange(0, sim, 1), index=np.arange(0, dimension[0], 1))
        mu,sigma = self.random_stochastic_parameters(sigma_j,dev,sd,dimension)
        diag = self.reverse_list(self.trian_diag(data_paid))
        data_paid_copy = data_paid.copy()
        if(len(dev)>mm):
            for k in range(mm+1,len(dev)+2):data_paid_copy[k] = np.nan
        for row in range(0,sim):
            m_i, sigma_i = mu.iloc[row,:],sigma.iloc[row,:]
            for j in range(len(m_i)):
                max_ind_row = np.max([0, mm - j - 1])
                for i in range(max_ind_row, mm):
                    VAR_i_j = sigma_i[j]/data_paid_copy.iloc[i,j]
                    lmean_i_j = np.log((m_i[j])**2 / (np.sqrt((m_i[j])**2 + VAR_i_j)))
                    lstdev_i_j = np.log((VAR_i_j) / ((m_i[j]) ** 2) + 1)
                    CL_i_j = lognormal(lmean_i_j,lstdev_i_j,1)
                    data_paid_copy.iloc[i,j+1] = round(data_paid_copy.iloc[i,j]*CL_i_j[0],3)
         #   print('data_paid_copy')
         #   print(data_paid_copy.to_string())
            Ultimate = data_paid_copy.iloc[:, data_paid_copy.columns[-1] - 1].to_list()
            BE = [x - y for x, y in zip(Ultimate, diag)]
            diag_1.iloc[:,row] = self.diag_1_triangle(data_paid_copy,nn,mm)
            Total_BE.append(sum(BE))
            BE_ForwardProjection_data.iloc[:,row] = BE
        return ([Total_BE,diag_1,BE_ForwardProjection_data])



    def statistic_C1(self,sigma_proj, Dev_pr,C_n):
        Dev_j_proj_kwadrat = [x ** 2 for x in Dev_pr[:-1]]
        mianownik = np.prod(Dev_j_proj_kwadrat[:-1])
        czesc_iloczynowa = []
        for k in range(len(Dev_pr)-1):
           licznik =  np.prod(Dev_pr[:k+1])*sigma_proj[k+1]*np.prod(Dev_j_proj_kwadrat[k+2:])
           czesc_iloczynowa.append(licznik/mianownik)
        E_C_1 = np.mean(C_n)/np.prod(Dev_pr)
        kk = E_C_1*np.sum(czesc_iloczynowa)
        V_C_1 = np.var(C_n)-kk
        return [E_C_1,V_C_1]

    def proj_C_1(self,E_C_1,Var_C_1,distribution,sim):
        if(distribution =='lognormal'):
            lmean = np.log((E_C_1**2)/np.sqrt((E_C_1**2+Var_C_1)))
            lstdev = np.sqrt(np.log(Var_C_1/(E_C_1)**2+1))
            seed(10)
            sim_dist = np.random.lognormal(lmean,lstdev,sim)
        if (distribution == 'gamma'):
            alpha = (E_C_1**2)/Var_C_1
            beta = Var_C_1/E_C_1
            seed(10)
            sim_dist = np.random.gamma(alpha,beta,sim)
        return(sim_dist)

    def add_nex_diagonal(self,trian_sim,d_sim):

        triangle_copy = trian_sim.copy()
        mm,nn = triangle_copy.shape[0],triangle_copy.shape[1]
        for k in range(nn + 1, nn + 2): triangle_copy[k] = np.nan
        triangle_copy.loc[len(triangle_copy)+1] = np.nan
        diag_rev = self.reverse_list(d_sim)
        #print(diag_rev)
        for i,j in zip(range(len(triangle_copy)-1,-1,-1),range(nn+1)):
            triangle_copy.iloc[i,j] = diag_rev[j]

        return(triangle_copy)

    def triangle_ReReserving(self,re_data,diag_sim,weights,diag_standard,number_simulation):
        BE_ReReserving_data = pd.DataFrame(0, columns=np.arange(0, number_simulation, 1), index=np.arange(0, re_data.shape[0], 1))
        MY_list = []
        for nr_sim_diag in range(number_simulation):
            new_triangle = self.add_nex_diagonal(re_data, diag_sim.iloc[:, nr_sim_diag].to_list())
            ind_all, m_i, m_first = self.index_all(new_triangle)
            l = self.l_i_j(new_triangle, ind_all)
            Dev_j_sim = self.Dev(new_triangle, weights, l, ind_all)
            for jjj in range(45): Dev_j_sim.append(1.00000006)
            proj_triangle_sim = self.triangle_forward(new_triangle, Dev_j_sim)
            Ultimate_ReReserving, MY = proj_triangle_sim.iloc[:-1, -1], proj_triangle_sim.iloc[-1, -1]
            BE_ReReserving_data.iloc[:,nr_sim_diag] = [x - y for x, y in zip(Ultimate_ReReserving, diag_standard)]
            MY_list.append(MY)
        return [MY_list,BE_ReReserving_data]

    def wspolczynik_shock(self,BE_PRR,BE_ReR,Scaling_factor):
        shock = []
        shock_scaled = []
        for i in range(BE_ReR.shape[1]):
            shock_all = [x-y for x,y in zip(BE_ReR.iloc[:,i].to_list(),BE_PRR)]
            shock.append(np.sum(shock_all))
            shock_scaled.append(np.sum(shock_all)*Scaling_factor)
        return [shock,shock_scaled]

    def data_for_lines(self,df):
        LoBs = np.unique(df['LoB'])
        list_triangle = []
        for LoB in LoBs:
            lies_bis = df.loc[df.iloc[:,0] == LoB]
            AY = np.unique(lies_bis['AY'])
            DY = (lies_bis.loc[df['AY'] == AY[0]].sort_values(by=['DY']))['DY']
            triangle_excel = pd.DataFrame(0, columns=DY,
                                          index=AY)
            for i in range(len(AY)):
                df2 = ((lies_bis.loc[df['AY'] == AY[i]].sort_values(by=['DY']))['Amount']).to_list()
                triangle_excel.iloc[i, 0:((len(df2)))] = df2
            list_triangle.append(triangle_excel)
        return ([LoBs, list_triangle])

    def show_triangle_for_libes(self,df_triangles, lob):
        Lobs, triangles = self.data_for_lines(df_triangles)
        ind, = np.where(Lobs == lob)
        return (triangles[ind[0]])

    def show_exposure_for_lib(self,df_triangles, lob):
        df_triangles=df_triangles.set_index(['LoB'])
        exposure = df_triangles.loc[lob]
        return (exposure)

    def simulated_observation(self,parameters, seed, simulation_n):
        np.random.seed(seed)
        if (parameters[0] == 'Poisson'):
            sim_d = np.random.poisson(parameters[1], simulation_n)
        elif (parameters[0] == 'Negative Binomial'):
            sim_d = np.random.negative_binomial(((parameters[1])**2)/(parameters[2]-parameters[1]),parameters[1]/parameters[2], simulation_n)
        elif (parameters[0] == 'LogNormal'):
            sim_d = np.random.lognormal(parameters[1],parameters[2], simulation_n)
        elif (parameters[0] == 'Weibull'):
            sim_d = np.random.weibull(parameters[1], parameters[2], simulation_n)
        return (sim_d)

    def mean_distribution(self,parameters):
        if (parameters[0] == 'Poisson'):
            mean_d = parameters[1]
        elif (parameters[0] == 'Negative Binomial'):
           r = ((parameters[1]) ** 2) / (parameters[2] - parameters[1])
           p = 1 - parameters[1] / parameters[2]
           mean_d = (r*(1-p))/p
        elif (parameters[0] == 'LogNormal'):
            mean_d = parameters[1]
        return (mean_d)

    def Large_Claims(self,Frequency, Sev, n_sim):
        L_C = []
        N_list = self.simulated_observation(Frequency, 10, n_sim)
        for N in N_list:
            L_C.append(np.sum(self.simulated_observation(Sev, 10, N)))
        return (L_C)

    def Attritional_Claims(self,Frequency, Sev, n_sim):
        mean_dd = self.mean_distribution(Sev)
        A_C = []
        N_list = self.simulated_observation(Frequency, 10, n_sim)
        for N in N_list:
            A_C.append(N*mean_dd)
        return (A_C)

    def check_value(self,data_vector,ind_var_factor, d_min, d_max):
        k = 0
        x_k_ind = []
        vector_value = []
        for x in data_vector:
            if (d_min < x < d_max):
                vector_value.append(x)
                x_k_ind.append(ind_var_factor[k])
            k=k+1
        return ([vector_value, x_k_ind])

    def wsp_w_k_sqr(self,f_input, se_factor, dopasowanie, n):
        if (dopasowanie == 'factor_CL'):
            w_k_sqr = [1 / np.sqrt(np.log(1 + ((se_f_k) / (f_k - 1) ** 2))) for f_k, se_f_k in zip(f_input, se_factor)]
        elif (dopasowanie == 'variance_CL'):
            w_k_sqr = [n - 1 - k for k in range(1, len(f_input) + 1)]
        elif (dopasowanie == 'factor P_to_I'):
            w_k_sqr = [1]*(len(f_input) + 1)
        elif (dopasowanie == 'variance_P_to_I'):
            w_k_sqr = [1]*(len(f_input) + 1)
        elif (dopasowanie == 'factor_LR'):
            w_k_sqr = [1 / np.sqrt(np.log(1 + ((se_f_k) / (f_k) ** 2))) for f_k, se_f_k in zip(f_input, se_factor)]
        elif (dopasowanie == 'variance_LR'):
            w_k_sqr = [n - k for k in range(1, len(f_input) + 1)]
        return (w_k_sqr)


    def fit_curve(self,data_input, sd_input,x_k, dopasowanie, n):
        se_factor = [x ** 2 for x in sd_input[:len(data_input)]]
        w_k_sqr = self.wsp_w_k_sqr(data_input, se_factor, dopasowanie, n)
        if (dopasowanie == 'factor_CL'):
            factor_input = [np.log(f - 1) for f in data_input]
        elif (dopasowanie == 'variance_CL'):
            factor_input = [np.log(sigma) for sigma in data_input]
        elif (dopasowanie == 'factor P_to_I'):
            factor_input = [np.log(1-r_j) for r_j in data_input]
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

    def wspolczynnik_reg(self,a, b,k_start,k_stop,dopasowanie):
        if (dopasowanie == 'factor_CL'):
            wartosci_reg = [1+np.exp(a * k + b) for k in range(k_start,k_stop+1)]
        elif (dopasowanie == 'variance_CL'):
            wartosci_reg = [np.exp(a * k + b) for k in range(k_start, k_stop + 1)]
        elif (dopasowanie == 'variance_CL'):
            wartosci_reg = [np.exp(a * k + b) for k in range(k_start, k_stop + 1)]
        elif (dopasowanie == 'factor P_to_I'):
            wartosci_reg = [1-np.exp(a * k + b) for k in range(k_start, k_stop + 1)]
        elif (dopasowanie == 'variance_P_to_I'):
            wartosci_reg = [np.exp(a * k + b) for k in range(k_start, k_stop + 1)]
        elif (dopasowanie == 'factor_LR'):
            wartosci_reg = [np.exp(a * k + b) for k in range(k_start, k_stop + 1)]
        elif (dopasowanie == 'variance_LR'):
            wartosci_reg = [np.exp(a * k + b) for k in range(k_start, k_stop + 1)]

        return (wartosci_reg)

####LOSS RATIO METHOD

    def wspolczynniki_LR_i_j(self, df_triangle, exposure):
        indeksy, _, _ = self.index_all(df_triangle)
        n = df_triangle.shape[1]
        df_trian_copy = df_triangle.copy()
        for i in range(n):
            if i ==0:
                ind = indeksy[i] +[indeksy[i][len(indeksy[i])-1]+1]
                df_trian_copy.iloc[ind, i] = [x/y for x,y in zip(df_triangle.iloc[ind, i],list(itemgetter(*ind)(exposure)))]
            else:
                b = df_triangle.iloc[indeksy[i-1], i ] - df_triangle.iloc[indeksy[i-1], i-1]
                if (len(b)>1):
                    b_exposure = [x/y for x,y in zip(b,list(itemgetter(*indeksy[i-1])(exposure)))]
                    df_trian_copy.iloc[indeksy[i-1], i] = b_exposure
                else:
                    b_exposure = [b/exposure[i - 1]]
                    df_trian_copy.iloc[indeksy[i - 1], i] = b_exposure
        return (df_trian_copy)

    def wspolczynnik_LR(self,data_LR_i_j, w, exposure):
        #dziala dobrze
        mm = w.shape[0]
        nn = w.shape[1]
        wsp_LR = []
        for j in range(nn):
            l_pom = []
            m_pom = []
            for i in range(mm-j):
                l_pom.append(data_LR_i_j.iloc[i, j] * w.iloc[i, j] * exposure[i])
                m_pom.append(exposure[i] * w.iloc[i, j])
                if (np.sum(l_pom)==0):
                    wartosc = 0
                else:
                    wartosc = np.sum(l_pom) / np.sum(m_pom)
            wsp_LR.append(wartosc)
        return (wsp_LR)

   #triangle_forward_loss_ratio

    def kwadrat_roznicy(self,data_frame, vector):
        mm = data_frame.shape[0]
        nn = data_frame.shape[1]
        l_triangl = pd.DataFrame(0, columns=data_frame.columns, index=data_frame.index)
        for j in range(nn-1):
            for i in range(mm - j):
                l_triangl.iloc[i, j] = (data_frame.iloc[i, j] - vector[j])**2
        return (l_triangl)

    def sigma_LRMPC(self, exponsure, w, wspo_LR_i_j, wsp__LR_j):
        #liczy dobrze
        mm = w.shape[0]
        nn = w.shape[1]
        sigma_j = []
        kw_d = self.kwadrat_roznicy(wspo_LR_i_j, wsp__LR_j)
        for j in range(nn):
            l_pom = []
            m_pom = []
            for i in range(mm - j):
                l_pom.append(kw_d.iloc[i, j] * w.iloc[i, j] * exponsure[i])
                m_pom.append(w.iloc[i, j])
            if ((np.sum(m_pom) - 1) > 0):
                sigma_j.append(np.sum(l_pom) / (np.sum(m_pom) - 1))
            else:
                sigma_j.append(0)
        return (sigma_j)

    def wspolczynnik_sd_LRMPC(self,wsp_sigma,w,exponsure):
        #liczy dobrze
        mm = w.shape[0]
        nn = w.shape[1]
        sd_j = []
        for j in range(nn):
            mianownik = [w.iloc[i, j] * exponsure[i] for i in range(mm-j)]
            if (np.sum(mianownik) == 0):
                sd_j.append(0)
            else:
                sd_j.append(np.sqrt(wsp_sigma[j]/np.sum(mianownik)))
        return (sd_j)

    def random_stochastic_parameters_Loss_ratio(self,sigma_j,dev,sd,dimension):
        stochastic_sigma = pd.DataFrame(0, columns=np.arange(0, dimension[3], 1), index=np.arange(0, dimension[2], 1))
        mu_j = pd.DataFrame(0, columns=np.arange(0, dimension[3], 1), index=np.arange(0, dimension[2], 1))
        for j in range(0,dimension[3]):
            st_swobody = np.max([1, dimension[0] - j - 1])
            stochastic_sigma.iloc[:,j] = (chisquare(st_swobody, size=dimension[2]) * sigma_j[j]) / st_swobody
            mu_j.iloc[:,j] = normal(dev[j],sd[j],size = dimension[2])
        return([mu_j,stochastic_sigma])

    def stochastic_triangle_forward_Loss_ratio(self,data_paid,sigma_j,dev,sd,sim,eksponsure):
        #pytanie czy sigma ma być z kwadratem czy bez
        BE_ForwardProjection_data = pd.DataFrame(0, columns=np.arange(0, sim, 1), index=np.arange(0, data_paid.shape[0], 1))
        mm,nn = data_paid.shape[0],data_paid.shape[1]
        dimension = [mm,nn,sim,len(dev)]
        Total_BE = []
        diag_1 = pd.DataFrame(0, columns=np.arange(0, sim, 1), index=np.arange(0, dimension[0], 1))
        mu,sigma = self.random_stochastic_parameters_Loss_ratio(sigma_j,dev,sd,dimension)
        diag = self.reverse_list(self.trian_diag(data_paid))
        data_paid_copy = data_paid.copy()
        if(len(dev)>mm):
            for k in range(mm+1,len(dev)+2):data_paid_copy[k] = np.nan
        for row in range(0,sim):
            m_i, sigma_i = mu.iloc[row,:],sigma.iloc[row,:]
            for j in range(len(m_i)-1):
                max_ind_row = np.max([0, mm - j - 1])
                for i in range(max_ind_row, mm):
                    VAR_i_j = ((sigma_i[j+1])**2)/eksponsure[i]
                    if (m_i[j]==0):
                        lmean_i_j = np.log((0.00000000001) ** 2 / (np.sqrt((0.00000000001) ** 2 + VAR_i_j)))
                        lstdev_i_j = np.log((VAR_i_j) / ((0.000000001) ** 2) + 1)
                    else:
                        lmean_i_j = np.log((m_i[j+1])**2 / (np.sqrt((m_i[j+1])**2 + VAR_i_j)))
                        lstdev_i_j = np.log((VAR_i_j) / ((m_i[j+1]) ** 2) + 1)
                    stochastic_LR_i_j = lognormal(lmean_i_j, lstdev_i_j, 1)
                    zb_1_dol = m_i[j + 1] - sd[j + 1]
                    zb_1_gor = m_i[j + 1] + sd[j + 1]
                    zb_2_dol = m_i[j + 1] - 2 * sd[j + 1]
                    zb_2_gor = m_i[j + 1] + 2 * sd[j + 1]
                    if (stochastic_LR_i_j > zb_1_dol and stochastic_LR_i_j < zb_1_gor):
                        data_paid_copy.iloc[i, j + 1] = data_paid_copy.iloc[i, j] + eksponsure[i] * stochastic_LR_i_j[0]
                    elif ((stochastic_LR_i_j > zb_2_dol and stochastic_LR_i_j < zb_1_dol) or (
                            stochastic_LR_i_j > zb_1_gor and stochastic_LR_i_j < zb_2_gor)):
                        data_paid_copy.iloc[i, j + 1] = data_paid_copy.iloc[i, j] + eksponsure[i] * (
                                0.5 * stochastic_LR_i_j[0])
                    else:
                        data_paid_copy.iloc[i, j + 1] = data_paid_copy.iloc[i, j] + eksponsure[i] * (
                                0 * stochastic_LR_i_j[0])

                    #data_paid_copy.iloc[i, j + 1] = data_paid_copy.iloc[i, j] + eksponsure[i] * stochastic_LR_i_j[0]
            Ultimate = data_paid_copy.iloc[:, data_paid_copy.shape[0] - 1].to_list()
            BE = [x - y for x, y in zip(Ultimate, diag)]
            diag_1.iloc[:, row] = self.diag_1_triangle(data_paid_copy, nn, mm)
            Total_BE.append(sum(BE))
            BE_ForwardProjection_data.iloc[:, row] = BE
        return ([Total_BE,diag_1,BE_ForwardProjection_data])

    #Połączenie metody multyplikatywnej z metodą LOSS RATIO

    def triangle_forward(self, df_data, f,k_forward_start):
        df_t_copy = df_data.copy()
        mm,nn = df_data.shape[0], df_data.shape[1]
        if (len(f) > mm):
            for k in range(mm + 1, len(f) + 2): df_t_copy[k] = np.nan
        for j in range(k_forward_start,len(f)):
            max_ind_row = np.max([0, mm - j - 1])
            for i in range(max_ind_row, mm):
                df_t_copy.iloc[i, j + 1] = df_t_copy.iloc[i, j] * f[j]
        return (df_t_copy)

    def triangle_inc_to_paid(self,df_inc,df_p,r):
        df_paid = df_p.copy()
        mm,nn = df_paid.shape[0], df_paid.shape[1]
        if (len(r) > mm):
            for k in range(mm + 1, len(r) + 2): df_paid[k] = np.nan
        for j in range(df_inc.shape[1]-1):
            max_ind_row = np.max([0, mm - j - 1])
            for i in range(max_ind_row, mm):
                df_paid.iloc[i, j + 1] = df_inc.iloc[i, j] * r[j]
        return (df_paid)

    def triangle_inc_to_paid_stochastic(self,df_inc,df_paid,data_r_i_j):
        mm,nn = df_paid.shape[0], df_paid.shape[1]
        for j in range(df_paid.shape[1]-1):
            max_ind_row = np.max([0, mm - j - 1])
            if (len(r) > mm):
                for k in range(mm + 1, len(r) + 2): df_paid[k] = np.nan
            for i in range(max_ind_row, mm):
                df_paid.iloc[i, j + 1] = df_inc.iloc[i, j+1] * data_r_i_j.iloc[i,j+1]
        return (df_paid)

    def triangle_forward_loss_ratio(self,df_data,LR_j,eksponsure,k_forward):
        print('df_data')
        print(df_data)
        print('LR_j')
        print(LR_j)
        print('eksponsure')
        print(eksponsure)
        #dobra funkcja
        mm,nn = df_data.shape[0], df_data.shape[1]
        df_t_copy = df_data.copy()
        if (len(LR_j) > mm):
            for k in range(mm + 1, len(LR_j) + 2): df_t_copy[k] = np.nan
        for j in range(k_forward):
            max_ind_row = np.max([0,mm-j-1])
            for i in range(max_ind_row,mm):
                df_t_copy.iloc[i, j + 1] = df_t_copy.iloc[i, j] + eksponsure[i]*LR_j[j+1]
        return (df_t_copy)

    def triangle_forward_LR_CL(self,df_data, f, LR_j, eksponsure, k):
        mm, nn = df_data.shape[0], df_data.shape[1]
        if (k==0):
            ilosc_LR = len(LR_j)-1
            data_output = self.triangle_forward_loss_ratio(df_data, LR_j, eksponsure,ilosc_LR)
        elif(k==1):
            ilosc_CL = 0
            data_output = self.triangle_forward(df_data,f,ilosc_CL)
        elif(k>1):
            ilosc_LR = k-2
            data_output_pom = self.triangle_forward_loss_ratio(df_data, LR_j, eksponsure, ilosc_LR)
            data_output = self.triangle_forward(data_output_pom, f, ilosc_LR)
        return(data_output)

#stochastic paid

    def wspolczynnik_r(self, d_paid, d_claim):
        wspol = []
        for i in range(d_paid.shape[1]):
            wspol.append(np.sum(d_paid.iloc[:,i])/np.sum(d_claim.iloc[:,i]))
        return(wspol)

#6.3.2
    def wspolczynnik_r_i_j(self,dd_paid,dd_claim):
        mm = dd_paid.shape[0]
        nn = dd_paid.shape[1]
        l_triangl = pd.DataFrame(0, columns=dd_paid.columns, index=dd_paid.index)
        for j in range(nn):
            for i in range(mm - j):
                l_triangl.iloc[i, j] = dd_paid.iloc[i, j]/dd_claim.iloc[i, j]
        return (l_triangl)

    def kwadrat_roznicy(self,data_frame, vector):
        mm = data_frame.shape[0]
        nn = data_frame.shape[1]
        l_triangl = pd.DataFrame(0, columns=data_frame.columns, index=data_frame.index)
        for j in range(nn-1):
            for i in range(mm - j):
                l_triangl.iloc[i, j] = (data_frame.iloc[i, j] - vector[j])**2
        return (l_triangl)

    def wspolczynnik_var_j(self,data_claim,wspo_r_i_j,wspo_r_j):
        mm = data_claim.shape[0]
        nn = data_claim.shape[1]
        var_j = []
        roznica_r_i_j = self.kwadrat_roznicy(wspo_r_i_j,wspo_r_j)
        for j in range(nn):
            var_j_pom = []
            for i in range(mm - j):
                var_j_pom.append(data_claim.iloc[i,j]*roznica_r_i_j.iloc[i,j])
            if (mm-j)==1:
                s_min = np.min([var_j[len(var_j) - 1], var_j[len(var_j) - 2],((var_j[len(var_j) - 1]) ** 2)/var_j[len(var_j) - 2]])
                var_j.append(s_min)
            else:
                var_j.append((np.sum(var_j_pom))/(mm-j))
        return(var_j)

    def wspolczynnik_res_i_j(self,wspo_r_i_j,wspo_r_j,wspo_var_j,data_frame):
        mm = data_frame.shape[0]
        nn = data_frame.shape[1]
        l_triangl = pd.DataFrame(np.NAN, columns=data_frame.columns, index=data_frame.index)
        for j in range(nn):
            for i in range(mm - j):
                licznik = wspo_r_i_j.iloc[i,j] - wspo_r_j[j]
                mianownik = np.sqrt(wspo_var_j[j]/data_frame.iloc[i,j])
                l_triangl.iloc[i, j] = licznik/mianownik
        return (l_triangl)

    def lambda_cerrelation(self,triangle_res):
        mm = triangle_res.shape[0]
        nn = triangle_res.shape[1]
        col_list = []
        col_kwadrat = []
        for i in range(mm-1):
            for j in range(nn-1-i):
                col_list.append(triangle_res.iloc[i,j]*triangle_res.iloc[i,j+1])
                col_kwadrat.append((triangle_res.iloc[i,j])**2)
        lambda_cor = np.sum(col_list)/np.sum(col_kwadrat)
        return(lambda_cor)

    def stochastic_r_i_j(self,wsp_r_j,var_j,data_c_i_j,wsp_res_i_j,lam_cor):
        stochastic_r_i_j = pd.DataFrame(0,columns=np.arange(0,data_c_i_j.shape[1],1),index=np.arange(0,data_c_i_j.shape[0],1))
        nn = data_c_i_j.shape[1]
        mm = data_c_i_j.shape[0]
        for col in range(1,mm+1):
            stochastic_r_i_j.iloc[mm-col,col] = wsp_r_j[col] + np.sqrt(var_j[col]/data_c_i_j.iloc[mm-col,col-1])*(np.random.normal(0,1)+wsp_res_i_j.iloc[mm-col,col-1]*lam_cor)
        start_col = 1
        for row in range(mm-1,-1,-1):
            start_col = start_col+1
            for col in range(start_col,nn-1):
                res_i_j = (stochastic_r_i_j.iloc[row, col-1]-wsp_r_j[col-1])/(np.sqrt(var_j[col-1]/data_c_i_j.iloc[row,col-1]))
                stochastic_r_i_j.iloc[row, col] = wsp_r_j[col] + np.sqrt(var_j[col]/data_c_i_j.iloc[mm-col,col-1])*(np.random.normal(0,1) +res_i_j*lam_cor)
        return(stochastic_r_i_j)

    def stochastic_triangle_paid_incurred(self,tr_paid_origin,data_paid, sigma_j,dev,sd,rj, varj,r_i_j, wsp_lam,sim):
        print(tr_paid_origin.to_string())
        BE_ForwardProjection_data = pd.DataFrame(0, columns=np.arange(0, sim, 1), index=np.arange(0, data_paid.shape[0], 1))
        seed(10)
        mm,nn = data_paid.shape[0],data_paid.shape[1]
        dimension = [mm,nn,sim,len(dev)]
        Total_BE = []
        diag_1 = pd.DataFrame(0, columns=np.arange(0, sim, 1), index=np.arange(0, dimension[0], 1))
        mu,sigma = self.random_stochastic_parameters(sigma_j,dev,sd,dimension)
        diag = self.reverse_list(self.trian_diag(data_paid))
        data_paid_copy = data_paid.copy()
        if(len(dev)>mm):
            for k in range(mm+1,len(dev)+2):data_paid_copy[k] = np.nan
        for row in range(0,sim):
            m_i, sigma_i = mu.iloc[row,:],sigma.iloc[row,:]
            for j in range(len(m_i)):
                max_ind_row = np.max([0, mm - j - 1])
                for i in range(max_ind_row, mm):
                    VAR_i_j = sigma_i[j]/data_paid_copy.iloc[i,j]
                    lmean_i_j = np.log((m_i[j])**2 / (np.sqrt((m_i[j])**2 + VAR_i_j)))
                    lstdev_i_j = np.log((VAR_i_j) / ((m_i[j]) ** 2) + 1)
                    CL_i_j = lognormal(lmean_i_j,lstdev_i_j,1)
                    data_paid_copy.iloc[i,j+1] = round(data_paid_copy.iloc[i,j]*CL_i_j[0],3)
         #   print('data_paid_copy')
         #   print(data_paid_copy.to_string())
            Ultimate = data_paid_copy.iloc[:, data_paid_copy.columns[-1] - 1].to_list()
            BE = [x - y for x, y in zip(Ultimate, diag)]
            diag_1.iloc[:,row] = self.diag_1_triangle(data_paid_copy,nn,mm)
            Total_BE.append(sum(BE))
            BE_ForwardProjection_data.iloc[:,row] = BE
        return ([Total_BE,diag_1,BE_ForwardProjection_data])
#6.3.6

    def roznica_macierz_listy(self,dane, vector):
        dane_copy = dane.copy()
        col = 0
        mm = dane_copy.shape[0]
        nn = dane_copy.shape[1]
        for j in range(nn):
            for i in range(mm-j):
                dane_copy.iloc[i,j] = dane.iloc[i,j]-vector[j]
        return(dane_copy)

    def roznica_macierz_listy(self,dane, vector):
        dane_copy = dane.iloc[:-1,:-1]
        for j in range(dane_copy.shape[1]):
            for i in range(dane_copy.shape[0]-j):
                dane_copy.iloc[i,j] = dane.iloc[i,j]-vector[i]
        return(dane_copy)

    def r_i_j_CL(self,l,Cl,data_paid,w,sd):
        pier_skl = self.roznica_macierz_listy(l, Cl)
        r_triangl = pd.DataFrame(0, columns=pier_skl.columns, index=pier_skl.index)
        for j in range(pier_skl.shape[1]):
            for i in range(pier_skl.shape[1]-j):
                licznik = data_paid.iloc[i,j]*w.iloc[i,j]
                r_triangl.iloc[i,j] = pier_skl.iloc[i,j]*np.sqrt(licznik/(float(sd[i]))**2)
        return (r_triangl)





