library(dplyr)
library(tidyr)
library(ggplot2)
library(parallel)
library(readxl)
library(ChainLadder)
set.seed(202260011)
library(openxlsx)
#################################3
setwd("I:/Ubezpieczeniowe/2024/PZU SA/7_Foldery_ZI/KS/Aplikacja RESQ")

library(readxl)
# Określ ścieżkę do pliku Excel
file_path <- "Kopia RESQ_CLAIMS_S2_2024.Q1.xlsx"

# Określ nazwę arkusza
sheet_name <- "DFM paid (1)"

# Wczytaj dane z określonego zakresu wierszy i kolumn
data <- read_excel(file_path, sheet = sheet_name,range = "B6:AI40")

#####################################
library(dplyr)
library(tidyr)
library(ggplot2)
library(parallel)
library(readxl)
set.seed(202260011)
PCG64Generator <- function(seed = 202260011) {
  set.seed(seed)
  list(
    random = function(size = 1) runif(size),
    normal = function(mu = 0, sigma = 1, size = 1) rnorm(size, mean = mu, sd = sigma),
    lognormal = function(mean = 0, sigma = 1, size = 1) rlnorm(size, meanlog = mean, sdlog = sigma),
    chi_squared = function(df, size = 1) rchisq(size, df = df)
  )
}

process_row <- function(args) {
  seed <- args[[1]]
  row <- args[[2]]
  mu <- args[[3]]
  sigma <- args[[4]]
  mm <- args[[5]]
  data_paid_copy <- args[[6]]
  Ultimate_Param_ReservingRisk <- args[[7]]
  rng <- PCG64Generator(seed)
  m_i <- mu
  sigma_i <- sigma
  for (j in seq_along(m_i)) {
    max_ind_row <- max(0, mm - j - 1)
    for (i in max_ind_row:mm) {
      VAR_i_j <- sigma_i[j] / data_paid_copy[i, j]
      lmean_i_j <- log((m_i[j]) ^ 2 / sqrt((m_i[j]) ^ 2 + VAR_i_j))
      lstdev_i_j <- log(1 + (VAR_i_j / (m_i[j]) ^ 2))
      CL_i_j <- rng$lognormal(lmean_i_j, lstdev_i_j, size = 1)
      if(TRUE){
        # Dodanie walidacji dla CL_i_j
        if (!is.na(CL_i_j) && !is.null(CL_i_j) && CL_i_j > m_i[j]-2*sqrt(VAR_i_j)&& CL_i_j < m_i[j]+2*sqrt(VAR_i_j)) {
          CL_i_j_choose <- CL_i_j
        }
        if(!is.na(CL_i_j) && !is.null(CL_i_j) && CL_i_j > m_i[j]-3*sqrt(VAR_i_j)&& CL_i_j < m_i[j]-2*sqrt(VAR_i_j)&& CL_i_j > m_i[j]+2*sqrt(VAR_i_j)&& CL_i_j < m_i[j]+3*sqrt(VAR_i_j)){
          CL_i_j_choose<-0.5*CL_i_j
        }
      }
      else{
        CL_i_j_choose<-CL_i_j
      }
      data_paid_copy[i, j + 1] <- data_paid_copy[i, j] * CL_i_j_choose
    }
  }
  u_i <- data_paid_copy[, ncol(data_paid_copy) - 1]
  result_j <- sum(u_i) - Ultimate_Param_ReservingRisk
  return(result_j)
}

random_stochastic_parameters <- function(sigma_j, dev, sd, dimension) {
  rng <- PCG64Generator()
  stochastic_sigma_j <- matrix(0, nrow = dimension[3], ncol = dimension[4])
  mu_j <- matrix(0, nrow = dimension[3], ncol = dimension[4])
  for (j in 1:dimension[4]) {
    mu_j[, j] <- rng$normal(dev[j], sd[j], size = dimension[3])
    st_swobody <- max(1, dimension[1] - j)
    chi_list <- rng$chi_squared(st_swobody, size = dimension[3])
    stochastic_sigma_j[, j] <- (ceiling(chi_list) * sigma_j[j]) / st_swobody
  }
  list(mu_j, stochastic_sigma_j)
}

stochastic_triangle_forward_test_szybki <- function(data_paid, sigma_j, dev, sd, sim, Ultimate_Param_ReservingRisk, num_sim, batch_size = 1000, result_file = "results.csv") {
  mm <- nrow(data_paid)
  nn <- ncol(data_paid)
  dimension <- c(mm, nn, sim, length(dev))
  # Wywołanie random_stochastic_parameters raz
  params <- random_stochastic_parameters(sigma_j, dev, sd, dimension)
  mu <- params[[1]]
  sigma <- params[[2]]
  results <- numeric()
  # Otwarcie pliku wynikowego
  write.table(NULL, file = result_file, sep = ",", col.names = FALSE, row.names = FALSE)
  
  for (i in seq(1, sim, by = batch_size)) {
    current_batch_size <- min(batch_size, sim - i + 1)
    data_paid_copy <- data_paid
    args <- lapply(1:current_batch_size, function(batch_row) {
      global_row <- i + batch_row - 1  # Przesunięcie indeksu, aby uwzględnić numer partii
      list(seed = 1, row = global_row, mu = mu[global_row, ], sigma = sigma[global_row, ], mm = mm, data_paid_copy = data_paid_copy, Ultimate_Param_ReservingRisk = Ultimate_Param_ReservingRisk)
    })
    num_cores <- detectCores() - 1
    cl <- makeCluster(num_cores)
    clusterExport(cl, list("process_row", "PCG64Generator"))
    Total_BE <- parLapply(cl, args, process_row)
    stopCluster(cl)
    batch_results <- unlist(Total_BE)
    results <- c(results, batch_results)
    # Zapis wyników do pliku po każdej partii
    write.table(batch_results, file = result_file, sep = ",", col.names = FALSE, row.names = FALSE, append = TRUE)
    # Debugowanie
    cat("Zamknięto klaster dla partii od", i, "do", i + current_batch_size - 1, "\n")
  }
  
  return(results)
}



#############################################################
#paid oliczenia

calculate_development_factors <- function(triangle) {
  n <- nrow(triangle)
  m <- ncol(triangle)
  development_factors <- matrix(NA, n, m)
  
  for (j in 2:m) {
    for (i in 1:(n-j+1)) {
      if (!is.na(triangle[i, j]) && !is.na(triangle[i, j-1]) && triangle[i, j-1] != 0) {
        development_factors[i, j-1] <- triangle[i, j] / triangle[i, j-1]
      }
    }
  }
  return(development_factors)
}
calculate_unbiased_development_factors <- function(triangle, development_factors,weights) {
  n <- nrow(triangle)
  m <- ncol(triangle)
  unbiased_factors <- numeric(m - 1)
  
  for (j in 2:m) {
    C_ij <- numeric()
    F_ij <- numeric()
    for (i in 1:(n - j + 1)) {
      if (!is.na(triangle[i, j]) && !is.na(triangle[i, j - 1]) && triangle[i, j - 1] != 0) {
        C_ij <- c(C_ij, triangle[i, j - 1]*weights[i, j - 1])
        F_ij <- c(F_ij, (triangle[i, j] / triangle[i, j - 1])*weights[i, j - 1])
      }
    }
    if (length(C_ij) > 0) {
      unbiased_factors[j - 1] <- sum(C_ij * F_ij) / sum(C_ij)
    } else {
      unbiased_factors[j - 1] <- NA
    }
  }
  return(unbiased_factors)
}
calculate_variance_factors <- function(triangle, development_factors,weight) {
  n <- nrow(triangle)
  m <- ncol(triangle)
  variance_factors <- numeric(m-1)
  
  for (j in 2:(m)) {
    numerator <- 0
    denominator <- sum(!is.na(weight[, j-1]))-2
    for (i in 1:(n-j+1)) {
      if (!is.na(triangle[i, j]) && !is.na(triangle[i, j-1]) && triangle[i, j-1] != 0) {
        C_ij <- triangle[i, j-1]
        F_ij <- triangle[i, j] / triangle[i, j-1]
        f_j <- development_factors[j-1]
        numerator <- numerator + weight[i, j-1]*C_ij * (F_ij - f_j)^2
        # denominator <- denominator + C_ij
      }
    }
    if(j==(m)){
      variance_factors[j-1] = min((variance_factors[j-2]^2)/(variance_factors[j-3]),min(variance_factors[j-3],variance_factors[j-2]))
    }
    else{
      variance_factors[j-1] <- numerator/denominator
      
    }
  }
  return(sqrt(variance_factors))
}
calculate_sd_factors <- function(tr_paid,sigma_in,weight) {
  n <- nrow(tr_paid)
  m <- ncol(tr_paid)
  sd_factors <- numeric(m-1)
  
  for (j in 2:(m)) {
    denominator<-0
    for (i in 1:(n-j+1)) {
      denominator <- denominator+tr_paid[i, j-1]*weight[i, j-1]
    }
    sd_factors[j-1]<-sigma_in[j-1]^2/denominator
  }
  return(sqrt(sd_factors))
}
####
create_pattern_matrix <- function(df_paid_l, pattern_number) {
  # Replace non-NA values with 1 and NA values with NaN
  reduced_df <- df_paid_l
  reduced_df[!is.na(reduced_df)] <- 1
  reduced_df[is.na(reduced_df)] <- NaN
  
  # Get the number of rows (assuming a square matrix)
  size <- nrow(reduced_df)
  reduced_df_copy <- reduced_df
  
  # Loop to set specific elements to 0
  for (j in 1:size) {
    for (i in 1:(size - pattern_number)) {
      if (i <= (size - pattern_number - j + 1)) {
        reduced_df_copy[i, j] <- 0
      }
    }
  }
  
  return(reduced_df_copy)
}


data_paid<-RAA
data_paid<-as.data.frame(data)
write.xlsx(data_paid, "linia_1.xlsx")

weight <- create_pattern_matrix(data_paid,10)
development_factorss<-calculate_development_factors(data_paid)
f_factor<-calculate_unbiased_development_factors(data_paid,development_factorss,weight)
sigma_j<-calculate_variance_factors(data_paid,f_factor,weight)
sd<-calculate_sd_factors(data_paid,sigma_j,weight)

sigma_j[33]<-0
sd[33]<-0

sigma_j
sd

result <- stochastic_triangle_forward_test_szybki(data_paid, sigma_j, f_factor, sd, 5000, 0,0)
hist(result)
quantile(result,0.995)
min(result)
rnorm(1,0.91,0)














#############################################################
reduced_df <- data_paid[1:(dim(data_paid)[1]-1),1:(dim(data_paid)[2]-1)]
reduced_df
weig_input<-create_pattern_matrix(data_paid,5)
colnames(data_paid)<-1:34
library(ChainLadder)
sum(getLatestCumulative(as.matrix(data_paid)))
mack<-MackChainLadder(data_paid,est.sigma = "Mack")
data_paid<-RAA
data_paid[1,9]<-18834
data_paid[2,9]<-18834
data_paid[1,8]<-18834
data_paid[2,8]<-18834
data_paid[2,8]<-18834

data_paid[1,1]<-1
data_paid[2,1]<-1
data_paid[3,1]<-1
data_paid[4,1]<-1
data_paid[5,1]<-1

boot_sim<-BootChainLadder(data_paid,R=5000)

IBNR_boot<-boot_sim$IBNR.Totals+3309420740
mean(IBNR_boot)
quantile(IBNR_boot,0.995)
hist(IBNR_boot)
max(IBNR_boot)
min(IBNR_boot)

result_out <-c()
options(digits = 10)
dev <- mack$f[-length(mack$f)]
sigma_j <- mack$sigma
sd <- mack$f.se

#symulacje bez podzialu
result <- stochastic_triangle_forward_test_szybki(data_paid, sigma_j, dev, sd, 5000, 160987,0)
x_sort<-sort(result)
x_sort
quantile(result,0.995)
hist(result)
max(result)
mean(result)



#########################

paid_triangle<-MCLpaid
inccured_triangle<-MCLincurred
munich<-MunichChainLadder(paid_triangle,inccured_triangle)
munich$QResiduals
#inccured
calculate_q <- function(paid_triangle, incurred_triangle) {
  n <- nrow(paid_triangle)
  m <- ncol(paid_triangle)
  q <- numeric(n - 1)
  
  for (j in 1:m) {
    sum_C_P <- 0
    sum_C_I <- 0
    
    for (i in 1:(n - j + 1)) {
      if (!is.na(paid_triangle[i, j]) && incurred_triangle[i, j] != 0) {
        sum_C_P <- sum_C_P + paid_triangle[i, j]
        sum_C_I <- sum_C_I + incurred_triangle[i, j]
      }
    }
    
    if (sum_C_I != 0) {
      q[j] <- sum_C_P / sum_C_I
    } else {
      q[j - 1] <- NA
    }
  }
  
  return(q)
}
calculate_q_factors <- function(paid_triangle, incurred_triangle) {
  n <- nrow(paid_triangle)
  m <- ncol(paid_triangle)
  q_factors <- matrix(NA, n, m)
  q_inverse_factors <- matrix(NA, n, m)
  
  for (j in 1:m) {
    for (i in 1:(n-j+1)) {
      if (!is.na(paid_triangle[i, j]) && !is.na(incurred_triangle[i, j]) && incurred_triangle[i, j] != 0) {
        q_factors[i, j] <- paid_triangle[i, j] / incurred_triangle[i, j]
        q_inverse_factors[i, j] <- incurred_triangle[i, j] / paid_triangle[i, j]
      }
    }
  }
  return(list(q_factors = q_factors, q_inverse_factors = q_inverse_factors))
}
calculate_variance_q <- function(incurred_triangle, q, q_factors) {
  n <- nrow(incurred_triangle)
  m <- ncol(incurred_triangle)
  variance_factors <- numeric(m)
  for (j in 1:m) {
    sum_C_I_Q_diff <- 0
    sum_C_I <- 0
    denominator <- n-j
    for (i in 1:(n - j + 1)) {
      if (!is.na(q_factors[i, j])) {
        C_I_ij <- incurred_triangle[i, j]
        Q_I_ij <- q_factors[i, j]
        q_j <- q[j]
        sum_C_I_Q_diff <- sum_C_I_Q_diff + C_I_ij * (Q_I_ij - q_j)^2
        sum_C_I <- sum_C_I + C_I_ij
      }
    }
    if(j==(m)){
      variance_factors[j] = min((variance_factors[j-1]^2)/(variance_factors[j-2]),min(variance_factors[j-2],variance_factors[j-1]))
    }
    else{
      variance_factors[j] <- sum_C_I_Q_diff/denominator
      
    }
  }
  return(variance_factors)
}
calculate_pearson_residuals_q <- function(incurred_triangle, q_factors, q, variance_q) {
  n <- nrow(incurred_triangle)
  m <- ncol(incurred_triangle)
  pearson_residuals_q <- matrix(NA, n, m)
  for (j in 1:m) {
    for (i in 1:(n - j+1)) {
      if (!is.na(incurred_triangle[i, j])) {
        Q_ij <- q_factors[i, j]
        q_j <- q[j]
        tau_I_j <- sqrt(variance_q[j])
        pearson_residuals_q[i, j] <- ((Q_ij - q_j) / tau_I_j)*sqrt(incurred_triangle[i, j])
      }
    }
  }
  return(pearson_residuals_q)
}
calculate_correlation<-function(res_data){
  n<-dim(res_data)[1]
  m<-dim(res_data)[2]
  licznik<-0
  mianownik<-0
  for(j in 1:(n-1)){
    for(i in 1:(n-j)){
      licznik<-licznik+res_data[i,j]*res_data[i,j+1]
      mianownik<-mianownik+res_data[i,j]^2
    }
  }
  return(licznik/mianownik)
}


q_calc<-calculate_q(paid_triangle,inccured_triangle)
qs_calc<-calculate_q_factors(paid_triangle,inccured_triangle)
var_j<-calculate_variance_q(inccured_triangle,q_calc,qs_calc$q_factors)
reas_pearson<-calculate_pearson_residuals_q(inccured_triangle,qs_calc$q_factors,q_calc,var_j)
cor_person<-calculate_correlation(reas_pearson)
cor_person








