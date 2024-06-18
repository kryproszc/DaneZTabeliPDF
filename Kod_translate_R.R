library(data.table)
library(ggplot2)
library(parallel)
library(readxl)

set.seed(202260011)

# Definicja generatora losowego
PCG64Generator <- function(seed = 202260011) {
  set.seed(seed)
  list(
    random = function(size = 1) runif(size),
    normal = function(mu = 0, sigma = 1, size = 1) rnorm(size, mean = mu, sd = sigma),
    lognormal = function(mean = 0, sigma = 1, size = 1) {
      if (length(mean) == 0 || length(sigma) == 0 || !is.finite(mean) || !is.finite(sigma)) {
        warning(paste("Invalid parameters for rlnorm: mean =", mean, "sigma =", sigma))
        return(rep(NA, size))
      }
      if (sigma <= 0) {
        warning(paste("Invalid parameter for rlnorm: sigma must be greater than zero. sigma =", sigma))
        return(rep(NA, size))
      }
      rlnorm(size, meanlog = mean, sdlog = sigma)
    },
    chi_squared = function(df, size = 1) rchisq(size, df = df)
  )
}

# Funkcja przetwarzająca jeden wiersz danych
process_row <- function(args) {
  seed <- args[[1]]
  row <- args[[2]]
  mu <- args[[3]]
  sigma <- args[[4]]
  mm <- args[[5]]
  data_paid_copy <- copy(args[[6]]) # Użyj data.table's copy
  Ultimate_Param_ReservingRisk <- args[[7]]
  
  rng <- PCG64Generator(seed)
  m_i <- mu[row, ]
  sigma_i <- sigma[row, ]
  
  for (j in seq_along(m_i)) {
    max_ind_row <- max(0, mm - j - 1)
    for (i in max_ind_row:mm) {
      VAR_i_j <- sigma_i[j] / data_paid_copy[i, j]
      lmean_i_j <- log((m_i[j]) ^ 2 / sqrt((m_i[j]) ^ 2 + VAR_i_j))
      lstdev_i_j <- log(1 + (VAR_i_j / (m_i[j]) ^ 2))
      if (!is.finite(lmean_i_j) || !is.finite(lstdev_i_j) || lstdev_i_j <= 0) {
        warning(paste("Skipping invalid parameters: lmean_i_j =", lmean_i_j, "lstdev_i_j =", lstdev_i_j))
        next
      }
      CL_i_j <- rng$lognormal(lmean_i_j, lstdev_i_j, size = 1)
      
      # Dodanie ograniczenia dla pierwszej przekątnej
      if (i == j) {  # warunek dla pierwszej przekątnej
        lower_bound <- m_i[j] - 2 * VAR_i_j
        upper_bound <- m_i[j] + 2 * VAR_i_j
        if (CL_i_j < lower_bound) {
          CL_i_j <- lower_bound
        } else if (CL_i_j > upper_bound) {
          CL_i_j <- upper_bound
        }
      }
      
      data_paid_copy[i, j + 1] <- data_paid_copy[i, j] * CL_i_j
    }
  }
  u_i <- data_paid_copy[, ncol(data_paid_copy) - 1]
  result_j <- sum(u_i) - Ultimate_Param_ReservingRisk
  return(result_j)
}

# Funkcja generująca losowe parametry
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

# Główna funkcja symulacyjna
stochastic_triangle_forward_test_szybki <- function(data_paid, sigma_j, dev, sd, sim, Ultimate_Param_ReservingRisk, num_sim) {
  mm <- nrow(data_paid)
  nn <- ncol(data_paid)
  dimension <- c(mm, nn, sim, length(dev))
  
  params <- random_stochastic_parameters(sigma_j, dev, sd, dimension)
  mu <- params[[1]]
  sigma <- params[[2]]
  data_paid_copy <- copy(data_paid)  # Użyj data.table's copy
  
  args <- lapply(1:sim, function(row) {
    list(seed = 202260011 + num_sim, row = row, mu = mu, sigma = sigma, mm = mm, data_paid_copy = copy(data_paid_copy), Ultimate_Param_ReservingRisk = Ultimate_Param_ReservingRisk)
  })
  
  num_cores <- max(detectCores() - 1, 2) # Użyj mniejszej liczby rdzeni
  cl <- makeCluster(num_cores)
  clusterExport(cl, list("process_row", "PCG64Generator", "copy"))
  Total_BE <- parLapply(cl, args, process_row)
  stopCluster(cl)
  
  unlist(Total_BE)
}

# Ustawienie katalogu roboczego i główne ustawienia
setwd("M:/Symulacje")
result_out <- c()
options(digits = 10)
wsp <- read_xlsx(path = "wspol_MTPL_F.xlsx")
dev <- as.numeric(wsp[1, 1:59])
sigma_j <- as.numeric(wsp[2, 1:59])
sd <- as.numeric(wsp[3, 1:59])
data_paid <- fread("reserv_data_MTPL_F.csv", sep = ";", dec = ",")[, -1, with = FALSE]

# Symulacje w partiach
ind = 0
for(sim in rep(100, 10)) {
  result <- stochastic_triangle_forward_test_szybki(data_paid, sigma_j, dev, sd, sim, 4232149669, ind)
  result_out <- c(result_out, result)
  ind = ind - 100
}

# Symulacja pełnego zestawu
result <- stochastic_triangle_forward_test_szybki(data_paid, sigma_j, dev, sd, 100000, 4232149669, 0)

# Wyniki
cat("Quantile 0.995:", quantile(result, 0.995), "\n")
cat("Quantile 0.996:", quantile(result, 0.996), "\n")
cat("Quantile 0.997:", quantile(result, 0.997), "\n")
cat("Quantile 0.998:", quantile(result, 0.998), "\n")
cat("Quantile 0.999:", quantile(result, 0.999), "\n")
cat("Mean:", mean(result), "\n")
cat("Difference (Quantile - Mean):", quantile(result, 0.995) - mean(result), "\n")
