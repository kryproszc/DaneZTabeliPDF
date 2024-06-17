library(dplyr)
library(tidyr)
library(ggplot2)
library(parallel)

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
  m_i <- mu[row, ]
  sigma_i <- sigma[row, ]
  
  for (j in seq_along(m_i)) {
    max_ind_row <- max(0, mm - j - 1)
    for (i in max_ind_row:mm) {
      VAR_i_j <- sigma_i[j] / data_paid_copy[i, j]
      lmean_i_j <- log((m_i[j]) ^ 2 / sqrt((m_i[j]) ^ 2 + VAR_i_j))
      lstdev_i_j <- log(1 + (VAR_i_j / (m_i[j]) ^ 2))
      CL_i_j <- rng$lognormal(lmean_i_j, lstdev_i_j, size = 1)
      data_paid_copy[i, j + 1] <- data_paid_copy[i, j] * CL_i_j
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

stochastic_triangle_forward_test_szybki <- function(data_paid, sigma_j, dev, sd, sim, Ultimate_Param_ReservingRisk) {
  mm <- nrow(data_paid)
  nn <- ncol(data_paid)
  dimension <- c(mm, nn, sim, length(dev))
  
  params <- random_stochastic_parameters(sigma_j, dev, sd, dimension)
  mu <- params[[1]]
  sigma <- params[[2]]
  data_paid_copy <- data_paid
  
  args <- lapply(1:sim, function(row) {
    list(seed = 202260011, row = row, mu = mu, sigma = sigma, mm = mm, data_paid_copy = data_paid_copy, Ultimate_Param_ReservingRisk = Ultimate_Param_ReservingRisk)
  })
  
  num_cores <- max(detectCores()/2-1, 4)
  cl <- makeCluster(num_cores)
  clusterExport(cl, list("process_row", "PCG64Generator"))
  Total_BE <- parLapply(cl, args, process_row)
  stopCluster(cl)
  
  unlist(Total_BE)
}

main <- function() {
  sim <- 10000
  options(digits = 10)
  wsp <- read.csv("wsp_csv.csv", sep = ";", dec = ",")
  dev <- as.numeric(wsp[1, 1:59])
  sigma_j <- as.numeric(wsp[2, 1:59])
  sd <- as.numeric(wsp[3, 1:59])
  data_paid <- read.csv("data.csv", sep = ";", dec = ",")[, -1]
  
  result <- stochastic_triangle_forward_test_szybki(data_paid, sigma_j, dev, sd, sim, 4232149669)
  
  cat("Quantile 0.995:", quantile(result, 0.995), "\n")
  cat("Quantile 0.996:", quantile(result, 0.996), "\n")
  cat("Quantile 0.997:", quantile(result, 0.997), "\n")
  cat("Quantile 0.998:", quantile(result, 0.998), "\n")
  cat("Quantile 0.999:", quantile(result, 0.999), "\n")
  cat("Mean:", mean(result), "\n")
  cat("Difference (Quantile - Mean):", quantile(result, 0.995) - mean(result), "\n")
  
 # pd_dev <- data.frame(BE = result)
 # write.csv(pd_dev, "BE_PCG.csv")
  
 # ggplot(pd_dev, aes(x = BE)) +
  ##  geom_histogram(bins = 50, color = "black", fill = "white") +
  #  labs(x = "Wynik", y = "Częstość", title = "Histogram wyników symulacji") +
   # theme_minimal()
}

main()
