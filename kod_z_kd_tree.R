# Rozpoczęcie pomiaru czasu
start_time <- Sys.time()

# Załadowanie potrzebnych bibliotek
library(readxl)
library(readr)
library(lubridate)
library(foreach)
library(doParallel)
library(openxlsx)
library(writexl)
library(Rcpp)

# Załadowanie kodu C++ do R
sourceCpp("/mnt/data/kod_z_kd_tree.cpp")


fire_probs <- as.matrix(read_excel("prawdopodobienstwa.xlsx", sheet = "PrawdPozaru", range = "B1:M18"))
fire_spread_probe <- as.matrix(read_excel("prawdopodobienstwa.xlsx", sheet = "Rozprzestrzenienia_Pozarow"))
exposure_data <- read_excel("dane_budynkow.xlsx")
exposure_data<-as.data.frame(na.omit(exponsure_data))
exposure_data[1:100,4]<-'2021-31-01'
exposure_data[100:length(exposure_data[,4]),4]<-'2022-01-01'
exposure_data[,5]<-sample(seq(as.Date('2022-01-01'), as.Date('2023-01-31'), by = "day"),replace = TRUE,400603+333)
exposure_data[,6]<-sample(seq(0, 17, by = 1),replace = TRUE,400603+333)
exposure_data[23:300,6]<-0
exposure_data<-as.matrix(exponsure_data)
modelled_year <- 2022
policy_start_month <- month(exposure_data[, 4])
policy_start_year <- year(exposure_data[, 4])
policy_end_month <- month(exposure_data[, 5])
policy_end_year <- year(exposure_data[, 5])
policy_start_month[which(policy_start_year < modelled_year)] <- 0
policy_start_month[which(policy_start_year > modelled_year)] <- 13
policy_end_month[which(policy_end_year > modelled_year)] <- 0
policy_end_month[which(policy_end_year > modelled_year)] <- 13

source_python('Kod_pythona_dla_pliku_kod_z_kd_tree.py')
coordinates <- cbind(as.numeric(gsub(",", ".", exposure_data[, 2])),as.numeric(gsub(",", ".", exposure_data[, 3])))
latitude <- convert_coordinates(coordinates)[,1]
latitude <- convert_coordinates(coordinates)[,2]
province <- as.numeric(exposure_data[, 6])
number_of_simulations <- 1 #(domyslnie 100000)
basic_exposure_list <- list()

number_of_basic_exposure <- matrix(nrow = 16, ncol = 12)
exposure_latitude <- list()
exposure_longitude <- list()
KD_tree_list <- list()

# Kontynuacja skryptu

# Inicjalizacja KD-Tree
for (i in 1:16) {
  basic_exposure_list[[i]] <- matrix(nrow = number_of_basic_exposure[i, 1], ncol = 12)
  exposure_latitude[[i]] <- vector("numeric", length = number_of_basic_exposure[i, 1])
  exposure_longitude[[i]] <- vector("numeric", length = number_of_basic_exposure[i, 1])
  KD_tree_list[[i]] <- vector("list", length = 31)
}

# Wykonaj obliczenia z wykorzystaniem KD-Tree
for (i in 1:16) {
  exposure_data_i <- exposure_data[sum(number_of_basic_exposure[1:(i - 1), 1]) + 1:sum(number_of_basic_exposure[1:i, 1]), ]
  
  # Przypisz dane
  basic_exposure_list[[i]] <- as.matrix(exposure_data_i)
  exposure_latitude[[i]] <- as.numeric(basic_exposure_list[[i]][, 2])
  exposure_longitude[[i]] <- as.numeric(basic_exposure_list[[i]][, 3])
  
  for (j in 1:31) {
    exposure_data_ij <- exposure_data_i[which(policy_start_month == (j - 1) & policy_end_month == j), ]
    coordinates_ij <- cbind(as.numeric(gsub(",", ".", exposure_data_ij[, 2])), as.numeric(gsub(",", ".", exposure_data_ij[, 3])))
    exposure_latitude_ij <- as.numeric(coordinates_ij[, 1])
    exposure_longitude_ij <- as.numeric(coordinates_ij[, 2])
    
    # Utwórz KD-Tree
    KD_tree_list[[i]][[j]] <- create_KD_tree(exposure_latitude_ij, exposure_longitude_ij)
  }
}

# Wykonaj symulacje pożaru
result <- run_fire_simulations(basic_exposure_list, exposure_latitude, exposure_longitude,
                                fire_probs, fire_spread_probabilities, number_of_simulations)

# Zakończenie pomiaru czasu
end_time <- Sys.time()
execution_time <- end_time - start_time

# Wyświetlenie wyników
print(paste("Czas wykonania: ", as.numeric(execution_time), " sekund"))
print(paste("Wynik: ", result))
