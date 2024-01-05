#include <Rcpp.h>
using namespace Rcpp;

// Funkcja do obliczania odległości
// [[Rcpp::export]]
NumericMatrix distance_calculation(NumericVector lati, NumericVector lat2, NumericVector long1, NumericVector long2) {
  int n = lati.size();
  NumericMatrix distances(n, n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double dlat = (lat2[j] - lati[i]) * (M_PI / 180);
      double dlong = (long2[j] - long1[i]) * (M_PI / 180);
      double a = sin(dlat / 2) * sin(dlat / 2) + cos(lati[i] * (M_PI / 180)) * cos(lat2[j] * (M_PI / 180)) * sin(dlong / 2) * sin(dlong / 2);
      double c = 2 * atan2(sqrt(a), sqrt(1 - a));
      distances(i, j) = 6371000 * c; // Promień Ziemi w metrach
    }
  }

  return distances;
}

// Funkcja do symulacji rozprzestrzeniania się pożaru
// [[Rcpp::export]]
List fire_spread_simulation(NumericVector lat, NumericVector lon, NumericVector exposure,
                            NumericVector exposure_lat, NumericVector exposure_lon, NumericVector fire_spread_probs) {
  int n = lat.size();
  int m = exposure_lat.size();
  
  // Tworzenie listy do przechowywania wyników
  List result(n);
  
  for (int i = 0; i < n; i++) {
    // Sprawdzenie, czy warunek (np. 1001 > 1000) jest spełniony
    if (1001 > 1000) {
      // Inicjalizacja struktur danych
      std::vector<std::vector<double>> ring_exposure_list(9);
      std::vector<int> ring_exposure_numbers(9, 0);
      
      // Przetwarzanie danych o odległościach budynków (zastąpić rzeczywistą logiką)
      NumericVector dist = exposure_lon; // Zastąpić rzeczywistą logiką obliczania odległości
      
      for (int l = 0; l < m; l++) {
        if (!NumericVector::is_na(dist[l])) {
          if (dist[l] == 0) {
            ring_exposure_numbers[0]++;
            ring_exposure_list[0].push_back(exposure[l]);
          } else {
            for (int j = 1; j < 9; j++) {
              double lower_bound = 25 * (j - 1);
              double upper_bound = 25 * j;
              if (dist[l] > lower_bound && dist[l] <= upper_bound) {
                ring_exposure_numbers[j]++;
                ring_exposure_list[j].push_back(exposure[l]);
                break; // Wyjście z pętli po dodaniu do odpowiedniego pierścienia
              }
            }
          }
        }
      }
      
      // Tworzenie wektorów do przechowywania wyników
      NumericVector number_of_fire_spreads(9);
      List fire_spreads_rings_list(9);
      
      for (int j = 0; j < 9; j++) {
        number_of_fire_spreads[j] = R::rbinom(1, ring_exposure_numbers[j], fire_spread_probs[j]);
        if (number_of_fire_spreads[j] > 0) {
          NumericVector sampled_values = wrap(rsample(ring_exposure_list[j], number_of_fire_spreads[j], false));
          fire_spreads_rings_list[j] = sampled_values;
        }
      }
      
      // Zapisywanie wyników w liście wyników
      List simulation_result = List::create(_["number_of_spreads"] = number_of_fire_spreads,
                                           _["spreads_list"] = fire_spreads_rings_list);
      
      result[i] = simulation_result;
    } else {
      // Jeśli warunek nie jest spełniony, zwróć pustą listę
      result[i] = List::create(_["number_of_spreads"] = 0,
                               _["spreads_list"] = NumericVector::create());
    }
  }
  
  return result;
}

// Funkcja do symulacji stochastycznego procesu pożaru
// [[Rcpp::export]]
List stochastic_fire_process_simulation(List exposure_list, List exposure_latitude, List exposure_longitude,
                                         List fire_probs_list, NumericVector fire_spread_probabilities,
                                         List kd_trees_vov) {
  int n_months = exposure_list.size();
  
  // Tworzenie wektorów i list do przechowywania wyników
  NumericVector exposure_number(n_months);
  List total_fire_list(n_months);
  NumericVector number_of_fires(n_months);
  List fire_sources_list(n_months);
  NumericVector number_of_fire_spreads;
  
  for (int i = 0; i < n_months; i++) {
    // Pobieranie danych dla bieżącego miesiąca
    NumericVector exposure = exposure_list[i];
    NumericVector exposure_lat = exposure_latitude[i];
    NumericVector exposure_lon = exposure_longitude[i];
    NumericVector fire_probs = fire_probs_list[i];
    
    List kd_trees_voiv_month = kd_trees_vov[i];
    
    exposure_number[i] = exposure.size();
    
    number_of_fires[i] = R::rbinom(1, exposure_number[i], fire_probs[0]); // Zakładając, że chcesz pierwsze prawdopodobieństwo
    
    fire_sources_list[i] = sample(exposure, number_of_fires[i], false);
    
    if (i == 0) {
      total_fire_list[i] = fire_sources_list[i];
    } else {
      fire_sources_list[i] = fire_sources_list[i][!in_element(fire_sources_list[i], total_fire_list[i - 1])];
      total_fire_list[i] = union_element(total_fire_list[i - 1], fire_sources_list[i]);
    }
    
    int n_sources = fire_sources_list[i].size();
    for (int m = 0; m < n_sources; m++) {
      double lat = exposure_lat[fire_sources_list[i][m]];
      double lon = exposure_lon[fire_sources_list[i][m]];
      
      // Wywołanie funkcji fire_spread_simulation w C++
      List fire_spreads = fire_spread_simulation(lat, lon, exposure, exposure_lat, exposure_lon, fire_spread_probabilities);
      
      List fire_spreads_list = fire_spreads[1];
      int fire_spreads_numbers = fire_spreads_list.size();
      
      number_of_fire_spreads.push_back(fire_spreads_numbers);
      
      total_fire_list[i] = union_element(total_fire_list[i], fire_spreads_list);
    }
  }
  
  List stochastic_fire_process_simulation = List::create(_["number_of_fire_spreads"] = number_of_fire_spreads);
  return stochastic_fire_process_simulation;
}

// Funkcja do uruchamiania symulacji pożaru
// [[Rcpp::export]]
NumericVector run_fire_simulations(List basic_exposure_list, List exposure_latitude, List exposure_longitude,
                                    List fire_probs, NumericVector fire_spread_probs, int number_of_simulations) {
  int num_cores = parallel::detectCores() - 1;
  parallel::makeCluster(num_cores, type = "PSOCK");
  doParallel::registerDoParallel(cl = num_cores);
  NumericVector result(number_of_simulations);
  
  for (int n = 0; n < number_of_simulations; ++n) {
    List fire_simulations;
    for (int k = 0; k < 16; ++k) {
      List simulation = stochastic_fire_process_simulation(basic_exposure_list[k], exposure_latitude[k],
                                                           exposure_longitude[k], fire_probs[k],
                                                           fire_spread_probs, kd_trees_vov[k]);
      fire_simulations.push_back(simulation);
    }
    
    double sum_fires = 0;
    for (int k = 0; k < 16; ++k) {
      NumericVector fires = fire_simulations[k];
      sum_fires += sum(fires);
    }
    result[n] = sum_fires;
  }
  parallel::stopImplicitCluster();
  return result;
}
