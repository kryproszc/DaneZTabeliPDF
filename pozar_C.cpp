#include <Rcpp.h>
using namespace Rcpp;

function fire_spread_simulation(lat, long, exposure, exposure_lat, exposure_long) {
  var long_arc = Math.acos(1 - (200 ** 2) / (2 * 4371000 ** 2 * (Math.PI / 2 - lat) ** 2));
  var exposure_helper = exposure_lat.map(function (value, index) {
    return (value - lat + 200 / 6371000) && (value >= lat - 200 / 6371000) && (exposure_long[index] <= long + long_arc) && (exposure_long[index] >= long - long_arc);
  });
  var exposure_spread_square = exposure.filter(function (value, index) {
    return exposure_helper[index];
  });
  if (exposure_spread_square.length > 1000) {
    var exposure_lat_spread_square = exposure_lat.filter(function (value, index) {
      return exposure_helper[index];
    });
    var exposure_long_spread_square = exposure_long.filter(function (value, index) {
      return exposure_helper[index];
    });
    var ring_exposure_list = [];
    var ring_exposure_numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (var j = 0; j < 9; j++) {
      ring_exposure_list[j] = [];
    }
    for (var i = 0; i < exposure_spread_square.length; i++) {
      var dist = distance_calculation(lat1 = lat, lat2 = exposure_lat_spread_square[i], long1 = long, long2 = exposure_long_spread_square[i]);
      if (!isNaN(dist)) {
        if (dist == 0) {
          ring_exposure_numbers[i] += 1;
          ring_exposure_list[i].push(exposure_spread_square[i]);
        } else {
          for (var j = 0; j < 8; j++) {
            if (dist > 25 * (j - 1) && dist <= 25 * j) {
              ring_exposure_numbers[j + 1] += 1;
              ring_exposure_list[j + 1].push(exposure_spread_square[i]);
            }
          }
        }
      }
    }
    var number_of_fire_spreads = [];
    var fire_spreads_rings_list = [];
    for (var j = 0; j < 9; j++) {
      number_of_fire_spreads[j] = rbinom(n = 1, size = ring_exposure_numbers[j], prob = fire_spread_probs[j]);
      if (number_of_fire_spreads[j] > 0) {
        fire_spreads_rings_list[j] = sample(x = ring_exposure_list[j], size = number_of_fire_spreads[j], replace = false);
      }
    }
    var fire_spread_simulation = [fire_spreads_rings_list.flat().length, fire_spreads_rings_list.flat()];
  } else {
    var fire_spread_simulation = [0, []];
  }
  return fire_spread_simulation;
}




#include <iostream>
#include <vector>
#include <algorithm>

std::vector<int> stochastic_fire_process_simulation(std::vector<std::vector<int>> exposure_list, std::vector<int> exposure_latitude, std::vector<int> exposure_longitude, std::vector<double> fire_probs_list, std::vector<double> fire_spread_probabilities) {
  std::vector<int> exposure_number(12);
  std::vector<std::vector<int>> total_fire_list(12);
  std::vector<int> number_of_fires(12);
  std::vector<std::vector<int>> fire_sources_list(12);
  std::vector<int> number_of_fire_spreads;
  
  for (int i = 0; i < 12; i++) {
    exposure_number[i] = exposure_list[i].size();
    number_of_fires[i] = rbinom(1, exposure_number[i], fire_probs_list[i]);
    fire_sources_list[i] = sample(exposure_list[i], number_of_fires[i], false);
    
    if (i == 0) {
      total_fire_list[i] = fire_sources_list[i];
    } else {
      std::vector<int> temp = fire_sources_list[i];
      fire_sources_list[i].erase(std::remove_if(fire_sources_list[i].begin(), fire_sources_list[i].end(), [&](int x) { return std::find(total_fire_list[i-1].begin(), total_fire_list[i-1].end(), x) != total_fire_list[i-1].end(); }), fire_sources_list[i].end());
      total_fire_list[i] = total_fire_list[i-1];
      total_fire_list[i].insert(total_fire_list[i].end(), fire_sources_list[i].begin(), fire_sources_list[i].end());
    }
    
    if (fire_sources_list[i].size() > 0) {
      for (int m = 0; m < fire_sources_list[i].size(); m++) {
        if (!std::isnan(latitude[fire_sources_list[i][m]]) && !std::isnan(longitude[fire_sources_list[i][m]])) {
          std::vector<std::vector<int>> fire_spreads = fire_spread_simulation(latitude[fire_sources_list[i][m]], longitude[fire_sources_list[i][m]], exposure_list[i], exposure_latitude[i], exposure_longitude[i]);
          std::vector<int> fire_spreads_list = fire_spreads[1];
          fire_spreads_list.erase(std::remove_if(fire_spreads_list.begin(), fire_spreads_list.end(), [&](int x) { return std::find(total_fire_list[i].begin(), total_fire_list[i].end(), x) != total_fire_list[i].end(); }), fire_spreads_list.end());
          int fire_spreads_numbers = fire_spreads[1].size();
          number_of_fire_spreads.push_back(fire_spreads_numbers);
          total_fire_list[i].insert(total_fire_list[i].end(), fire_spreads_list.begin(), fire_spreads_list.end());
        }
      }
    }
  }
  
  return number_of_fire_spreads;
}

int main() {
  // Define input variables
  std::vector<std::vector<int>> exposure_list;
  std::vector<int> exposure_latitude;
  std::vector<int> exposure_longitude;
  std::vector<double> fire_probs_list;
  std::vector<double> fire_spread_probabilities;
  
  // Call the function
  std::vector<int> result = stochastic_fire_process_simulation(exposure_list, exposure_latitude, exposure_longitude, fire_probs_list, fire_spread_probabilities);
  
  // Print the result
  for (int i = 0; i < result.size(); i++) {
    std::cout << result[i] << " ";
  }
  
  return 0;
}




