std::vector<ThreadData> threadDataVec(liczba_watkow);


for(int i = 0; i < liczba_watkow; ++i) {
  int start = i * rozmiar_danych_do_przetworzenia_przez_watek;
  int koniec = std::min(start + rozmiar_danych_do_przetworzenia_przez_watek, n);
  
  
  pool.push([&](int start, int koniec, int threadIndex) {
    for(int j = start; j < koniec; ++j) {
      bool logical_value = !((exposure_longitude[j] > east_lon) || (exposure_longitude[j] < west_lon) || (exposure_latitude[j] < south_lat) || (exposure_latitude[j] > north_lat));
      if(logical_value) {
        threadDataVec[threadIndex].lat_sub.push_back(exposure_latitude[j]);
        threadDataVec[threadIndex].lon_sub.push_back(exposure_longitude[j]);
        threadDataVec[threadIndex].insu_sub.push_back(exposure_insurance[j]);
        threadDataVec[threadIndex].reas_sub.push_back(exposure_reassurance[j]);
        threadDataVec[threadIndex].premium_sub.push_back(exposure_sum_value[j]);
      }
    }
  }, start, koniec, i);
}


pool.wait();


// Scalamy dane z wszystkich wątków
for (const auto& threadData : threadDataVec) {
  lat_sub.insert(lat_sub.end(), threadData.lat_sub.begin(), threadData.lat_sub.end());
  lon_sub.insert(lon_sub.end(), threadData.lon_sub.begin(), threadData.lon_sub.end());
  insu_sub.insert(insu_sub.end(), threadData.insu_sub.begin(), threadData.insu_sub.end());
  reas_sub.insert(reas_sub.end(), threadData.reas_sub.begin(), threadData.reas_sub.end());
  premium_sub.insert(premium_sub.end(), threadData.premium_sub.begin(), threadData.premium_sub.end());
}
