for(int i = 0; i < liczba_watkow; ++i) {
  int start = i * rozmiar_danych_do_przetworzenia_przez_watek;
  int koniec = std::min(start + rozmiar_danych_do_przetworzenia_przez_watek, n);
  
  
  
  
  pool.push([&](int start, int koniec, int threadIndex) {
    for(int j = start; j < koniec; ++j) {
      bool logical_value = !((exposure_longitude[j] > east_lon) || (exposure_longitude[j] < west_lon) || (exposure_latitude[j] < south_lat) || (exposure_latitude[j] > north_lat));
      if(logical_value) {
        
        
        vect[i] = ({exposure_latitude[j],exposure_longitude[j], exposure_insurance[j], exposure_reassurance[j] , exposure_sum_value[j]});//(i);
      }
    }
    
    
    vect.push_ranges({30000*threadIndex, 30000*threadIndex+koniec});
    
    
    
    
  }, start, koniec, i);
}




pool.wait();