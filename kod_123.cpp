

auto start = std::chrono::high_resolution_clock::now();


struct Data {
  double lat_sub;
  double lon_sub;
  double insu_sub;
  double reas_sub;
  double premium_sub;
}


struct ThreadRange {
  int start;
  int end;
  
  
};


template<typename T, size_t Size>
class CustomVect {
public:
  std::vector<ThreadRange> ranges ;
  mutable std::mutex mutex;
  CustomVect() : size(0) {}
  
  
  void push_back(const T& value)  noexcept {
    data[size++] = value;
  }
  
  
  void push_ranges(const ThreadRange& value) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    ranges.push_back(value);
    std::sort(ranges.begin(), ranges.end(), [](const ThreadRange& a, const ThreadRange& b) {
      return a.start < b.start;
    });
  }
  
  
  void printRanges() const {
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto& range : ranges) {
      std::cout << "Start: " << range.start << ", End: " << range.end << std::endl;
    }
  }
  T& operator[](size_t index) {
    
    
    return data[index];
  }
  
  
  size_t getSize() const {
    return size;
  }
  
  
private:
  size_t size;
  
  
  T data[Size];
};






CustomVect<Data, 400000> vect;


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




auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = end - start;
Rcpp::Rcout << "ThreadPool time: " << elapsed.count() << " s\n";