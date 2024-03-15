

// [[Rcpp::export]]
void testIterative()
{
  std::vector<size_t> x(1000000, 1);
  for(size_t i = 0; i < x.size(); ++i) {
    x[i] = 2 * x[i];
  }
}




// [[Rcpp::export]]
void testThreadPoolParallelFor()
{
  ThreadPool pool; // tworzy pulę wątków
  std::vector<size_t> x(1000000, 1); // wektor danych
  
  
  auto dummy = [&](size_t i) -> void {
    x[i] = 2 * x[i]; // operacja na elemencie wektora
  };
  
  
  pool.parallelFor(0, x.size(), dummy); // wykonanie operacji równolegle
  pool.join(); // oczekiwanie na zakończenie wszystkich wątków
}




// mierzymy czas
system.time(testIterative())
system.time(testThreadPoolParallelFor())