// [[Rcpp::export]]
void testThreadPoolParallelForNumericVector()
{
  ThreadPool pool; // tworzy pulę wątków
  NumericVector x(1000000000, 1.0); // wektor danych typu NumericVector
  
  
  auto dummy = [&](size_t i) -> void {
    x[i] = 2 * x[i]; // operacja na elemencie wektora
  };
  
  
  pool.parallelFor(0, x.size(), dummy); // wykonanie operacji równolegle
  pool.join(); // oczekiwanie na zakończenie wszystkich wątków
}
