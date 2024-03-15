// [[Rcpp::export]]
void testThreadPoolFactorial()
{
  ThreadPool pool;
  const size_t size = 20;
  std::vector<unsigned long long> results(size);
  
  
  auto factorialOperation = [&](size_t i) -> void {
    results[i] = factorial(i + 10); 
  };
  
  
  for (size_t i = 0; i < size; ++i) {
    pool.push(factorialOperation, i);
  }
  
  
  pool.join(); 