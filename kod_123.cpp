

// [[Rcpp::export]]
void testIterativeFactorial()
{
  const size_t size = 20; 
  std::vector<unsigned long long> results(size);
  
  
  for (size_t i = 0; i < size; ++i) {
    results[i] = factorial(i + 10);
  }
}
