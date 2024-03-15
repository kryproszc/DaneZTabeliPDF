auto start = std::chrono::high_resolution_clock::now();





  
  auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = end - start;
Rcpp::Rcout << "ThreadPool time: " << elapsed.count() << " s\n";








start = std::chrono::high_resolution_clock::now();




wywolanie funkcji  iteracyjnej
  
  
  end = std::chrono::high_resolution_clock::now();
elapsed = end - start;
Rcpp::Rcout << "Iterative time: " << elapsed.count() << " s\n";