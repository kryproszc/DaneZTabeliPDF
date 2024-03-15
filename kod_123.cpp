#include <iostream>
#include <vector>
#include <chrono>


#include <iostream>
#include <vector>
#include <chrono>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>


class FlatVector {
private:
  std::vector<double> data;
  size_t sizeX, sizeY, sizeZ;
  
  
public:
  FlatVector(size_t x, size_t y, size_t z) : sizeX(x), sizeY(y), sizeZ(z), data(x * y * z) {}
  
  
  double& operator()(size_t x, size_t y, size_t z) {
    return data[x * sizeY * sizeZ + y * sizeZ + z];
  }
};


int main() {
  std::vector<std::vector<std::vector<double>>> vect; 
  vect.resize(100); 
  
  
  auto start = std::chrono::high_resolution_clock::now();
  
  
  for (int i = 0; i < 100; ++i) {
    vect[i].resize(100); 
    for (int j = 0; j < 100; ++j) {
      
      
      for (int k = 0; k < 100000; ++k) {
        vect[i][j].emplace_back(i * j * k); 
      }
    }
  }
  
  
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time taken by Triple Vector: " 
            << duration.count() << " microseconds" << std::endl;
  
  
  FlatVector vectt(100, 100, 100000);
  start = std::chrono::high_resolution_clock::now();
  
  
  for (int i = 0; i < 100; ++i) {
    for (int j = 0; j < 100; ++j) {
      for (int k = 0; k < 100000; ++k) {
        vectt(i, j, k) = i * j * k;
      }
    }
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time taken by Flat Vector:   " 
            << duration.count() << " microseconds" << std::endl;
  
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 99);
  std::uniform_int_distribution<> dis_k(0, 99999);
  
  
  
  
  
  start = std::chrono::high_resolution_clock::now();
  for (int n = 0; n < 10000; ++n) { 
    int i = dis(gen), j = dis(gen), k = dis_k(gen);
    auto val = vect[i][j][k]; 
  }
  stop = std::chrono::high_resolution_clock::now();
  auto durationVect = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  
  
  start = std::chrono::high_resolution_clock::now();
  for (int n = 0; n < 10000; ++n) { 
    int i = dis(gen), j = dis(gen), k = dis_k(gen);
    auto val = vectt(i, j, k); 
  }
  stop = std::chrono::high_resolution_clock::now();
  auto durationFlatVect = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  
  
  std::cout << "Random access time for Triple Vector: " << durationVect.count() << " microseconds" << std::endl;
  std::cout << "Random access time for Flat Vector:   " << durationFlatVect.count() << " microseconds" << std::endl;
  
  
}
