#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <iterator>
#include <random>
#include <numeric>
#include <sstream>

using namespace std;

const unsigned int fixedSeed = 123456789;
std::mt19937 gen(fixedSeed);
std::uniform_real_distribution<double> distribution;

struct Point
{
  long double x, y;
  int insu;
  int reas;
  double premium;
  
  Point(long double x, long double y, int insu, int reas, double premium)
  {
    this->x = x;
    this->y = y;
    this->insu = insu;
    this->reas = reas;
    this->premium = premium;
  }
};

struct Rect
{
  double x1, y1, x2, y2;
  
  double width() const { return std::abs(x2 - x1); }
  double height() const { return std::abs(y2 - y1); }
  
  bool contains(const Point &p) const
  {
    return (p.x >= std::min(x1, x2) && p.x <= std::max(x1, x2) &&
            p.y >= std::min(y1, y2) && p.y <= std::max(y1, y2));
  }
  
  bool intersects(const Rect &range) const
  {
    return !(range.x1 > std::max(x1, x2) || range.x2 < std::min(x1, x2) ||
             range.y1 > std::max(y1, y2) || range.y2 < std::min(y1, y2));
  }
  bool contains(const Rect &other) const
  {
    return (std::min(x1, x2) <= std::min(other.x1, other.x2) &&
            std::max(x1, x2) >= std::max(other.x1, other.x2) &&
            std::min(y1, y2) <= std::min(other.y1, other.y2) &&
            std::max(y1, y2) >= std::max(other.y1, other.y2));
  }
};

class Grid2D
{
private:
  int rows, cols;
  Rect bounds;
  std::vector<std::vector<std::vector<Point>>> grid;
  
public:
  Grid2D(int rows, int cols, Rect bounds) : rows(rows), cols(cols), bounds(bounds)
  {
    grid.resize(rows, std::vector<std::vector<Point>>(cols));
  }
  
  bool insert(const Point &p)
  {
    if (!bounds.contains(p))
      return false;
    
    int col = static_cast<int>((p.x - bounds.x1) / ((bounds.x2 - bounds.x1) / cols));
    int row = static_cast<int>((p.y - bounds.y1) / ((bounds.y2 - bounds.y1) / rows));
    if (row >= 0 && row < rows && col >= 0 && col < cols)
    {
      grid[row][col].push_back(p);
      return true;
    }
    return false;
  }
  
  void getPointsInRange(const Rect &range, std::vector<long double> &lat_sub,
                        std::vector<long double> &lon_sub,
                        std::vector<int> &insu_sub,
                        std::vector<int> &reas_sub,
                        std::vector<double> &premium_sub) const
  {
    int startCol = std::max(0, int((range.x1 - bounds.x1) / ((bounds.x2 - bounds.x1) / cols)));
    int endCol = std::min(cols - 1, int((range.x2 - bounds.x1) / ((bounds.x2 - bounds.x1) / cols)));
    int startRow = std::max(0, int((range.y1 - bounds.y1) / ((bounds.y2 - bounds.y1) / rows)));
    int endRow = std::min(rows - 1, int((range.y2 - bounds.y1) / ((bounds.y2 - bounds.y1) / rows)));
    for (int row = startRow; row <= endRow; ++row)
    {
      for (int col = startCol; col <= endCol; ++col)
      {
        for (const auto &p : grid[row][col])
        {
          if (range.contains(p))
          {
            lat_sub.push_back(p.x);
            lon_sub.push_back(p.y);
            insu_sub.push_back(p.insu);
            reas_sub.push_back(p.reas);
            premium_sub.push_back(p.premium);
          }
        }
      }
    }
  }
  std::vector<Point> query(const Rect &range)
  {
    std::vector<Point> result;
    int startCol = std::max(0, int((range.x1 - bounds.x1) / ((bounds.x2 - bounds.x1) / cols)));
    int endCol = std::min(cols - 1, int((range.x2 - bounds.x1) / ((bounds.x2 - bounds.x1) / cols)));
    int startRow = std::max(0, int((range.y1 - bounds.y1) / ((bounds.y2 - bounds.y1) / rows)));
    int endRow = std::min(rows - 1, int((range.y2 - bounds.y1) / ((bounds.y2 - bounds.y1) / rows)));
    for (int row = startRow; row <= endRow; ++row)
    {
      for (int col = startCol; col <= endCol; ++col)
      {
        for (const auto &p : grid[row][col])
        {
          if (range.contains(p))
          {
            result.push_back(p);
          }
        }
      }
    }
    return result;
  }
};

class QuadTree
{
private:
  static const int CAPACITY = 5000;
  Rect boundary;
  std::vector<Point> points;
  QuadTree *nw;
  QuadTree *ne;
  QuadTree *sw;
  QuadTree *se;
  bool divided;
  
public:
  QuadTree(Rect boundary) : boundary(boundary), divided(false), nw(nullptr), ne(nullptr), sw(nullptr), se(nullptr)
  {
    points.reserve(10000);
  }
  
  ~QuadTree()
  {
    delete nw;
    delete ne;
    delete sw;
    delete se;
  }
  
  bool insert(const Point &p)
  {
    if (!boundary.contains(p))
    {
      return false;
    }
    
    if (points.size() < CAPACITY)
    {
      points.emplace_back(p);
      return true;
    }
    
    if (!divided)
    {
      subdivide();
    }
    
    if (p.x < (boundary.x1 + boundary.x2) / 2)
    {
      if (p.y < (boundary.y1 + boundary.y2) / 2)
      {
        return nw->insert(p);
      }
      else
      {
        return sw->insert(p);
      }
    }
    else
    {
      if (p.y < (boundary.y1 + boundary.y2) / 2)
      {
        return ne->insert(p);
      }
      else
      {
        return se->insert(p);
      }
    }
  }
  
  void subdivide()
  {
    double midX = (boundary.x1 + boundary.x2) / 2;
    double midY = (boundary.y1 + boundary.y2) / 2;
    
    Rect nwBoundary = {boundary.x1, boundary.y1, midX, midY};
    Rect neBoundary = {midX, boundary.y1, boundary.x2, midY};
    Rect swBoundary = {boundary.x1, midY, midX, boundary.y2};
    Rect seBoundary = {midX, midY, boundary.x2, boundary.y2};
    
    nw = new QuadTree(nwBoundary);
    ne = new QuadTree(neBoundary);
    sw = new QuadTree(swBoundary);
    se = new QuadTree(seBoundary);
    
    divided = true;
  }
  
  void getPointsInRange(const Rect &range, std::vector<long double> &lat_sub,
                        std::vector<long double> &lon_sub,
                        std::vector<int> &insu_sub,
                        std::vector<int> &reas_sub,
                        std::vector<double> &premium_sub) const
  {
    
    if (!boundary.intersects(range))
    {
      return;
    }
    
    for (const Point &p : points)
    {
      if (range.contains(p))
      {
        lat_sub.push_back(p.x);
        lon_sub.push_back(p.y);
        insu_sub.push_back(p.insu);
        reas_sub.push_back(p.reas);
        premium_sub.push_back(p.premium);
      }
    }
    
    if (divided)
    {
      nw->getPointsInRange(range, lat_sub, lon_sub, insu_sub, reas_sub, premium_sub);
      ne->getPointsInRange(range, lat_sub, lon_sub, insu_sub, reas_sub, premium_sub);
      sw->getPointsInRange(range, lat_sub, lon_sub, insu_sub, reas_sub, premium_sub);
      se->getPointsInRange(range, lat_sub, lon_sub, insu_sub, reas_sub, premium_sub);
    }
  }
};

QuadTree &getQuadTree(std::vector<QuadTree> &quadtreeVector, int woj, int month)
{
  int index = woj * 12 + month;
  return quadtreeVector[index];
}

Rect boundary = {49.0, 16.07, 54.5, 24.09};
std::vector<QuadTree> quadtree(17 * 12, QuadTree(boundary));
std::vector<std::vector<std::vector<long double>>> exposure_longitude;
std::vector<std::vector<std::vector<long double>>> exposure_latitude;
std::vector<std::vector<long double>> fire_probs;
std::vector<std::vector<long double>> fire_spread_prob_vec;
std::vector<long double> conditional_mean_trend_parameters;
std::vector<long double> conditional_Cov;
std::vector<std::vector<std::vector<int>>> exposure_insurance;
std::vector<std::vector<std::vector<int>>> exposure_reassurance;
std::vector<std::vector<std::vector<double>>> exposure_sum_value;
std::vector<std::vector<double>> wielkosc_pozaru;
std::vector<std::vector<double>> fakultatywna_input_num;
std::vector<std::vector<std::vector<double>>> fakultatywna_input_val;
std::vector<std::vector<double>> obligatoryjna_input_risk;
std::vector<std::vector<double>> vec_obligat_insur_event;
int sim;
int kat_val;
int ilosc_ubezpieczycieli;

class ThreadPool
{
public:
  ThreadPool(size_t num_threads = thread::hardware_concurrency())
  {
    for (size_t i = 0; i < num_threads; ++i)
    {
      threads_.emplace_back([this]
      {
        while (true) {
          function<void()> task;
          {
            unique_lock<mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] {
              return !tasks_.empty() || stop_;
            });
            
            if (stop_ && tasks_.empty()) {
              return;
            }
            
            task = move(tasks_.front());
            tasks_.pop();
          }
          
          task();
          {
            unique_lock<mutex> lock(queue_mutex_);
            --tasks_count_;
            if (tasks_count_ == 0) {
              cv_task_done_.notify_one(); 
            }
          }
        } });
    }
  }
  
  ~ThreadPool()
  {
    {
      unique_lock<mutex> lock(queue_mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &thread : threads_)
    {
      thread.join();
    }
  }
  
  void enqueue(function<void()> task)
  {
    {
      unique_lock<mutex> lock(queue_mutex_);
      tasks_.emplace(move(task));
      ++tasks_count_;
    }
    cv_.notify_one();
  }
  
  void wait_for_tasks()
  {
    unique_lock<mutex> lock(queue_mutex_);
    cv_task_done_.wait(lock, [this]()
    { return tasks_count_ == 0; });
  }
  
private:
  vector<thread> threads_;
  queue<function<void()>> tasks_;
  mutex queue_mutex_;
  condition_variable cv_;
  condition_variable cv_task_done_;
  atomic<size_t> tasks_count_ = 0;
  bool stop_ = false;
};

ThreadPool pool(thread::hardware_concurrency() / 2);

namespace sftrabbit
{

template <typename RealType = double>
class beta_distribution
{
public:
  typedef RealType result_type;
  
  class param_type
  {
  public:
    typedef beta_distribution distribution_type;
    
    explicit param_type(RealType a = 2.0, RealType b = 2.0)
      : a_param(a), b_param(b) {}
    
    RealType a() const { return a_param; }
    RealType b() const { return b_param; }
    
    bool operator==(const param_type &other) const
    {
      return (a_param == other.a_param &&
              b_param == other.b_param);
    }
    
    bool operator!=(const param_type &other) const
    {
      return !(*this == other);
    }
    
  private:
    RealType a_param, b_param;
  };
  
  explicit beta_distribution(RealType a = 2.0, RealType b = 2.0)
    : a_gamma(a), b_gamma(b) {}
  explicit beta_distribution(const param_type &param)
    : a_gamma(param.a()), b_gamma(param.b()) {}
  
  void reset() {}
  
  param_type param() const
  {
    return param_type(a(), b());
  }
  
  void param(const param_type &param)
  {
    a_gamma = gamma_dist_type(param.a());
    b_gamma = gamma_dist_type(param.b());
  }
  
  template <typename URNG>
  result_type operator()(URNG &engine)
  {
    return generate(engine, a_gamma, b_gamma);
  }
  
  template <typename URNG>
  result_type operator()(URNG &engine, const param_type &param)
  {
    gamma_dist_type a_param_gamma(param.a()),
    b_param_gamma(param.b());
    return generate(engine, a_param_gamma, b_param_gamma);
  }
  
  result_type min() const { return 0.0; }
  result_type max() const { return 1.0; }
  
  RealType a() const { return a_gamma.alpha(); }
  RealType b() const { return b_gamma.alpha(); }
  
  bool operator==(const beta_distribution<result_type> &other) const
  {
    return (param() == other.param() &&
            a_gamma == other.a_gamma &&
            b_gamma == other.b_gamma);
  }
  
  bool operator!=(const beta_distribution<result_type> &other) const
  {
    return !(*this == other);
  }
  
private:
  typedef std::gamma_distribution<result_type> gamma_dist_type;
  
  gamma_dist_type a_gamma, b_gamma;
  
  template <typename URNG>
  result_type generate(URNG &engine,
                       gamma_dist_type &x_gamma,
                       gamma_dist_type &y_gamma)
  {
    result_type x = x_gamma(engine);
    return x / (x + y_gamma(engine));
  }
};

template <typename CharT, typename RealType>
std::basic_ostream<CharT> &operator<<(std::basic_ostream<CharT> &os,
                                      const beta_distribution<RealType> &beta)
{
  os << "~Beta(" << beta.a() << "," << beta.b() << ")";
  return os;
}

template <typename CharT, typename RealType>
std::basic_istream<CharT> &operator>>(std::basic_istream<CharT> &is,
                                      beta_distribution<RealType> &beta)
{
  std::string str;
  RealType a, b;
  if (std::getline(is, str, '(') && str == "~Beta" &&
      is >> a && is.get() == ',' && is >> b && is.get() == ')')
  {
    beta = beta_distribution<RealType>(a, b);
  }
  else
  {
    is.setstate(std::ios::failbit);
  }
  return is;
}

}

class VectorSim
{
public:
  std::vector<std::vector<double>> data;
  mutable std::mutex mtx;
  
  VectorSim() : data(30, std::vector<double>()) {}
  
  void addDataVec(int insurane, double value)
  {
    std::lock_guard<std::mutex> lock(mtx);
    
    data[insurane].push_back(value);
  }
  
  std::vector<std::vector<double>> returnVectorSim()
  {
    std::lock_guard<std::mutex> lock(mtx);
    
    return (data);
  }
  
  void clearVector(int num_vec)
  {
    
    data[num_vec].clear();
  }
};

class VectorPozarPierwotny
{
public:
  std::vector<std::vector<long double>> build_fire;
  VectorPozarPierwotny() : build_fire(9, std::vector<long double>()) {}
  
  void addPozarPierwotny(int insurancer, long double exposure_longitude_one,
                         long double exposure_latitude_one, int woj, int mies, double exposire_sum_one, int index_table, double wielkosc_pozar_kwota,
                         double reas_fire)
  {
    build_fire[0].push_back(insurancer);
    build_fire[1].push_back(exposure_longitude_one);
    build_fire[2].push_back(exposure_latitude_one);
    build_fire[3].push_back(woj + 1);
    build_fire[4].push_back(mies + 1);
    build_fire[5].push_back(exposire_sum_one);
    build_fire[6].push_back(index_table);
    build_fire[7].push_back(wielkosc_pozar_kwota);
    build_fire[8].push_back(reas_fire);
  }
  std::vector<std::vector<long double>> returnPozarPierwotny()
  {
    return (build_fire);
  }
};
class VectorPozarRozprzestrzeniony
{
public:
  std::vector<std::vector<double>> build_fire_rozprzestrzeniony;
  VectorPozarRozprzestrzeniony() : build_fire_rozprzestrzeniony(11, std::vector<double>()) {}
  
  void addPozarRozprzestrzeniony(std::vector<std::vector<double>> spread_one_building)
  {
    for (int ttt = 0; ttt < 11; ttt++)
    {
      build_fire_rozprzestrzeniony[ttt].insert(
          build_fire_rozprzestrzeniony[ttt].end(),
          spread_one_building[ttt].begin(),
          spread_one_building[ttt].end());
    }
  }
  std::vector<std::vector<double>> returnPozarRozprzestrzeniony()
  {
    return (build_fire_rozprzestrzeniony);
  }
};
struct Data
{
  double lat_sub;
  double lon_sub;
  double insu_sub;
  double reas_sub;
  double premium_sub;
};

int randBin(int size_exp, double prob_size)
{
  
  std::binomial_distribution<> distrib(size_exp, prob_size);
  
  return distrib(gen);
}

double randZeroToOne(int a, int b)
{
  
  distribution.param(std::uniform_real_distribution<double>::param_type(a, b));
  return distribution(gen);
}
std::vector<int> sample_vec(std::vector<int> &population, int sampleSize)
{
  std::vector<int> sampleData(sampleSize);
  
  for (int i = 0; i < sampleSize; i++)
  {
    int randomIndex = randZeroToOne(0.0, population.size() - 1);
    sampleData[i] = population[randomIndex];
  }
  return sampleData;
}

int search_closest(const std::vector<double> &sorted_array, double x)
{
  auto iter_geq = std::lower_bound(
    sorted_array.begin(),
    sorted_array.end(),
    x);
  
  if (iter_geq == sorted_array.begin())
  {
    
    return 0;
  }
  else if (iter_geq == sorted_array.end())
  {
    
    return sorted_array.size() - 1;
  }
  
  double a = *(iter_geq - 1);
  double b = *(iter_geq);
  
  if (fabs(x - a) < fabs(x - b))
  {
    return iter_geq - sorted_array.begin() - 1;
  }
  return iter_geq - sorted_array.begin();
}

double percentage_of_loss(std::vector<std::vector<double>> wielkosc_pozaru)
{
  int ind_prob;
  double exp_sen;
  double val_dist;
  val_dist = randZeroToOne(0, 1);
  std::vector<double> probability;
  probability = wielkosc_pozaru[0];
  std::vector<double> exponsure_sensitiv;
  exponsure_sensitiv = wielkosc_pozaru[1];
  ind_prob = search_closest(probability, val_dist);
  exp_sen = exponsure_sensitiv[ind_prob];
  return (exp_sen);
}

double calc_reas_bligator(std::vector<double> vec_obligat_insur_risk, double sum_prem)
{
  double out_obl = 0.0;
  if (sum_prem < vec_obligat_insur_risk[0])
  {
    out_obl = vec_obligat_insur_risk[2] * sum_prem;
  }
  else if (sum_prem > vec_obligat_insur_risk[0] && sum_prem < vec_obligat_insur_risk[1])
  {
    out_obl = vec_obligat_insur_risk[2] * vec_obligat_insur_risk[0];
  }
  else if (sum_prem > vec_obligat_insur_risk[1])
  {
    out_obl = sum_prem - (vec_obligat_insur_risk[1] - vec_obligat_insur_risk[0]);
  }
  return (out_obl);
}

double haversine_cpp(double lat1, double long1,
                     double lat2, double long2,
                     double earth_radius = 6378137)
{
  
  double distance;
  
  if (!((long1 > 360) || (long2 > 360) || (lat1 > 90) || (lat2 > 90)))
  {
    double deg_to_rad = 0.0174532925199432957;
    double delta_phi = (lat2 - lat1) * deg_to_rad;
    double delta_lambda = (long2 - long1) * deg_to_rad;
    double phi1 = lat1 * deg_to_rad;
    double phi2 = lat2 * deg_to_rad;
    double term1 = pow(sin(delta_phi * .5), 2);
    double term2 = cos(phi1) * cos(phi2) * pow(sin(delta_lambda * .5), 2);
    double delta_sigma = 2 * atan2(sqrt(term1 + term2), sqrt(1 - term1 - term2));
    distance = earth_radius * delta_sigma;
  }
  else
  {
    distance = NAN;
  }
  return distance;
}
std::vector<std::vector<double>> haversine_loop_cpp_vec(
    
    double radius,
    int n1,
    
    int woj, int mies)
  
{
  long double lat_center = exposure_latitude[woj][mies][n1];
  long double lon_center = exposure_longitude[woj][mies][n1];
  
  double south_lat = lat_center - radius;
  double north_lat = lat_center + radius;
  double west_lon = lon_center - radius;
  double east_lon = lon_center + radius;
  
  int n = exposure_longitude.size();
  
  std::vector<long double> lat_sub;
  std::vector<long double> lon_sub;
  std::vector<int> insu_sub;
  std::vector<int> reas_sub;
  std::vector<double> premium_sub;
  std::vector<int> ind_spread_build;
  std::vector<std::vector<double>> ind_ring(11);
  bool logical_value;
  double left = lat_center - radius;
  double top = lon_center - radius;
  for (int i = 0; i < n; i++)
  {
    logical_value = !((exposure_longitude[woj][mies][i] > east_lon) || (exposure_longitude[woj][mies][i] < west_lon) || (exposure_latitude[woj][mies][i] < south_lat) || (exposure_latitude[woj][mies][i] > north_lat));
    if (logical_value)
    {
      
      lat_sub.push_back(exposure_latitude[woj][mies][i]);
      lon_sub.push_back(exposure_longitude[woj][mies][i]);
      insu_sub.push_back(exposure_insurance[woj][mies][i]);
      reas_sub.push_back(exposure_reassurance[woj][mies][i]);
      premium_sub.push_back(exposure_sum_value[woj][mies][i]);
    }
  }
  
  if (lat_sub.size() > 0)
  {
    
    std::vector<std::vector<double>> distance_res(9);
    std::vector<std::vector<long double>> lat_ring(9);
    std::vector<std::vector<long double>> lon_ring(9);
    std::vector<std::vector<int>> insu_ring(9);
    std::vector<std::vector<int>> reas_ring(9);
    std::vector<std::vector<double>> exponsure_sum_ring(9);
    int n1 = lon_sub.size();
    if (n1 > 0)
    {
      for (int i = 0; i < n1; ++i)
      {
        double res = haversine_cpp(lat_center, lon_center, lat_sub[i], lon_sub[i]);
        int index = -1;
        
        if (res < 0.005)
        {
          index = 0;
        }
        else if (res >= 0.005 && res < 25)
        {
          index = 1;
        }
        else if (res >= 25 && res < 50)
        {
          index = 2;
        }
        else if (res >= 50 && res < 75)
        {
          index = 3;
        }
        else if (res >= 75 && res < 100)
        {
          index = 4;
        }
        else if (res >= 100 && res < 125)
        {
          index = 5;
        }
        else if (res >= 125 && res < 150)
        {
          index = 6;
        }
        else if (res >= 150 && res < 175)
        {
          index = 7;
        }
        else if (res >= 175 && res < 200)
        {
          index = 8;
        }
        
        if (index != -1)
        {
          distance_res[index].push_back(res);
          lat_ring[index].push_back(lat_sub[i]);
          lon_ring[index].push_back(lon_sub[i]);
          insu_ring[index].push_back(insu_sub[i]);
          reas_ring[index].push_back(reas_sub[i]);
          exponsure_sum_ring[index].push_back(premium_sub[i]);
        }
      }
    }
    
    int exposure_number;
    std::vector<double> out_distance;
    
    std::vector<std::vector<double>> out_data(11);
    std::vector<int> number_of_fire_spreads(9);
    std::vector<int> fire_spreads_indicator(9);
    std::vector<double> conditional_mean(9);
    std::vector<double> alpha(9);
    std::vector<double> beta(9);
    std::vector<double> simulated_probability(9);
    std::vector<std::vector<int>> fire_spreads_rings_list(9);
    for (int j = 0; j < 9; j++)
    {
      std::vector<int> ring_exposure_list(lat_ring[j].size());
      std::iota(std::begin(ring_exposure_list), std::end(ring_exposure_list), 0);
      if (lat_ring[j].size() > 0)
      {
        if (j == 0)
        {
          fire_spreads_indicator[j] = randBin(1, fire_spread_prob_vec[0][j]);
        }
        else if (j == 1)
        {
          fire_spreads_indicator[j] = randBin(1, fire_spread_prob_vec[0 + fire_spreads_indicator[j - 1]][j]);
        }
        else
        {
          fire_spreads_indicator[j] = randBin(1, fire_spread_prob_vec[0 + 2 * fire_spreads_indicator[j - 1] + fire_spreads_indicator[j - 2]][j]);
        }
        if (fire_spreads_indicator[j] > 0)
        {
          if (lat_ring[j].size() == 1)
          {
            number_of_fire_spreads[j] = 1;
            fire_spreads_rings_list[j] = sample_vec(ring_exposure_list, number_of_fire_spreads[j]);
          }
          else if (lat_ring[j].size() == 2)
          {
            conditional_mean[j] = conditional_mean_trend_parameters[0] * std::pow(lat_ring[j].size() - 1, conditional_mean_trend_parameters[1]);
            number_of_fire_spreads[j] = 1 + randBin(1, conditional_mean[j]);
            fire_spreads_rings_list[j] = sample_vec(ring_exposure_list, number_of_fire_spreads[j]);
          }
          else if (lat_ring[j].size() == 3)
          {
            conditional_mean[j] = conditional_mean_trend_parameters[0] * std::pow(lat_ring[j].size() - 1, conditional_mean_trend_parameters[1]);
            alpha[j] = conditional_mean[j] * (lat_ring[j].size() - 1 - conditional_mean[j] - conditional_Cov[0]) / (conditional_mean[j] + (lat_ring[j].size() - 1) * (conditional_Cov[0] - 1));
            beta[j] = (lat_ring[j].size() - 1 - conditional_mean[j]) * (lat_ring[j].size() - 1 - conditional_mean[j] - conditional_Cov[0]) / (conditional_mean[j] + (lat_ring[j].size() - 1) * (conditional_Cov[0] - 1));
            sftrabbit::beta_distribution<> dist(alpha[j], beta[j]);
            simulated_probability[j] = dist(gen);
            number_of_fire_spreads[j] = 1 + randBin(lat_ring[j].size() - 1, simulated_probability[j]);
            fire_spreads_rings_list[j] = sample_vec(ring_exposure_list, number_of_fire_spreads[j]);
          }
          else
          {
            conditional_mean[j] = conditional_mean_trend_parameters[0] * std::pow(lat_ring[j].size() - 1, conditional_mean_trend_parameters[1]);
            alpha[j] = conditional_mean[j] * (lat_ring[j].size() - 1 - conditional_mean[j] - conditional_Cov[1]) / (conditional_mean[j] + (lat_ring[j].size() - 1) * (conditional_Cov[1] - 1));
            beta[j] = (lat_ring[j].size() - 1 - conditional_mean[j]) * (lat_ring[j].size() - 1 - conditional_mean[j] - conditional_Cov[1]) / (conditional_mean[j] + (lat_ring[j].size() - 1) * (conditional_Cov[1] - 1));
            
            number_of_fire_spreads[j] = 1 + randBin(lat_ring[j].size() - 1, simulated_probability[j]);
            fire_spreads_rings_list[j] = sample_vec(ring_exposure_list, number_of_fire_spreads[j]);
          }
        }
        if (fire_spreads_rings_list[j].size() > 0)
        {
          for (auto it = std::begin(fire_spreads_rings_list[j]); it != std::end(fire_spreads_rings_list[j]); ++it)
          {
            double wielkosc_pozar_procent;
            double wielkosc_pozar_kwota;
            wielkosc_pozar_procent = percentage_of_loss(wielkosc_pozaru);
            wielkosc_pozar_kwota = wielkosc_pozar_procent * exponsure_sum_ring[j][*it];
            if (wielkosc_pozar_kwota < 500.0)
            {
              wielkosc_pozar_kwota = 500.0;
            }
            out_data[0].push_back(distance_res[j][*it]);
            out_data[1].push_back(lat_ring[j][*it]);
            out_data[2].push_back(lon_ring[j][*it]);
            out_data[3].push_back(insu_ring[j][*it]);
            out_data[4].push_back(reas_ring[j][*it]);
            out_data[5].push_back(exponsure_sum_ring[j][*it]);
            out_data[6].push_back(wielkosc_pozar_kwota);
          }
        }
      }
    }
    return (out_data);
  }
}

int exposure_number;
double reas_fire;
double sum_vec_out;
double sum_vec_kat_out;
double sum_netto_out;
double sum_netto_kat_out;
VectorSim out_brutto_final;
VectorSim out_brutto_kat_final;
VectorSim out_netto_final;
VectorSim out_netto_kat_final;
VectorSim sim_brutto_final;
VectorSim sim_brutto_kat_final;
VectorSim sim_netto_final;
VectorSim sim_netto_kat_final;

void initializeFireRiskSimulator(
    const std::vector<std::vector<std::vector<long double>>> &exposure_longitude_data,
    const std::vector<std::vector<std::vector<long double>>> &exposure_latitude_data,
    const std::vector<std::vector<long double>> &fire_probs_data,
    const std::vector<std::vector<long double>> &fire_spread_prob_vec_data,
    std::vector<long double> &conditional_mean_trend_parameters_data,
    std::vector<long double> &conditional_Cov_data,
    const std::vector<std::vector<std::vector<int>>> &exposure_insurance_data,
    const std::vector<std::vector<std::vector<int>>> &exposure_reassurance_data,
    const std::vector<std::vector<std::vector<double>>> &exposure_sum_value_data,
    const std::vector<std::vector<double>> &wielkosc_pozaru_data,
    const std::vector<std::vector<double>> &fakultatywna_input_num_data,
    const std::vector<std::vector<std::vector<double>>> fakultatywna_input_val_data,
    const std::vector<std::vector<double>> &obligatoryjna_input_risk_data,
    const std::vector<std::vector<double>> vec_obligat_insur_event_data,
    const int &sim_data,
    const int &kat_val_data,
    const int &ilosc_ubezpieczycieli_data)
{
  exposure_longitude = exposure_longitude_data;
  exposure_latitude = exposure_latitude_data;
  fire_probs = fire_probs_data;
  fire_spread_prob_vec = fire_spread_prob_vec_data;
  conditional_mean_trend_parameters = conditional_mean_trend_parameters_data;
  conditional_Cov = conditional_Cov_data;
  exposure_insurance = exposure_insurance_data;
  exposure_reassurance = exposure_reassurance_data;
  exposure_sum_value = exposure_sum_value_data;
  wielkosc_pozaru = wielkosc_pozaru_data;
  fakultatywna_input_num = fakultatywna_input_num_data;
  fakultatywna_input_val = fakultatywna_input_val_data;
  obligatoryjna_input_risk = obligatoryjna_input_risk_data;
  vec_obligat_insur_event = vec_obligat_insur_event_data;
  sim = sim_data;
  kat_val = kat_val_data;
  ilosc_ubezpieczycieli = ilosc_ubezpieczycieli_data;
}

std::vector<std::vector<std::vector<double>>> calc_brutto_ring(std::vector<double> data_input,
                                                               std::vector<double> insurance, double kat_val, int ilosc_ubezpieczycieli)
{
  std::vector<std::vector<std::vector<double>>> out_final(6);
  std::vector<std::vector<double>> out_brutto(ilosc_ubezpieczycieli);
  std::vector<std::vector<double>> out_kat_brutto(ilosc_ubezpieczycieli);
  std::vector<std::vector<double>> ind_brutto(ilosc_ubezpieczycieli);
  std::vector<std::vector<double>> ind_kat_brutto(ilosc_ubezpieczycieli);
  std::vector<std::vector<double>> out_sum_brutto(ilosc_ubezpieczycieli);
  std::vector<std::vector<double>> out_sum_kat_brutto(ilosc_ubezpieczycieli);
  int ind_next = 0;
  for (auto it = std::begin(insurance); it != std::end(insurance); ++it)
  {
    out_brutto[*it].push_back(data_input[ind_next]);
    ind_brutto[*it].push_back(ind_next);
    if (data_input[ind_next] > kat_val)
    {
      out_kat_brutto[*it].push_back(data_input[ind_next]);
      ind_kat_brutto[*it].push_back(ind_next);
    }
    ind_next += 1;
  }
  for (int i = 0; i < ilosc_ubezpieczycieli; i++)
  {
    double sum_brutto = accumulate(out_brutto[i].begin(), out_brutto[i].end(), 0.0);
    double sum_kat_brutto = accumulate(out_kat_brutto[i].begin(), out_kat_brutto[i].end(), 0.0);
    out_sum_brutto[i].push_back(sum_brutto);
    out_sum_kat_brutto[i].push_back(sum_kat_brutto);
  }
  out_final[0] = out_brutto;
  out_final[1] = out_kat_brutto;
  out_final[2] = out_sum_brutto;
  out_final[3] = out_sum_kat_brutto;
  out_final[4] = ind_brutto;
  out_final[5] = ind_kat_brutto;
  return (out_final);
}

double calc_res_bligator(std::vector<double> vec_obligat_insur_risk, double sum_prem)
{
  double out_obl = 0.0;
  if (sum_prem < vec_obligat_insur_risk[0])
  {
    out_obl = vec_obligat_insur_risk[2] * sum_prem;
  }
  else if (sum_prem > vec_obligat_insur_risk[0] && sum_prem < vec_obligat_insur_risk[1])
  {
    out_obl = vec_obligat_insur_risk[2] * vec_obligat_insur_risk[0];
  }
  else if (sum_prem > vec_obligat_insur_risk[1])
  {
    out_obl = sum_prem - (vec_obligat_insur_risk[1] - vec_obligat_insur_risk[0]);
  }
  return (out_obl);
}
std::vector<std::vector<double>> reasurance_risk(std::vector<std::vector<double>> out_exp_sum_kwota_insurancers,
                                                 std::vector<double> out_reas,
                                                 std::vector<std::vector<double>> fakultatywna_input_num,
                                                 std::vector<std::vector<std::vector<double>>> fakultatywna_input_val,
                                                 std::vector<std::vector<double>> obligatoryina_input_risk,
                                                 int ilosc_ubezpieczyciell)
{
  
  double exp_fire_pre;
  double reas_oblig;
  double b_f;
  double reas_fakultat;
  std::vector<double> vec_fakul_insur_num;
  std::vector<double> vec_obligat_insur_risk;
  std::vector<std::vector<double>> vec_fakul_insur_val;
  std::vector<std::vector<double>> sum_prem_out_res(ilosc_ubezpieczycieli);
  std::vector<std::vector<double>> ind_prem_out_res(ilosc_ubezpieczycieli);
  std::vector<double> vec_final_premium;
  for (int kk = 0; kk < ilosc_ubezpieczycieli; kk++)
  {
    std::vector<double> input_one_insurance = out_exp_sum_kwota_insurancers[kk];
    int len_insurance = input_one_insurance.size();
    for (int i = 0; i < len_insurance; i++)
    {
      exp_fire_pre = input_one_insurance[i];
      vec_obligat_insur_risk = obligatoryjna_input_risk[kk];
      reas_fakultat = exp_fire_pre;
      reas_oblig = exp_fire_pre;
      if ((out_reas[i] < 9000))
      {
        vec_fakul_insur_num = fakultatywna_input_num[kk];
        vec_fakul_insur_val = fakultatywna_input_val[kk];
        b_f = vec_fakul_insur_val[out_reas[i]][0];
        if (std::find(vec_fakul_insur_num.begin(), vec_fakul_insur_num.end(), out_reas[i]) != vec_fakul_insur_num.end())
        {
          reas_fakultat = exp_fire_pre * b_f + std::max(0.0, (1 - b_f) * exp_fire_pre - vec_fakul_insur_val[out_reas[i]][1]);
          reas_oblig = reas_fakultat;
        }
        else
        {
          reas_fakultat = std::min(exp_fire_pre, vec_fakul_insur_val[out_reas[i]][0]) + std::max(0.0, exp_fire_pre - vec_fakul_insur_val[out_reas[i]][1]);
          reas_oblig = reas_fakultat;
        }
      }
      if (floor(vec_obligat_insur_risk[0]) >= 0)
      {
        reas_oblig = calc_res_bligator(vec_obligat_insur_risk, reas_fakultat);
      }
      sum_prem_out_res[kk].push_back(reas_oblig);
      ind_prem_out_res[kk].push_back(i);
    }
  }
  return (sum_prem_out_res);
}

std::vector<std::vector<double>> calc_reas_obliga_event(int ins_ind,
                                                        double fire_prem,
                                                        std::vector<std::vector<double>> num_reas_insurances,
                                                        std::vector<std::vector<double>> val_reas_insurances,
                                                        int size_vec,
                                                        std::vector<std::vector<double>> vec_obligat_insur_event, int ilosc_ubezpieczycieli)
{
  std::vector<std::vector<double>> vec_reas_final(3);
  std::vector<double> reas_spread(size_vec);
  std::vector<double> val_reas_insurance;
  std::vector<double> num_reas_insurance;
  std::vector<double> vec_obligat;
  std::vector<double> val_sums_insur;
  
  double reas_oblig;
  double sum_of_elems;
  double sum_of_elems_fire_el;
  for (int i = 0; i < ilosc_ubezpieczycieli; i++)
  {
    double sum_value = 0;
    val_reas_insurance = val_reas_insurances[i];
    num_reas_insurance = num_reas_insurances[i];
    vec_obligat = vec_obligat_insur_event[i];
    sum_of_elems = std::accumulate(val_reas_insurance.begin(), val_reas_insurance.end(), 0);
    int size_vec_reas;
    size_vec_reas = num_reas_insurance.size();
    if ((size_vec_reas == 0) && (ins_ind == i))
    {
      vec_reas_final[0].push_back(fire_prem);
    }
    else if ((size_vec_reas >= 1) && (ins_ind == i))
    {
      sum_of_elems_fire_el = sum_of_elems + fire_prem;
      reas_oblig = calc_reas_bligator(vec_obligat, sum_of_elems_fire_el);
      if (sum_of_elems_fire_el != reas_oblig)
      {
        for (auto it = std::begin(num_reas_insurance); it != std::end(num_reas_insurance); ++it)
        {
          reas_spread[*it] = sum_of_elems_fire_el / (size_vec_reas + 1);
          sum_value += sum_of_elems_fire_el / (size_vec_reas + 1);
        }
        vec_reas_final[0].push_back(sum_of_elems_fire_el / (size_vec_reas + 1));
      }
      else
      {
        int kk = 0;
        for (auto it = std::begin(num_reas_insurance); it != std::end(num_reas_insurance); ++it)
        {
          reas_spread[*it] = val_reas_insurance[kk];
          sum_value += val_reas_insurance[kk];
          kk = kk + 1;
        }
        vec_reas_final[0].push_back(fire_prem);
      }
    }
    else if ((size_vec_reas > 1) && (ins_ind != 1))
    {
      reas_oblig = calc_reas_bligator(vec_obligat, sum_of_elems);
      if (sum_of_elems != reas_oblig)
      {
        int kk = 0;
        for (auto it = std::begin(num_reas_insurance); it != std::end(num_reas_insurance); ++it)
        {
          reas_spread[*it] = sum_of_elems / size_vec_reas;
          kk = kk + 1;
        }
        vec_reas_final[0].push_back(sum_of_elems / size_vec_reas);
      }
      else
      {
        int kk = 0;
        for (auto it = std::begin(num_reas_insurance); it != std::end(num_reas_insurance); ++it)
        {
          reas_spread[*it] = val_reas_insurance[kk];
          sum_value += val_reas_insurance[kk];
          kk = kk + 1;
        }
      }
    }
    else
    {
      int kk = 0;
      for (auto it = std::begin(num_reas_insurance); it != std::end(num_reas_insurance); ++it)
      {
        reas_spread[*it] = val_reas_insurance[kk];
        kk = kk + 1;
      }
    }
    val_sums_insur.push_back(sum_value);
  }
  vec_reas_final[1] = reas_spread;
  vec_reas_final[2] = val_sums_insur;
  
  return (vec_reas_final);
}

struct Results
{
  int binom_fire;
  std::vector<int> fire_sources_list;
  std::vector<std::vector<std::vector<double>>> spread_results;
};
#include <map>

Results &getResults(std::vector<Results> &res, int woj, int month)
{
  int index = woj * 12 + month;
  return res[index];
}
void simulateExponsure()
{
  std::vector<Results> resVect(17 * 12);
  
  std::map<std::pair<size_t, int>, Results> results_map;
  
  for (size_t woj = 0; woj < 17; woj++)
  {
    for (int mies = 0; mies < 12; mies++)
    {
      pool.enqueue([&resVect, &results_map, woj, mies]
      {
        int index_table = 0;
        if (exposure_longitude[woj][mies].size() > 0)
        {
          Results results;
          
          int binom_fire = randBin(exposure_longitude[woj][mies].size(), fire_probs[woj][mies]);
          results.binom_fire = binom_fire;
          
          if (binom_fire > 0)
          {
            std::vector<int> fire_sources_list(binom_fire);
            
            std::vector<int> pom_index_fire(exposure_longitude[woj][mies].size());
            std::iota(std::begin(pom_index_fire), std::end(pom_index_fire), 0);
            
            fire_sources_list = sample_vec(pom_index_fire, binom_fire);
            results.fire_sources_list = fire_sources_list;
            
            for (int nr_budynku : fire_sources_list) {
              results.spread_results.push_back(haversine_loop_cpp_vec(5.767, nr_budynku, woj, mies));
            }
            
            getResults(resVect,woj,mies) = results;
            
          }
        } });
    }
  }
  
  pool.wait_for_tasks();
  
  for (size_t woj = 0; woj < 17; woj++)
  {
    for (int mies = 0; mies < 12; mies++)
    {
      int index_table = 0;
      if (exposure_longitude[woj][mies].size() > 0)
      {
        
        auto ress =   getResults(resVect,woj,mies) ;
        
        int binom_fire = ress.binom_fire;
        if (binom_fire > 0)
        {
          std::vector<int> fire_sources_list = ress.fire_sources_list;
          
          for (size_t budynki = 0; budynki < fire_sources_list.size(); budynki++)
          {
            auto nr_budynku = fire_sources_list[budynki];
            
            std::vector<std::vector<double>> spread_one_building = ress.spread_results[budynki];
            
            int insurancer = exposure_insurance[woj][mies][nr_budynku];
            double wielkosc_pozar_procent = percentage_of_loss(wielkosc_pozaru);
            double wielkosc_pozar_kwota = wielkosc_pozar_procent * exposure_sum_value[woj][mies][nr_budynku];
            if (wielkosc_pozar_kwota < 500.0)
              wielkosc_pozar_kwota = 500.0;
            
            double b_f;
            double reas_fakultat = wielkosc_pozar_kwota;
            double reas_oblig = wielkosc_pozar_kwota;
            if ((exposure_reassurance[woj][mies][nr_budynku] < 9000))
            {
              b_f = fakultatywna_input_val[exposure_insurance[woj][mies][nr_budynku]][exposure_reassurance[woj][mies][nr_budynku]][0];
              if (std::find(fakultatywna_input_num[exposure_insurance[woj][mies][nr_budynku]].begin(), fakultatywna_input_num[exposure_insurance[woj][mies][nr_budynku]].end(), exposure_reassurance[woj][mies][nr_budynku]) != fakultatywna_input_num[exposure_insurance[woj][mies][nr_budynku]].end())
              {
                reas_fakultat = wielkosc_pozar_kwota * b_f + std::max(0.0, (1 - b_f) * wielkosc_pozar_kwota - fakultatywna_input_val[exposure_insurance[woj][mies][nr_budynku]][exposure_reassurance[woj][mies][nr_budynku]][1]);
                reas_oblig = reas_fakultat;
              }
              else
              {
                reas_fakultat = std::min(wielkosc_pozar_kwota, fakultatywna_input_val[exposure_insurance[woj][mies][nr_budynku]][exposure_reassurance[woj][mies][nr_budynku]][0]) +
                  std::max(0.0,
                           wielkosc_pozar_kwota - fakultatywna_input_val[exposure_insurance[woj][mies][nr_budynku]][exposure_reassurance[woj][mies][nr_budynku]][0] - fakultatywna_input_val[exposure_insurance[woj][mies][nr_budynku]][exposure_reassurance[woj][mies][nr_budynku]][1]);
                reas_oblig = reas_fakultat;
              }
            }
            
            if (floor(obligatoryjna_input_risk[exposure_insurance[woj][mies][nr_budynku]][0]) >= 0)
              reas_oblig = calc_reas_bligator(obligatoryjna_input_risk[exposure_insurance[woj][mies][nr_budynku]], reas_fakultat);
            
            reas_fire = reas_oblig;
            sim_brutto_final.addDataVec(insurancer, wielkosc_pozar_kwota);
            
            sim_netto_final.addDataVec(insurancer, wielkosc_pozar_kwota);
            
            double reas_fire_kat = 0.0;
            if (wielkosc_pozar_kwota > kat_val)
            {
              sim_brutto_kat_final.addDataVec(insurancer, wielkosc_pozar_kwota);
              reas_fire_kat = reas_fire;
            }
            
            int len_spread = 0;
            len_spread = spread_one_building[4].size();
            if (len_spread > 0)
            {
              std::vector<std::vector<std::vector<double>>> out_vec_brutto(6);
              out_vec_brutto = calc_brutto_ring(spread_one_building[6], spread_one_building[3], kat_val,
                                                ilosc_ubezpieczycieli);
              for (int pp = 0; pp < ilosc_ubezpieczycieli; pp++)
              {
                sim_brutto_final.addDataVec(pp, out_vec_brutto[2][pp][0]);
              }
              
            }
            index_table += 1;
          }
        }
      }
    }
  }
}

std::vector<std::vector<double>> returnVAL()
{
  std::vector<std::vector<double>> out = out_brutto_final.returnVectorSim();
  return (out);
}

std::vector<std::vector<double>> zwroc_test(std::vector<std::vector<std::vector<long double>>> &exponsure_longitude,
                                            std::vector<std::vector<std::vector<long double>>> &exponsure_latitude,
                                            std::vector<std::vector<long double>> &list_list_wyb,
                                            std::vector<std::vector<long double>> fire_spread_prob_vec,
                                            std::vector<long double> conditional_mean_trend_parameters,
                                            std::vector<long double> conditional_Cov,
                                            std::vector<std::vector<std::vector<int>>> exponsure_insurance,
                                            std::vector<std::vector<std::vector<int>>> exponsure_reassurance,
                                            std::vector<std::vector<std::vector<double>>> exponsure_sum_value,
                                            std::vector<std::vector<double>> wielkosc_pozaru,
                                            std::vector<std::vector<double>> fakultatywna_input_num,
                                            std::vector<std::vector<std::vector<double>>> fakultatywna_input_val,
                                            std::vector<std::vector<double>> obligatoryjna_input_risk,
                                            std::vector<std::vector<double>> obligatoryjna_input_event,
                                            int sim,
                                            int kat_val,
                                            int ilosc_ubezpieczycieli)
{
  
  for (int sim_num = 0; sim_num < sim; sim_num++)
  {
    
    simulateExponsure();
    
    std::vector<std::vector<double>> out_sum_vec_out = sim_brutto_final.returnVectorSim();
    
    for (int kk = 0; kk < ilosc_ubezpieczycieli; kk++)
    {
      sim_brutto_final.clearVector(kk);
      out_brutto_final.addDataVec(kk, accumulate(out_sum_vec_out[kk].begin(), out_sum_vec_out[kk].end(), 0.0));
    }
  }
  std::vector<std::vector<double>> out = returnVAL();
  return (out);
}

std::vector<std::vector<std::vector<double>>> generateRandomData(int numRegions, int numMonths, std::mt19937 &gen, std::uniform_real_distribution<> &dist)
{
  std::vector<std::vector<std::vector<double>>> data(numRegions, std::vector<std::vector<double>>(numMonths));
  for (int i = 0; i < numRegions; ++i)
  {
    for (int j = 0; j < numMonths; ++j)
    {
      int n_num = static_cast<int>(dist(gen) * 10000);
      data[i][j].resize(n_num);
      for (int k = 0; k < n_num; ++k)
      {
        data[i][j][k] = dist(gen);
      }
    }
  }
  return data;
}

std::vector<double> generateSingleVector(std::mt19937 &gen, std::uniform_real_distribution<> &dist, int size)
{
  std::vector<double> v(size);
  for (auto &elem : v)
  {
    elem = dist(gen);
  }
  return v;
}

std::vector<double> generateRandomDoubles(int count, double min, double max)
{
  std::random_device rd;
  
  std::uniform_real_distribution<> dist(min, max);
  std::vector<double> values(count);
  for (auto &val : values)
  {
    val = dist(gen);
  }
  return values;
}

std::vector<int> generateRandomInts(int count, int min, int max)
{
  std::random_device rd;
  
  std::uniform_int_distribution<> dist(min, max);
  std::vector<int> values(count);
  for (auto &val : values)
  {
    val = dist(gen);
  }
  return values;
}

std::vector<long double> generateRandomLongDoubles(int count, long double min, long double max)
{
  
  std::uniform_real_distribution<long double> dist(min, max);
  std::vector<long double> values(count);
  for (auto &val : values)
  {
    val = dist(gen);
  }
  return values;
}

int main()
{
  
  const int numRegions = 17;
  const int numMonths = 12;
  
  std::vector<std::vector<std::vector<long double>>> exponsure_longitude(numRegions);
  std::vector<std::vector<std::vector<long double>>> exponsure_latitude(numRegions);
  std::vector<std::vector<long double>> list_list_wyb(numRegions);
  std::vector<std::vector<long double>> fire_spread_prob_vec(4);
  std::vector<long double> conditional_mean_trend_parameters = {0.14L, 0.44L};
  std::vector<long double> conditional_Cov = {1.25L, 2.29L};
  std::vector<std::vector<std::vector<int>>> exponsure_insurance(numRegions);
  std::vector<std::vector<std::vector<int>>> exponsure_reassurance(numRegions);
  std::vector<std::vector<std::vector<double>>> exponsure_sum_value(numRegions);
  std::vector<std::vector<double>> wielkosc_pozaru(2);
  std::vector<std::vector<double>> fakultatywna_input_num(4);
  std::vector<std::vector<std::vector<double>>> fakultatywna_input_val(4);
  
  for (int woj = 0; woj < numRegions; ++woj)
  {
    exponsure_longitude[woj].resize(numMonths);
    exponsure_latitude[woj].resize(numMonths);
    exponsure_insurance[woj].resize(numMonths);
    exponsure_reassurance[woj].resize(numMonths);
    exponsure_sum_value[woj].resize(numMonths);
    list_list_wyb[woj] = generateRandomLongDoubles(numMonths, 5.44e-05L, 9.44e-05L);
    
    for (int month = 0; month < numMonths; ++month)
    {
      int n_num = generateRandomInts(1,  150000,  150000)[0];
      
      exponsure_latitude[woj][month] = generateRandomLongDoubles(n_num, 49.0L, 54.5L);
      exponsure_longitude[woj][month] = generateRandomLongDoubles(n_num, 16.07L, 24.09L);
      exponsure_sum_value[woj][month] = generateRandomDoubles(n_num, 2000.0, 500000.0);
      exponsure_insurance[woj][month] = generateRandomInts(n_num, 0, 3);
      exponsure_reassurance[woj][month] = generateRandomInts(n_num, 0, 3);
      
      for (long int ic = 0; ic < n_num; ic++)
        getQuadTree(quadtree, woj, month).insert({exponsure_latitude[woj][month][ic], exponsure_longitude[woj][month][ic], exponsure_insurance[woj][month][ic], exponsure_reassurance[woj][month][ic], exponsure_sum_value[woj][month][ic]});
    }
  }
  
  std::cout << "ZAKONCZONO GENEROWANIE DANYCH " << std::endl;
  
  for (size_t i = 0; i < fire_spread_prob_vec.size(); ++i)
  {
    fire_spread_prob_vec[i] = generateRandomLongDoubles(9, 0.009L, 0.4L);
  }
  
  for (size_t i = 0; i < wielkosc_pozaru.size(); ++i)
  {
    wielkosc_pozaru[i] = generateRandomDoubles(5000, 0.0, 1.0);
  }
  
  for (size_t i = 0; i < fakultatywna_input_num.size(); ++i)
  {
    fakultatywna_input_num[i] = {0, 1, 2, 3};
    fakultatywna_input_val[i].resize(4);
    for (size_t j = 0; j < 4; ++j)
    {
      fakultatywna_input_val[i][j] = {0.2, 10000000.0, 0.4, 10000000.0, 0.2, 10000000.0, 0.8, 10000000.0};
    }
  }
  
  std::vector<std::vector<double>> obligatoryjna_input_risk = {
    {2500000, 10000000, 1, 1},
    {2500000, 10000000, 1, 1},
    {2500000, 10000000, 1, 1},
    {2500000, 10000000, 1, 1}};
  
  std::vector<std::vector<double>> obligatoryjna_input_event = {
    {25000000, 100000000, 1, 1},
    {25000000, 100000000, 1, 1},
    {25000000, 100000000, 1, 1},
    {25000000, 100000000, 1, 1}};
  
  
  sim = 1;
  
  auto start = std::chrono::high_resolution_clock::now();
  initializeFireRiskSimulator(exponsure_longitude,
                              exponsure_latitude,
                              list_list_wyb,
                              fire_spread_prob_vec,
                              conditional_mean_trend_parameters,
                              conditional_Cov,
                              exponsure_insurance,
                              exponsure_reassurance,
                              exponsure_sum_value,
                              wielkosc_pozaru,
                              fakultatywna_input_num,
                              fakultatywna_input_val,
                              obligatoryjna_input_risk,
                              obligatoryjna_input_event,
                              sim,
                              500000,
                              4);
  
  auto vec = zwroc_test(exponsure_longitude, exponsure_latitude, list_list_wyb, fire_spread_prob_vec, conditional_mean_trend_parameters, conditional_Cov, exponsure_insurance, exponsure_reassurance,
                        exponsure_sum_value, wielkosc_pozaru, fakultatywna_input_num, fakultatywna_input_val, obligatoryjna_input_risk, obligatoryjna_input_event, 100, 5000000, 4);
  
  auto stop = std::chrono::high_resolution_clock::now();
  
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  
  std::cout << "Function took " << duration.count() << " milliseconds to execute." << std::endl;
  
  std::cout << "Contents of the vector of vectors:" << std::endl;
  for (const auto &subVec : vec)
  {
    for (double value : subVec)
    {
      std::cout << value << " ";
    }
  }
  
  return 0;
}