#include <Rcpp.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <iterator>

using namespace Rcpp;
using namespace std;


// [[Rcpp:plugins(cpp11)]]


class VectorSim{
public:
  // tutaj rezerwuje 30 bo bede mial ok 30 ubezpieczycieli. Ale pogram moze byc tez wykonywany dla mniejszej ilosci ubezpieczycieli
  std::vector<std::vector<double>> data;
  VectorSim(): data (30, std::vector<double>()) {}
  
  void addDataVec (int insurane, double value) { 
    data [insurane].push_back(value);
  }
  
  std::vector<std::vector<double>> returnVectorSim() {
    return (data);
  }
  
  void clearVector (int num_vec) {
    data[num_vec].clear();
  }
};

class VectorPozarPierwotny{
public:
  std::vector<std::vector<long double>>build_fire;
  VectorPozarPierwotny():build_fire(9,std::vector<long double>()){}
  
  void addPozarPierwotny(int insurancer, long double exposure_longitude_one,
                         long double exposure_latitude_one, int woj, int mies, double exposire_sum_one, int index_table, double wielkosc_pozar_kwota,
                         double reas_fire){
    build_fire[0].push_back(insurancer);
    build_fire[1].push_back(exposure_longitude_one);
    build_fire[2].push_back(exposure_latitude_one);
    build_fire[3].push_back(woj+1);
    build_fire[4].push_back(mies+1);
    build_fire[5].push_back(exposire_sum_one);
    build_fire[6].push_back(index_table);
    build_fire[7].push_back(wielkosc_pozar_kwota);
    build_fire[8].push_back(reas_fire);
  }
  std::vector<std::vector<long double>> returnPozarPierwotny(){
    return (build_fire);
  }
};
class VectorPozarRozprzestrzeniony{
public:
  std::vector<std::vector<double>> build_fire_rozprzestrzeniony;
  VectorPozarRozprzestrzeniony():build_fire_rozprzestrzeniony(11,std::vector<double>()){}
  
  void addPozarRozprzestrzeniony(std::vector<std::vector<double>>spread_one_building){
    for ( int ttt=0; ttt<11;ttt++){
      build_fire_rozprzestrzeniony[ttt].insert(
          build_fire_rozprzestrzeniony[ttt].end(),
          spread_one_building[ttt].begin(),
          spread_one_building[ttt].end());
    }
  }
  std::vector<std::vector<double>> returnPozarRozprzestrzeniony(){
    return(build_fire_rozprzestrzeniony);
  }
};
struct Data{
  double lat_sub;
  double lon_sub;
  double insu_sub;
  double reas_sub;
  double premium_sub;
};



class FireRiskSimulator {
private:
  std::vector <std::vector<std::vector < long double>>>exposure_longitude;
  std::vector <std::vector<std::vector < long double>>> exposure_latitude;
  std::vector <std::vector<long double>> fire_probs;
  std::vector<std::vector<long double>> fire_spread_prob_vec;
  std::vector<long double> conditional_mean_trend_parameters;
  std::vector<long double> conditional_Cov;
  std::vector <std::vector<std::vector < int>>>exposure_insurance;
  std::vector <std::vector<std::vector < int>>>exposure_reassurance;
  std::vector <std::vector<std::vector < double>>>exposure_sum_value;
  std::vector <std::vector<double>> wielkosc_pozaru;
  std::vector <std::vector<double>> fakultatywna_input_num;
  std::vector <std::vector<std::vector < double>>>fakultatywna_input_val;
  std::vector <std::vector<double>> obligatoryjna_input_risk;
  std::vector <std::vector<double>> vec_obligat_insur_event;
  int sim;
  double kat_val;
  int ilosc_ubezpieczycieli;
  VectorSim out_brutto_final;
  VectorSim out_brutto_kat_final;
  VectorSim out_netto_final;
  VectorSim out_netto_kat_final;
  VectorSim sim_brutto_final;
  VectorSim sim_brutto_kat_final;
  VectorSim sim_netto_final;
  VectorSim sim_netto_kat_final;
  VectorPozarPierwotny buildPierwotny;
  VectorPozarRozprzestrzeniony buildRozprzestrzeniony;
  std::vector<long double> exposure_longitude_vec;
  std::vector<long double> exposure_latitude_vec;
  std::vector<int> exposure_insurance_vec;
  std::vector<int> exposure_reassurance_vec;
  std::vector<double> exposure_sum_one;
  
  int exposure_number;
  int binom_fire;
  double wielkosc_pozar_procent;
  double wielkosc_pozar_kwota;
  double reas_fire;
  int insurancer;
  double reas_fire_kat;
  int len_spread;
  double sum_vec_out;
  double sum_vec_kat_out;
  double sum_netto_out;
  double sum_netto_kat_out;
  
  double randZeroToOne(int size, int a, int b) {
    Rcpp::NumericVector val_dist = Rcpp::runif(size, a, b);
    return (val_dist[0]);
  }
  
  std::vector<int> sample_vec(std::vector<int> &population, int sampleSize) {
    std::vector<int> sampleData(sampleSize);
    for (int i = 0; i < sampleSize; i++) {
      int randomIndex = randZeroToOne(1, 0.0, population.size() - 1);
      sampleData[i] = population[randomIndex];
    }
    return sampleData;
  }
  
  int randBin(int size_exp, double prob_size) {
    Rcpp::NumericVector
    val_dist = Rcpp::rbinom(1, size_exp, prob_size);
    return (val_dist[0]);
  }
  
  bool contains(std::vector<bool> vec, int elem) {
    bool result = false;
    if (find(vec.begin(), vec.end(), elem) != vec.end()) {
      result = true;
    }
    return result;
  }
  
  int search_closest(const std::vector<double> &sorted_array, double x) {
    auto iter_geq = std::lower_bound(
      sorted_array.begin(),
      sorted_array.end(),
      x
    );
    
    if (iter_geq == sorted_array.begin()) {
      return 0;
    }
    
    double a = *(iter_geq - 1);
    double b = *(iter_geq);
    if (fabs(x - a) < fabs(x - b)) {
      return iter_geq - sorted_array.begin() - 1;
    }
    return iter_geq - sorted_array.begin();
  }
  
  double percentage_of_loss(std::vector <std::vector<double>> wielkosc_pozaru) {
    int ind_prob;
    double exp_sen;
    double val_dist;
    val_dist = randZeroToOne(1, 0, 1);
    std::vector<double> probability;
    probability = wielkosc_pozaru[0];
    std::vector<double> exponsure_sensitiv;
    exponsure_sensitiv = wielkosc_pozaru[1];
    ind_prob = search_closest(probability, val_dist);
    exp_sen = exponsure_sensitiv[ind_prob];
    return (exp_sen);
  }
  
  double calc_reas_bligator(std::vector<double> vec_obligat_insur_risk, double sum_prem) {
    double out_obl = 0.0;
    if (sum_prem < vec_obligat_insur_risk[0]) {
      out_obl = vec_obligat_insur_risk[2] * sum_prem;
    } else if (sum_prem > vec_obligat_insur_risk[0] && sum_prem < vec_obligat_insur_risk[1]) {
      out_obl = vec_obligat_insur_risk[2] * vec_obligat_insur_risk[0];
    } else if (sum_prem > vec_obligat_insur_risk[1]) {
      out_obl = sum_prem - (vec_obligat_insur_risk[1] - vec_obligat_insur_risk[0]);
    }
    return (out_obl);
    
  }
  
  double reasecuration_build_fire(double exp_fire_pre,
                                  double reas,
                                  std::vector<double> vec_fakul_insur_num,
                                  std::vector <std::vector<double>> vec_fakul_insur_val,
                                  std::vector<double> vec_obligat_insur_risk) {
    double reas_oblig;
    double b_f;
    double reas_fakultat;
    reas_fakultat = exp_fire_pre;
    reas_oblig = exp_fire_pre;
    if ((reas < 9000)) {
      b_f = vec_fakul_insur_val[reas][0];
      if (std::find(vec_fakul_insur_num.begin(), vec_fakul_insur_num.end(), reas) != vec_fakul_insur_num.end()) {
        reas_fakultat=exp_fire_pre * b_f + std::max(0.0, (1 - b_f) * exp_fire_pre - vec_fakul_insur_val[reas][1]);
        reas_oblig = reas_fakultat;
      } else {
        reas_fakultat = std::min(exp_fire_pre, vec_fakul_insur_val[reas][0]) +
          std::max(0.0,
                   exp_fire_pre - vec_fakul_insur_val[reas][0] - vec_fakul_insur_val[reas][1]);
        reas_oblig = reas_fakultat;
      }
    }
    if (floor(vec_obligat_insur_risk[0]) >= 0) {
      reas_oblig = calc_reas_bligator(vec_obligat_insur_risk, reas_fakultat);
    }
    return (reas_oblig);
  }
  
  std::vector<std::vector<double>>index_spread_build (long double lat_center,
                                                      long double lon_center,
                                                      std::vector<std::vector<double>> distance_res,
                                                      std::vector<std::vector<long double>>lat_ring,
                                                      std::vector<std::vector<long double>>lon_ring,
                                                      std::vector<std::vector<int>>insu_ring,
                                                      std::vector<std::vector<int>>reas_ring,
                                                      std::vector<std::vector<double>>exposure_sum_ring,
                                                      std::vector<std::vector<long double>>fire_spread_indicator_probs,
                                                      std::vector<long double> conditional_mean_trend_parameters,
                                                      std::vector<long double> conditional_CoV,
                                                      std::vector<std::vector<double>> wielkosc_pozaru) {
    int exposure_number;
    //int biom_fire;
    std::vector<double> out_distance;
    std::vector<int> ind_number_pom;
    std::vector<double> distance_res_pom;
    std::vector<long double> out_lat_pom;
    std::vector<long double> out_lon_pom;
    std::vector<int> out_insu_pom;
    std::vector<int> out_reas_pom;
    std::vector<double> out_exp_sum_pom;
    std::vector<std::vector<double>> out_data(11);
    std::vector<std::vector<std::vector<double>>> amount_of_loss_vec;
    std::vector<int> number_of_fire_spreads(9);
    std::vector<int> fire_spreads_indicator(9);
    std::vector<double> conditional_mean(9);
    std::vector<double> alpha(9);
    std::vector<double> beta(9);
    std::vector<double> simulated_probability(9);
    std::vector<std::vector<int>> fire_spreads_rings_list(9);
    for ( int j = 0; j<9; j++){
      std::vector<int> output_sources_list;
      exposure_number = lat_ring[j].size();
      out_lat_pom = lat_ring[j];
      out_lon_pom = lon_ring[j];
      out_insu_pom=insu_ring[j];
      out_reas_pom=reas_ring[j];
      out_exp_sum_pom = exposure_sum_ring[j];
      distance_res_pom=distance_res[j];
      std::vector<int> ring_exposure_list (exposure_number) ;
      std::iota (std::begin(ring_exposure_list), std::end(ring_exposure_list), 0);
      if(exposure_number > 0){
        if(j==0){
          fire_spreads_indicator[j] = randBin(1,fire_spread_indicator_probs[0][j]);
        }
        else if (j == 1){
          fire_spreads_indicator[j] = randBin(1,fire_spread_indicator_probs[0 + fire_spreads_indicator[j-1]][j]);
        }
        else{
          fire_spreads_indicator[j] = randBin(1,fire_spread_indicator_probs[0 + 2* fire_spreads_indicator[j-1] + fire_spreads_indicator[j-2]][j]);
        }
        if(fire_spreads_indicator[j]>0){
          if(exposure_number==1){
            number_of_fire_spreads[j]=1;
            fire_spreads_rings_list[j]=sample_vec(ring_exposure_list,number_of_fire_spreads[j]);
          }
          else if (exposure_number == 2){
            conditional_mean[j]= conditional_mean_trend_parameters[0] * std::pow(exposure_number -1, conditional_mean_trend_parameters[1]);
            number_of_fire_spreads[j]=1 + randBin(1,conditional_mean[j]);
            fire_spreads_rings_list[j]=sample_vec(ring_exposure_list,number_of_fire_spreads[j]);
          }
          else if( exposure_number==3){
            conditional_mean[j]= conditional_mean_trend_parameters[0] * std::pow(exposure_number -1, conditional_mean_trend_parameters[1]);
            alpha[j] = conditional_mean[j]*(exposure_number - 1 - conditional_mean[j]-conditional_CoV[0])/(conditional_mean[j]+(exposure_number - 1) * (conditional_CoV[0]-1));
            beta[j]= (exposure_number - 1 - conditional_mean[j]) * (exposure_number -1 - conditional_mean[j]- conditional_CoV[0]) / (conditional_mean[j] + (exposure_number - 1) * (conditional_CoV[0]-1));
            Rcpp::NumericVector val_rbeta = Rcpp::rbeta(1,alpha[j],beta[j]);
            simulated_probability[j] = val_rbeta[0];
            number_of_fire_spreads[j] = 1 + randBin(exposure_number-1, simulated_probability[j]);
            fire_spreads_rings_list[j]=sample_vec(ring_exposure_list,number_of_fire_spreads[j]);
          }
          else{
            conditional_mean[j]= conditional_mean_trend_parameters[0] * std::pow(exposure_number -1, conditional_mean_trend_parameters[1]);
            alpha[j] = conditional_mean[j]*(exposure_number - 1 - conditional_mean[j]-conditional_CoV[1])/(conditional_mean[j]+(exposure_number - 1) * (conditional_CoV[1]-1));
            beta[j]= (exposure_number - 1 - conditional_mean[j]) * (exposure_number -1 - conditional_mean[j]- conditional_CoV[1]) / (conditional_mean[j] + (exposure_number - 1) * (conditional_CoV[1]-1));
            Rcpp::NumericVector val_rbeta = Rcpp::rbeta(1,alpha[j],beta[j]);
            number_of_fire_spreads[j] = 1 + randBin(exposure_number-1, simulated_probability[j]);
            fire_spreads_rings_list[j]=sample_vec(ring_exposure_list,number_of_fire_spreads[j]);
          }
        }
        if (fire_spreads_rings_list[j].size()>0){
          for ( auto it = std::begin(fire_spreads_rings_list[j]); it != std::end(fire_spreads_rings_list[j]); ++it){
            double wielkosc_pozar_procent;
            double wielkosc_pozar_kwota;
            wielkosc_pozar_procent = percentage_of_loss(wielkosc_pozaru);
            wielkosc_pozar_kwota= wielkosc_pozar_procent*out_exp_sum_pom[*it];
            if (wielkosc_pozar_kwota <500.0){
              wielkosc_pozar_kwota = 500.0;
            }
            out_data[0].push_back(distance_res_pom[*it]);
            out_data[1].push_back(out_lat_pom[*it]);
            out_data[2].push_back(out_lon_pom[*it]);
            out_data[3].push_back(out_insu_pom[*it]);
            out_data[4].push_back(out_reas_pom[*it]);
            out_data[5].push_back(out_exp_sum_pom[*it]);
            out_data[6].push_back(wielkosc_pozar_kwota);
          }
        }
      }
    }
    return(out_data);
  }
  
  double haversine_cpp(double lat1, double long1,
                       double lat2, double long2,
                       double earth_radius = 6378137){
    
    double distance;
    
    if (!((long1 > 360) || (long2 > 360) || (lat1 > 90) || (lat2 > 90))){
      double deg_to_rad = 0.0174532925199432957; // i.e. pi/180 (multiplication is faster than division)
      double delta_phi = (lat2 - lat1) * deg_to_rad;
      double delta_lambda = (long2 - long1) * deg_to_rad;
      double phi1 = lat1 * deg_to_rad;
      double phi2 = lat2 * deg_to_rad;
      double term1 = pow(sin(delta_phi * .5), 2);
      double term2 = cos(phi1) * cos(phi2) * pow(sin(delta_lambda * .5), 2);
      double delta_sigma = 2 * atan2(sqrt(term1 + term2), sqrt(1 - term1 - term2));
      distance = earth_radius * delta_sigma;
    } else {
      distance = NAN;
    }
    return distance;
  }
  
  std::vector<std::vector<double>> index_in_ring(long double lat_center,
                                                 long double lon_center,
                                                 std::vector<long double> lat_sub,
                                                 std::vector<long double> lon_sub,
                                                 std::vector<std::vector<long double>>   fire_spread_prob_vec,
                                                 std::vector<long double> conditional_mean_trend_parameters,
                                                 std::vector<long double> conditional_Cov,
                                                 std::vector<int> insu_sub,
                                                 std::vector<int> reas_sub,
                                                 std::vector<double> exponsure_sum_value,
                                                 std::vector<std::vector<double>> wielkosc_pozaru){
    std::vector<std::vector<int>> ind_number(9);
    std::vector<std::vector<double>> distance_res(9);
    std::vector<std::vector<long double>> lat_ring(9);
    std::vector<std::vector<long double>> lon_ring(9);
    std::vector<std::vector<int>> insu_ring(9);
    std::vector<std::vector<int>> reas_ring(9);
    std::vector<std::vector<double>> exponsure_sum_ring(9);
    std::vector<std::vector<double>> ind_after_prob(9);
    int n1 = lon_sub.size();
    if(n1>0){
      for(int i =0;i<n1;++i){
        double res = haversine_cpp(lat_center,lon_center,lat_sub[i],lon_sub[i]);
        if(res<0.005){
          distance_res[0].push_back(res);
          lat_ring[0].push_back(lat_sub[i]);
          lon_ring[0].push_back(lon_sub[i]);
          insu_ring[0].push_back(insu_sub[i]);
          reas_ring[0].push_back(reas_sub[i]);
          exponsure_sum_ring[0].push_back(exponsure_sum_value[i]);
        }
        if(res>0.005 &&res<25){
          distance_res[1].push_back(res);
          lat_ring[1].push_back(lat_sub[i]);
          lon_ring[1].push_back(lon_sub[i]);
          insu_ring[1].push_back(insu_sub[i]);
          reas_ring[1].push_back(reas_sub[i]);
          exponsure_sum_ring[1].push_back(exponsure_sum_value[i]);
        }
        if(res>25 &&res<50){
          distance_res[2].push_back(res);
          lat_ring[2].push_back(lat_sub[i]);
          lon_ring[2].push_back(lon_sub[i]);
          insu_ring[2].push_back(insu_sub[i]);
          reas_ring[2].push_back(reas_sub[i]);
          exponsure_sum_ring[2].push_back(exponsure_sum_value[i]);
        }
        if(res>50 &&res<75){
          distance_res[3].push_back(res);
          lat_ring[3].push_back(lat_sub[i]);
          lon_ring[3].push_back(lon_sub[i]);
          insu_ring[3].push_back(insu_sub[i]);
          reas_ring[3].push_back(reas_sub[i]);
          exponsure_sum_ring[3].push_back(exponsure_sum_value[i]);
        }
        if(res>75 &&res<100){
          distance_res[4].push_back(res);
          lat_ring[4].push_back(lat_sub[i]);
          lon_ring[4].push_back(lon_sub[i]);
          insu_ring[4].push_back(insu_sub[i]);
          reas_ring[4].push_back(reas_sub[i]);
          exponsure_sum_ring[4].push_back(exponsure_sum_value[i]);
        }
        if(res>100 &&res<125){
          distance_res[5].push_back(res);
          lat_ring[5].push_back(lat_sub[i]);
          lon_ring[5].push_back(lon_sub[i]);
          insu_ring[5].push_back(insu_sub[i]);
          reas_ring[5].push_back(reas_sub[i]);
          exponsure_sum_ring[5].push_back(exponsure_sum_value[i]);
        }
        if(res>125 &&res<150){
          distance_res[6].push_back(res);
          lat_ring[6].push_back(lat_sub[i]);
          lon_ring[6].push_back(lon_sub[i]);
          insu_ring[6].push_back(insu_sub[i]);
          reas_ring[6].push_back(reas_sub[i]);
          exponsure_sum_ring[6].push_back(exponsure_sum_value[i]);
        }
        if(res>150 &&res<175){
          distance_res[7].push_back(res);
          lat_ring[7].push_back(lat_sub[i]);
          lon_ring[7].push_back(lon_sub[i]);
          insu_ring[7].push_back(insu_sub[i]);
          reas_ring[7].push_back(reas_sub[i]);
          exponsure_sum_ring[7].push_back(exponsure_sum_value[i]);
        }
        if(res>175 &&res<200){
          distance_res[8].push_back(res);
          lat_ring[8].push_back(lat_sub[i]);
          lon_ring[8].push_back(lon_sub[i]);
          insu_ring[8].push_back(insu_sub[i]);
          reas_ring[8].push_back(reas_sub[i]);
          exponsure_sum_ring[8].push_back(exponsure_sum_value[i]);
        }}}
    ind_after_prob = index_spread_build(lat_center,lon_center,distance_res,lat_ring,lon_ring,insu_ring,
                                        reas_ring,exponsure_sum_ring,  fire_spread_prob_vec,
                                        conditional_mean_trend_parameters,
                                        conditional_Cov,wielkosc_pozaru);
    return(ind_after_prob);
    
  }
  
  std::vector<std::vector<double>> haversine_loop_cpp_vec(std::vector<long double> exposure_longitude,
                                                          std::vector<long double> exposure_latitude,
                                                          std::vector<std::vector<long double>>   fire_spread_prob_vec,
                                                          std::vector<long double> conditional_mean_trend_parameters,
                                                          std::vector<long double> conditional_Cov,
                                                          double radius,
                                                          int n1,
                                                          std::vector<int> exposure_insurance,
                                                          std::vector<int> exposure_reassuramce,
                                                          std::vector<double> exposure_sum_value,
                                                          std::vector<std::vector<double>> wielkosc_pozaru
  ) {
    long double lat_center = exposure_latitude[n1];
    long double lon_center = exposure_longitude[n1];
    int circumference_earth_in_meters = 40075000;
    double one_lat_in_meters = circumference_earth_in_meters * 0.002777778;  // 0.002777778 is used instead of 1/360;
    double one_lon_in_meters = circumference_earth_in_meters * cos(lat_center * 0.01745329) * 0.002777778;
    double south_lat = lat_center - radius / one_lat_in_meters;
    double north_lat = lat_center + radius / one_lat_in_meters;
    double west_lon = lon_center - radius / one_lon_in_meters;
    double east_lon = lon_center + radius / one_lon_in_meters;
    int n = exposure_longitude.size();
    
    std::vector<long double> lat_sub;
    std::vector<long double> lon_sub;
    std::vector<int> insu_sub;
    std::vector<int> reas_sub;
    std::vector<double> premium_sub;
    std::vector<int> ind_spread_build;
    std::vector<std::vector<double>> ind_ring(11);
    bool logical_value;
    
    for ( int i = 0; i < n; i++ ){
      logical_value = !((exposure_longitude[i] > east_lon) || (exposure_longitude[i] < west_lon) || (exposure_latitude[i] < south_lat) || (exposure_latitude[i] > north_lat));
      if(logical_value){
        lat_sub.push_back(exposure_latitude[i]);
        lon_sub.push_back(exposure_longitude[i]);
        insu_sub.push_back(exposure_insurance[i]);
        reas_sub.push_back(exposure_reassuramce[i]);
        premium_sub.push_back(exposure_sum_value[i]);
      }}
    
    if(lat_sub.size()>0){
      ind_ring = index_in_ring(lat_center,lon_center,lat_sub,lon_sub, fire_spread_prob_vec,
                               conditional_mean_trend_parameters,
                               conditional_Cov,insu_sub,reas_sub,premium_sub,wielkosc_pozaru);
    }
    return(ind_ring);
  }
  
  std::vector<std::vector<std::vector<double>>> calc_brutto_ring(std::vector<double> data_input,
                                                                 std::vector<double> insurance, double kat_val, int ilosc_ubezpieczycieli){
    std::vector<std::vector<std::vector<double>>> out_final(6);
    std::vector<std::vector<double>> out_brutto(ilosc_ubezpieczycieli);
    std::vector<std::vector<double>> out_kat_brutto(ilosc_ubezpieczycieli);
    std::vector<std::vector<double>> ind_brutto(ilosc_ubezpieczycieli);
    std::vector<std::vector<double>> ind_kat_brutto(ilosc_ubezpieczycieli);
    std::vector<std::vector<double>> out_sum_brutto(ilosc_ubezpieczycieli);
    std::vector<std::vector<double>> out_sum_kat_brutto(ilosc_ubezpieczycieli);
    int ind_next = 0;
    for ( auto it = std::begin(insurance); it!= std::end(insurance); ++it){
      out_brutto[*it].push_back(data_input[ind_next]);
      ind_brutto[*it].push_back(ind_next);
      if(data_input[ind_next]>kat_val){
        out_kat_brutto[*it].push_back(data_input[ind_next]);
        ind_kat_brutto[*it].push_back(ind_next);
      }
      ind_next +=1;
    }
    for ( int i = 0 ; i < ilosc_ubezpieczycieli; i ++){
      double sum_brutto = accumulate(out_brutto[i].begin(),out_brutto[i].end(),0.0);
      double sum_kat_brutto = accumulate ( out_kat_brutto[i].begin(), out_kat_brutto[i].end(),0.0);
      out_sum_brutto[i].push_back(sum_brutto);
      out_sum_kat_brutto[i].push_back(sum_kat_brutto);
    }
    out_final[0] = out_brutto;
    out_final[1] = out_kat_brutto;
    out_final[2] = out_sum_brutto;
    out_final[3] = out_sum_kat_brutto;
    out_final[4] = ind_brutto;
    out_final[5] = ind_kat_brutto;
    return ( out_final);
  }
  
  double calc_res_bligator(std::vector<double> vec_obligat_insur_risk,double sum_prem){
    double out_obl=0.0;
    if(sum_prem<vec_obligat_insur_risk[0]){
      out_obl = vec_obligat_insur_risk[2]*sum_prem;
    }
    else if(sum_prem>vec_obligat_insur_risk[0] && sum_prem<vec_obligat_insur_risk[1]){
      out_obl = vec_obligat_insur_risk[2]*vec_obligat_insur_risk[0];
    }
    else if(sum_prem>vec_obligat_insur_risk[1]){
      out_obl = sum_prem - (vec_obligat_insur_risk[1] - vec_obligat_insur_risk[0]);
    }
    return(out_obl);
  }
  
  std::vector<std::vector<double>> reasurance_risk(std:: vector<std:: vector<double>> out_exp_sum_kwota_insurancers,
                                                   std::vector<double> out_reas,
                                                   std::vector<std::vector<double>> fakultatywna_input_num,
                                                   std::vector<std::vector<std::vector<double>>>fakultatywna_input_val,
                                                   std::vector<std::vector<double>> obligatoryina_input_risk,
                                                   int ilosc_ubezpieczyciell){
    
    //int len_exp_sum;
    double  exp_fire_pre;
    double reas_oblig;
    double b_f;
    double reas_fakultat;
    std::vector<double> vec_fakul_insur_num;
    std::vector<double> vec_obligat_insur_risk;
    std::vector<std::vector<double>> vec_fakul_insur_val;
    std::vector<std::vector<double>> sum_prem_out_res(ilosc_ubezpieczycieli);
    std::vector<std::vector<double>> ind_prem_out_res(ilosc_ubezpieczycieli);
    std::vector<double> vec_final_premium;
    for(int kk =0 ; kk<ilosc_ubezpieczycieli;kk++){
      std::vector<double> input_one_insurance = out_exp_sum_kwota_insurancers[kk];
      int len_insurance = input_one_insurance.size();
      for ( int i = 0 ; i <len_insurance ; i ++){
        exp_fire_pre = input_one_insurance[i];
        vec_obligat_insur_risk = obligatoryjna_input_risk[kk];
        reas_fakultat = exp_fire_pre;
        reas_oblig = exp_fire_pre;
        if((out_reas[i]<9000)){
          vec_fakul_insur_num = fakultatywna_input_num[kk];
          vec_fakul_insur_val = fakultatywna_input_val[kk];
          b_f = vec_fakul_insur_val[out_reas[i]][0];
          if ( std::find( vec_fakul_insur_num.begin(),vec_fakul_insur_num.end(),out_reas[i])!= vec_fakul_insur_num.end()){
            reas_fakultat = exp_fire_pre*b_f + std::max (0.0,(1-b_f)*exp_fire_pre-vec_fakul_insur_val[out_reas[i]][1]);
            reas_oblig=reas_fakultat;
          }
          else{
            reas_fakultat = std::min(exp_fire_pre,vec_fakul_insur_val[out_reas[i]][0])+ std::max (0.0 , exp_fire_pre-vec_fakul_insur_val[out_reas[i]][1]); 
            reas_oblig=reas_fakultat;
          }
          
        }
        if(floor(vec_obligat_insur_risk[0])>= 0){
          reas_oblig = calc_res_bligator(vec_obligat_insur_risk,reas_fakultat);
        }
        sum_prem_out_res[kk].push_back(reas_oblig);
        ind_prem_out_res[kk].push_back(i);
      }
    }
    return(sum_prem_out_res);
  }
  
  std::vector <std::vector<double>> calc_reas_obliga_event( int ins_ind,
                                                            double fire_prem,
                                                            std::vector<std::vector<double>> num_reas_insurances,
                                                            std::vector<std::vector<double>> val_reas_insurances,
                                                            int size_vec,
                                                            std::vector<std::vector<double>> vec_obligat_insur_event, int ilosc_ubezpieczycieli){
    std::vector <std::vector<double>> vec_reas_final(3);
    std::vector<double> reas_spread(size_vec);
    std::vector<double> val_reas_insurance;
    std::vector<double> num_reas_insurance;
    std::vector<double> vec_obligat;
    std::vector<double> val_sums_insur;
    
    double reas_oblig;
    double sum_of_elems;
    double sum_of_elems_fire_el;
    for (int i = 0; i < ilosc_ubezpieczycieli; i++) {
      double sum_value = 0;
      val_reas_insurance = val_reas_insurances[i];
      num_reas_insurance = num_reas_insurances[i];
      vec_obligat = vec_obligat_insur_event[i];
      sum_of_elems = std::accumulate(val_reas_insurance.begin(), val_reas_insurance.end(), 0);
      int size_vec_reas;
      size_vec_reas = num_reas_insurance.size();
      if ((size_vec_reas == 0) && (ins_ind == i)) {
        vec_reas_final[0].push_back(fire_prem);
      } else if ((size_vec_reas >= 1) && (ins_ind == i)) {
        sum_of_elems_fire_el = sum_of_elems + fire_prem;
        reas_oblig = calc_reas_bligator(vec_obligat, sum_of_elems_fire_el);
        if (sum_of_elems_fire_el != reas_oblig) {
          for (auto it = std::begin(num_reas_insurance); it != std::end(num_reas_insurance); ++it) {
            reas_spread[*it] = sum_of_elems_fire_el/(size_vec_reas + 1);
            sum_value +=sum_of_elems_fire_el/(size_vec_reas + 1);
          }
          vec_reas_final[0].push_back(sum_of_elems_fire_el / (size_vec_reas + 1));
        }
        else{
          int kk = 0;
          for (auto it = std::begin(num_reas_insurance); it != std::end(num_reas_insurance); ++it) {
            reas_spread[*it] = val_reas_insurance[kk];
            sum_value += val_reas_insurance[kk];
            kk = kk + 1;
          }
          vec_reas_final[0].push_back(fire_prem);
        }
      }
      else if((size_vec_reas>1) && (ins_ind != 1)) {
        reas_oblig = calc_reas_bligator (vec_obligat, sum_of_elems);
        if (sum_of_elems!=reas_oblig) {
          int kk = 0;
          for (auto it = std::begin (num_reas_insurance); it != std::end (num_reas_insurance); ++it)
          { reas_spread [*it] = sum_of_elems/size_vec_reas;
            kk=kk+1;
          }
          vec_reas_final[0].push_back(sum_of_elems/size_vec_reas);
        }
        else{
          int kk = 0;
          for (auto it = std::begin (num_reas_insurance); it != std::end (num_reas_insurance); ++it)
          { reas_spread [*it] = val_reas_insurance [kk];
            sum_value+=val_reas_insurance[kk];
            kk=kk+1;
          }
        }
      }
      else{
        int kk = 0;
        for (auto it = std::begin (num_reas_insurance); it != std::end (num_reas_insurance); ++it)
        { reas_spread [*it] = val_reas_insurance[kk];
          kk=kk+1;
        }
      }
      val_sums_insur.push_back(sum_value);
    }
    vec_reas_final [1] =reas_spread;
    vec_reas_final [2] = val_sums_insur;
    
    return(vec_reas_final);
  }
  
  void simulateExponsure (size_t woj, int mies, int index_table) {
    exposure_longitude_vec = exposure_longitude[woj][mies];
    exposure_number = exposure_longitude_vec.size();
    if (exposure_number > 0) {
      exposure_latitude_vec = exposure_latitude[woj][mies];
      exposure_insurance_vec = exposure_insurance[woj][mies];
      exposure_reassurance_vec = exposure_reassurance[woj][mies];
      exposure_sum_one = exposure_sum_value[woj][mies];
      binom_fire = randBin(exposure_number, fire_probs[woj][mies]);
      if (binom_fire > 0) {
        std::vector<int> fire_sources_list(binom_fire);
        std::vector<int> pom_index_fire(exposure_number);
        std::iota(std::begin(pom_index_fire), std::end(pom_index_fire), 0);
        fire_sources_list = sample_vec(pom_index_fire, binom_fire);
        for (auto it = fire_sources_list.begin(); it != fire_sources_list.end(); ++it) {
          insurancer = exposure_insurance_vec[*it];
          wielkosc_pozar_procent = percentage_of_loss(wielkosc_pozaru);
          wielkosc_pozar_kwota = wielkosc_pozar_procent*exposure_sum_one[*it];
          if (wielkosc_pozar_kwota < 500.0) {
            wielkosc_pozar_kwota = 500.0;
          }
          reas_fire = reasecuration_build_fire(wielkosc_pozar_kwota, exposure_reassurance_vec[*it],
                                               fakultatywna_input_num[exposure_insurance_vec[*it]],
                                                                     fakultatywna_input_val[exposure_insurance_vec[*it]],
                                                                                           obligatoryjna_input_risk[exposure_insurance_vec[*it]]);
          buildPierwotny.addPozarPierwotny(insurancer, exposure_longitude_vec[*it],
                                           exposure_latitude_vec[*it],
                                                                woj + 1, mies + 1, exposure_sum_one[*it], index_table,
                                                                wielkosc_pozar_kwota, reas_fire);
          sim_brutto_final.addDataVec(insurancer, wielkosc_pozar_kwota);
          sim_netto_final.addDataVec(insurancer, wielkosc_pozar_kwota);
          reas_fire_kat = 0.0;
          if (wielkosc_pozar_kwota > kat_val) {
            sim_brutto_kat_final.addDataVec(insurancer, wielkosc_pozar_kwota);
            reas_fire_kat = reas_fire;
          }
          std::vector <std::vector<double>> spread_one_building(11);
          spread_one_building = haversine_loop_cpp_vec(exposure_longitude_vec, exposure_latitude_vec,
                                                       fire_spread_prob_vec,
                                                       conditional_mean_trend_parameters,
                                                       conditional_Cov, 200,
                                                       *it, exposure_insurance_vec, exposure_reassurance_vec,
                                                       exposure_sum_one, wielkosc_pozaru);
          len_spread = 0;
          len_spread = spread_one_building[4].size();
          if (len_spread > 0) {
            std::vector < std::vector < std::vector < double>>> out_vec_brutto(6);
            out_vec_brutto = calc_brutto_ring(spread_one_building[6], spread_one_building[3], kat_val,
                                              ilosc_ubezpieczycieli);
            std::vector <std::vector<double>> reas_risk = reasurance_risk(out_vec_brutto[0],
                                                                          spread_one_building[4],
                                                                                             fakultatywna_input_num,
                                                                                             fakultatywna_input_val,
                                                                                             obligatoryjna_input_risk,
                                                                                             ilosc_ubezpieczycieli);
            std::vector <std::vector<double>> reas_event = calc_reas_obliga_event(insurancer, reas_fire,
                                                                                  out_vec_brutto[4],
                                                                                                reas_risk, len_spread,
                                                                                                vec_obligat_insur_event,
                                                                                                ilosc_ubezpieczycieli);
            sim_netto_final.addDataVec(insurancer, reas_event[0][0]);
            std::vector <std::vector<double>> reas_risk_kat = reasurance_risk(out_vec_brutto[1],
                                                                              spread_one_building[4],
                                                                                                 fakultatywna_input_num,
                                                                                                 fakultatywna_input_val,
                                                                                                 obligatoryjna_input_risk,
                                                                                                 ilosc_ubezpieczycieli);
            std::vector <std::vector<double>> reas_event_kat = calc_reas_obliga_event(insurancer,
                                                                                      reas_fire_kat,
                                                                                      out_vec_brutto[5],
                                                                                                    reas_risk_kat,
                                                                                                    len_spread,
                                                                                                    vec_obligat_insur_event,
                                                                                                    ilosc_ubezpieczycieli);
            sim_netto_kat_final.addDataVec(insurancer, reas_event_kat[0][0]);
            for (int pp = 0; pp < ilosc_ubezpieczycieli; pp++) {
              sim_brutto_final.addDataVec(pp, out_vec_brutto[2][pp][0]);
              sim_brutto_kat_final.addDataVec(pp, out_vec_brutto[3][pp][0]);
              sim_netto_final.addDataVec(pp, reas_event[2][pp]);
              sim_netto_kat_final.addDataVec(pp, reas_event_kat[2][pp]);
            }
            spread_one_building[10].insert(
                spread_one_building[10].begin(),
                reas_event[1].begin(),
                reas_event[1].end());
            spread_one_building[7].insert(spread_one_building[7].end(), len_spread, index_table);
            spread_one_building[8].insert(spread_one_building[8].end(), len_spread, woj + 1);
            spread_one_building[9].insert(spread_one_building[9].end(), len_spread, mies + 1);
          }
          index_table += 1;
        }
      }
    }
  }
public:
  
  FireRiskSimulator (
      const std::vector<std::vector<std::vector<long double>>>& exposure_longitude_data,
      const std::vector<std::vector<std::vector<long double>>>& exposure_latitude_data,
      const std::vector<std::vector<long double>>& fire_probs_data,
      const std::vector<std::vector<long double>> &fire_spread_prob_vec_data,
      std::vector<long double> &conditional_mean_trend_parameters_data,
      std::vector<long double> &conditional_Cov_data,
      const std::vector<std::vector<std::vector<int>>>& exposure_insurance_data,
      const std::vector<std::vector<std::vector<int>>>& exposure_reassurance_data,
      const std::vector<std::vector<std::vector<double>>>& exposure_sum_value_data,
      const std::vector<std::vector<double>>& wielkosc_pozaru_data,
      const std::vector<std::vector<double>>& fakultatywna_input_num_data,
      const std::vector<std::vector<std::vector<double>>> fakultatywna_input_val_data,
      const std::vector<std::vector<double>>& obligatoryjna_input_risk_data,
      const std::vector<std::vector<double>> vec_obligat_insur_event_data,
      const int& sim,
      const int& kat_val,
      const int& ilosc_ubezpieczycieli
  ):
  exposure_longitude (exposure_longitude_data),
  exposure_latitude (exposure_latitude_data),
  fire_probs (fire_probs_data),
  fire_spread_prob_vec (fire_spread_prob_vec_data),
  conditional_mean_trend_parameters(conditional_mean_trend_parameters_data),
  conditional_Cov(conditional_Cov_data),
  exposure_insurance (exposure_insurance_data),
  exposure_reassurance (exposure_reassurance_data),
  exposure_sum_value (exposure_sum_value_data),
  wielkosc_pozaru (wielkosc_pozaru_data),
  fakultatywna_input_num (fakultatywna_input_num_data),
  fakultatywna_input_val (fakultatywna_input_val_data),
  obligatoryjna_input_risk (obligatoryjna_input_risk_data),
  vec_obligat_insur_event (vec_obligat_insur_event_data),
  sim (sim),
  kat_val (kat_val),
  ilosc_ubezpieczycieli (ilosc_ubezpieczycieli) {}
  
  
  void RunSimulation () {
    int ind = 0;
    // tutaj ide po symulacjach i przeczytaj komentarz 834
    for (int sim_num = 0; sim_num < sim; sim_num++) {
      for (size_t woj = 0;woj < 17; woj++) {
        for (int mies = 0; mies < 12; mies++) {
          simulateExponsure(woj, mies, ind);
        }}
      std::vector <std::vector<double>> out_sum_vec_out = sim_brutto_final.returnVectorSim();
      std::vector < std::vector < double >> sim_brutto_kat_final_out = sim_brutto_kat_final.returnVectorSim();
      std::vector <std::vector<double>> sim_netto_final_out = sim_netto_final.returnVectorSim();
      std::vector <std::vector<double>> sim_netto_kat_final_out = sim_netto_kat_final.returnVectorSim();
      // to co chcialbym zapisywac to dla kazdego ubezpieczyciela na koniec zapisac
      //out_brutto_final, out_brutto_kat_final, out_netto_final, out_netto_kat_final. Miec te 4 wektory w excel lub csv i kazdego ubezpieczyciela w osobnym.
      // przyjmijmy, ze bedziemy mieli 10 000 symulacji. Jak wezmiemy 0.5 % to wychodzi nam 50. i teraz to co chcialbym miec tez zapisane, to
      // budynki pierwotne i rozprzestrzenione dla kazdego ubezpieczyciela dla tych 50 symulacji na podstawie sum_vec_out, sum_vec_kat_out, sum_netto_out, sum_netto_kat_out.
      // przyklad. ustalam ubezpieczyciela i zapisuje dla niego budynki rozprzestrzenione i pierwotne dla 50 symulacji, w ktorych sum_vec_out jest najwieksze, 
      // czyli to trzeba by trzymac gdzies te budynki i sprawdzac czy otrzymane sum_vec_out jest wieksze od tych, co juz sa trzymane. i tak samo dla oraz sum_vec_kat_out, sum_netto_out, sum_netto_kat_out
      // to pewnie bedzie czasochlonne wiec program powinien miec opcje tego zapisu lub bez.
      for (int kk = 0; kk < ilosc_ubezpieczycieli; kk++) {
        sum_vec_out = accumulate(out_sum_vec_out[kk].begin(), out_sum_vec_out[kk].end(), 0.0);
        sim_brutto_final.clearVector(kk);
        out_brutto_final.addDataVec(kk, sum_vec_out);
        sum_vec_kat_out = accumulate(sim_brutto_kat_final_out[kk].begin(),
                                     sim_brutto_kat_final_out[kk].end(), 0.0);
        sim_brutto_kat_final.clearVector(kk);
        out_brutto_kat_final.addDataVec(kk, sum_vec_kat_out);
        sum_netto_out = accumulate(sim_netto_final_out[kk].begin(), sim_netto_final_out[kk].end(), 0.0);
        sim_netto_final.clearVector(kk);
        out_netto_final.addDataVec(kk, sum_netto_out);
        sum_netto_kat_out = accumulate(sim_netto_kat_final_out[kk].begin(),
                                       sim_netto_kat_final_out[kk].end(), 0.0);
        sim_netto_kat_final.clearVector(kk);
        out_netto_kat_final.addDataVec(kk, sum_netto_kat_out);
      }
      
    }
    
  }
  std::vector<std::vector<double>> returnVAL(){
    std::vector<std::vector<double>> out = out_brutto_final.returnVectorSim();
    return ( out);
  }
};




//[[Rcpp::export]]
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
                                            int ilosc_ubezpieczycieli){
  FireRiskSimulator fireriskSimulator(exponsure_longitude,
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
                                      ilosc_ubezpieczycieli);
  fireriskSimulator.RunSimulation();
  std::vector<std::vector<double>> out = fireriskSimulator.returnVAL();
  return(out);
}

