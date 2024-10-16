#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <iterator>
#include <random>
#include <numeric>
#include <sstream>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <iterator>
#include <random>
#include <numeric>
#include <chrono>
#include <atomic>
#include <mutex>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <iterator>
#include <random>
#include <numeric>
#include <sstream>
#include <deque>
#include <condition_variable>
#include <thread>
#include <functional>
#include <future>
#include <sstream>
#include <iostream>
#include <string>
#include <map>
#include <chrono>
#include <vector>
#include "csvstream.hpp"

class ThreadPool
{
public:
    ThreadPool(unsigned int n);

    template <class F>
    void enqueue(F &&f);
    void waitFinished();
    ~ThreadPool();

    unsigned int getProcessed() const { return processed; }

private:
    std::vector<std::thread> workers;
    std::deque<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable cv_task;
    std::condition_variable cv_finished;
    std::atomic_uint processed;
    unsigned int busy;
    bool stop;

    void thread_proc();
};

ThreadPool::ThreadPool(unsigned int n)
    : busy(), processed(), stop()
{
    for (unsigned int i = 0; i < n; ++i)
        workers.emplace_back(std::bind(&ThreadPool::thread_proc, this));
}

ThreadPool::~ThreadPool()
{

    std::unique_lock<std::mutex> latch(queue_mutex);
    stop = true;
    cv_task.notify_all();
    latch.unlock();

    for (auto &t : workers)
        t.join();
}

void ThreadPool::thread_proc()
{
    while (true)
    {
        std::unique_lock<std::mutex> latch(queue_mutex);
        cv_task.wait(latch, [this]()
                     { return stop || !tasks.empty(); });
        if (!tasks.empty())
        {

            ++busy;

            auto fn = tasks.front();
            tasks.pop_front();

            latch.unlock();

            fn();
            ++processed;

            latch.lock();
            --busy;
            cv_finished.notify_one();
        }
        else if (stop)
        {
            break;
        }
    }
}

template <class F>
void ThreadPool::enqueue(F &&f)
{
    std::unique_lock<std::mutex> lock(queue_mutex);
    tasks.emplace_back(std::forward<F>(f));
    cv_task.notify_one();
}

void ThreadPool::waitFinished()
{
    std::unique_lock<std::mutex> lock(queue_mutex);
    cv_finished.wait(lock, [this]()
                     { return tasks.empty() && (busy == 0); });
}

ThreadPool pool(1);

using namespace std;

const unsigned int fixedSeed = 123456789;
std::mt19937 gen(fixedSeed);

const int numRegions = 17;
const int numMonths = 12;

std::vector<std::vector<std::vector<long double>>> exponsure_longitude(numRegions);
std::vector<std::vector<std::vector<long double>>> exponsure_latitude(numRegions);
std::vector<std::vector<long double>> list_list_wyb(numRegions);
std::vector<std::vector<long double>> fire_spread_prob_vec(4);
std::vector<long double> conditional_mean_trend_parameters(2);
std::vector<long double> conditional_Cov(2);
std::vector<std::vector<std::vector<int>>> exponsure_insurance(numRegions);
std::vector<std::vector<std::vector<int>>> exponsure_reassurance(numRegions);
std::vector<std::vector<std::vector<double>>> exponsure_sum_value(numRegions);
std::vector<std::vector<double>> wielkosc_pozaru(2);
std::vector<std::vector<double>> fakultatywna_input_num;
std::vector<std::vector<std::vector<double>>> fakultatywna_input_val;
std::vector<std::vector<double>> obligatoryjna_input_risk;
std::vector<std::vector<double>> obligatoryjna_input_event;

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
    VectorSim() : data(30, std::vector<double>()) {}

    void addDataVec(int insurane, double value)
    {
        data[insurane].push_back(value);
    }

    std::vector<std::vector<double>> returnVectorSim()
    {
        return (data);
    }

    void clearVector(int num_vec)
    {
        data[num_vec].clear();
    }
};

VectorSim out_brutto_final;

class VectorPozarPierwotny
{
public:
    std::vector<std::vector<long double>> build_fire;
    VectorPozarPierwotny() : build_fire(9, std::vector<long double>()) {}

    void addPozarPierwotny(int insurancer, int nr_budynku, int woj, int mies, int index_table, double wielkosc_pozar_kwota,
                           double reas_fire)
    {
        build_fire[0].push_back(insurancer);
        build_fire[1].push_back(exponsure_longitude[woj - 1][mies - 1][nr_budynku]);
        build_fire[2].push_back(exponsure_latitude[woj - 1][mies - 1][nr_budynku]);
        build_fire[3].push_back(woj + 1);
        build_fire[4].push_back(mies + 1);
        build_fire[5].push_back(exponsure_sum_value[woj - 1][mies - 1][nr_budynku]);
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

double randZeroToOne(int a, int b)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution;
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

int randBin(int size_exp, double prob_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::binomial_distribution<> distrib(size_exp, prob_size);

    return distrib(gen);
}

bool contains(std::vector<bool> vec, int elem)
{
    bool result = false;
    if (find(vec.begin(), vec.end(), elem) != vec.end())
    {
        result = true;
    }
    return result;
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

double reasecuration_build_fire(double exp_fire_pre, int woj, int mies, int nr_budynku)
{

    double reas = exponsure_reassurance[woj][mies][nr_budynku];
    std::vector<double> vec_fakul_insur_num = fakultatywna_input_num[exponsure_insurance[woj][mies][nr_budynku]];
    std::vector<std::vector<double>> vec_fakul_insur_val = fakultatywna_input_val[exponsure_insurance[woj][mies][nr_budynku]];
    std::vector<double> vec_obligat_insur_risk = obligatoryjna_input_risk[exponsure_insurance[woj][mies][nr_budynku]];
    double reas_oblig;
    double b_f;
    double reas_fakultat;
    reas_fakultat = exp_fire_pre;
    reas_oblig = exp_fire_pre;
    if ((reas < 9000))
    {
        b_f = vec_fakul_insur_val[reas][0];
        if (std::find(vec_fakul_insur_num.begin(), vec_fakul_insur_num.end(), reas) != vec_fakul_insur_num.end())
        {
            reas_fakultat = exp_fire_pre * b_f + std::max(0.0, (1 - b_f) * exp_fire_pre - vec_fakul_insur_val[reas][1]);
            reas_oblig = reas_fakultat;
        }
        else
        {
            reas_fakultat = std::min(exp_fire_pre, vec_fakul_insur_val[reas][0]) +
                            std::max(0.0,
                                     exp_fire_pre - vec_fakul_insur_val[reas][0] - vec_fakul_insur_val[reas][1]);
            reas_oblig = reas_fakultat;
        }
    }
    if (floor(vec_obligat_insur_risk[0]) >= 0)
    {
        reas_oblig = calc_reas_bligator(vec_obligat_insur_risk, reas_fakultat);
    }
    return (reas_oblig);
}

std::vector<std::vector<double>> index_spread_build(
    long double lat_center,
    long double lon_center,
    const std::vector<std::vector<double>> &distance_res,
    const std::vector<std::vector<long double>> &lat_ring,
    const std::vector<std::vector<long double>> &lon_ring,
    const std::vector<std::vector<int>> &insu_ring,
    const std::vector<std::vector<int>> &reas_ring,
    const std::vector<std::vector<double>> &exposure_sum_ring)
{
    int exposure_number;

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
    for (int j = 0; j < 9; j++)
    {
        std::vector<int> output_sources_list;
        exposure_number = lat_ring[j].size();
        out_lat_pom = lat_ring[j];
        out_lon_pom = lon_ring[j];
        out_insu_pom = insu_ring[j];
        out_reas_pom = reas_ring[j];
        out_exp_sum_pom = exposure_sum_ring[j];
        distance_res_pom = distance_res[j];
        std::vector<int> ring_exposure_list(exposure_number);
        std::iota(std::begin(ring_exposure_list), std::end(ring_exposure_list), 0);
        if (exposure_number > 0)
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
                if (exposure_number == 1)
                {
                    number_of_fire_spreads[j] = 1;
                    fire_spreads_rings_list[j] = sample_vec(ring_exposure_list, number_of_fire_spreads[j]);
                }
                else if (exposure_number == 2)
                {
                    conditional_mean[j] = conditional_mean_trend_parameters[0] * std::pow(exposure_number - 1, conditional_mean_trend_parameters[1]);
                    number_of_fire_spreads[j] = 1 + randBin(1, conditional_mean[j]);
                    fire_spreads_rings_list[j] = sample_vec(ring_exposure_list, number_of_fire_spreads[j]);
                }
                else if (exposure_number == 3)
                {
                    conditional_mean[j] = conditional_mean_trend_parameters[0] * std::pow(exposure_number - 1, conditional_mean_trend_parameters[1]);
                    alpha[j] = conditional_mean[j] * (exposure_number - 1 - conditional_mean[j] - conditional_Cov[0]) / (conditional_mean[j] + (exposure_number - 1) * (conditional_Cov[0] - 1));
                    beta[j] = (exposure_number - 1 - conditional_mean[j]) * (exposure_number - 1 - conditional_mean[j] - conditional_Cov[0]) / (conditional_mean[j] + (exposure_number - 1) * (conditional_Cov[0] - 1));
                    sftrabbit::beta_distribution<> dist(alpha[j], beta[j]);
                    simulated_probability[j] = dist(gen);
                    number_of_fire_spreads[j] = 1 + randBin(exposure_number - 1, simulated_probability[j]);
                    fire_spreads_rings_list[j] = sample_vec(ring_exposure_list, number_of_fire_spreads[j]);
                }
                else
                {
                    conditional_mean[j] = conditional_mean_trend_parameters[0] * std::pow(exposure_number - 1, conditional_mean_trend_parameters[1]);
                    alpha[j] = conditional_mean[j] * (exposure_number - 1 - conditional_mean[j] - conditional_Cov[1]) / (conditional_mean[j] + (exposure_number - 1) * (conditional_Cov[1] - 1));
                    beta[j] = (exposure_number - 1 - conditional_mean[j]) * (exposure_number - 1 - conditional_mean[j] - conditional_Cov[1]) / (conditional_mean[j] + (exposure_number - 1) * (conditional_Cov[1] - 1));

                    number_of_fire_spreads[j] = 1 + randBin(exposure_number - 1, simulated_probability[j]);
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
                    wielkosc_pozar_kwota = wielkosc_pozar_procent * out_exp_sum_pom[*it];
                    if (wielkosc_pozar_kwota < 500.0)
                    {
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
    return (out_data);
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

std::vector<std::vector<double>> index_in_ring(
    long double lat_center,
    long double lon_center,
    const std::vector<long double> &lat_sub,
    const std::vector<long double> &lon_sub,
    const std::vector<int> &insu_sub,
    const std::vector<int> &reas_sub,
    const std::vector<double> &exponsure_sum_value)
{
    std::vector<std::vector<double>> distance_res(9);
    std::vector<std::vector<long double>> lat_ring(9);
    std::vector<std::vector<long double>> lon_ring(9);
    std::vector<std::vector<int>> insu_ring(9);
    std::vector<std::vector<int>> reas_ring(9);
    std::vector<std::vector<double>> exponsure_sum_ring(9);
    std::vector<std::vector<double>> ind_after_prob(9);
    int n1 = lon_sub.size();
    if (n1 > 0)
    {
        for (int i = 0; i < n1; ++i)
        {
            double res = haversine_cpp(lat_center, lon_center, lat_sub[i], lon_sub[i]);
            if (res < 0.005)
            {
                distance_res[0].push_back(res);
                lat_ring[0].push_back(lat_sub[i]);
                lon_ring[0].push_back(lon_sub[i]);
                insu_ring[0].push_back(insu_sub[i]);
                reas_ring[0].push_back(reas_sub[i]);
                exponsure_sum_ring[0].push_back(exponsure_sum_value[i]);
            }
            if (res > 0.005 && res < 25)
            {
                distance_res[1].push_back(res);
                lat_ring[1].push_back(lat_sub[i]);
                lon_ring[1].push_back(lon_sub[i]);
                insu_ring[1].push_back(insu_sub[i]);
                reas_ring[1].push_back(reas_sub[i]);
                exponsure_sum_ring[1].push_back(exponsure_sum_value[i]);
            }
            if (res > 25 && res < 50)
            {
                distance_res[2].push_back(res);
                lat_ring[2].push_back(lat_sub[i]);
                lon_ring[2].push_back(lon_sub[i]);
                insu_ring[2].push_back(insu_sub[i]);
                reas_ring[2].push_back(reas_sub[i]);
                exponsure_sum_ring[2].push_back(exponsure_sum_value[i]);
            }
            if (res > 50 && res < 75)
            {
                distance_res[3].push_back(res);
                lat_ring[3].push_back(lat_sub[i]);
                lon_ring[3].push_back(lon_sub[i]);
                insu_ring[3].push_back(insu_sub[i]);
                reas_ring[3].push_back(reas_sub[i]);
                exponsure_sum_ring[3].push_back(exponsure_sum_value[i]);
            }
            if (res > 75 && res < 100)
            {
                distance_res[4].push_back(res);
                lat_ring[4].push_back(lat_sub[i]);
                lon_ring[4].push_back(lon_sub[i]);
                insu_ring[4].push_back(insu_sub[i]);
                reas_ring[4].push_back(reas_sub[i]);
                exponsure_sum_ring[4].push_back(exponsure_sum_value[i]);
            }
            if (res > 100 && res < 125)
            {
                distance_res[5].push_back(res);
                lat_ring[5].push_back(lat_sub[i]);
                lon_ring[5].push_back(lon_sub[i]);
                insu_ring[5].push_back(insu_sub[i]);
                reas_ring[5].push_back(reas_sub[i]);
                exponsure_sum_ring[5].push_back(exponsure_sum_value[i]);
            }
            if (res > 125 && res < 150)
            {
                distance_res[6].push_back(res);
                lat_ring[6].push_back(lat_sub[i]);
                lon_ring[6].push_back(lon_sub[i]);
                insu_ring[6].push_back(insu_sub[i]);
                reas_ring[6].push_back(reas_sub[i]);
                exponsure_sum_ring[6].push_back(exponsure_sum_value[i]);
            }
            if (res > 150 && res < 175)
            {
                distance_res[7].push_back(res);
                lat_ring[7].push_back(lat_sub[i]);
                lon_ring[7].push_back(lon_sub[i]);
                insu_ring[7].push_back(insu_sub[i]);
                reas_ring[7].push_back(reas_sub[i]);
                exponsure_sum_ring[7].push_back(exponsure_sum_value[i]);
            }
            if (res > 175 && res < 200)
            {
                distance_res[8].push_back(res);
                lat_ring[8].push_back(lat_sub[i]);
                lon_ring[8].push_back(lon_sub[i]);
                insu_ring[8].push_back(insu_sub[i]);
                reas_ring[8].push_back(reas_sub[i]);
                exponsure_sum_ring[8].push_back(exponsure_sum_value[i]);
            }
        }
    }
    ind_after_prob = index_spread_build(lat_center, lon_center, distance_res, lat_ring, lon_ring, insu_ring,
                                        reas_ring, exponsure_sum_ring);
    return (ind_after_prob);
}

std::vector<std::vector<double>> haversine_loop_cpp_vec(
    double radius,
    int n1, int woj, int mies)

{
    long double lat_center = exponsure_latitude[woj][mies][n1];
    long double lon_center = exponsure_longitude[woj][mies][n1];
    int circumference_earth_in_meters = 40075000;
    double one_lat_in_meters = circumference_earth_in_meters * 0.002777778;
    double one_lon_in_meters = circumference_earth_in_meters * cos(lat_center * 0.01745329) * 0.002777778;
    double south_lat = lat_center - radius / one_lat_in_meters;
    double north_lat = lat_center + radius / one_lat_in_meters;
    double west_lon = lon_center - radius / one_lon_in_meters;
    double east_lon = lon_center + radius / one_lon_in_meters;
    int n = exponsure_longitude[woj][mies].size();

    std::vector<long double> lat_sub;
    std::vector<long double> lon_sub;
    std::vector<int> insu_sub;
    std::vector<int> reas_sub;
    std::vector<double> premium_sub;
    std::vector<int> ind_spread_build;
    std::vector<std::vector<double>> ind_ring(11);
    bool logical_value;

    for (int i = 0; i < n; i++)
    {
        logical_value = !((exponsure_longitude[woj][mies][i] > east_lon) || (exponsure_longitude[woj][mies][i] < west_lon) || (exponsure_latitude[woj][mies][i] < south_lat) || (exponsure_latitude[woj][mies][i] > north_lat));
        if (logical_value)
        {
            lat_sub.push_back(exponsure_latitude[woj][mies][i]);
            lon_sub.push_back(exponsure_longitude[woj][mies][i]);
            insu_sub.push_back(exponsure_insurance[woj][mies][i]);
            reas_sub.push_back(exponsure_reassurance[woj][mies][i]);
            premium_sub.push_back(exponsure_sum_value[woj][mies][i]);
        }
    }

    if (lat_sub.size() > 0)
    {
        ind_ring = index_in_ring(lat_center, lon_center, lat_sub, lon_sub,

                                 insu_sub, reas_sub, premium_sub);
    }
    return (ind_ring);
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
                                                 int ilosc_ubezpieczycieli)
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
                                                        int size_vec, int ilosc_ubezpieczycieli)
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
        vec_obligat = obligatoryjna_input_event[i];
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

std::mutex g_num_mutex;
VectorSim out_brutto_kat_final;
VectorSim out_netto_kat_final;
VectorSim out_netto_final;

void simulateExponsure(int sim, double kat_val, int ilosc_ubezpieczycieli)
{
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

    VectorSim sim_brutto_final;
    VectorSim sim_brutto_kat_final;
    VectorSim sim_netto_final;
    VectorSim sim_netto_kat_final;
    VectorPozarPierwotny buildPierwotny;

    for (size_t woj = 0; woj < 17; woj++)
    {
        for (int mies = 0; mies < 12; mies++)
        {
            int index_table = 0;
            exposure_number = exponsure_longitude[woj][mies].size();
            if (exposure_number > 0)
            {
                binom_fire = randBin(exposure_number, list_list_wyb[woj][mies]);
                if (binom_fire > 0)
                {
                    std::vector<int> fire_sources_list(binom_fire);

                    std::vector<int> pom_index_fire(exposure_number);
                    std::iota(std::begin(pom_index_fire), std::end(pom_index_fire), 0);

                    fire_sources_list = sample_vec(pom_index_fire, binom_fire);

                    for (size_t itx = 0; itx < fire_sources_list.size(); itx++)
                    {
                        auto nr_budynku = fire_sources_list[itx];

                        std::vector<std::vector<double>> spread_one_building(11);

                        spread_one_building = haversine_loop_cpp_vec(200,
                                                                     nr_budynku,
                                                                     woj, mies);

                        wielkosc_pozar_procent = percentage_of_loss(wielkosc_pozaru);
                        wielkosc_pozar_kwota = wielkosc_pozar_procent * exponsure_sum_value[woj][mies][nr_budynku];

                        if (wielkosc_pozar_kwota < 500.0)
                            wielkosc_pozar_kwota = 500.0;

                        reas_fire = reasecuration_build_fire(wielkosc_pozar_kwota, woj, mies, nr_budynku);

                        insurancer = exponsure_insurance[woj][mies][nr_budynku];
                        buildPierwotny.addPozarPierwotny(insurancer, nr_budynku,
                                                         woj + 1, mies + 1, index_table,
                                                         wielkosc_pozar_kwota, reas_fire);
                        sim_brutto_final.addDataVec(insurancer, wielkosc_pozar_kwota);
                        sim_netto_final.addDataVec(insurancer, wielkosc_pozar_kwota);
                        reas_fire_kat = 0.0;
                        if (wielkosc_pozar_kwota > kat_val)
                        {
                            sim_brutto_kat_final.addDataVec(insurancer, wielkosc_pozar_kwota);
                            reas_fire_kat = reas_fire;
                        }

                        len_spread = 0;
                        len_spread = spread_one_building[4].size();
                        if (len_spread > 0)
                        {
                            std::vector<std::vector<std::vector<double>>> out_vec_brutto(6);
                            out_vec_brutto = calc_brutto_ring(spread_one_building[6], spread_one_building[3], kat_val,
                                                              ilosc_ubezpieczycieli);
                            std::vector<std::vector<double>> reas_risk = reasurance_risk(out_vec_brutto[0],
                                                                                         spread_one_building[4],
                                                                                         ilosc_ubezpieczycieli);
                            std::vector<std::vector<double>> reas_event = calc_reas_obliga_event(insurancer, reas_fire,
                                                                                                 out_vec_brutto[4],
                                                                                                 reas_risk, len_spread,
                                                                                                 ilosc_ubezpieczycieli);
                            sim_netto_final.addDataVec(insurancer, reas_event[0][0]);
                            std::vector<std::vector<double>> reas_risk_kat = reasurance_risk(out_vec_brutto[1],
                                                                                             spread_one_building[4],
                                                                                             ilosc_ubezpieczycieli);
                            std::vector<std::vector<double>> reas_event_kat = calc_reas_obliga_event(insurancer,
                                                                                                     reas_fire_kat,
                                                                                                     out_vec_brutto[5],
                                                                                                     reas_risk_kat,
                                                                                                     len_spread,
                                                                                                     ilosc_ubezpieczycieli);
                            sim_netto_kat_final.addDataVec(insurancer, reas_event_kat[0][0]);
                            for (int pp = 0; pp < ilosc_ubezpieczycieli; pp++)
                            {

                                sim_brutto_final.addDataVec(pp, out_vec_brutto[2][pp][0]);
                                sim_brutto_kat_final.addDataVec(pp, out_vec_brutto[3][pp][0]);
                                sim_netto_final.addDataVec(pp, reas_event[2][pp]);
                                sim_netto_kat_final.addDataVec(pp, reas_event_kat[2][pp]);
                            }

                        }
                        index_table += 1;
                    }
                }
            }
        }
    }

    std::vector<std::vector<double>> out_sum_vec_out = sim_brutto_final.returnVectorSim();
    std::vector<std::vector<double>> sim_brutto_kat_final_out = sim_brutto_kat_final.returnVectorSim();
    std::vector<std::vector<double>> sim_netto_final_out = sim_netto_final.returnVectorSim();
    std::vector<std::vector<double>> sim_netto_kat_final_out = sim_netto_kat_final.returnVectorSim();

    g_num_mutex.lock();
    for (int kk = 0; kk < ilosc_ubezpieczycieli; kk++)
    {
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

    g_num_mutex.unlock();
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
    std::random_device rd;

    std::uniform_real_distribution<long double> dist(min, max);
    std::vector<long double> values(count);
    for (auto &val : values)
    {
        val = dist(gen);
    }
    return values;
}

const std::string FOLDER_REAS = "csv/Reasekuracja/";
const std::string FOLDER_UBEZP = "csv/Ubezpieczyciele/";

void processPrPozaru(const std::string &filename)
{
    csvstream csvin(filename);

    std::map<std::string, std::string> row;

    try
    {
        while (csvin >> row)
        {
            list_list_wyb[0].emplace_back(std::stod(row["1"]));
            list_list_wyb[1].emplace_back(std::stod(row["2"]));
            list_list_wyb[2].emplace_back(std::stod(row["3"]));
            list_list_wyb[3].emplace_back(std::stod(row["4"]));
            list_list_wyb[4].emplace_back(std::stod(row["5"]));
            list_list_wyb[5].emplace_back(std::stod(row["6"]));
            list_list_wyb[6].emplace_back(std::stod(row["7"]));
            list_list_wyb[7].emplace_back(std::stod(row["8"]));
            list_list_wyb[8].emplace_back(std::stod(row["9"]));
            list_list_wyb[9].emplace_back(std::stod(row["10"]));
            list_list_wyb[10].emplace_back(std::stod(row["11"]));
            list_list_wyb[11].emplace_back(std::stod(row["12"]));
            list_list_wyb[12].emplace_back(std::stod(row["13"]));
            list_list_wyb[13].emplace_back(std::stod(row["14"]));
            list_list_wyb[14].emplace_back(std::stod(row["15"]));
            list_list_wyb[15].emplace_back(std::stod(row["16"]));
            list_list_wyb[16].emplace_back(std::stod(row["17"]));
        }

    }
    catch (const std::invalid_argument &e)
    {
        std::cerr << "Error: Invalid argument for stoi or stod conversion 2." << std::endl;
    }
}

void processPrRozprzestrzenienia(const std::string &filename)
{
    csvstream csvin(filename);

    std::map<std::string, std::string> row;

    try
    {
        int cnt = 0;
        while (csvin >> row)
        {
            fire_spread_prob_vec[cnt].emplace_back(std::stod(row["0"]));
            fire_spread_prob_vec[cnt].emplace_back(std::stod(row["(0,25]"]));
            fire_spread_prob_vec[cnt].emplace_back(std::stod(row["(25,50]"]));
            fire_spread_prob_vec[cnt].emplace_back(std::stod(row["(50,75]"]));
            fire_spread_prob_vec[cnt].emplace_back(std::stod(row["(75,100]"]));
            fire_spread_prob_vec[cnt].emplace_back(std::stod(row["(100,125]"]));
            fire_spread_prob_vec[cnt].emplace_back(std::stod(row["(125,150]"]));
            fire_spread_prob_vec[cnt].emplace_back(std::stod(row["(150,175]"]));
            fire_spread_prob_vec[cnt].emplace_back(std::stod(row["(175,200]"]));

            if (cnt == 0)
            {
                conditional_mean_trend_parameters[0] = (std::stod(row["a1"]));
                conditional_mean_trend_parameters[1] = (std::stod(row["b1"]));

                conditional_Cov[0] = (std::stod(row["a2"]));
                conditional_Cov[1] = (std::stod(row["b2"]));
            }
            cnt++;
        }

    }

    catch (const std::invalid_argument &e)
    {
        std::cerr << "Error: Invalid argument for stoi or stod conversion 2." << std::endl;
    }
}

void processPrWielkoscPozaru(const std::string &filename)
{

    csvstream csvin(filename);

    std::map<std::string, std::string> row;

    while (csvin >> row)
    {
        wielkosc_pozaru[0].emplace_back(std::stod(row["Rozmiar"]));
        wielkosc_pozaru[1].emplace_back(std::stod(row["Prawdopodobienstwo"]));
    }

}

void processOblig(const std::vector<std::string> &filename)
{

    for (int i = 0; i < filename.size(); i++)
    {
        obligatoryjna_input_risk.push_back(std::vector<double>());
        obligatoryjna_input_event.push_back(std::vector<double>());

        for (int j = 0; j < 4; j++)
        {
            obligatoryjna_input_risk[i].push_back(0);
            obligatoryjna_input_event[i].push_back(0);
        }
    }

    for (int i = 0; i < filename.size(); i++)
    {
        csvstream csvin(FOLDER_REAS + filename[i] + ".csv");

        std::map<std::string, std::string> row;

        int cnt = 0;
        while (csvin >> row)
        {
            if (cnt == 0)
            {
                obligatoryjna_input_risk[i][3] = std::stod(row["Udzial (ryzyko)"]);
                obligatoryjna_input_event[i][3] = std::stod(row["Udzial (zdarzenie)"]);
            }
            else
            {
                obligatoryjna_input_risk[i][2] = std::stod(row["Udzial (ryzyko)"]);
                obligatoryjna_input_event[i][2] = std::stod(row["Udzial (zdarzenie)"]);
            }
            obligatoryjna_input_risk[i][0] = std::stod(row["Od (ryzyko)"]);
            obligatoryjna_input_risk[i][1] = std::stod(row["Do (ryzyko)"]);

            obligatoryjna_input_event[i][0] = std::stod(row["Od (zdarzenie)"]);
            obligatoryjna_input_event[i][1] = std::stod(row["Do (zdarzenie)"]);
            cnt++;
            if (cnt == 2)
                break;
        }
    }

}

int extractMonth(const std::string &date)
{
    std::istringstream dateStream(date);
    std::string segment;
    std::getline(dateStream, segment, '.');
    std::getline(dateStream, segment, '.');
    return std::stoi(segment);
}

void processRow(const std::string &startDate, const std::string &endDate, int region, double latitude, double longitude, int reassurance, double sumValue, int insurance)
{
    int startMonth = extractMonth(startDate) - 1;
    int endMonth = extractMonth(endDate) - 1;

    for (int month = startMonth; month <= endMonth; ++month)
    {
        exponsure_latitude[region][month].push_back(latitude);
        exponsure_longitude[region][month].push_back(longitude);
        exponsure_insurance[region][month].push_back(insurance);
        exponsure_reassurance[region][month].push_back(reassurance);
        exponsure_sum_value[region][month].push_back(sumValue);
    }
}

bool is_date_in_year(const std::string &date_str, int year)
{
    int date_year = std::stoi(date_str.substr(6, 4));
    return date_year == year;
}

void get_dates_within_year(std::string &date_str_a, std::string &date_str_b, int year)
{
    std::string start_of_year = "01.01." + std::to_string(year);
    std::string end_of_year = "31.12." + std::to_string(year);

    bool date_a_in_year = is_date_in_year(date_str_a, year);
    bool date_b_in_year = is_date_in_year(date_str_b, year);

    if (date_a_in_year && date_b_in_year)
    {
        return;
    }
    else if (date_a_in_year)
    {
        date_str_b = end_of_year;
        return;
    }
    else if (date_b_in_year)
    {
        date_str_a = start_of_year;
        return;
    }
    else
    {
        date_str_a = start_of_year;
        date_str_b = end_of_year;
        return;
    }
}

void processBudynki(const std::vector<std::string> &filename, std::string year)
{

    for (int i = 0; i < filename.size(); i++)
    {
        csvstream csvin(FOLDER_UBEZP + filename[i] + ".csv");

        std::map<std::string, std::string> row;

        int id_ubezp = 0;
        try
        {
            while (csvin >> row)
            {

                std::string dataPoczatku = row["DataPoczatku"];
                std::string dataKonca = row["DataKonca"];
                int reasekuracjaf = 9999;
                try
                {
                    reasekuracjaf = std::stoi(row["ReasekuracjaF"]);
                }
                catch (const std::invalid_argument &e)
                {
                    reasekuracjaf = 9999;
                }
                get_dates_within_year(dataPoczatku, dataKonca, std::stoi(year));
                processRow(
                    dataPoczatku,
                    dataKonca,
                    std::stoi(row["WojUjednolicone"]),
                    std::stod(row["Szerokosc"]),
                    std::stod(row["Dlugosc"]),
                    reasekuracjaf,
                    std::stod(row["SumaUbezpieczenia"]),
                    id_ubezp);
            }
        }
        catch (const std::invalid_argument &e)
        {
            std::cerr << "Error: Invalid argument for stoi or stod conversion 1." << std::endl;
        }
    }
}

void print3DVector(const std::vector<std::vector<std::vector<double>>> &vec)
{
    for (const auto &matrix : vec)
    {
        for (const auto &row : matrix)
        {
            for (double element : row)
            {
                std::cout << element << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "----" << std::endl;
    }
}

void processReas(const std::vector<std::string> &filename)
{

    fakultatywna_input_num.resize(filename.size());
    for (int i = 0; i < filename.size(); i++)
    {
        csvstream csvin(FOLDER_REAS + filename[i] + ".csv");

        std::map<std::string, std::string> row;

        std::vector<std::vector<double>> first_outer_vector;

        while (csvin >> row)
        {
            if ((row["ZachowekKwota"]) == "")
            {

                fakultatywna_input_num[i].push_back(std::stoi(row["Lp"]));
                first_outer_vector.push_back({std::stod(row["ZachowekProcent"]), std::stod(row["Pojemnosc"])});
            }
            else
            {
                first_outer_vector.push_back({std::stod(row["ZachowekKwota"]), std::stod(row["Pojemnosc"])});
            }
        }

        fakultatywna_input_val.push_back(first_outer_vector);
    }

    print3DVector(fakultatywna_input_val);
}

int main()
{

    std::random_device rd;

    std::setlocale(LC_ALL, "nb_NO.UTF-8");

    for (int woj = 0; woj < 17; ++woj)
    {
        exponsure_longitude[woj].resize(12);
        exponsure_latitude[woj].resize(12);
        exponsure_insurance[woj].resize(12);
        exponsure_reassurance[woj].resize(12);
        exponsure_sum_value[woj].resize(12);
    }

    std::string line;
    std::vector<std::string> fileNames;
    std::string year;
    std::cout << "Podaj rok, ktory ma byc brany pod uwage: ";
    std::getline(std::cin, year);

    std::cout << "Wprowadz nazwy plikow po spacji: ";
    std::getline(std::cin, line);

    std::istringstream iss(line);
    std::string fileName;
    while (iss >> fileName)
    {
        fileNames.push_back(fileName);
    }

    std::cout << "Indeksy przydzielone plikom:\n";
    for (int i = 0; i < fileNames.size(); ++i)
    {
        std::cout << "Indeks " << i << ": " << fileNames[i] << std::endl;
    }

    processReas(fileNames);
    processOblig(fileNames);
    processBudynki(fileNames, year);

    processPrPozaru("csv/Pr_pozaru.csv");
    processPrRozprzestrzenienia("csv/pr_rozprzestrzenienia.csv");
    processPrWielkoscPozaru("csv/pr_wielkosc_pozaru.csv");

    auto start = std::chrono::high_resolution_clock::now();

    int sim = 100;
    double kat_val = 500000;
    int ilosc_ubezpieczycieli = fileNames.size();

    for (int sim_num = 0; sim_num < sim; sim_num++)
    {
        pool.enqueue([sim, kat_val, ilosc_ubezpieczycieli]()
                     { simulateExponsure(sim, kat_val, ilosc_ubezpieczycieli); });
    }

    pool.waitFinished();

    std::vector<std::vector<double>> vec = out_brutto_final.returnVectorSim();

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