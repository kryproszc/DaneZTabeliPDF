#include "csvstream.hpp"
#include <iostream>
#include <string>
#include <map>
#include <chrono>
#include <vector>

std::vector<std::vector<long double>> list_list_wyb(17);

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

        std::cout << list_list_wyb[0][0] << std::endl;
    }
    catch (const std::invalid_argument &e)
    {
        std::cerr << "Error: Invalid argument for stoi or stod conversion 2." << std::endl;
    }
}









std::vector<std::vector<long double>> fire_spread_prob_vec(4);
std::vector<long double> conditional_mean_trend_parameters(2);
std::vector<long double> conditional_Cov(2);

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

        std::cout << conditional_mean_trend_parameters[0] << std::endl;

        for (const auto& inner_vec : fire_spread_prob_vec) {
            for (const auto& value : inner_vec) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }
    }

    catch (const std::invalid_argument &e)
    {
        std::cerr << "Error: Invalid argument for stoi or stod conversion 2." << std::endl;
    }
}








std::vector<std::vector<double>> wielkosc_pozaru(2);
void processPrWielkoscPozaru(const std::string &filename)
{

    csvstream csvin(filename);

    std::map<std::string, std::string> row;

    while (csvin >> row)
    {
        wielkosc_pozaru[0].emplace_back(std::stod(row["Rozmiar"]));
        wielkosc_pozaru[1].emplace_back(std::stod(row["Prawdopodobienstwo"]));
    }

    std::cout << wielkosc_pozaru[0][0] << std::endl;
}









int liczba_ubezp = 2;
std::vector<std::vector<double>> obligatoryjna_input_risk;
std::vector<std::vector<double>> obligatoryjna_input_event;

void processOblig(const std::string &filename)
{

    for (int i = 0; i < liczba_ubezp; i++)
    {
        obligatoryjna_input_risk.push_back(std::vector<double>());
        obligatoryjna_input_event.push_back(std::vector<double>());

        for (int j = 0; j < 4; j++)
        {
            obligatoryjna_input_risk[i].push_back(0);
            obligatoryjna_input_event[i].push_back(0);
        }
    }

    csvstream csvin(filename);

    std::map<std::string, std::string> row;

    int cnt = 0;
    while (csvin >> row)
    {
        std::cout << std::stod(row["ZachowekProcent"]) << "\n";

        if (cnt == 0)
        {
            obligatoryjna_input_risk[0][3] = std::stod(row["Udzial (ryzyko)"]);
        }
        else
        {
            obligatoryjna_input_risk[0][2] = std::stod(row["Udzial (ryzyko)"]);
        }
        obligatoryjna_input_risk[0][0] = std::stod(row["Od (ryzyko)"]);
        obligatoryjna_input_risk[0][1] = std::stod(row["Do (ryzyko)"]);
        cnt++;
        if (cnt == 2)
            break;
    }

    while (csvin >> row)
    {
        if (!row["ZachowekProcent"].empty()) {
            std::cout << std::stod(row["ZachowekProcent"]) << "\n";

        }
    }

   // std::cout << obligatoryjna_input_risk[0][0] << std::endl;
}






const int numRegions = 17;

std::vector<std::vector<std::vector<long double>>> exponsure_longitude(numRegions);
std::vector<std::vector<std::vector<long double>>> exponsure_latitude(numRegions);
std::vector<std::vector<std::vector<int>>> exponsure_insurance(numRegions);
std::vector<std::vector<std::vector<int>>> exponsure_reassurance(numRegions);
std::vector<std::vector<std::vector<double>>> exponsure_sum_value(numRegions);

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
    int startMonth = extractMonth(startDate)-1;
    int endMonth = extractMonth(endDate)-1;

    for (int month = startMonth; month <= endMonth; ++month)
    {
        exponsure_latitude[region][month].push_back(latitude);
        exponsure_longitude[region][month].push_back(longitude);
        exponsure_insurance[region][month].push_back(insurance);
        exponsure_reassurance[region][month].push_back(reassurance);
        exponsure_sum_value[region][month].push_back(sumValue);
    }
}

void processBudynki(const std::string &filename)
{

    csvstream csvin(filename);

    std::map<std::string, std::string> row;

    int id_ubezp = 0;
    try
    {
        while (csvin >> row)
        {
              std::cout << std::setprecision(15)<< std::stod(row["Dlugosc"]) << std::endl;
            processRow(
                row["DataPoczatku"],
                row["DataKonca"],
                std::stoi(row["WojUjednolicone"]),
                std::stod(row["Szerokosc"]),
                std::stod(row["Dlugosc"]),
                std::stoi(row["ReasekuracjaF"]),
                std::stod(row["SumaUbezpieczenia"]),
                id_ubezp);
        }
    }

    catch (const std::invalid_argument &e)
    {
        std::cerr << "Error: Invalid argument for stoi or stod conversion 1." << std::endl;
    }
}

int main()
{
    std::setlocale(LC_ALL, "nb_NO.UTF-8");


    for (int woj = 0; woj < 17; ++woj)
    {
        exponsure_longitude[woj].resize(12);
        exponsure_latitude[woj].resize(12);
        exponsure_insurance[woj].resize(12);
        exponsure_reassurance[woj].resize(12);
        exponsure_sum_value[woj].resize(12);
    }


   // processPrPozaru("csv/pr_pozaru_new.csv");
   // processPrRozprzestrzenienia("csv/pr_rozprzestrzenienia_new.csv");
    processOblig("csv/Reasekuracaja_new.csv");
    //processBudynki("csv/UNIQA_real.csv");
    //processPrWielkoscPozaru("csv/pr_wielkosc_pozaru_new.csv");



    return 0;
}