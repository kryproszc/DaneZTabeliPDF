#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include "libxl.h"

using namespace libxl;

const double PI = 3.14159265359;
const double EARTH_RADIUS = 6371000;

double distanceCalculation(double lat1Rad, double lat2Rad, double deltaLong) {
    double a = sin(deltaLat/2) * sin(deltaLat/2) + cos(lat1Rad) * cos(lat2Rad) * sin(deltaLong/2) * sin(deltaLong/2);
    double c = 2 * atan2(sqrt(a), sqrt(1-a));
    return EARTH_RADIUS * c;
}

std::vector<int> fireSpreadSimulation(double lat, double lon, const std::vector<double> &exposure, const std::vector<double> &exposureLat, const std::vector<double> &exposureLon) {
    double latRad = lat * PI / 180.0;
    double longArc = acos(1 - (200.0 * 200.0) / (2 * EARTH_RADIUS * EARTH_RADIUS * (PI/2 - latRad) * (PI/2 - latRad)));
    double latMinus200 = lat - 200.0 / EARTH_RADIUS;
    double latPlus200 = lat + 200.0 / EARTH_RADIUS;
    double lonMinusArc = lon - longArc;
    double lonPlusArc = lon + longArc;

    std::vector<int> exposureHelper;
    for (int i = 0; i < exposureLat.size(); ++i) {
        double dist = distanceCalculation(latRad, exposureLat[i] * PI / 180.0, (exposureLon[i] - lon) * PI / 180.0);
        if (dist >= latMinus200 && dist <= latPlus200 && exposureLon[i] <= lonPlusArc && exposureLon[i] >= lonMinusArc) {
            exposureHelper.push_back(i);
        }
    }

    std::vector<int> exposureSpreadSquare;
    if (exposureHelper.size() > 1000) {
        std::vector<std::vector<int>> ringExposureList(9, std::vector<int>());
        std::vector<int> ringExposureNumbers(9, 0);

        for (int i = 0; i < exposureHelper.size(); ++i) {
            double dist = distanceCalculation(latRad, exposureLat[exposureHelper[i]] * PI / 180.0, (exposureLon[exposureHelper[i]] - lon) * PI / 180.0);
            int ringIndex = std::min(static_cast<int>(dist / 25), 8);
            ringExposureNumbers[ringIndex]++;
            ringExposureList[ringIndex].push_back(exposureHelper[i]);
        }

        std::vector<int> numberOfFireSpreads(9, 0);
        std::vector<std::vector<int>> fireSpreadsRingsList(9, std::vector<int>());

        for (int j = 0; j < 9; ++j) {
            numberOfFireSpreads[j] = std::rand() % ringExposureNumbers[j];
            if (numberOfFireSpreads[j] > 0) {
                std::vector<int> indices(ringExposureNumbers[j]);
                for (int i = 0; i < ringExposureNumbers[j]; ++i) {
                    indices[i] = i;
                }
                std::random_shuffle(indices.begin(), indices.end());
                for (int i = 0; i < numberOfFireSpreads[j]; ++i) {
                    fireSpreadsRingsList[j].push_back(ringExposureList[j][indices[i]]);
                }
            }
        }

        exposureSpreadSquare.push_back(fireSpreadsRingsList.size());
        for (int j = 0; j < fireSpreadsRingsList.size(); ++j) {
            for (int i = 0; i < fireSpreadsRingsList[j].size(); ++i) {
                exposureSpreadSquare.push_back(fireSpreadsRingsList[j][i]);
            }
        }
    }

    return exposureSpreadSquare;
}

std::vector<int> stochasticFireProcessSimulation(int level, const std::vector<std::vector<int>> &exposureList, const std::vector<std::vector<double>> &exposureLatitude,
                                                  const std::vector<std::vector<double>> &exposureLongitude, const std::vector<std::vector<double>> &fireProbsList) {
    std::vector<int> exposureNumber(level, 0);
    std::vector<std::vector<int>> totalFireList(level, std::vector<int>());
    std::vector<int> numberOfFires(level, 0);
    std::vector<std::vector<int>> fireSourcesList(level, std::vector<int>());
    std::vector<int> numberOfFireSpreads;

    for (int i = 0; i < level; ++i) {
        exposureNumber[i] = exposureList[i].size();
        numberOfFires[i] = std::rand() % exposureNumber[i];
        fireSourcesList[i].resize(numberOfFires[i]);
        std::random_shuffle(exposureList[i].begin(), exposureList[i].end());
        for (int j = 0; j < numberOfFires[i]; ++j) {
            fireSourcesList[i][j] = exposureList[i][j];
        }
        if (i == 0) {
            totalFireList[i] = fireSourcesList[i];
        } else {
            std::vector<int> temp;
            for (int j = 0; j < fireSourcesList[i].size(); ++j) {
                if (std::find(totalFireList[i - 1].begin(), totalFireList[i - 1].end(), fireSourcesList[i][j]) == totalFireList[i - 1].end()) {
                    temp.push_back(fireSourcesList[i][j]);
                }
            }
            fireSourcesList[i] = temp;
            totalFireList[i].insert(totalFireList[i].end(), fireSourcesList[i].begin(), fireSourcesList[i].end());
        }
        if (fireSourcesList[i].size() > 0) {
            for (int m = 0; m < fireSourcesList[i].size(); ++m) {
                if (fireSourcesList[i][m] >= 0 && fireSourcesList[i][m] < exposureLatitude.size()) {
                    std::vector<int> fireSpreads = fireSpreadSimulation(exposureLatitude[fireSourcesList[i][m]], exposureLongitude[fireSourcesList[i][m]],
                                                                      exposureList[i], exposureLatitude[i], exposureLongitude[i]);
                    std::vector<int> temp;
                    for (int j = 0; j < fireSpreads.size(); ++j) {
                        if (std::find(totalFireList[i].begin(), totalFireList[i].end(), fireSpreads[j]) == totalFireList[i].end()) {
                            temp.push_back(fireSpreads[j]);
                        }
                    }
                    numberOfFireSpreads.push_back(temp.size());
                    totalFireList[i].insert(totalFireList[i].end(), temp.begin(), temp.end());
                }
            }
        }
    }
    return numberOfFireSpreads;
}

void processSimulationResults(const std::vector<int>& results) {
    std::cout << "Wyniki symulacji: ";
    for (int i = 0; i < results.size(); ++i) {
        std::cout << results[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    Book* book = xlCreateBook();
    if (book) {
        if (book->load(L"prawdopodobienstwa.xlsx")) {
            Sheet* sheet1 = book->getSheet(0);
            Sheet* sheet2 = book->getSheet(1);
            Sheet* sheet3 = book->getSheet(2);

            if (sheet1 && sheet2 && sheet3) {
                std::vector<std::vector<double>> prawdopodobienstwa;
                for (int row = 0; row < sheet1->lastRow(); ++row) {
                    std::vector<double> rowValues;
                    for (int col = 0; col < sheet1->lastCol(); ++col) {
                        double value = sheet1->readNum(row, col);
                        rowValues.push_back(value);
                    }
                    prawdopodobienstwa.push_back(rowValues);
                }

                std::vector<std::vector<double>> rozprzestrzenianiePozarow;
                for (int row = 0; row < sheet2->lastRow(); ++row) {
                    std::vector<double> rowValues;
                    for (int col = 0; col < sheet2->lastCol(); ++col) {
                        double value = sheet2->readNum(row, col);
                        rowValues.push_back(value);
                    }
                    rozprzestrzenianiePozarow.push_back(rowValues);
                }

                std::vector<std::vector<double>> daneBudynkow;
                for (int row = 0; row < sheet3->lastRow(); ++row) {
                    double szerokosc = sheet3->readNum(row, 1);
                    double dlugosc = sheet3->readNum(row, 2);
                    std::vector<double> budynek = {szerokosc, dlugosc};
                    daneBudynkow.push_back(budynek);
                }

                int level = prawdopodobienstwa.size();
                for (int k = 0; k < level; ++k) {
                    // Brakujący fragment kodu R zastąpiony funkcją stochasticFireProcessSimulation
                    std::vector<int> result = stochasticFireProcessSimulation(k, basicExposureList[k], exposureLatitude[k],
                                                                              exposureLongitude[k], fireProbs[k]);
                    processSimulationResults(result);
                }
            }
        } else {
            std::cerr << "Nie udało się wczytać pliku Excela." << std::endl;
        }

        book->release();
    } else {
        std::cerr << "Nie udało się utworzyć obiektu książki." << std::endl;
    }

    return 0;
}
