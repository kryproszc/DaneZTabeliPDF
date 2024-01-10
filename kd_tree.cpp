#include <iostream>
#include <vector>
#include <xlsxio_read.h>

using namespace std;

vector<vector<double>> readExcel(string filename, string sheetname, string range) {
    xlsxioreader xls;
    vector<vector<double>> data;

    xls = xlsxioread_open(filename.c_str());
    if (xls == NULL) {
        cerr << "Błąd podczas otwierania pliku Excel." << endl;
        return data;
    }

    if (xlsxioread_list_sheets(xls)) {
        cerr << "Błąd podczas listowania arkuszy." << endl;
        xlsxioread_close(xls);
        return data;
    }

    if (xlsxioread_process(xls, XLSXIOREAD_SKIP_EMPTY_ROWS) != XLSXIOREAD_SUCCESS) {
        cerr << "Błąd podczas przetwarzania pliku Excel." << endl;
        xlsxioread_close(xls);
        return data;
    }

    while (xlsxioread_process(xls, XLSXIOREAD_NEXT_ROW) == XLSXIOREAD_SUCCESS) {
        vector<double> row;
        if (range.empty() || xlsxioread_sheet_open(xls, sheetname.c_str(), XLSXIOREAD_SKIP_EMPTY_ROWS) == XLSXIOREAD_SUCCESS) {
            while (xlsxioread_process(xls, XLSXIOREAD_NEXT_CELL) == XLSXIOREAD_SUCCESS) {
                if (xlsxioread_sheet_name(xls) == sheetname) {
                    double value;
                    if (xlsxioread_get_float(xls, &value, NULL)) {
                        row.push_back(value);
                    }
                }
            }
            if (!row.empty()) {
                data.push_back(row);
            }
            if (!range.empty()) {
                xlsxioread_sheet_close(xls);
                break;
            }
        }
    }

    xlsxioread_close(xls);
    return data;
}

vector<vector<double>> readExposureData(string filename) {
    xlsxioreader xls;
    vector<vector<double>> data;

    xls = xlsxioread_open(filename.c_str());
    if (xls == NULL) {
        cerr << "Błąd podczas otwierania pliku Excel." << endl;
        return data;
    }

    // Zakładamy, że dane znajdują się na arkuszu o nazwie "Sheet1" (możesz zmienić na odpowiednią nazwę arkusza)
    const char* sheetname = "Sheet1";

    if (xlsxioread_sheet_open(xls, sheetname, XLSXIOREAD_SKIP_EMPTY_ROWS) != XLSXIOREAD_SUCCESS) {
        cerr << "Błąd podczas otwierania arkusza '" << sheetname << "'." << endl;
        xlsxioread_close(xls);
        return data;
    }

    while (xlsxioread_process(xls, XLSXIOREAD_NEXT_ROW) == XLSXIOREAD_SUCCESS) {
        vector<double> row;
        while (xlsxioread_process(xls, XLSXIOREAD_NEXT_CELL) == XLSXIOREAD_SUCCESS) {
            double value;
            if (xlsxioread_get_float(xls, &value, NULL)) {
                row.push_back(value);
            }
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }

    xlsxioread_close(xls);
    return data;
}

int binomialDistribution(int n, double p) {
    random_device rd;
    mt19937 gen(rd());
    binomial_distribution<int> dist(n, p);
    return dist(gen);
}

vector<double> sample(vector<double>& population, int sampleSize) {
    random_device rd;
    mt19937 gen(rd());
    vector<double> sampleData(sampleSize);
    for (int i = 0; i < sampleSize; i++) {
        uniform_real_distribution<double> dist(0.0, population.size() - 1);
        int randomIndex = dist(gen);
        sampleData[i] = population[randomIndex];
    }
    return sampleData;
}

int countTotalFireSpreads(vector<int>& numberOfFireSpreads) {
    int total = 0;
    for (int count : numberOfFireSpreads) {
        total += count;
    }
    return total;
}

vector<double> flattenFireSpreadsRingsList(vector<vector<double>>& fireSpreadsRingsList) {
    vector<double> flattened;
    for (const auto& ring : fireSpreadsRingsList) {
        flattened.insert(flattened.end(), ring.begin(), ring.end());
    }
    return flattened;
}

int sumVector(vector<int>& vec) {
    int sum = 0;
    for (int value : vec) {
        sum += value;
    }
    return sum;
}

void print(vector<int>& vec) {
    for (int value : vec) {
        cout << value << " ";
    }
    cout << endl;
}

// Funkcja do obliczania odległości między dwoma punktami na sferze
double distanceCalculation(double lat1, double lat2, double long1, double long2) {
    double pi = 3.14159265358979323846;
    return 6371000 * sqrt(pow((pi/2 - lat1), 2) + pow((pi/2 - lat2), 2) - 2 * (pi/2 - lat1) * (pi/2 - lat2) * cos(long1 - long2));
}

struct Building {
    int index;
    double latitude;
    double longitude;
};

struct KDNode {
    vector<double> point; // Współrzędne punktu
    int axis;             // Wymiar (0 dla x, 1 dla y, 2 dla z, itd.)
    KDNode* left;         // Lewe dziecko
    KDNode* right;        // Prawe dziecko

    KDNode(vector<double>& p, int a) : point(p), axis(a), left(nullptr), right(nullptr) {}
};

vector<double> fireSpreadSimulation(double lat, double lon, vector<double>& exposure, vector<double>& exposureLat, vector<double>& exposureLon, vector<vector<int>>& kdTreesVoivMonth) {
    if (1001 > 1000) {
        vector<vector<double>> ringExposureList(9);
        vector<int> ringExposureNumbers(9, 0);

        for (int j = 0; j < 9; ++j) {
            ringExposureList[j] = vector<double>();
        }

        vector<pair<int, double>> dataBuildingDistance = getBuildingsWithUnDistance();
        vector<int> ind = extractIndices(dataBuildingDistance);
        vector<double> dist = extractDistances(dataBuildingDistance);
        vector<double> exposureSpreadSquare = extractExposureValues(exposure, ind);

        for (int l = 0; l < dist.size(); ++l) {
            if (!isnan(dist[l])) {
                if (dist[l] == 0) {
                    ringExposureNumbers[l] += 1;
                    ringExposureList[0].push_back(exposureSpreadSquare[l]);
                } else {
                    for (int j = 0; j < 8; ++j) {
                        if (dist[l] > 25 * j && dist[l] <= 25 * (j + 1)) {
                            ringExposureNumbers[j + 1] += 1;
                            ringExposureList[j + 1].push_back(exposureSpreadSquare[l]);
                        }
                    }
                }
            }
        }

        vector<int> numberOfFireSpreads(9);
        vector<vector<double>> fireSpreadsRingsList(9);

        for (int j = 0; j < 9; ++j) {
            numberOfFireSpreads[j] = binomialDistribution(1, ringExposureNumbers[j], fireSpreadProbs[j]);
            if (numberOfFireSpreads[j] > 0) {
                fireSpreadsRingsList[j] = sample(ringExposureList[j], numberOfFireSpreads[j]);
            }
        }

        vector<double> fireSpreadSimulationResult;
        fireSpreadSimulationResult.push_back(countTotalFireSpreads(numberOfFireSpreads));
        fireSpreadSimulationResult.push_back(flattenFireSpreadsRingsList(fireSpreadsRingsList));

        return fireSpreadSimulationResult;
    } else {
        vector<double> fireSpreadSimulationResult = {0, vector<double>()};
        return fireSpreadSimulationResult;
    }
}

void stochasticFireProcessSimulation(vector<vector<double>>& exposureList, vector<vector<double>>& exposureLatitude, vector<vector<double>>& exposureLongitude, vector<vector<double>>& fireProbsList, vector<vector<double>>& fireSpreadProbabilities, vector<vector<vector<int>>>& kdTreesVoivMonth) {
    vector<int> exposureNumber(12);
    vector<vector<int>> totalFireList(12);

    vector<int> numberOfFires(12);
    vector<vector<int>> fireSourcesList(12);
    vector<int> numberOfFireSpreads;

    for (int i = 0; i < 12; ++i) {
        kdTreesVoivMonth = kdTreesVoiv[i];
        exposureNumber[i] = exposureList[i].size();
        numberOfFires[i] = binomialDistribution(1, exposureNumber[i], fireProbsList[i]);
        fireSourcesList[i] = sample(exposureList[i], numberOfFires[i]);

        if (i == 0) {
            totalFireList[i] = fireSourcesList[i];
        } else {
            vector<int> diff = setDifference(fireSourcesList[i], totalFireList[i - 1]);
            fireSourcesList[i] = diff;
            totalFireList[i] = vectorUnion(totalFireList[i - 1], fireSourcesList[i]);
        }

        if (fireSourcesList[i].size() > 0) {
            for (int m = 0; m < fireSourcesList[i].size(); ++m) {
                double lat = exposureLatitude[i][fireSourcesList[i][m]];
                double lon = exposureLongitude[i][fireSourcesList[i][m]];
                vector<double> fireSpreads = fireSpreadSimulation(lat, lon, exposureList[i], exposureLatitude[i], exposureLongitude[i], kdTreesVoivMonth);
                vector<double> fireSpreadsList = setDifference(fireSpreads[1], totalFireList[i]);
                int fireSpreadsNumbers = fireSpreadsList.size();
                numberOfFireSpreads.push_back(fireSpreadsNumbers);
                totalFireList[i] = vectorUnion(totalFireList[i], fireSpreadsList);
            }
        }
    }
    vector<int> stochasticFireProcessSimulationResult = {sumVector(numberOfFireSpreads)};
    print(stochasticFireProcessSimulationResult);
    time_t end_time;
    time(&end_time);
    cout << "Czas wykonania: " << difftime(end_time, start_time) << " sekundy" << endl;
}

int main() {
    time_t start_time;
    time(&start_time);
    vector<vector<double>> fireProbs = readExcel("prawdopodobienstwa.xlsx", "PrawdPozaru", "B1:M18");
    vector<vector<double>> fireSpreadProbs = readExcel("prawdopodobienstwa.xlsx", "Rozprzestrzenienia_Pozarow", "");
    vector<vector<double>> exposureData = readExposureData("dane_budynkow.xlsx");

    if (!exposureData.empty()) {
        for (double value : exposureData[0]) {
            cout << value << " ";
        }
        cout << endl;
    }
    
    for (int k = 0; k < 17; ++k) {
        for (int n = 0; n < 100000; ++n) {
            for (int month = 1; month <= 12; ++month) {
                vector<int> activePolices = buildingsWithUnDistance[k * 12 + month - 1];
                int numBuildings = activePolices.size();
                vector<vector<vector<int>>> kdTreesVoivMonth(16);
                for (int k = 0; k < 16; ++k) {
                    for (int n = 0; n < number_of_simulations; ++n) {
                        stochasticFireProcessSimulation(basic_exposure_list[k], exposure_latitude[k], exposure_longitude[k], fireProbs[k], fireSpreadProbs, kdTreesVoivMonth);
                    }
                }
                vector<int> buildingsInPreviousMonths = getNonBurningBuildings(buildingsWithFires, exposureData, month);
                for (int buildingIndex : buildingsWithFires) {
                    double lat = coordinates[buildingIndex][0];
                    double lon = coordinates[buildingIndex][1];
                    vector<int> buildingsWithinDistance = getBuildingsWithinDistance(coordinates, lat, lon, 200.0);
                    vector<int> spreadCounts = getRandomSpreadCounts(spreadProbs[k], buildingsWithinDistance.size());
                    vector<int> spreadBuildings = getSpreadBuildings(buildingsWithinDistance, spreadCounts);
                    cout << "Building Index: " << buildingIndex << endl;
                    cout << "Spread to Buildings: ";
                    for (int spreadBuilding : spreadBuildings) {
                        cout << spreadBuilding << " ";
                    }
                    cout << endl;
                }
            }
        }
    }

    // Zakończenie i wydruk czasu
    time_t end_time;
    time(&end_time);
    cout << "Czas wykonania: " << difftime(end_time, start_time) << " sekundy" << endl;

    return 0;
}