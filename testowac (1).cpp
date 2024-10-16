#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <deque>
#include <algorithm>
#include <filesystem>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <pqxx/pqxx>
#include <sstream>
#include <thread>
#include <vector>
#include <iomanip>
#include <unordered_map>
#include <string>
#include <iomanip>
#include <vector>
#include <random>
#include "csvstream.hpp"
#include "json.hpp"
#include <mutex>
#include <sstream>
#include <utility>
#include <stdexcept>

const int NUM_THREADS = 1;
const char *HOST = "127.0.0.1";

const char *PORT = "9741";
const char *PORT2 = "9741";
std::string DB_PORT = "5432";
std::atomic<int> counterz(0);

std::vector<std::vector<std::string>> allStrings(NUM_THREADS);
std::vector<std::vector<std::string>> allStringsId(NUM_THREADS);

std::unordered_map<std::string, int> postalCodeCount;
int totalCodes = 0;
std::vector<std::string> codes;
std::vector<double> probabilities;

std::random_device rd;
std::mt19937 gen(rd());

pqxx::connection c(
    "dbname=nominatim user=nominatim password=nominatim host=localhost "
    "port=5432");
pqxx::work txn(c);

std::map<std::tuple<std::string, std::string, int>, std::pair<int, std::string>>
    occurrences;

std::map<std::tuple<std::string, std::string, std::string>, int> wordCnt;

// Deklaracja mutexa
std::mutex occurrencesMutex;
void addOccurrence(const std::string &city, const std::string &postcode,
                   int flag, int vecPos)
{
    if (!city.empty() || !postcode.empty())
    {
        // Blokada mutexa
        std::lock_guard<std::mutex> lock(occurrencesMutex);

        auto &entry = occurrences[{city, postcode, flag}];

        entry.first++;
        entry.second += std::to_string(vecPos) + " ";
    }
}

std::pair<std::string, int> getWojewodztwoMapa(const std::string &kodPocztowy)
{
    int kod;
    try
    {
        kod = std::stoi(kodPocztowy.substr(0, 2)) * 1000;
    }
    catch (const std::invalid_argument &e)
    {
        return {"Nieznane województwo", 0};
    }
    catch (const std::out_of_range &e)
    {
        return {"Nieznane województwo", 0};
    }

    std::map<std::string, std::pair<int, int>> wojewodztwa = {
        {"Dolnośląskie", {50000, 59999}},
        {"Kujawsko-Pomorskie", {85000, 89999}},
        {"Lubelskie", {20000, 23999}},
        {"Lubuskie", {65000, 69999}},
        {"Łódzkie", {90000, 99999}},
        {"Małopolskie", {30000, 34999}},
        {"Mazowieckie", {0, 9999}},
        {"Opolskie", {45000, 49999}},
        {"Podkarpackie", {35000, 39999}},
        {"Podlaskie", {15000, 19999}},
        {"Pomorskie", {80000, 84999}},
        {"Śląskie", {40000, 44999}},
        {"Świętokrzyskie", {25000, 29999}},
        {"Warmińsko-Mazurskie", {10000, 14999}},
        {"Wielkopolskie", {60000, 64999}},
        {"Zachodniopomorskie", {70000, 78999}}};

    int numer = 1;
    for (const auto &woj : wojewodztwa)
    {
        if (kod >= woj.second.first && kod <= woj.second.second)
        {
            return {woj.first, numer};
        }
        ++numer;
    }

    return {"Nieznane województwo", 0};
}

struct Address
{
    std::string lp;
    std::string ulica;
    std::string kodPocztowy;
    std::string miasto;
    std::string wojewodztwo;
    std::string kraj;
    std::string lot;
    std::string lat;
    int flaga1;
    int flaga2;
    std::string sklejone;
    std::string numerUmowy;
    std::string dataPoczatku;
    std::string dataKonca;
    std::string sumaUbezpieczenia;
    std::string odnowienia;
    std::string reasekuracjaO;
    std::string reasekuracjaF;
    std::string adresujedn;
    int proby = 0;
    bool dbProcess = false;
};

std::vector<Address> toProcess;
std::vector<Address> dataToCSV;
std::vector<Address> rozklad;
std::vector<Address> flaga0;
std::vector<Address> flaga1;
std::vector<Address> flaga2;
std::vector<Address> flaga3;
std::vector<Address> flaga4;
std::vector<Address> flaga5;
std::vector<Address> flaga6;

const double PI = 3.14159265358979323846;
const double R = 6371.0;

const char SAFE[256] =
    {
        /*      0 1 2 3  4 5 6 7  8 9 A B  C D E F */
        /* 0 */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        /* 1 */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        /* 2 */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        /* 3 */ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,

        /* 4 */ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        /* 5 */ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        /* 6 */ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        /* 7 */ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,

        /* 8 */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        /* 9 */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        /* A */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        /* B */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

        /* C */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        /* D */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        /* E */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        /* F */ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

std::string url_encode(const std::string &sSrc)
{
    const char DEC2HEX[16 + 1] = "0123456789ABCDEF";
    const unsigned char *pSrc = (const unsigned char *)sSrc.c_str();
    const int SRC_LEN = sSrc.length();
    unsigned char *const pStart = new unsigned char[SRC_LEN * 3];
    unsigned char *pEnd = pStart;
    const unsigned char *const SRC_END = pSrc + SRC_LEN;

    for (; pSrc < SRC_END; ++pSrc)
    {
        if (SAFE[*pSrc])
            *pEnd++ = *pSrc;
        else
        {
            // escape this char
            *pEnd++ = '%';
            *pEnd++ = DEC2HEX[*pSrc >> 4];
            *pEnd++ = DEC2HEX[*pSrc & 0x0F];
        }
    }

    std::string sResult((char *)pStart, (char *)pEnd);
    delete[] pStart;
    return sResult;
}
// std::string url_encode(const std::string &value)
// {
//     std::ostringstream encoded;
//     encoded.fill('0');
//     encoded << std::nouppercase << std::hex;

//     for (char c : value)
//     {
//         if (c == ';')
//         {
//             encoded << "%3B";
//         }
//         else if (!isalnum(static_cast<unsigned char>(c)) && c != '-' &&
//                  c != '_' && c != '.' && c != '~')
//         {
//             encoded << '%' << std::setw(2)
//                     << static_cast<int>(static_cast<unsigned char>(c));
//         }
//         else
//         {
//             encoded << c;
//         }
//     }

//     return encoded.str();
// }

int create_nonblocking_socket(const char *host, const char *port)
{
    struct addrinfo hints, *res;
    int sockfd;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(host, port, &hints, &res) != 0)
    {
        perror("getaddrinfo");
        return -1;
    }

    sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sockfd == -1)
    {
        perror("socket");
        freeaddrinfo(res);
        return -1;
    }

    int flags = fcntl(sockfd, F_GETFL, 0);
    if (fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) == -1)
    {
        perror("fcntl");
        close(sockfd);
        freeaddrinfo(res);
        return -1;
    }

    if (connect(sockfd, res->ai_addr, res->ai_addrlen) == -1)
    {
        if (errno != EINPROGRESS)
        {
            perror("connect");
            close(sockfd);
            freeaddrinfo(res);
            return -1;
        }
    }

    freeaddrinfo(res);
    return sockfd;
}
std::vector<std::vector<Address>> all(NUM_THREADS);
std::vector<std::vector<Address>> all2(NUM_THREADS);
std::vector<std::vector<Address>> all3(NUM_THREADS);
std::vector<std::vector<Address>> all4(NUM_THREADS);

void add(const Address &pos)
{
    toProcess.push_back(pos);
}

void add(const Address &pos, int &currentIndex)
{
    all[currentIndex].push_back(pos);
    currentIndex = (currentIndex + 1) % all.size();
}
std::string removeUpToFirstSemicolon(const std::string &input)
{
    size_t pos = input.find(';');

    if (pos != std::string::npos)
    {
        return input.substr(pos + 1);
    }

    return "";
}

void perform_requests4(int thread_id,
                       std::shared_ptr<std::vector<std::string>> &data)
{
    int NUM_REQUESTS = all4[thread_id].size();

    int epoll_fd = epoll_create1(0);
    if (epoll_fd == -1)
    {
        perror("epoll_create1");
        exit(1);
    }
    int valuer = ++counterz;

    int sockfd;

    sockfd = create_nonblocking_socket(HOST, PORT);

    if (sockfd == -1)
    {
        exit(1);
    }

    epoll_event ev, events[NUM_REQUESTS];

    ev.events = EPOLLOUT | EPOLLET;
    ev.data.fd = sockfd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, sockfd, &ev) == -1)
    {
        perror("epoll_ctl: EPOLLOUT");
        close(sockfd);
        exit(1);
    }

    int requests_sent = 0;
    bool keep_going = true;
    int id = 0;
    int lp = 0;

    while (keep_going)
    {
        int nfds = epoll_wait(epoll_fd, events, NUM_REQUESTS, -1);
        if (nfds == -1)
        {
            perror("epoll_wait");
            exit(1);
        }

        for (int n = 0; n < nfds; ++n)
        {
            if (events[n].events & EPOLLOUT && requests_sent < NUM_REQUESTS)
            {
                std::string address = url_encode(all4[thread_id][id++].sklejone);

                std::string request =
                    "GET /search.php?q=" + address + "&format=json&limit=1" +
                    " HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nConnection: "
                    "keep-alive\r\n\r\n";
                ssize_t bytes_sent = send(sockfd, request.c_str(), request.length(), 0);
                if (bytes_sent == -1)
                {
                    perror("send");
                    keep_going = false;
                    break;
                }
                requests_sent++;
                ev.events = EPOLLIN | EPOLLET;
                ev.data.fd = sockfd;
                if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, sockfd, &ev) == -1)
                {
                    perror("epoll_ctl: EPOLLIN");
                    keep_going = false;
                    break;
                }
            }
            else if (events[n].events & EPOLLIN)
            {
                char buffer[4096];
                std::string response;
                ssize_t bytes_read;
                while ((bytes_read = read(sockfd, buffer, sizeof(buffer))) > 0)
                {
                    response.append(buffer, bytes_read);
                }

                if (bytes_read == -1 && errno != EAGAIN && errno != EWOULDBLOCK)
                {
                    perror("read");
                    keep_going = false;
                }
                else if (bytes_read == 0)
                {
                    keep_going = false;
                }
                else
                {
                    if (requests_sent < NUM_REQUESTS)
                    {
                        ev.events = EPOLLOUT | EPOLLET;
                        ev.data.fd = sockfd;
                        if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, sockfd, &ev) == -1)
                        {
                            perror("epoll_ctl: EPOLLOUT");
                            keep_going = false;
                        }
                    }
                    else
                    {
                        keep_going = false;
                    }
                }

                if (!response.empty())
                {
                    std::string http_response = response;

                    std::string::size_type json_start = http_response.find('[');

                    if (json_start != std::string::npos)
                    {
                        std::string json_str = http_response.substr(json_start);

                        nlohmann::json j = nlohmann::json::parse(json_str);

                        if (!j[0]["lat"].is_null())
                        {
                            std::string lat = "" + std::string(j[0]["lat"]);
                            std::string lon = "" + std::string(j[0]["lon"]);
                            std::string name = "" + std::string(j[0]["display_name"]);
                            all4[thread_id][lp].lat = lat;
                            all4[thread_id][lp].lot = lon;
                            all4[thread_id][lp].adresujedn = name;
                            all4[thread_id][lp].flaga2 = 3;

                            dataToCSV[std::stoi(all4[thread_id][lp].lp)] =
                                std::move(all4[thread_id][lp]);
                        }
                        else
                        {
                            all4[thread_id][lp].lat = "0";
                            all4[thread_id][lp].lot = "0";
                            all4[thread_id][lp].adresujedn = "Brak danych";
                            all4[thread_id][lp].flaga2 = 3;

                            dataToCSV[std::stoi(all4[thread_id][lp].lp)] =
                                std::move(all4[thread_id][lp]);
                        }
                        lp++;
                    }
                }
            }
        }
    }

    close(sockfd);
    close(epoll_fd);
}

void perform_requests3(int thread_id,
                       std::shared_ptr<std::vector<std::string>> &data)
{
    int NUM_REQUESTS = all3[thread_id].size();

    int epoll_fd = epoll_create1(0);
    if (epoll_fd == -1)
    {
        perror("epoll_create1");
        exit(1);
    }
    int valuer = ++counterz;

    int sockfd;

    sockfd = create_nonblocking_socket(HOST, PORT);

    if (sockfd == -1)
    {
        exit(1);
    }

    epoll_event ev, events[NUM_REQUESTS];

    ev.events = EPOLLOUT | EPOLLET;
    ev.data.fd = sockfd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, sockfd, &ev) == -1)
    {
        perror("epoll_ctl: EPOLLOUT");
        close(sockfd);
        exit(1);
    }

    int requests_sent = 0;
    bool keep_going = true;
    int id = 0;
    int lp = 0;

    while (keep_going)
    {
        int nfds = epoll_wait(epoll_fd, events, NUM_REQUESTS, -1);
        if (nfds == -1)
        {
            perror("epoll_wait");
            exit(1);
        }

        for (int n = 0; n < nfds; ++n)
        {
            if (events[n].events & EPOLLOUT && requests_sent < NUM_REQUESTS)
            {
                std::string address = url_encode(all3[thread_id][id++].sklejone);

                std::string request =
                    "GET /search.php?q=" + address + "&format=json&limit=1" +
                    " HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nConnection: "
                    "keep-alive\r\n\r\n";
                ssize_t bytes_sent = send(sockfd, request.c_str(), request.length(), 0);
                if (bytes_sent == -1)
                {
                    perror("send");
                    keep_going = false;
                    break;
                }
                requests_sent++;
                ev.events = EPOLLIN | EPOLLET;
                ev.data.fd = sockfd;
                if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, sockfd, &ev) == -1)
                {
                    perror("epoll_ctl: EPOLLIN");
                    keep_going = false;
                    break;
                }
            }
            else if (events[n].events & EPOLLIN)
            {
                char buffer[4096];
                std::string response;
                ssize_t bytes_read;
                while ((bytes_read = read(sockfd, buffer, sizeof(buffer))) > 0)
                {
                    response.append(buffer, bytes_read);
                }

                if (bytes_read == -1 && errno != EAGAIN && errno != EWOULDBLOCK)
                {
                    perror("read");
                    keep_going = false;
                }
                else if (bytes_read == 0)
                {
                    keep_going = false;
                }
                else
                {
                    if (requests_sent < NUM_REQUESTS)
                    {
                        ev.events = EPOLLOUT | EPOLLET;
                        ev.data.fd = sockfd;
                        if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, sockfd, &ev) == -1)
                        {
                            perror("epoll_ctl: EPOLLOUT");
                            keep_going = false;
                        }
                    }
                    else
                    {
                        keep_going = false;
                    }
                }

                if (!response.empty())
                {
                    std::string http_response = response;

                    std::string::size_type json_start = http_response.find('[');

                    if (json_start != std::string::npos)
                    {
                        std::string json_str = http_response.substr(json_start);

                        nlohmann::json j = nlohmann::json::parse(json_str);

                        if (!j[0]["lat"].is_null())
                        {
                            std::string lat = "" + std::string(j[0]["lat"]);
                            std::string lon = "" + std::string(j[0]["lon"]);
                            std::string name = "" + std::string(j[0]["display_name"]);

                            all3[thread_id][lp].adresujedn = name;
                            all3[thread_id][lp].flaga2 = 2;

                            if (all3[thread_id][lp].flaga1 == 2)
                            {
                                addOccurrence(all3[thread_id][lp].miasto,
                                              all3[thread_id][lp].kodPocztowy, 2,
                                              flaga2.size());
                                flaga2.push_back(all3[thread_id][lp]);

                                all3[thread_id].erase(all3[thread_id].begin() + lp);

                                lp--;
                                id--;
                            }
                            else if (all3[thread_id][lp].flaga1 == 6)
                            {
                                addOccurrence(all3[thread_id][lp].miasto,
                                              all3[thread_id][lp].kodPocztowy, 6,
                                              flaga6.size());

                                flaga6.push_back(all3[thread_id][lp]);
                                all3[thread_id].erase(all3[thread_id].begin() + lp);

                                lp--;
                                id--;
                            }
                            else
                            {
                                all3[thread_id][lp].lat = lat;
                                all3[thread_id][lp].lot = lon;
                                dataToCSV[std::stoi(all3[thread_id][lp].lp)] =
                                    std::move(all3[thread_id][lp]);
                            }
                        }
                        else
                        {
                            all3[thread_id][lp].sklejone =
                                removeUpToFirstSemicolon(all3[thread_id][lp].sklejone);

                            all4[thread_id].push_back(all3[thread_id][lp]);

                            all3[thread_id].erase(all3[thread_id].begin() + lp);

                            lp--;
                            id--;
                        }
                        lp++;
                    }
                }
            }
        }
    }

    close(sockfd);
    close(epoll_fd);

    if (all4[thread_id].size() > 0)
        perform_requests4(thread_id, data);
}

void perform_requests2(int thread_id,
                       std::shared_ptr<std::vector<std::string>> &data)
{
    (*data)[thread_id] +=
        "Dane z wątku2222 " + std::to_string(thread_id) + "\n\n";

    int NUM_REQUESTS = all2[thread_id].size();

    int epoll_fd = epoll_create1(0);
    if (epoll_fd == -1)
    {
        perror("epoll_create1");
        exit(1);
    }
    int valuer = ++counterz;

    int sockfd;

    sockfd = create_nonblocking_socket(HOST, PORT);

    if (sockfd == -1)
    {
        exit(1);
    }

    epoll_event ev, events[NUM_REQUESTS];

    ev.events = EPOLLOUT | EPOLLET;
    ev.data.fd = sockfd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, sockfd, &ev) == -1)
    {
        perror("epoll_ctl: EPOLLOUT");
        close(sockfd);
        exit(1);
    }

    int requests_sent = 0;
    bool keep_going = true;
    int id = 0;
    int lp = 0;

    while (keep_going)
    {
        int nfds = epoll_wait(epoll_fd, events, NUM_REQUESTS, -1);
        if (nfds == -1)
        {
            perror("epoll_wait");
            exit(1);
        }

        for (int n = 0; n < nfds; ++n)
        {
            if (events[n].events & EPOLLOUT && requests_sent < NUM_REQUESTS)
            {
                std::string address = url_encode(all2[thread_id][id++].sklejone);

                std::string request =
                    "GET /search.php?q=" + address + "&format=json&limit=1" +
                    " HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nConnection: "
                    "keep-alive\r\n\r\n";
                ssize_t bytes_sent = send(sockfd, request.c_str(), request.length(), 0);
                if (bytes_sent == -1)
                {
                    perror("send");
                    keep_going = false;
                    break;
                }
                requests_sent++;
                ev.events = EPOLLIN | EPOLLET;
                ev.data.fd = sockfd;
                if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, sockfd, &ev) == -1)
                {
                    perror("epoll_ctl: EPOLLIN");
                    keep_going = false;
                    break;
                }
            }
            else if (events[n].events & EPOLLIN)
            {
                char buffer[4096];
                std::string response;
                ssize_t bytes_read;
                while ((bytes_read = read(sockfd, buffer, sizeof(buffer))) > 0)
                {
                    response.append(buffer, bytes_read);
                }

                if (bytes_read == -1 && errno != EAGAIN && errno != EWOULDBLOCK)
                {
                    perror("read");
                    keep_going = false;
                }
                else if (bytes_read == 0)
                {
                    keep_going = false;
                }
                else
                {
                    if (requests_sent < NUM_REQUESTS)
                    {
                        ev.events = EPOLLOUT | EPOLLET;
                        ev.data.fd = sockfd;
                        if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, sockfd, &ev) == -1)
                        {
                            perror("epoll_ctl: EPOLLOUT");
                            keep_going = false;
                        }
                    }
                    else
                    {
                        keep_going = false;
                    }
                }

                if (!response.empty())
                {
                    std::string http_response = response;

                    std::string::size_type json_start = http_response.find('[');

                    if (json_start != std::string::npos)
                    {
                        std::string json_str = http_response.substr(json_start);

                        nlohmann::json j = nlohmann::json::parse(json_str);

                        if (!j[0]["lat"].is_null())
                        {
                            std::string lat = "" + std::string(j[0]["lat"]);
                            std::string lon = "" + std::string(j[0]["lon"]);
                            std::string name = "" + std::string(j[0]["display_name"]);

                            all2[thread_id][lp].adresujedn = name;
                            all2[thread_id][lp].flaga2 = 1;

                            if (all2[thread_id][lp].flaga1 == 2)
                            {
                                addOccurrence(all2[thread_id][lp].miasto,
                                              all2[thread_id][lp].kodPocztowy, 2,
                                              flaga2.size());

                                flaga2.push_back(all2[thread_id][lp]);

                                all2[thread_id].erase(all2[thread_id].begin() + lp);

                                lp--;
                                id--;
                            }
                            else if (all2[thread_id][lp].flaga1 == 3)
                            {
                                addOccurrence(all2[thread_id][lp].miasto,
                                              all2[thread_id][lp].kodPocztowy, 3,
                                              flaga3.size());

                                flaga3.push_back(all2[thread_id][lp]);

                                all2[thread_id].erase(all2[thread_id].begin() + lp);

                                lp--;
                                id--;
                            }
                            else if (all2[thread_id][lp].flaga1 == 6)
                            {
                                addOccurrence(all2[thread_id][lp].miasto,
                                              all2[thread_id][lp].kodPocztowy, 6,
                                              flaga6.size());

                                flaga6.push_back(all2[thread_id][lp]);
                                all2[thread_id].erase(all2[thread_id].begin() + lp);

                                lp--;
                                id--;
                            }
                            else
                            {
                                all2[thread_id][lp].lat = lat;
                                all2[thread_id][lp].lot = lon;

                                dataToCSV[std::stoi(all2[thread_id][lp].lp)] =
                                    std::move(all2[thread_id][lp]);
                            }
                        }
                        else
                        {
                            all2[thread_id][lp].sklejone =
                                removeUpToFirstSemicolon(all2[thread_id][lp].sklejone);

                            all3[thread_id].push_back(all2[thread_id][lp]);

                            all2[thread_id].erase(all2[thread_id].begin() + lp);

                            lp--;
                            id--;
                        }
                        lp++;
                    }
                }
            }
        }
    }

    close(sockfd);
    close(epoll_fd);

    if (all3[thread_id].size() > 0)
        perform_requests3(thread_id, data);
}

void perform_requests2X(int thread_id,
                        std::vector<Address> &nextProcess)
{

    int NUM_REQUESTS = nextProcess.size();

    int from = 0;

    int epoll_fd = epoll_create1(0);
    if (epoll_fd == -1)
    {
        perror("epoll_create1");
        exit(1);
    }

    int sockfd;

    sockfd = create_nonblocking_socket(HOST, PORT);

    if (sockfd == -1)
    {
        exit(1);
    }

    epoll_event ev, events[NUM_REQUESTS];

    ev.events = EPOLLOUT | EPOLLET;
    ev.data.fd = sockfd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, sockfd, &ev) == -1)
    {
        perror("epoll_ctl: EPOLLOUT");
        close(sockfd);
        exit(1);
    }

    int requests_sent = 0;
    bool keep_going = true;
    int id = 0;

    while (keep_going)
    {
        int nfds = epoll_wait(epoll_fd, events, NUM_REQUESTS, -1);
        if (nfds == -1)
        {
            perror("epoll_wait");
            exit(1);
        }

        for (int n = 0; n < nfds; ++n)
        {
            if (events[n].events & EPOLLOUT && requests_sent < NUM_REQUESTS)
            {
                std::string address = url_encode(toProcess[from].sklejone);

                std::string request =
                    "GET /search.php?q=" + address + "&json_callback=XDEAD" + std::to_string(from) + "BEEFX" + "&format=json&limit=1" +
                    " HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nConnection: "
                    "keep-alive\r\n\r\n";

#ifdef DEBUG
                std::cout << request << std::endl;
#endif
                // if(toProcess[from].dbProcess == true)
                //     addOccurrence(toProcess[from].miasto, toProcess[from].kodPocztowy, toProcess[from].flaga1, from);

                from++;
                ssize_t bytes_sent = send(sockfd, request.c_str(), request.length(), 0);
                if (bytes_sent == -1)
                {
                    perror("send");
                    keep_going = false;
                    break;
                }
                requests_sent++;
                ev.events = EPOLLIN | EPOLLET;
                ev.data.fd = sockfd;
                if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, sockfd, &ev) == -1)
                {
                    perror("epoll_ctl: EPOLLIN");
                    keep_going = false;
                    break;
                }
            }
            else if (events[n].events & EPOLLIN)
            {
                char buffer[4096];
                std::string response;
                ssize_t bytes_read;
                while ((bytes_read = read(sockfd, buffer, sizeof(buffer))) > 0)
                {
                    response.append(buffer, bytes_read);
                }

                if (bytes_read == -1 && errno != EAGAIN && errno != EWOULDBLOCK)
                {
                    perror("read");
                    keep_going = false;
                }
                else if (bytes_read == 0)
                {
                    keep_going = false;
                }
                else
                {
                    if (requests_sent < NUM_REQUESTS)
                    {
                        ev.events = EPOLLOUT | EPOLLET;
                        ev.data.fd = sockfd;
                        if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, sockfd, &ev) == -1)
                        {
                            perror("epoll_ctl: EPOLLOUT");
                            keep_going = false;
                        }
                    }
                    else
                    {
                        keep_going = false;
                    }
                }

                if (!response.empty())
                {
                    std::string http_response = response;
                    http_response.erase(http_response.length() - 1, 1);
#ifdef DEBUG
                    std::cout << http_response << std::endl;
#endif

                    std::string::size_type json_start = http_response.find('[');

                    if (json_start != std::string::npos)
                    {
                        std::string json_str = http_response.substr(json_start);

#ifdef DEBUG
                        std::cout << json_str << std::endl;
#endif
                        size_t posStart = http_response.find("XDEAD");
                        size_t posEnd = http_response.find("BEEFX");

                        // Sprawdź, czy znaleziono oba podciągi i 'BEEFX' jest po 'XDEAD'
                        // Wyciągnij liczbę między 'XDEAD' i 'BEEFX'
                        int number = std::stoi(http_response.substr(posStart + 5, posEnd - posStart - 5));
                        // std::cout << "Liczba: " << number << std::endl;

                        // // Usuń podciąg od 'XDEAD' do 'BEEFX('
                        // json_str.erase(posStart, posEnd - posStart + 6);

                        // // Usuń końcowe ')'
                        // size_t posCloseParen = json_str.rfind(')');
                        //     json_str.erase(posCloseParen, 1);
#ifdef DEBUG
                        std::cout << json_str << std::endl;
#endif

                        nlohmann::json j = nlohmann::json::parse(json_str);

                        if (!j[0]["lat"].is_null())
                        {
                            std::string lat = "" + std::string(j[0]["lat"]);
                            std::string lon = "" + std::string(j[0]["lon"]);
                            std::string name = "" + std::string(j[0]["display_name"]);

                            toProcess[number].lat = lat;
                            toProcess[number].lot = lon;
                            toProcess[number].flaga2 = 0;
                            toProcess[number].adresujedn = name;

                            // all[thread_id][lp].adresujedn = name;
                            // all[thread_id][lp].flaga2 = 0;

                            // if (all[thread_id][lp].flaga1 == 0)
                            // {
                            //     addOccurrence(all[thread_id][lp].miasto,
                            //                   all[thread_id][lp].kodPocztowy, 0, flaga0.size());
                            //     flaga0.push_back(all[thread_id][lp]);
                            //     all[thread_id].erase(all[thread_id].begin() + lp);

                            //     lp--;
                            //     id--;
                            // }
                            // else if (all[thread_id][lp].flaga1 == 5)
                            // {
                            //     addOccurrence(all[thread_id][lp].miasto,
                            //                   all[thread_id][lp].kodPocztowy, 5, flaga5.size());
                            //     flaga5.push_back(all[thread_id][lp]);
                            //     all[thread_id].erase(all[thread_id].begin() + lp);

                            //     lp--;
                            //     id--;
                            // else
                            // {

                            //     all[thread_id][lp].lat = lat;
                            //     all[thread_id][lp].lot = lon;
                            //     dataToCSV[std::stoi(all[thread_id][lp].lp)] =
                            //         std::move(all[thread_id][lp]);
                            // // }
                        }
                        else
                        {
                            // std::cout << " KCKZCZ" << std::endl;
                            // all[thread_id][lp].sklejone =
                            toProcess[number].sklejone = removeUpToFirstSemicolon(toProcess[number].sklejone);
                            toProcess[number].flaga2 += 1;
                            nextProcess.push_back(toProcess[number]);

                            // if(toProcess[number].flaga1 == 2) {
                            //     addOccurrence(toProcess[number].miasto, toProcess[number].kodPocztowy, toProcess[number].flaga1, number);
                            // } else {

                            // //                   all[thread_id][lp].kodPocztowy, 5, flaga5.size());
                            // // all2[thread_id].push_back(all[thread_id][lp]);
                            // }
                            // all[thread_id].erase(all[thread_id].begin() + lp);
                        }
                    }
                }
            }
        }
    }

    close(sockfd);
    close(epoll_fd);

    if (nextProcess.size() > 0)
        perform_requests2X(thread_id, nextProcess);
}

// ##############
// ##############
// ##############
// ##############
// ##############
// ##############
// ##############
// ##############
// ##############
// ##############
// ##############
// ##############
// ##############
// ##############
// ##############
void perform_requests(int thread_id,
                      std::shared_ptr<std::vector<std::string>> &data,
                      std::pair<int, int> range)
{

    std::vector<Address> nextProcess;

    int NUM_REQUESTS = range.second - range.first;

    int from = range.first;
    int to = range.second;

    int epoll_fd = epoll_create1(0);
    if (epoll_fd == -1)
    {
        perror("epoll_create1");
        exit(1);
    }

    int sockfd;

    sockfd = create_nonblocking_socket(HOST, PORT);

    if (sockfd == -1)
    {
        exit(1);
    }

    epoll_event ev, events[NUM_REQUESTS];

    ev.events = EPOLLOUT | EPOLLET;
    ev.data.fd = sockfd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, sockfd, &ev) == -1)
    {
        perror("epoll_ctl: EPOLLOUT");
        close(sockfd);
        exit(1);
    }

    int requests_sent = 0;
    bool keep_going = true;
    int id = 0;

    while (keep_going)
    {
        int nfds = epoll_wait(epoll_fd, events, NUM_REQUESTS, -1);
        if (nfds == -1)
        {
            perror("epoll_wait");
            exit(1);
        }

        for (int n = 0; n < nfds; ++n)
        {
            if (events[n].events & EPOLLOUT && requests_sent < NUM_REQUESTS)
            {
                std::string address = url_encode(toProcess[from].sklejone);

                std::string request =
                    "GET /search.php?q=" + address + "&json_callback=XDEAD" + std::to_string(from) + "BEEFX" + "&format=json&limit=1" +
                    " HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nConnection: "
                    "keep-alive\r\n\r\n";
#ifdef DEBUG
                std::cout << request << std::endl;
#endif
                from++;
                ssize_t bytes_sent = send(sockfd, request.c_str(), request.length(), 0);
                if (bytes_sent == -1)
                {
                    perror("send");
                    keep_going = false;
                    break;
                }
                requests_sent++;
                ev.events = EPOLLIN | EPOLLET;
                ev.data.fd = sockfd;
                if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, sockfd, &ev) == -1)
                {
                    perror("epoll_ctl: EPOLLIN");
                    keep_going = false;
                    break;
                }
            }
            else if (events[n].events & EPOLLIN)
            {
                char buffer[4096];
                std::string response;
                ssize_t bytes_read;
                while ((bytes_read = read(sockfd, buffer, sizeof(buffer))) > 0)
                {
                    response.append(buffer, bytes_read);
                }

                if (bytes_read == -1 && errno != EAGAIN && errno != EWOULDBLOCK)
                {
                    perror("read");
                    keep_going = false;
                }
                else if (bytes_read == 0)
                {
                    keep_going = false;
                }
                else
                {
                    if (requests_sent < NUM_REQUESTS)
                    {
                        ev.events = EPOLLOUT | EPOLLET;
                        ev.data.fd = sockfd;
                        if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, sockfd, &ev) == -1)
                        {
                            perror("epoll_ctl: EPOLLOUT");
                            keep_going = false;
                        }
                    }
                    else
                    {
                        keep_going = false;
                    }
                }

                if (!response.empty())
                {
                    std::string http_response = response;
                    http_response.erase(http_response.length() - 1, 1);
#ifdef DEBUG
                    std::cout << http_response << std::endl;
#endif
                    std::string::size_type json_start = http_response.find('[');

                    if (json_start != std::string::npos)
                    {
                        std::string json_str = http_response.substr(json_start);
#ifdef DEBUG
                        std::cout << json_str << std::endl;
#endif
                        size_t posStart = http_response.find("XDEAD");
                        size_t posEnd = http_response.find("BEEFX");

                        // Sprawdź, czy znaleziono oba podciągi i 'BEEFX' jest po 'XDEAD'
                        // Wyciągnij liczbę między 'XDEAD' i 'BEEFX'
                        int number = std::stoi(http_response.substr(posStart + 5, posEnd - posStart - 5));
// std::cout << "Liczba: " << number << std::endl;

// // Usuń podciąg od 'XDEAD' do 'BEEFX('
// json_str.erase(posStart, posEnd - posStart + 6);

// // Usuń końcowe ')'
// size_t posCloseParen = json_str.rfind(')');
//     json_str.erase(posCloseParen, 1);
#ifdef DEBUG
                        std::cout << json_str << std::endl;
#endif

                        nlohmann::json j = nlohmann::json::parse(json_str);

                        if (!j[0]["lat"].is_null())
                        {
                            std::string lat = "" + std::string(j[0]["lat"]);
                            std::string lon = "" + std::string(j[0]["lon"]);
                            std::string name = "" + std::string(j[0]["display_name"]);

                            toProcess[number].lat = lat;
                            toProcess[number].lot = lon;
                            toProcess[number].flaga2 = 0;
                            toProcess[number].adresujedn = name;

                            // all[thread_id][lp].adresujedn = name;
                            // all[thread_id][lp].flaga2 = 0;

                            // if (all[thread_id][lp].flaga1 == 0)
                            // {
                            //     addOccurrence(all[thread_id][lp].miasto,
                            //                   all[thread_id][lp].kodPocztowy, 0, flaga0.size());
                            //     flaga0.push_back(all[thread_id][lp]);
                            //     all[thread_id].erase(all[thread_id].begin() + lp);

                            //     lp--;
                            //     id--;
                            // }
                            // else if (all[thread_id][lp].flaga1 == 5)
                            // {
                            //     addOccurrence(all[thread_id][lp].miasto,
                            //                   all[thread_id][lp].kodPocztowy, 5, flaga5.size());
                            //     flaga5.push_back(all[thread_id][lp]);
                            //     all[thread_id].erase(all[thread_id].begin() + lp);

                            //     lp--;
                            //     id--;
                            // else
                            // {

                            //     all[thread_id][lp].lat = lat;
                            //     all[thread_id][lp].lot = lon;
                            //     dataToCSV[std::stoi(all[thread_id][lp].lp)] =
                            //         std::move(all[thread_id][lp]);
                            // // }
                        }
                        else
                        {
                            // std::cout << " KCKZCZ" << std::endl;
                            // all[thread_id][lp].sklejone =
                            toProcess[number].sklejone = removeUpToFirstSemicolon(toProcess[number].sklejone);
                            toProcess[number].flaga2 += 1;

                            if (toProcess[number].flaga1 == 2)
                            {
                                addOccurrence(toProcess[number].miasto, "", toProcess[number].flaga1, number);
                            }
                            else if (toProcess[number].flaga1 == 3)
                            {
                                addOccurrence("", toProcess[number].kodPocztowy, toProcess[number].flaga1, number);
                            }
                            else if (toProcess[number].flaga1 == 1)
                            {
                                addOccurrence(toProcess[number].miasto, toProcess[number].kodPocztowy, toProcess[number].flaga1, number);

                                // nextProcess.push_back(toProcess[number]);
                            }
                            // if(toProcess[number].flaga1 == 2) {
                            //     addOccurrence(toProcess[number].miasto, toProcess[number].kodPocztowy, toProcess[number].flaga1, number);
                            // } else {

                            // //                   all[thread_id][lp].kodPocztowy, 5, flaga5.size());
                            // // all2[thread_id].push_back(all[thread_id][lp]);
                            // }
                            // all[thread_id].erase(all[thread_id].begin() + lp);
                        }
                    }
                }
            }
        }
    }

    close(sockfd);
    close(epoll_fd);

    // if (nextProcess.size() > 0)
    //     perform_requests2X(thread_id, nextProcess);
}

std::string removeAfterSlash(std::string address)
{
    size_t pos;

    pos = address.find('/');
    if (pos != std::string::npos)
    {
        address = address.substr(0, pos);
    }

    return address;
}

std::string removeWord(const std::string &input,
                       const std::string &wordToRemove)
{
    std::istringstream stream(input);
    std::string word;
    std::string result;

    while (stream >> word)
    {
        if (word != wordToRemove)
        {
            result += word + " ";
        }
    }

    if (!result.empty())
    {
        result.pop_back();
    }

    return result;
}

std::string removeWordsWithDot(const std::string &input)
{
    std::istringstream stream(input);
    std::string word;
    std::string result;

    while (stream >> word)
    {
        if (word.find('.') == std::string::npos)
        {
            result += word + " ";
        }
    }

    if (!result.empty())
    {
        result.pop_back();
    }

    return result;
}

bool containsWordIgnoreCase(const std::string &text, const std::string &word)
{
    std::string lowerText = text;
    std::string lowerWord = word;

    std::transform(lowerText.begin(), lowerText.end(), lowerText.begin(),
                   ::tolower);
    std::transform(lowerWord.begin(), lowerWord.end(), lowerWord.begin(),
                   ::tolower);

    return lowerText.find(lowerWord) != std::string::npos;
}

void addStringId(std::vector<std::vector<std::string>> &allStringsId,
                 const std::string &str, int &currentIndex)
{
    allStringsId[currentIndex].push_back(str);
    currentIndex = (currentIndex + 1) % allStringsId.size();
}

void addString(std::vector<std::vector<std::string>> &allStrings,
               const std::string &str, int &currentIndex)
{
    allStrings[currentIndex].push_back(str);
    currentIndex = (currentIndex + 1) % allStrings.size();
}

bool hasNumber(const std::string &str)
{
    return str.find_first_of("0123456789") != std::string::npos;
}

std::string removeSpecificTitles(const std::string &str)
{
    std::string result = str;
    std::vector<std::string> titles = {"doktora"};

    for (const auto &title : titles)
    {
        size_t pos = result.find(title);
        while (pos != std::string::npos)
        {
            result.erase(pos, title.length());
            pos = result.find(title);
        }
    }

    return result;
}

void wypiszAdresy(const std::vector<Address> &addresses,
                  const std::string &nazwaWektora)
{
    std::cout << "Wektor: " << nazwaWektora << std::endl;
    for (const auto &address : addresses)
    {
        std::cout << "Kod pocztowy: " << address.kodPocztowy
                  << ", Miasto: " << address.miasto
                  << ", Flaga1: " << address.flaga1 << std::endl;
    }
    std::cout << std::endl;
}

void saveToCSV(const std::vector<Address> &dataToCSV,
               const std::string &filename)
{
    std::ofstream file("/mnt/c/Output_geokodowanie/" + filename);

    if (!file.is_open())
    {
        std::cerr << "Nie można otworzyć pliku do zapisu!" << std::endl;
        return;
    }

    file << "Lp;DataPoczatku;DataKonca;SumaUbezpieczenia;Odnowienia;"
            "Ulica;KodPocztowy;Miasto;Wojewodztwo;Kraj;ReasekuracjaO;"
            "ReasekuracjaF;Szerokosc;Dlugosc;Flaga_1;Flaga_2;Nr_Wojewodztwa\n";

    for (size_t i = 0; i < dataToCSV.size(); ++i)
    {
        const auto &address = dataToCSV[i];

        auto [wojewodztwo, nr_woj] = getWojewodztwoMapa(address.kodPocztowy);

        file << address.lp << ";"
             << address.dataPoczatku << ";" << address.dataKonca << ";"
             << address.sumaUbezpieczenia << ";" << address.odnowienia << ";"
             << address.ulica << ";" << address.kodPocztowy << ";" << address.miasto
             << ";" << wojewodztwo << ";" << address.kraj << ";"
             << address.reasekuracjaO << ";" << address.reasekuracjaF << ";"
             << address.lat << ";" << address.lot
             << ";" << address.flaga1 << ";" << address.flaga2 << ";" << nr_woj << ";" << "\n";
    }

    file.close();
}

int pobierzLiczbe(const std::string &ciag, int indeks)
{
    std::istringstream iss(ciag);
    std::string liczba;
    int licznik = 0;

    while (iss >> liczba)
    {
        if (licznik == indeks)
        {
            return std::stoi(liczba);
        }
        licznik++;
    }

    throw std::out_of_range("Nie znaleziono liczby o podanym indeksie. Ciag: \"" + ciag + "\", Indeks: " + std::to_string(indeks));

    // throw std::out_of_range("Nie znaleziono liczby o podanym indeksie.");
}

// #include <iostream>
// #include <deque>
// #include <thread>
// #include <mutex>
// #include <string>

bool isStringStreamEmpty(const std::stringstream &ss)
{
    return ss.str().empty();
}

class ThreadSafeDeque
{
public:
    std::stringstream pop_front()
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!data.empty())
        {
            std::stringstream value;
            value << data.front();
            data.pop_front();
            return value;
        }
        return std::stringstream(); // Zwracamy pusty stringstream, gdy deque jest pusty
    }

    void push_back(const std::stringstream &value)
    {
        std::lock_guard<std::mutex> lock(mtx);
        data.push_back(value.str());
    }

    size_t size() const
    {
        return data.size();
    }

private:
    std::deque<std::string> data;
    std::mutex mtx;
};

ThreadSafeDeque workers_data;

// void consumer(ThreadSafeDeque& deq) {
//     for (int i = 0; i < 10; ++i) {
//         std::string value = deq.pop_front();
//         if (!value.empty()) {
//             std::cout << "Popped: " << value << std::endl;
//         } else {
//             std::cout << "Deque is empty." << std::endl;
//         }
//     }
// }

// int main() {
//     ThreadSafeDeque deq;

//     // Dodajemy dane do deque
//     for (int i = 0; i < 10; ++i) {
//         deq.push_back("String " + std::to_string(i));
//     }

//     std::thread t1(consumer, std::ref(deq));
//     t1.join();

//     return 0;
// }

std::mutex mtx_word;
void test_perf()
{
    while (workers_data.size() > 0)
    {
        auto dd = workers_data.pop_front();
        if (!dd.str().empty())
        {
            std::string query = dd.str();
            pqxx::result r = txn.exec(query);

            for (const auto &row : r)
            {
                std::string city, street, postcode, housenumber, place;

                size_t pos = 0;

                std::string input = row["address"].c_str();
                // std::cout << input << std::endl;
                while (pos != std::string::npos)
                {
                    size_t key_start = input.find('"', pos);
                    if (key_start == std::string::npos)
                        break;
                    size_t key_end = input.find('"', key_start + 1);
                    if (key_end == std::string::npos)
                        break;

                    std::string key = input.substr(key_start + 1, key_end - key_start - 1);

                    size_t value_start = input.find('"', key_end + 1);
                    if (value_start == std::string::npos)
                        break;
                    size_t value_end = input.find('"', value_start + 1);
                    if (value_end == std::string::npos)
                        break;

                    std::string value =
                        input.substr(value_start + 1, value_end - value_start - 1);

                    if (key == "city")
                    {
                        city = value;
                    }
                    else if (key == "street")
                    {
                        street = value;
                    }
                    else if (key == "postcode")
                    {
                        postcode = value;
                    }
                    else if (key == "housenumber")
                    {
                        housenumber = value;
                    }
                    else if (key == "place")
                    {
                        place = value;
                    }

                    pos = value_end + 1;
                }

                if (city == "")
                    city = place;

                int flag = row["Flag"].as<int>();
                std::string vecPositions = row["VecPos"].c_str();
                double lat = row["lat"].as<double>();
                double lon = row["lon"].as<double>();
                int liczba = -1;
                {
                    std::lock_guard<std::mutex> lock(mtx_word);

                    wordCnt[{vecPositions, city, postcode}] += 1;
                    // std::cout << vecPositions << " : " <<  wordCnt[{vecPositions, city, postcode}] - 1 << std::endl;
                    liczba = pobierzLiczbe(vecPositions,
                                           wordCnt[{vecPositions, city, postcode}] - 1);
                }
#ifdef DEBUG
                std::cout << "liczba, rozmiar vec " << liczba << ":" << toProcess.size() << std::endl;
#endif

                if (toProcess.size() > liczba)
                {
                    toProcess[liczba].miasto = city;
                    toProcess[liczba].kodPocztowy = postcode;
                    toProcess[liczba].ulica = street + " " + housenumber;
                    toProcess[liczba].lat = std::to_string(lat);
                    toProcess[liczba].lot = std::to_string(lon);
                }
            }
        }
    }
}

std::mutex occurrencesMutex2;
void addOccurrenceTemp(std::map<std::tuple<std::string, std::string, int>, std::pair<int, std::string>> &occurrences2, const std::string &city, const std::string &postcode,
                       int flag, int vecPos)
{
    if (!city.empty() || !postcode.empty())
    {
        // Blokada mutexa
        std::lock_guard<std::mutex> lock(occurrencesMutex2);

        auto &entry = occurrences2[{city, postcode, flag}];

        entry.first++;
        entry.second += std::to_string(vecPos) + " ";
    }
}

void perform_random(int threadid)
{

    pqxx::connection c(
        "dbname=nominatim user=nominatim password=nominatim host=localhost "
        "port=5432");
    pqxx::work txn(c);

    while (true)
    {
        auto dd = workers_data.pop_front();
        if (!dd.str().empty())
        {
            // std::cout << "THREAD ID: " << threadid << dd.str() << std::endl;

            pqxx::result r = txn.exec(dd);
            // std::cout << "rozmiar " << r.size() << std::endl;
            if (r.size() == 0)
            {
                // std::cout << dd.str() << std::endl;
                std::vector<int> vecPosNumbers;

                size_t start = dd.str().find("'");
                size_t end = dd.str().find("'", start + 1);

                if (start != std::string::npos && end != std::string::npos)
                {
                    std::string numbers = dd.str().substr(start + 1, end - start - 1);
                    std::stringstream ss(numbers);
                    std::string number;

                    while (ss >> number)
                    {
                        vecPosNumbers.push_back(std::stoi(number));
                    }
                }

                std::map<std::tuple<std::string, std::string, int>, std::pair<int, std::string>> temp_occurrences;

                for (const auto &num : vecPosNumbers)
                {
                    if (toProcess[num].flaga1 == 2)
                    {

                        if (probabilities.size() > 0)
                        {
                            std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
                            std::string pCode = codes[dist(gen)];

                            toProcess[num].kodPocztowy = pCode;
                            // toProcess[num].flaga2 += 1;
                            addOccurrenceTemp(temp_occurrences, "", pCode, 2, num);
                        }
                    }
                    else if (toProcess[num].flaga1 == 3)
                    {

                        if (probabilities.size() > 0)
                        {
                            std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
                            std::string pCode = codes[dist(gen)];

                            toProcess[num].kodPocztowy = pCode;
                            // toProcess[num].flaga2 += 1;
                            addOccurrenceTemp(temp_occurrences, "", pCode, 3, num);
                        }
                    }
                    else if (toProcess[num].flaga1 == 4 && toProcess[num].proby == 0)
                    {

                        toProcess[num].proby += 1;
                        // toProcess[num].flaga2 += 1;
                        addOccurrenceTemp(temp_occurrences, toProcess[num].miasto, "", 4, num);
                    }
                    else if (toProcess[num].flaga1 == 4 && toProcess[num].proby >= 1)
                    {
                        if (probabilities.size() > 0)
                        {
                            std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
                            std::string pCode = codes[dist(gen)];

                            toProcess[num].kodPocztowy = pCode;
                            toProcess[num].proby += 1;
                            toProcess[num].flaga2 += 1;
                            addOccurrenceTemp(temp_occurrences, "", toProcess[num].kodPocztowy, 4, num);
                        }
                    }
                    else if (toProcess[num].flaga1 == 5 && toProcess[num].proby == 0)
                    {

                        // toProcess[num].flaga2 += 1;
                        toProcess[num].proby += 1;

                        addOccurrenceTemp(temp_occurrences, toProcess[num].miasto, "", 5, num);
                    }
                    else if (toProcess[num].flaga1 == 5 && toProcess[num].proby == 1)
                    {
                        toProcess[num].proby += 1;
                        toProcess[num].flaga2 += 1;
                        addOccurrenceTemp(temp_occurrences, "", toProcess[num].kodPocztowy, 5, num);
                    }
                    else if (toProcess[num].flaga1 == 5 && toProcess[num].proby >= 2)
                    {

                        if (probabilities.size() > 0)
                        {
                            std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
                            std::string pCode = codes[dist(gen)];
                            toProcess[num].proby += 1;

                            toProcess[num].kodPocztowy = pCode;
                            toProcess[num].flaga2 += 1;
                            addOccurrenceTemp(temp_occurrences, "", pCode, 5, num);
                        }
                    }
                    else if (toProcess[num].flaga1 == 6)
                    {

                        if (probabilities.size() > 0)
                        {
                            std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
                            std::string pCode = codes[dist(gen)];

                            toProcess[num].kodPocztowy = pCode;
                            // toProcess[num].flaga2 += 1;
                            addOccurrenceTemp(temp_occurrences, "", pCode, 6, num);
                        }
                    }
                    else if (toProcess[num].flaga1 == 1 && toProcess[num].proby == 0)
                    {
                        // toProcess[num].flaga2 += 1;
                        toProcess[num].proby += 1;
                        addOccurrenceTemp(temp_occurrences, toProcess[num].miasto, toProcess[num].kodPocztowy, 1, num);
                    }
                    else if (toProcess[num].flaga1 == 1 && toProcess[num].proby == 1)
                    {

                        toProcess[num].proby += 1;
                        toProcess[num].flaga2 += 1;
                        addOccurrenceTemp(temp_occurrences, "", toProcess[num].kodPocztowy, 1, num);
                    }
                    else if (toProcess[num].flaga1 == 1 && toProcess[num].proby >= 2)
                    {

                        if (probabilities.size() > 0)
                        {
                            std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
                            std::string pCode = codes[dist(gen)];

                            toProcess[num].proby += 1;
                            toProcess[num].kodPocztowy = pCode;
                            toProcess[num].flaga2 += 1;
                            addOccurrenceTemp(temp_occurrences, "", pCode, 1, num);
                        }
                    }
                    else if (toProcess[num].flaga1 == 0)
                    {

                        if (probabilities.size() > 0)
                        {
                            std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
                            std::string pCode = codes[dist(gen)];

                            toProcess[num].proby += 1;
                            toProcess[num].kodPocztowy = pCode;
                            // toProcess[num].flaga2 += 1;
                            addOccurrenceTemp(temp_occurrences, "", pCode, 0, num);
                        }
                    }
                }
                for (const auto &entry : temp_occurrences)
                {
                    std::stringstream query;
                    const auto &key = entry.first;
                    const auto &value = entry.second;

                    const std::string &city = std::get<0>(key);
                    const std::string &postcode = std::get<1>(key);
                    int flag = std::get<2>(key);
                    int count = value.first;
                    std::string vecPos = value.second;

                    if (count > 0 && (!city.empty() || !postcode.empty()))
                    {
                        query // << "(\n"
                            << "    SELECT \n"
                            << "        " << flag << " AS Flag,\n"
                            << "        " << '\'' << vecPos << '\'' << " AS VecPos,\n"
                            << "        address,\n"
                            << "        ST_Y(ST_Centroid(centroid)) AS lat,\n"
                            << "        ST_X(ST_Centroid(centroid)) AS lon\n"
                            << "    FROM \n"
                            << "        placex\n"
                            << "    WHERE \n";

                        bool hasCondition = false;
                        if (!postcode.empty())
                        {
                            query << "        address -> 'postcode' ILIKE '" << postcode << "'\n";
                            hasCondition = true;
                        }
                        if (!city.empty())
                        {
                            if (hasCondition)
                            {
                                query << "        AND ";
                            }
                            // query << "address -> 'city' ILIKE '" << city << "'\n";
                            query << "(address -> 'city' ILIKE '" << city << "' OR address -> 'place' ILIKE '" << city << "')\n";
                            hasCondition = true;
                        }

                        if (hasCondition)
                        {
                            query << "        AND ";
                        }
                        query << "address -> 'housenumber' IS NOT NULL AND address -> "
                                 "'housenumber' != ''\n";
                        query << "    ORDER BY RANDOM() " << "\n";
                        query << "    LIMIT " << count << ";\n\n";
                        workers_data.push_back(query);
                    }
                }

                // std::cout << "Liczby dla VecPos: ";
                // for (const auto& num : vecPosNumbers) {
                //     std::cout << num << " ";
                // }
                // std::cout << std::endl;

                continue;
            }

            int liczba = -1;
            int nr_liczby = 0;

            for (const auto &row : r)
            {
                std::string city, street, postcode, housenumber, place;

                size_t pos = 0;

                std::string input = row["address"].c_str();
                // std::cout << input << std::endl;
                while (pos != std::string::npos)
                {
                    size_t key_start = input.find('"', pos);
                    if (key_start == std::string::npos)
                        break;
                    size_t key_end = input.find('"', key_start + 1);
                    if (key_end == std::string::npos)
                        break;

                    std::string key = input.substr(key_start + 1, key_end - key_start - 1);

                    size_t value_start = input.find('"', key_end + 1);
                    if (value_start == std::string::npos)
                        break;
                    size_t value_end = input.find('"', value_start + 1);
                    if (value_end == std::string::npos)
                        break;

                    std::string value =
                        input.substr(value_start + 1, value_end - value_start - 1);

                    if (key == "city")
                    {
                        city = value;
                    }
                    else if (key == "street")
                    {
                        street = value;
                    }
                    else if (key == "postcode")
                    {
                        postcode = value;
                    }
                    else if (key == "housenumber")
                    {
                        housenumber = value;
                    }
                    else if (key == "place")
                    {
                        place = value;
                    }

                    pos = value_end + 1;
                }

                if (city == "")
                    city = place;

                int flag = row["Flag"].as<int>();
                std::string vecPositions = row["VecPos"].c_str();
                // std::cout << vecPositions << " : ";
                double lat = row["lat"].as<double>();
                double lon = row["lon"].as<double>();

                // {
                //     std::lock_guard<std::mutex> lock(mtx_word);

                // wordCnt[{vecPositions, city, postcode}] += 1;
                liczba = pobierzLiczbe(vecPositions,
                                       nr_liczby++);
// std:: cout << vecPositions << ":" << liczba << " -  " << nr_liczby << std::endl;
// }
#ifdef DEBUG
                std::cout << "liczba, rozmiar vec " << liczba << ":" << toProcess.size() << std::endl;
#endif

                // std::cout << liczba << " : " << street << " : " << housenumber << " : " << place << std::endl;
                if (toProcess.size() > liczba)
                {
                    toProcess[liczba].miasto = city;
                    toProcess[liczba].kodPocztowy = postcode;
                    toProcess[liczba].ulica = street + " " + housenumber;
                    toProcess[liczba].lat = std::to_string(lat);
                    toProcess[liczba].lot = std::to_string(lon);
                }
            }
        }
        else
        {
            break;
        }
    }
}

namespace fs = std::filesystem;

int main()
{
    std::string path = "/mnt/c/Input_geokodowanie";

    int fileCount = 0;
    int currentFileCnt = 1;
    // zliczanie plików
    // for (const auto& entry : fs::directory_iterator(path)) {
    //     if (fs::is_regular_file(entry)) {
    //         ++fileCount;
    //     }
    // }

    std::cout << "Polaczony z " << c.dbname() << "\n\n";

    // for (const auto &entry : fs::directory_iterator(path))
    // {
        postalCodeCount.clear();
        occurrences.clear();
        rozklad.clear();
        toProcess.clear();

        totalCodes = 0;
        codes.clear();
        probabilities.clear();
        

        // std::cout << "Plik " << currentFileCnt << "/" << fileCount << ", obecnie przetwarzany jest: " << entry.path().filename().string() << std::endl;
        //  pqxx::connection c("dbname=nominatim user=nominatim password=nominatim host=localhost port=5432");
        //     std::cout << "Connected to " << c.dbname() << '\n';

        //                 pqxx::work txn(c);

        //      std::string queryr = R"(
        //     SELECT
        //         3 AS Flag,
        //         '12 83 ' AS VecPos,
        //         address,
        //         ST_Y(ST_Centroid(centroid)) AS lat,
        //         ST_X(ST_Centroid(centroid)) AS lon
        //     FROM
        //         placex
        //     WHERE
        //         address -> 'postcode' ILIKE '05-825'
        //         AND address -> 'housenumber' IS NOT NULL AND address -> 'housenumber' != ''
        //     ORDER BY RANDOM()
        //     LIMIT 2;
        //         )";

        //         // Wykonanie zapytania
        //         pqxx::result rd = txn.exec(queryr);

        //          for (const auto& row : rd) {
        //             std::string address = row["address"].c_str();
        //             double lat = row["lat"].as<double>();
        //             double lon = row["lon"].as<double>();

        //             std::cout << "Address: " << address << ", Lat: " << lat << ", Lon: " << lon << '\n';
        //         }
        //         for(;;) {}

        int currentIndex = 0;
        int currentIndexId = 0;

        // csvstream dorozkladu("proba2.csv");

        std::vector<std::pair<std::string, std::string>> row;

        auto start = std::chrono::high_resolution_clock::now();

        int rows_cnt = 1;

        // while (dorozkladu >> row)
        // {
        //     std::string kodPocztowy = row[7].second;

        //     if(kodPocztowy != "")
        //        postalCodeCount[kodPocztowy] += 1;

        //     rows_cnt++;
        // }

        // for (const auto &entry : postalCodeCount)
        // {
        //     totalCodes += entry.second;
        // }

        // for (const auto &entry : postalCodeCount)
        // {
        //     double probability = static_cast<double>(entry.second) / totalCodes;
        //     codes.push_back(entry.first);
        //     probabilities.push_back(probability);
        // }

        // dataToCSV.resize(rows_cnt);
        // for (int i = 0; i < rows_cnt; ++i)
        // {
        //     dataToCSV[i] = {};
        // }

        // -----------
        csvstream csvin("uciety.csv");

        while (csvin >> row)
        {
            std::string sklejone = "";
            std::string lp = row[0].second;
            std::string numerUmowy = row[1].second;
            std::string dataPoczatku = row[2].second;
            std::string dataKonca = row[3].second;
            std::string sumaUbezpieczenia = row[4].second;
            std::string odnowienia = row[5].second;
            std::string ulica = row[6].second;
            std::string kodPocztowy = row[7].second;
            std::string miasto = row[8].second;
            std::string wojewodztwo = row[9].second;
            std::string kraj = row[10].second;
            std::string reasekuracjaO = row[11].second;
            std::string reasekuracjaF = row[12].second;

            if (kodPocztowy != "")
                postalCodeCount[kodPocztowy] += 1;

            int flaga1 = -1;
            int flaga2 = 0;

            ulica = removeWord(ulica, "nr");
            ulica = removeAfterSlash(ulica);

            // if (ulica != "" && kodPocztowy != "" && miasto != "" && wojewodztwo != "" &&
            //     kraj != "" && ulica == miasto) {
            //   flaga1 = 1;
            //   sklejone = ulica + ";" + kodPocztowy + ";" + wojewodztwo + ";" + kraj;
            // } else if (ulica != "" && kodPocztowy != "" && miasto != "" &&
            //            wojewodztwo != "" && kraj != "" &&
            //            ulica.find(miasto) != std::string::npos) {
            //   flaga1 = 1;
            //   sklejone = ulica + ";" + kodPocztowy + ";" + wojewodztwo + ";" + kraj;
            // } else if (ulica != "" && kodPocztowy != "" && miasto == "" &&
            //            wojewodztwo != "" && kraj != "") {
            //   flaga1 = 6;
            //   sklejone = ulica + ";" + kodPocztowy + ";" + wojewodztwo + ";" + kraj;
            // } else if (ulica == "" && miasto == "" && wojewodztwo == "" && kraj == "" &&
            //            kodPocztowy != "") {
            //   flaga1 = 4;
            //   sklejone = kodPocztowy;
            // } else if (ulica == "" && miasto == "" && wojewodztwo == "" && kraj != "" &&
            //            kodPocztowy == "") {
            //   flaga1 = 5;
            // std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
            // std::string drawnCode = codes[dist(gen)];

            //   sklejone = drawnCode;
            // } else if (ulica == "" && miasto != "" && wojewodztwo != "" && kraj != "" &&
            //            kodPocztowy != "") {
            //   sklejone = kodPocztowy + ";" + miasto + ";" + wojewodztwo + ";" + kraj;
            //   flaga1 = 2;
            // } else if (ulica == "" && miasto != "" && wojewodztwo != "" && kraj != "" &&
            //            kodPocztowy == "") {
            //   sklejone = miasto + ";" + wojewodztwo + ";" + kraj;
            //   flaga1 = 3;
            // } else {
            //   sklejone = ulica + ";" + kodPocztowy + ";" + miasto + ";" + wojewodztwo +
            //              ";" + kraj;
            //   flaga1 = 0;
            // }

            if (ulica == "" && kodPocztowy == "" && miasto == "")
            {
                flaga1 = 0;
                // tu będzie losowanie z rozkłądu kodu pocztowego i pozniej z tego kodu pocztowego losowanie budynku
                // tu zawsze bedzie ok
                flaga2 = 0;

                // std::cout << " ddddddddddddddddddddd " << probabilities.size() << std::endl;

                // if( probabilities.size( ) > 0 ) {
                // std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
                // std::string pCode = codes[dist(gen)];

                // sklejone = pCode;
                // kodPocztowy = pCode;
                // // std::cout << sklejone << std::endl;
                // } else {
                // }

                // std::cout << " EST " << std::endl;
            }
            else if (ulica != "" && kodPocztowy == "" && miasto != "" && ulica != miasto)
            {
                sklejone = ulica + ";" + miasto;
                flaga1 = 2;
                // koduje

                // jesli nie zageokoduje to ucinamy ulice i geokodujemy miasto i z tego miasta losujemy budynek
                // dajemy flage2 = 1
            }
            else if (ulica != "" && kodPocztowy != "" && miasto != "" && ulica == miasto)
            {
                flaga1 = 4;
                sklejone = kodPocztowy + ";" + miasto;
            }
            else if (ulica != "" && kodPocztowy != "" && miasto != "" && ulica.find(miasto) != std::string::npos)
            {
                sklejone = ulica + ";" + kodPocztowy;
                flaga1 = 3;
            }
            else if (ulica == "" && kodPocztowy != "" && miasto != "")
            {
                sklejone = kodPocztowy + ";" + miasto;
                flaga1 = 5;
                // losujemy budynek z sklejone czyli kodPocztowy + ";" + miasto;

                // OK
            }
            else if (ulica == "" && kodPocztowy != "" && miasto == "")
            {
                sklejone = kodPocztowy;
                flaga1 = 6;
                // losujemy budynek z sklejone, czyli sam kod pocztowy

                // OK
            }
            else if (ulica == "" && kodPocztowy == "" && miasto == "")
            {
                flaga1 = 7;
                // losujemy z rozkladu kod pocztwoy i z niego budynki
                // tutaj nie wazne co jest w wojewodztwie i kraju zawsze bedzie losowanie

                // OK
            }
            else
            {
                sklejone = ulica + ";" + miasto;
                flaga1 = 1;

                // // jesli nie zageokoduje to ucinamy ulice i geokodujemy

                // sklejone = kodPocztowy + ";" + miasto;

                // // dajemy flage2 = 1

                // // Jesli sie nie uda zageokodowac to ucinamy miasto
                // sklejone = kodPocztowy;

                // // losujemy budynek z kodu Pocztowego
                // // dajemy flaga2=2

                // // Jesli sie nie uda zageokodowac to
                // sklejone = miasto;

                // losujemy budynek z kodu miasto
                // dajemy flaga2=3
            }

#ifdef DEBUG
            std::cout << flaga1 << " " << flaga2 << std::endl;
#endif
            // add(Address{lp, ulica, kodPocztowy, miasto, wojewodztwo, kraj,
            //             "brak danych", "brak danych", flaga1, flaga2, sklejone,
            //             numerUmowy, dataPoczatku, dataKonca, sumaUbezpieczenia,
            //             odnowienia, reasekuracjaO, reasekuracjaF, ""},
            //     currentIndex);

            // if (flaga1 == 3)
            // {
            //     addOccurrence(ulica, kodPocztowy, flaga1, std::stoi(lp) - 1);
            // }
            if (flaga1 == 4)
            {
                addOccurrence(miasto, kodPocztowy, flaga1, std::stoi(lp) - 1);
            }
            else if (flaga1 == 5)
            {
                addOccurrence(miasto, kodPocztowy, flaga1, std::stoi(lp) - 1);
            }
            else if (flaga1 == 6)
            {
                addOccurrence("", kodPocztowy, flaga1, std::stoi(lp) - 1);
            }
            else if (flaga1 == 0 || flaga1 == 7)
            {
                rozklad.push_back(Address{lp, ulica, kodPocztowy, miasto, wojewodztwo, kraj,
                                          "brak danych", "brak danych", flaga1, flaga2, sklejone,
                                          numerUmowy, dataPoczatku, dataKonca, sumaUbezpieczenia,
                                          odnowienia, reasekuracjaO, reasekuracjaF, ""});
            }

            add(Address{lp, ulica, kodPocztowy, miasto, wojewodztwo, kraj,
                        "brak danych", "brak danych", flaga1, flaga2, sklejone,
                        numerUmowy, dataPoczatku, dataKonca, sumaUbezpieczenia,
                        odnowienia, reasekuracjaO, reasekuracjaF, ""});

            rows_cnt++;

            //  std::cout << "Flaga 1: " << flaga1 << " flaga2: " << flaga2 << std::endl;
        }

        for (const auto &entry : postalCodeCount)
        {
            totalCodes += entry.second;
        }

        for (const auto &entry : postalCodeCount)
        {
            double probability = static_cast<double>(entry.second) / totalCodes;
            codes.push_back(entry.first);
            probabilities.push_back(probability);
        }

        // dataToCSV.resize(rows_cnt);
        // for (int i = 0; i < rows_cnt; ++i)
        // {
        //     dataToCSV[i] = {};
        // }

        for (int i = 0; i < rozklad.size(); i++)
        {
            if (probabilities.size() > 0)
            {
                std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
                std::string pCode = codes[dist(gen)];

                int number = std::stoi(rozklad[i].lp) - 1;
                toProcess[number].kodPocztowy = pCode;
                toProcess[number].sklejone = pCode;
                // toProcess[number].dbProcess = true;

                if (toProcess[number].flaga1 == 0 || toProcess[number].flaga1 == 7)
                {
                    addOccurrence("", toProcess[number].kodPocztowy, toProcess[number].flaga1, number);
                }
            }
            // else
            // {
            //     toProcess[std::stoi(rozklad[i].lp) - 1].kodPocztowy = "00-000";
            //     toProcess[std::stoi(rozklad[i].lp) - 1].sklejone = "00-000";
            // }
        }

        // std::cout << "size " << toProcess.size() << std::endl;

        //------------------------

        std::vector<std::thread> threads;
        auto data = std::make_shared<std::vector<std::string>>(NUM_THREADS);

        int M = NUM_THREADS;      // liczba wątków
        int N = toProcess.size(); // rozmiar wektora

        std::vector<std::pair<int, int>> ranges;
        int chunkSize = N / M;

        for (int i = 0; i < M; ++i)
        {
            int start = i * chunkSize;
            int end = (i == M - 1) ? N : start + chunkSize;
            ranges.push_back({start, end});
        }

        for (int i = 0; i < NUM_THREADS; i++)
        {
            threads.emplace_back(perform_requests, i, std::ref(data), ranges[i]);
        }

        for (auto &t : threads)
        {
            t.join();
        }

        // std::vector<std::stringstream> queries;

        // queries.reserve(occurrences.size());

        bool first = true;
        bool isFirst = false;
        int liczbalosowan = 0;
        for (const auto &entry : occurrences)
        {
            std::stringstream query;
            const auto &key = entry.first;
            const auto &value = entry.second;

            const std::string &city = std::get<0>(key);
            const std::string &postcode = std::get<1>(key);
            int flag = std::get<2>(key);
            int count = value.first;
            std::string vecPos = value.second;

            if (count > 0 && (!city.empty() || !postcode.empty()))
            {
                // if (!first)
                // {
                //     query << "UNION ALL\n";
                // }
#ifdef UPDATE
                if (!isFirst)
                {
                    query
                        << "CREATE INDEX idx_postcode ON placex ((address -> 'postcode'));\n"
                        << "CREATE INDEX idx_housenumber ON placex ((address -> 'housenumber'));\n"
                        << "CREATE INDEX idx_city ON placex ((address -> 'city'));\n";
                    isFirst = true;
                }
#endif

                query // << "(\n"
                    << "    SELECT \n"
                    << "        " << flag << " AS Flag,\n"
                    << "        " << '\'' << vecPos << '\'' << " AS VecPos,\n"
                    << "        address,\n"
                    << "        ST_Y(ST_Centroid(centroid)) AS lat,\n"
                    << "        ST_X(ST_Centroid(centroid)) AS lon\n"
                    << "    FROM \n"
                    << "        placex\n"
                    << "    WHERE \n";

                bool hasCondition = false;
                if (!postcode.empty())
                {
                    query << "        address -> 'postcode' ILIKE '" << postcode << "'\n";
                    hasCondition = true;
                }
                if (!city.empty())
                {
                    if (hasCondition)
                    {
                        query << "        AND ";
                    }
                    query << "(address -> 'city' ILIKE '" << city << "' OR address -> 'place' ILIKE '" << city << "')\n";
                    hasCondition = true;
                }

                if (hasCondition)
                {
                    query << "        AND ";
                }
                query << "address -> 'housenumber' IS NOT NULL AND address -> "
                         "'housenumber' != ''\n";
                query << "    ORDER BY RANDOM() " << "\n";
                query << "    LIMIT " << count << ";\n\n";
                // liczbalosowan += count;
                //   << ")\n";
                // queries.push_back(query);
                //  std::cout<<query.str()<<std::endl;
                workers_data.push_back(query);
                // queries.push_back(std::move(query)); // Przenosimy zamiast kopiować
                // first = false;
            }
        }

// std::cout << "LOPS " << workers_data.size() << " : " << liczbalosowan << std::endl;

//   for (int i = 0; i < 2; ++i) {
//     std::stringstream value = workers_data.pop_front();
//     if (!value.str().empty()) {
//         std::cout << "Popped: " << value.str() << std::endl;
//     } else {
//         std::cout << "Deque is empty." << std::endl;
//     }
// }
//    std::ofstream outFile("plik.txt");

// // Sprawdzanie, czy plik został poprawnie otwarty
// if (outFile.is_open()) {
//     // Zapis stringa do pliku
//     outFile << query.str();

//     // Zamknięcie pliku
//     outFile.close();
//     std::cout << "Zapisano tekst do pliku." << std::endl;
// } else {
//     std::cerr << "Nie można otworzyć pliku do zapisu." << std::endl;
// }
#ifdef DEBUG
        std::cout << "=======================" << std::endl;
#endif

        // test_perf();

        std::vector<std::thread> workers;

        for (int i = 0; i < NUM_THREADS; i++)
        {
            workers.emplace_back(perform_random, i);
        }

        for (auto &t : workers)
        {
            t.join();
        }

        // pqxx::result r = txn.exec(queries[0]);
        // std::cout << "rozmiar " << r.size() << std::endl;

        // for (const auto &row : r)
        // {
        //     std::string city, street, postcode, housenumber, place;

        //     size_t pos = 0;

        //     std::string input = row["address"].c_str();
        // #ifdef DEBUG
        //     std::cout << input << std::endl;
        // #endif
        //     while (pos != std::string::npos)
        //     {
        //         size_t key_start = input.find('"', pos);
        //         if (key_start == std::string::npos)
        //             break;
        //         size_t key_end = input.find('"', key_start + 1);
        //         if (key_end == std::string::npos)
        //             break;

        //         std::string key = input.substr(key_start + 1, key_end - key_start - 1);

        //         size_t value_start = input.find('"', key_end + 1);
        //         if (value_start == std::string::npos)
        //             break;
        //         size_t value_end = input.find('"', value_start + 1);
        //         if (value_end == std::string::npos)
        //             break;

        //         std::string value =
        //             input.substr(value_start + 1, value_end - value_start - 1);

        //         if (key == "city")
        //         {
        //             city = value;
        //         }
        //         else if (key == "street")
        //         {
        //             street = value;
        //         }
        //         else if (key == "postcode")
        //         {
        //             postcode = value;
        //         }
        //         else if (key == "housenumber")
        //         {
        //             housenumber = value;
        //         }
        //         else if (key == "place")
        //         {
        //             place = value;
        //         }

        //         pos = value_end + 1;
        //     }

        //     if (city == "")
        //         city = place;

        //     int flag = row["Flag"].as<int>();
        //     std::string vecPositions = row["VecPos"].c_str();
        //     double lat = row["lat"].as<double>();
        //     double lon = row["lon"].as<double>();

        //     wordCnt[{vecPositions, city, postcode}] += 1;
        //     int liczba = pobierzLiczbe(vecPositions,
        //                                wordCnt[{vecPositions, city, postcode}] - 1);
        //     #ifdef DEBUG
        //     std::cout << "liczba, rozmiar vec " << liczba << ":" << toProcess.size() << std::endl;
        //     #endif

        //     if (toProcess.size() > liczba)
        //         {
        //             toProcess[liczba].miasto = city;
        //             toProcess[liczba].kodPocztowy = postcode;
        //             toProcess[liczba].ulica = street + " " + housenumber;
        //             toProcess[liczba].lat = std::to_string(lat);
        //             toProcess[liczba].lot = std::to_string(lon);
        //         }
        // }

        // std::cout << "Rozmiar flaga0: " << flaga0.size() << std::endl;
        // std::cout << "Rozmiar flaga2: " << flaga2.size() << std::endl;
        // std::cout << "Rozmiar flaga3: " << flaga3.size() << std::endl;
        // std::cout << "Rozmiar flaga4: " << flaga4.size() << std::endl;
        // std::cout << "Rozmiar flaga5: " << flaga5.size() << std::endl;
        // std::cout << "Rozmiar flaga6: " << flaga6.size() << std::endl;

        // for (auto &adres : flaga0)
        // {
        //     int index = std::stoi(adres.lp);
        //     dataToCSV[index] = std::move(adres);
        // }

        // for (auto &adres : flaga2)
        // {
        //     int index = std::stoi(adres.lp);
        //     dataToCSV[index] = std::move(adres);
        // }

        // for (auto &adres : flaga3)
        // {
        //     int index = std::stoi(adres.lp);
        //     dataToCSV[index] = std::move(adres);
        // }

        // for (auto &adres : flaga4)
        // {
        //     int index = std::stoi(adres.lp);

        //     dataToCSV[index] = std::move(adres);
        // }

        // for (auto &adres : flaga5)
        // {
        //     int index = std::stoi(adres.lp);

        //     dataToCSV[index] = std::move(adres);
        // }

        // for (auto &adres : flaga6)
        // {
        //     int index = std::stoi(adres.lp);

        //     dataToCSV[index] = std::move(adres);
        // }

        // saveToCSV(toProcess, entry.path().filename().string());
        saveToCSV(toProcess,"uciety2_OUT.csv");

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Czas : " << duration.count() << " sekund, zapisano " << rows_cnt-1 << " rekordow\n" << std::endl;

        currentFileCnt++;
        // std::cout << entry.path().filename().string() << std::endl;
    // }

    return 0;
}
