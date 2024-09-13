#include <iostream>
#include <vector>
#include <thread>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <fstream>
#include <chrono>
#include <memory>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include "csvstream.hpp"

const int NUM_THREADS = 2;
const char *HOST = "127.0.0.1";

const char *PORT = "8080";

std::vector<std::vector<std::string>> allStrings(NUM_THREADS);

std::string url_encode(const std::string& value) {
    std::ostringstream encoded;
    encoded.fill('0');
    encoded << std::nouppercase << std::hex;

    for (char c : value) {
        if (c == ';') {
            encoded << "%3B";
        } else if (!isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_' && c != '.' && c != '~') {
            encoded << '%' << std::setw(2) << static_cast<int>(static_cast<unsigned char>(c));
        } else {
            encoded << c;
        }
    }

    return encoded.str();
}

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

void perform_requests(int thread_id, std::shared_ptr<std::vector<std::string>> &data)
{

    int NUM_REQUESTS = allStrings[thread_id].size();
    // std::cout << " SIZE " << thread_id << " " << NUM_REQUESTS << std::endl;
    (*data)[thread_id] = "Dane z wątku " + std::to_string(thread_id) + "\n\n";

    int epoll_fd = epoll_create1(0);
    if (epoll_fd == -1)
    {
        perror("epoll_create1");
        exit(1);
    }

    int sockfd = create_nonblocking_socket(HOST, PORT);
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
                 std::string address = url_encode(allStrings[thread_id][id++]);
                //  std::cout << address << std::endl;
    std::string request = "GET /search.php?q=" + address + " HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nConnection: keep-alive\r\n\r\n";
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
                    (*data)[thread_id] += response;
                }
            }
        }
    }

    close(sockfd);
    close(epoll_fd);
}

std::string removeApartmentNumber(std::string address)
{

    std::string toRemove = "nr";
    size_t pos = address.find(toRemove);
    if (pos != std::string::npos)
    {
        address.erase(pos, toRemove.length());
    }

    pos = address.find('/');
    if (pos != std::string::npos)
    {
        address = address.substr(0, pos); 
    }

    return address;
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

    std::transform(lowerText.begin(), lowerText.end(), lowerText.begin(), ::tolower);
    std::transform(lowerWord.begin(), lowerWord.end(), lowerWord.begin(), ::tolower);

    return lowerText.find(lowerWord) != std::string::npos;
}

void addString(std::vector<std::vector<std::string>>& allStrings, const std::string& str, int& currentIndex) {
    allStrings[currentIndex].push_back(str);
    currentIndex = (currentIndex + 1) % allStrings.size(); 
}

int main()
{

    for(int i = 0; i < allStrings.size(); i++)
        allStrings.reserve(100000);

    int currentIndex = 0;

    csvstream csvin("proba.csv");

    std::vector<std::pair<std::string, std::string>> row;

    std::string sklejone = "";
    auto start = std::chrono::high_resolution_clock::now();

    while (csvin >> row)
    {
        for (const auto &[key, value] : row)
        {
            if (key == "Ulica")
            {
                sklejone = "";
                sklejone += removeWordsWithDot(removeApartmentNumber(value)) + ";";
            }
            else if (key == "KodPocztowy")
            {
                sklejone += value + ";";
            }
            else if (key == "Miasto")
            {
                    sklejone += value + ";";
            }
            else if (key == "Wojewodztwo")
            {
                sklejone += value + ";";
            }
            else if (key == "Kraj")
            {
                sklejone += value; 

                addString((allStrings), sklejone, (currentIndex));

            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Czas: " << duration.count() << " sekund\n";

    // for (size_t i = 0; i < allStrings.size(); ++i) {
    //         std::cout << "Vector " << i << ": ";
    //         for (const auto& s : allStrings[i]) {
    //             std::cout << s << " ";
    //         }
    //         std::cout << std::endl;
    //     }

    std::vector<std::thread> threads;
    auto data = std::make_shared<std::vector<std::string>>(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; i++) {
        threads.emplace_back(perform_requests, i, std::ref(data));
    }

    for (auto& t : threads) {
        t.join();
    }

    for (size_t i = 0; i < data->size(); i++) {
        std::string filename = "thread_" + std::to_string(i) + ".txt";
        std::ofstream output_file(filename);

        if (!output_file.is_open()) {
            std::cerr << "Nie można otworzyć pliku: " << filename << std::endl;
            continue;
        }

        output_file << (*data)[i];
        output_file.close();
    }

    return 0;
}