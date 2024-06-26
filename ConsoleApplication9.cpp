#include <winsock2.h>
#include <ws2tcpip.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

#pragma comment(lib, "Ws2_32.lib")

#define DEFAULT_BUFLEN 512

void initialize_winsock() {
    WSADATA wsaData;
    int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (result != 0) {
        std::cerr << "WSAStartup failed: " << result << std::endl;
        exit(1);
    }
}

SOCKET create_socket() {
    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) {
        std::cerr << "Error at socket(): " << WSAGetLastError() << std::endl;
        WSACleanup();
        exit(1);
    }
    return sock;
}

void connect_socket(SOCKET& sock, const std::string& host, const std::string& port) {
    sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(std::stoi(port));
    inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr);

    int res = connect(sock, (sockaddr*)&server_addr, sizeof(server_addr));
    if (res == SOCKET_ERROR) {
        std::cerr << "Unable to connect to server: " << WSAGetLastError() << std::endl;
        closesocket(sock);
        WSACleanup();
        exit(1);
    }
}

void send_request(SOCKET& sock, const std::string& request) {
    int res = send(sock, request.c_str(), static_cast<int>(request.length()), 0);
    if (res == SOCKET_ERROR) {
        std::cerr << "send failed: " << WSAGetLastError() << std::endl;
        closesocket(sock);
        WSACleanup();
        exit(1);
    }
}

std::string receive_response(SOCKET& sock) {
    char buffer[DEFAULT_BUFLEN];
    int res = recv(sock, buffer, DEFAULT_BUFLEN, 0);
    if (res > 0) {
        return std::string(buffer, res);
    }
    else if (res == 0) {
        std::cerr << "Connection closed" << std::endl;
    }
    else {
        std::cerr << "recv failed: " << WSAGetLastError() << std::endl;
    }
    return "";
}

std::string get_line(const std::string& response, int lineNumber) {
    std::istringstream responseStream(response);
    std::string line;
    for (int i = 0; i < lineNumber; ++i) {
        if (!std::getline(responseStream, line)) {
            return "Line not found";
        }
    }
    return line;
}

int main() {
    initialize_winsock();

    SOCKET sock = create_socket();
    connect_socket(sock, "172.25.9.44", "8080");

    std::string request = "GET /search.php?q=WANDY%20RUTKIEWICZ%20A%20Wroc%C5%82aw HTTP/1.1\r\n"
        "Host: 172.25.9.44:8080\r\n"
        "Connection: keep-alive\r\n"
        "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7\r\n"
        "Accept-Encoding: gzip, deflate\r\n"
        "Accept-Language: pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7\r\n"
        "Cache-Control: max-age=0\r\n"
        "Upgrade-Insecure-Requests: 1\r\n"
        "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36\r\n"
        "\r\n";


    auto start = std::chrono::high_resolution_clock::now();
    std::ofstream outFile("responses.txt", std::ios::app);

    for (int i = 0; i < 100000; ++i) {
        send_request(sock, request);
        std::string response = receive_response(sock);
      //  std::string line = get_line(response, 10);
        outFile << "Response " << i + 1 << ": " << response << std::endl;
        std::cout << "Response " << i + 1 << ": " << response << std::endl;

        //if (i == 100000 - 1) {
        //    std::cout << "Response " << i + 1 << ": " << response << std::endl;
        //}
    }
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

    std::cout << "Czas wykonania: " << duration.count() << " sekund" << std::endl;

    closesocket(sock);
    WSACleanup();
    return 0;
}