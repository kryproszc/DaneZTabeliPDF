﻿#define _WIN32_WINNT 0x0600 
#include <ws2tcpip.h>
#include <winsock2.h>
#include <iostream>
#include <string>


//#pragma comment(lib, "ws2_32.lib")

void printLastError(const std::string& function) {
    int error = WSAGetLastError();
    LPVOID lpMsgBuf;
    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        error,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR) &lpMsgBuf,
        0, NULL);

    std::cerr << function << " failed with error: " << error << " - " << (char*)lpMsgBuf << std::endl;
    LocalFree(lpMsgBuf);
}

int main() {
    WSADATA wsaData;
    SOCKET ConnectSocket = INVALID_SOCKET;
    struct sockaddr_in serverAddr;
    int iResult;

    // Initialize Winsock
    iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (iResult != 0) {
        printLastError("WSAStartup");
        return 1;
    }
    std::cerr << "WSAStartup succeeded" << std::endl;

    // Create a SOCKET for connecting to server
    ConnectSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (ConnectSocket == INVALID_SOCKET) {
        printLastError("socket");
        WSACleanup();
        return 1;
    }
    std::cerr << "Socket created" << std::endl;

    // Setup the sockaddr_in structure
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(8080);
    iResult = inet_pton(AF_INET, "172.25.9.44", &serverAddr.sin_addr);
    if (iResult <= 0) {
        if (iResult == 0) {
            std::cerr << "inet_pton failed: invalid address string" << std::endl;
        } else {
            printLastError("inet_pton");
        }
        closesocket(ConnectSocket);
        WSACleanup();
        return 1;
    }

    std::cerr << "Server address setup succeeded" << std::endl;

    // Connect to server
    iResult = connect(ConnectSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
    if (iResult == SOCKET_ERROR) {
        printLastError("connect");
        closesocket(ConnectSocket);
        WSACleanup();
        return 1;
    }
    std::cerr << "Successfully connected to the server" << std::endl;

    // Cleanup
    closesocket(ConnectSocket);
    WSACleanup();
    return 0;
}