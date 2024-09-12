// EnvUtils.cpp
#include "EnvUtils.h"
#include <iostream>
#include <cstdlib>

bool getAndVerifyEnvDirs(std::string& dataDir, std::string& outDir) {
    const char* DATA_DIR = std::getenv("DATA_DIR");
    const char* OUT_DIR = std::getenv("OUT_DIR");

    if (DATA_DIR == nullptr) {
        std::cerr << "Error: DATA_DIR environment variable not set." << std::endl;
        return false;
    }

    if (OUT_DIR == nullptr) {
        std::cerr << "Error: OUT_DIR environment variable not set." << std::endl;
        return false;
    }

    dataDir = DATA_DIR;
    outDir = OUT_DIR;

    // Create the output directory if it doesn't exist
    std::string mkdirCommand = "mkdir -p " + outDir;
    int mkdirResult = system(mkdirCommand.c_str());
    if (mkdirResult != 0) {
        std::cerr << "Error: Failed to create output directory." << std::endl;
        return false;
    }

    // Create the log directory if it doesn't exist
    std::string logDir = outDir + "/log";
    std::string mkdirLogCommand = "mkdir -p " + logDir;
    int mkdirLogResult = system(mkdirLogCommand.c_str());
    if (mkdirLogResult != 0) {
        std::cerr << "Error: Failed to create log directory." << std::endl;
        return false;
    }

    return true;
}
