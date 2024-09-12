
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <thread>
#include <mutex>
#include <vector>

#include "EnvUtils.cpp"
#include "TH1F.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TFile.h"
#include "TTree.h"


// Filepath and directory variables
std::string dataDir;
std::string outDir;
int RECORD_LENGTH = 1024;


// Define a struct that will hold your event data
class EventData {
public:
    Float_t max_value;
    Int_t max_value_index;
    Float_t sum_values;
    Float_t values[1024];
    Int_t event_number;
    Int_t channel;

    void Clear() {
        max_value = 0;
        max_value_index = 0;
        sum_values = 0;
        memset(values, 0, sizeof(values));
        event_number = 0;
    }
};

void plot_event(const EventData& event, int event_number) {
    TCanvas *c1 = new TCanvas("c1", "Event Plot", 800, 600);
    TGraph *gr = new TGraph(RECORD_LENGTH);
    
    for (int i = 0; i < RECORD_LENGTH; ++i) {
        gr->SetPoint(i, i, event.values[i]);
    }
    
    gr->SetTitle(Form("Event %d (Channel %d)", event_number, event.channel));
    gr->GetXaxis()->SetTitle("Sample");
    gr->GetYaxis()->SetTitle("Amplitude (background subtracted)");
    gr->SetLineColor(kBlue);
    gr->Draw("AL");

    
    std::string plotFilePath = outDir + "/event_" + std::to_string(event_number) + "_plot.png";
    c1->SaveAs(plotFilePath.c_str());
    
    delete gr;
    delete c1;
}

void create_root_from_txt() {
    if (!getAndVerifyEnvDirs(dataDir, outDir)) {
        return;
    }

    std::string filePath = dataDir + "/set2/27";
    std::vector<std::string> fileNames = {"/wave_0.txt", "/wave_1.txt"};
    std::vector<int> channels = {0, 1};

    std::string outputFilePath = outDir + "/output.root";
    TFile f(outputFilePath.c_str(), "recreate");
    TTree t1("t1", "Event data");

    EventData eventData;
    t1.Branch("max_value", &eventData.max_value, "max_value/F");
    t1.Branch("max_value_index", &eventData.max_value_index, "max_value_index/I");
    t1.Branch("sum_values", &eventData.sum_values, "sum_values/F");
    t1.Branch("values", eventData.values, Form("values[%d]/F", RECORD_LENGTH));
    t1.Branch("event_number", &eventData.event_number, "event_number/I");
    t1.Branch("channel", &eventData.channel, "channel/I");

    std::vector<float> eventValues(RECORD_LENGTH);
    std::string line;
    float value;

    for (size_t i = 0; i < fileNames.size(); ++i) {
        std::string fullPath = filePath + fileNames[i];
        std::ifstream inFile(fullPath);
        if (!inFile) {
            std::cerr << "Error opening file: " << fullPath << std::endl;
            continue;
        }

        int eventCount = 0;
        int lineCount = 0;

        while (inFile) {
            eventValues.clear();
            eventValues.reserve(RECORD_LENGTH);

            // Read RECORD_LENGTH values
            for (int j = 0; j < RECORD_LENGTH && std::getline(inFile, line); ++j) {
                if (line.empty() || line.find(":") != std::string::npos) {
                    --j;
                    continue;
                }
                value = std::stof(line);
                eventValues.push_back(value);
                ++lineCount;
            }

            if (eventValues.size() == RECORD_LENGTH) {
                eventData.Clear();
                
                // Calculate background (average of first 30 values)
                float background = std::accumulate(eventValues.begin(), eventValues.begin() + 30, 0.0f) / 30;

                // Subtract background and fill eventData
                for (int k = 0; k < RECORD_LENGTH; ++k) {
                    eventData.values[k] = eventValues[k] - background;
                }

                eventData.max_value = *std::max_element(eventData.values, eventData.values + RECORD_LENGTH);
                eventData.max_value_index = std::distance(eventData.values, std::max_element(eventData.values, eventData.values + RECORD_LENGTH));
                eventData.sum_values = std::accumulate(eventData.values, eventData.values + RECORD_LENGTH, 0.0f);
                eventData.event_number = eventCount;
                eventData.channel = channels[i];

                t1.Fill();
                ++eventCount;
                if (eventCount % 5000 ==0) {std::cout << "Events processed: "<< eventCount << std::endl;}

            }
        }

        std::cout << "File: " << fullPath << std::endl;
        std::cout << "Total lines read: " << lineCount << std::endl;
        std::cout << "Processed events: " << eventCount << std::endl;
    }

    t1.Write();
    f.Close();

    std::cout << "ROOT file created: " << outputFilePath << std::endl;
}