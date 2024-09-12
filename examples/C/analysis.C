#include "TH1F.h"
#include "TCanvas.h"
#include "TTree.h"
#include "TFile.h"

ROOT::RDataFrame df(100);

gRandom->SetSeed(1);
auto df_1 = df.Define("rnd", []() { return gRandom->Gaus(); });

df_1.Snapshot("randomNumbers", "df008_createDataSetFromScratch.root");
