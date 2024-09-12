#include "TFile.h"
#include "TTree.h"
#include <vector>
#include <memory>


void create_root() {

std::vector<double> data = {1.2, 3.4, 5.6, 7.8, 9.0}; 

std::unique_ptr<TFile> myFile( TFile::Open("waveDump.root", "RECREATE") );

TTree* tree = new TTree("tree", "Example Tree"); 

tree->Branch("data", &data);
tree->Fill(); 
tree->Write(); 
myFile->Close();

}