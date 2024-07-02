#!/bin/bash

# Define the directories
DATA_DIR="./data/" 
OUT_DIR="./outputs/"

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$OUT_DIR"

# Print the directories to verify
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUT_DIR"