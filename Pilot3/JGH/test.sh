#!/bin/bash

# Start the process
echo "STARTING THE WRANGLING PROCESS"
echo "CALLING THE SB FILES NOW..."
echo ""

echo "RUNNING: single-model.py ../../Data/Pilot3/P3B3_data/ ./ 0 0 0"
python single-model.py ../../Data/Pilot3/P3B3_data/ ./ 0 0 0

echo "RUNNING: single-model.py ../../Data/Pilot3/P3B3_data/ ./ 0 1 1"
python single-model.py ../../Data/Pilot3/P3B3_data/ ./ 0 1 1

echo "RUNNING: single-model.py ../../Data/Pilot3/P3B3_data/ ./ 0 2 2"
python single-model.py ../../Data/Pilot3/P3B3_data/ ./ 0 2 2

echo "RUNNING: single-model.py ../../Data/Pilot3/P3B3_data/ ./ 0 3 3"
python single-model.py ../../Data/Pilot3/P3B3_data/ ./ 0 3 3

echo "RUNNING: single-model.py ../../Data/Pilot3/P3B3_data/ ./ 0 4 4"
python single-model.py ../../Data/Pilot3/P3B3_data/ ./ 0 4 4

echo ""
echo "======================"
echo "FINISHED RUNNING FILES"
echo "======================"
