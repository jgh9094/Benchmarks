#!/bin/bash

# Start the process
echo "STARTING PROCESS"
echo ""

echo "CREATING SINGLE MODELS FOR TASK 1"

echo "RUNNING: single-model.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ ./ 0 0 0 0"
python single-model.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ ./ 0 0 0 0

echo "RUNNING: single-model.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ ./ 0 1 1 0"
python single-model.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ ./ 0 1 1 0

echo "RUNNING: single-model.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ ./ 0 2 2 0"
python single-model.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ ./ 0 2 2 0

echo "RUNNING: single-model.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ ./ 0 3 3 0"
python single-model.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ ./ 0 3 3 0

echo "RUNNING: single-model.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ ./ 0 4 4 0"
python single-model.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ ./ 0 4 4 0

echo "CREATING ENSEMBLE FOR TASK 1"

echo "RUNNING: ensemble-model.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ ./ ./ 0 5 0"
python ensemble-model.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ ./ ./ 0 5 0

echo "DISTILLING KNOWLEDGE INTO SINGLE CNN"

echo "RUNNING: python distiller.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ ./ ./ 0 0"
python distiller.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ ./ ./ 0 0

echo "CREATING MTCNN"

echo "RUNNING: python mtcnn-model.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ ./ 0 0"
python mtcnn-model.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ ./ 0 0

echo "DISTILLING MTCNN INTO SMALLER MTCNN"

echo "RUNNING: python distiller-mtcnn.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ /home/60h/GEM-SUMMER-2021/Benchmarks/Pilot3/JGH/MTModel-0/ ./  0 0 7"
python distiller-mtcnn.py /home/60h/GEM-SUMMER-2021/Benchmarks/Data/P3B3_data/ /home/60h/GEM-SUMMER-2021/Benchmarks/Pilot3/JGH/MTModel-0/ ./  0 0 7
echo ""
echo "======================"
echo "FINISHED RUNNING FILES"
echo "======================"
