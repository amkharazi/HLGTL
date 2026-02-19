#!/bin/bash

# این دستور باعث می‌شود اگر یکی از اسکریپت‌ها با ارور مواجه شد،
# کل پروسه متوقف شود تا وقت تلف نشود.
set -e

echo "=========================================================="
echo "Starting Experiment 1: TEST_001 (Base Model - 3 Seeds)"
echo "=========================================================="
python TEST_001_efficientNet_25_seed.py

echo ""
echo "=========================================================="
echo "Starting Experiment 2: TEST_002 (TRL Model - 3 Seeds)"
echo "=========================================================="
python TEST_002_efficientNet_28_seed.py

echo ""
echo "=========================================================="
echo "Starting Experiment 3: TEST_003 (Our Method - 3 Seeds)"
echo "=========================================================="
python TEST_003_efficientNet_32_seed.py

echo ""
echo "=========================================================="
echo "All experiments finished successfully!"
echo "=========================================================="