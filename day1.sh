#!/bin/bash
set -e  # 遇到错误自动停止
echo "=== $(date): Start job 1 ==="
python ParamSearch.py --cfg configs/GPS/arxiv-GPS.yaml 

echo "=== $(date): Job 1 finished, start job 2 ==="
python ParamSearch.py --cfg configs/Graphormer/arxiv-Graphormer.yaml  
