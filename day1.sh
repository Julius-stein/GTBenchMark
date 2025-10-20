#!/bin/bash
set -e  # 遇到错误自动停止
echo "=== $(date): Start job 1 ==="
python main.py --cfg configs/GPS/zinc-GPS.yaml 

echo "=== $(date): Job 1 finished, start job 2 ==="
python main.py --cfg configs/Exphormer/zinc-Exphormer.yaml 
