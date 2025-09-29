#!/usr/bin/env bash
set -euo pipefail

# ===== 默认参数（允许有进程）=====
MIN_FREE_MB=${MIN_FREE_MB:-20000}   # 至少空闲显存(MiB)，默认 10GB
MAX_UTIL=${MAX_UTIL:-10}            # GPU利用率上限(%)，默认 <=10%
INTERVAL=${INTERVAL:-60}            # 轮询间隔(秒)

log() { echo "[$(date '+%F %T')] $*"; }

# 防止同机重复挂同一个 watcher（需要可留，不要可删）
exec 9>/tmp/wait_gpu.lock 2>/dev/null || true
flock -n 9 || { log "已有 GPU watcher 在运行，退出。"; exit 0; }

command -v nvidia-smi >/dev/null || { echo "未找到 nvidia-smi"; exit 1; }

if [[ $# -eq 0 ]]; then
  echo "用法：$0 -- <训练命令...>"
  echo "例： $0 -- python train.py --cfg config.yaml"
  exit 1
fi

pick_gpu() {
  # 选择满足条件里“空闲显存最大”的那张卡
  mapfile -t rows < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
                      --format=csv,noheader,nounits)
  best_idx=""
  best_free=-1
  for r in "${rows[@]}"; do
    IFS=',' read -r idx free util <<<"$r"
    free=${free//[[:space:]]/}
    util=${util//[[:space:]]/}
    if (( free >= MIN_FREE_MB && util <= MAX_UTIL )); then
      if (( free > best_free )); then
        best_free=$free
        best_idx=${idx//[[:space:]]/}
      fi
    fi
  done
  [[ -n "$best_idx" ]] && echo "$best_idx" || return 1
}

log "等待GPU：min_free=${MIN_FREE_MB}MiB, max_util=${MAX_UTIL}%（允许有进程）"
while true; do
  if idx=$(pick_gpu); then
    # 二次确认避免竞态
    sleep 1
    idx2=$(pick_gpu || true)
    if [[ "$idx" != "${idx2:-}" ]]; then
      log "刚被人抢走了，继续等..."
      sleep "$INTERVAL"
      continue
    fi
    export CUDA_VISIBLE_DEVICES="$idx"
    log "已选中 GPU $idx，启动命令：$*"
    exec "$@"
  else
    brief=$(nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu \
            --format=csv,noheader,nounits | tr -s ' ')
    log "尚未满足条件，${INTERVAL}s 后重试。概况："
    echo "$brief" | sed 's/^/  /'
    sleep "$INTERVAL"
  fi
done
