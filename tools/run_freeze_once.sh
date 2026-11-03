#!/bin/bash
# 补跑首次观察基线：已存在基线指针则跳过（可用于一次性 cron）。
cd "$(dirname "$0")/.." || exit 1
if [ -f .regression-control/baseline.json ]; then
  echo "观察基线已存在，跳过。"
  exit 0
fi
exec ./start.sh freeze-baseline
