#!/bin/bash
cp "$HOME/qrl/results/noise_feedback_v2/noise_feedback_v2_results.json" \
  "/mnt/c/Users/HZR/Desktop/揭榜挂帅擂台赛/qrl-fix-732/results/noise_feedback_v2/noise_feedback_v2_results.json"
cp "$HOME/qrl/results/noise_feedback_v2/noise_feedback_v2_report.md" \
  "/mnt/c/Users/HZR/Desktop/揭榜挂帅擂台赛/qrl-fix-732/results/noise_feedback_v2/noise_feedback_v2_report.md" 2>/dev/null || true
echo "结果已复制回 Windows"
