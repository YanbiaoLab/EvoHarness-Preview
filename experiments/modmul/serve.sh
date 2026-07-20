#!/usr/bin/env bash
# 起 modmul 评估服务(GPU 机或本机),从仓库根运行。
# task_version 绑定 third_party 克隆 commit —— 裁定/评分逻辑变了就换版本号。
cd "$(dirname "$0")/.." || exit 1
exec python -m evoharness.evoserve \
    --grade-fn modmul.grade:grade_fn \
    --task-version "modmul-fb558ce-v1" \
    --eval-set-version "public-t1-3-n50-w124" \
    --port "${PORT:-8321}" \
    --max-workers "${MAX_WORKERS:-2}" \
    ${EVOSERVE_TOKEN:+--token "$EVOSERVE_TOKEN"}
