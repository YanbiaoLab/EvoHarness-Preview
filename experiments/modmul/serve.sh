#!/usr/bin/env bash
# 起 modmul 评估服务(GPU 机或本机),从仓库根运行。
#
# ⚠️ Round-1 起,主线**不再走这条路径**:基因组已经是多文件 workspace,而评估
# 协议 v1 只在 wire 上传 main_text()(evocore/remote.py::_submit),侧文件过不去。
# 主跑请在评测机上直接跑 run_evolution(in-process WorkspaceGradeFnGrader):
#
#     python -m experiments.run_evolution --recipe e3r --task modmul --live \
#         --run-dir /root/userdata/modmul_r1 \
#         --config experiments/modmul/experiments/gpu_run3.yaml
#
# 这个脚本保留给单文件候选的远程评测(协议 v2 落地前的兼容路径)。
# task_version 绑定 third_party 克隆 commit —— 裁定/评分逻辑变了就换版本号。
cd "$(dirname "$0")/.." || exit 1
exec python -m evoharness.evoserve \
    --grade-fn modmul.grade:grade_fn \
    --task-version "modmul-fb558ce-v2" \
    --eval-set-version "public-t1-10-asha-h90" \
    --port "${PORT:-8321}" \
    --max-workers "${MAX_WORKERS:-3}" \
    ${EVOSERVE_TOKEN:+--token "$EVOSERVE_TOKEN"}
