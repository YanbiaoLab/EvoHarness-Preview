# E5 对照实验计划(经验闭环验收)

> **2026-08-04 状态**:e5s 已越过"对照实验"阶段直接进入生产(IMO E5 系列 run、
> modmul r14–r15),E-deadend 给出跨轮次的自然实验证据(8/16→0/16)。但
> **严格的 E5r/E5s vs E3r 同种子配对从未跑完** —— 该验收现并入更大的
> e6p vs e5s 配对实验(docs/EvoHarness.md §3.5),不再单独排期。本文的
> 实验设计(臂定义、判读口径)在配对实验设计时直接复用。

2026-07-24。回答的问题:**LLM 归因(L1/L2)比机械记录(L0)多赚多少;
定向指令与算子 bandit 各自再叠加多少。** 前置:buffer 清单 1–10 +
两个可叠臂已落码全绿;`--experience-mode` / `--lesson-directive` /
`--operator-bandit` / `--reflect-batch-size` 均进 manifest。

## 0. 实验约束(先认清再设计)

- spec 冻结:每 run `max_candidates=15`(2 种子 + 13 后代),
  val 12 题、单次评估 → **噪声地板 ~0.083**。小于它的差异不可读。
- reflect_batch_size=4(默认已改):首批 lessons 在第 ~5 个候选后生效,
  覆盖后段 ~8 个提案。batch=8 会让 E5 臂形同虚设(只剩 5 个提案受益)。
- 巩固阈值 40 在 15 候选 run 中不会触发——本轮实验实际上验证的是
  L0+L1+scratchpad,Consolidation 要等大 run 或多域。
- 定价字段为 null → BudgetMeter 记 0,**成本盯 provider 控制台**。
- directive 臂依赖 lessons 数据,只能叠在 lessons 臂上;bandit 可叠任意臂。

## 1. 阶段划分

### Stage 0 — live 冒烟(1 run,先跑这个)

```bash
python -m experiments.imo_proof.run run --run-dir results/smoke_e5s \
  --seed 0 --experience-mode lessons+scratchpad
```

验收清单(不看分数,只看机制):
- [ ] `results/smoke_e5s/experience.jsonl` 有 `"kind": "lesson"` 行,
      且 verdict/tags 合理(真实模型的 JSON 能被 `_normalize` 吃下);
- [ ] 日志无连续 "reflection batch skipped"(偶发可容忍,连续=格式不合);
- [ ] checkpoint.json 的 MutationReflector state 里 scratchpad 非空、
      三分区结构成立;
- [ ] 至少一次提案的 system prompt 出现 "# Lessons from past mutations"
      (翻 run_dir 提案 trace);
- [ ] 反思调用总成本占比 < 10%(控制台估)。
任何一条不过 → 修完再进 Stage 1(大概率是反思 prompt 要按真实模型
的输出习惯微调)。

### Stage 1 — 主对照(6 runs):E3r vs E5s × 种子 {0,1,2}

```bash
# arm=retrieval / lessons+scratchpad ; seed=0/1/2,配对使用相同种子
python -m experiments.imo_proof.run run --run-dir results/e3r_s{K} \
  --seed {K} --experience-mode retrieval
python -m experiments.imo_proof.run run --run-dir results/e5s_s{K} \
  --seed {K} --experience-mode lessons+scratchpad
```

E5r(不带 scratchpad)本轮不跑:E5s 是完整主张,若 E5s 赢再消融
拆分贡献;若 E5s 输,E5r 大概率也不赢,省 3 个 run。

### Stage 2 — 可叠臂(条件触发,4 runs):仅当 E5s ≥ E3r

```bash
python -m experiments.imo_proof.run run --run-dir results/e5sD_s{K} \
  --seed {K} --experience-mode lessons+scratchpad --lesson-directive
python -m experiments.imo_proof.run run --run-dir results/e5sB_s{K} \
  --seed {K} --experience-mode lessons+scratchpad --operator-bandit
```

种子 {0,1};双开(D+B)等单开结果出来再定,避免一次烧 6 个 run。
若 Stage 1 打平:bandit 可单独叠在 E3r 上再试(它不依赖 lessons)。

## 2. 指标与判定规则

**主指标**:val best fitness;`summary["test_points_percentage"]`
(终选候选的 test 分)。
**机制指标**(每 run 从 manifest/history/experience.jsonl 提取):
novelty_rejections 总数、proposal_failed 数、lessons 中 noise 占比、
(bandit 臂)算子份额随代数的漂移、(directive 臂)指令触发次数。

**判定规则(抗噪纪律,提前写死防事后择优)**:
- 配对比较:同种子 arm 间相减,不做跨种子平均比大小;
- 判"赢"需同时满足:≥2/3 种子方向一致 **且** 配对均值差 > 0.083
  (val 噪声地板);
- 终选候选加测:winner 的 test 评估重复 3 次取均值再报
  (治 0.595→0.389 型单次幻觉;backlog#5 的临时手工版);
- 不满足即判平局,平局取更便宜的臂(E3r 无反思成本)。

**结果去向**:更新 [[imo-evolution-results]] 记忆 + 设计文档 §5 打勾;
E5s 赢 → 下一步做"lesson 升格"的默认化评估;输/平 → 反思 prompt
迭代一轮(读 smoke 的 lessons 质量找病因)再战,而不是直接放弃架构。

## 3. 总量与成本

最多 1 + 6 + 4 = 11 runs(Stage 2 条件触发)。单 run 量级:
13 次 agentic 提案会话 + 15×12 题 solver/grader 评估 + 2~3 次反思调用。
以历史两轮 run 的实测花费为基准 ×11 估算总预算,超预算先砍种子 2。
