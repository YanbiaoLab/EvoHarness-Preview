# modmul:进化有效性验证协议

2026-07-26。**判据在开跑前固定**,事后不得移动。

## 为什么需要这份文件

到今天为止,进化在这个项目上**没有任何有效性证据**:

- modmul:一次完整 run 都没跑过,`h90 = 9` 全部是手写 seed 的成绩;
- IMO:四个 run、约 14 小时开销,但 **seed 从未在 test 划分上评过**,
  效应量没有测量值;四个 run 的 test 分数彼此之差小于一个标准误。

而且那四个 IMO run 期间大部分机制是死的(算子对优化器不可见、novelty 门
0 次触发、迁移 0 次、bandit/lesson-directive 从未开启、27% 候选被预算 bug
错记、L1 归因编造因果并注入下游)。所以此前既没有证据说它有用,也没有
证据说它没用 —— 问题**根本没被问出来过**。

modmul 是第一次能真正问出来的战场:目标离散可验证、缺口是具体的可变异
常数、而且有一个公开的 h90=10 存在证明说明天花板不在这里。

## 对照组(control)

`limb_horner` 种子,完整 rungs,L40S:

```
fitness 0.8993   h90 9   overall 0.892   params 91,841
tier 1-8 100%    tier 9 92%    tier 10 0%(从未运行,预算耗尽)
perturbation_random_acc 0.0 vs original 0.9733
```

正在跑第二次读数(n=2)。若两次 `h90` 不一致,对照组取**较低**的那个,
并在报告里写明散布 —— 拿高的那个当基线是在给自己放水。

## 判据一:SOTA(拿分)

**任一候选的 leaderboard key `(h90, overall_accuracy)` 严格大于对照组的
`(9, 0.892)`。** 即 `h90 = 10`,或 `h90 = 9` 且 `overall > 0.892`。

用 `leaderboard_key()` 判,不用 `fitness` —— fitness 是搜索信号,已经
刻意与排名键不同构。

必须同时满足:
- `passed = True`(扰动闸通过,即答案来自训练参数);
- 静态检查通过;
- **换一个随机种子复测仍然成立**(单次读数不作数)。

## 判据二:进化是否真的在起作用(这才是主问题)

拿分可以靠"注入的知识 + 手写 seed 已经很强"达成,那不证明搜索有用。
所以另设一条:

**搜索是否独立到达了两条我知道、但刻意没写进 prompt 的杠杆?**

注入的是**现象**(前沿是时间预算不是精度;步数同时买时间与可靠性;
状态宽余量是未测轴且他人测到过两个数量级的摆动),**没有注入设置**
(radix 设多少、余量加多少)。

判定方式(读 `run.db` 的 diff 与 `visible_metrics`):

| 结果 | 判定 |
|---|---|
| 有候选改了 radix 或状态宽余量**并因此提分** | **强证据:搜索找到了没被告知的东西** |
| 提分了,但全部来自训练配方/超参微调 | 弱证据:搜索在利用注入的知识,没有发现新东西 |
| 没有候选超过对照组 | **零结果 —— 走下面的诊断路径** |

第二列那条要写清楚,不许事后美化成成功。

## 零结果时的诊断路径(按顺序,每步都要有测量)

1. **活性审计**(`scripts/audit.py`)—— 先确认机制在转,而不是又一次
   "单元测试全绿但机制没触发"。重点查:`warm_start` 是否真的出现 `full`、
   `budget_headroom` 是否真的被算出来、岛间迁移在 gen 10 是否发生、
   算子是否产出不同的编辑。
2. **变异是否触及关键文件** —— 统计 `change_summary` 里 `arch.py` /
   `train.py` / `model.py` 的分布。若 `train.py` 几乎不被改,说明
   "一次只改一个文件"的例外没生效或没被理解。
3. **选择压力** —— 统计亲本复用次数与岛内多样性。若又是同一亲本连选
   5-6 次(IMO 上的实测形态),问题在选择而不是变异。
4. **信号** —— `fitness` 的实际分布。若所有候选都挤在对照组附近,说明
   软化后的度量在这个区间仍然太平。
5. 以上都健康却仍无提升 → 才是"进化在这个任务上确实没用"的证据,
   而且此时才知道**是哪一环没用**。

## 运行配置

```bash
source /root/l40s_env.sh && set +u
cd /root/EvoHarness && set -a && source .env && set +a
export PYTHONPATH=/root/EvoHarness/experiments:${PYTHONPATH:-}
nohup .venv/bin/python -m experiments.run_evolution \
  --recipe e3r --task modmul --live \
  --run-dir /root/userdata/evoharness_results/modmul_r1 \
  --config experiments/modmul/experiments/gpu_run3.yaml \
  > /root/userdata/evoharness_results/modmul_r1.log 2>&1 &
```

`gpu_run3.yaml`:120 代 / 3 岛(limb_horner, horner_cell, serial_ar)/
`eval_batch_size 3` / 迁移间隔 10 / archive 60。全部 12 个键已对过当前配置类。

⚠️ **时间估计要重算**:yaml 里的 40-70 小时是**没有权重继承**时的经验值。
继承会让更多候选晋级到 R2(90 分钟档)而不是死在 R0(8 分钟档),
每代墙钟会变长。跑起来后按前 5 代的实测重新外推,别拿旧数字规划。

checkpoint 每代落盘,同一 run-dir 重跑即续跑。

## 已知会污染结论的因素(先写下来,免得事后找借口)

- **对照组只有 n=1~2**,而 tier 9 的 92% 是 50 例上的读数(官方 100 例)。
  若某候选拿到 h90=9/overall 0.90,可能只是噪声。所以判据一要求换种子复测。
- **h90=9 本身靠批内超支过关**(tier 9 超预算 29%,预算检查在批次之间)。
  对照组和候选都受同一条规则,组内可比;但对外报 SOTA 时必须写明。
- **权重继承让 fitness 路径依赖** —— 谱系的分数反映累积算力而非仅基因,
  跨岛不可比。冠军出来后要**冷启动重测**一次,量它基因本身的价值。
- `holdout = unavailable`(私有集没生成),没有过拟合预警。
