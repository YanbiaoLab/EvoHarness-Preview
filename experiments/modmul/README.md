<!-- modmul 实验入口文档 v1（2026-07-26）。
     定位：目录边界 + 当前状态 + 对照系拆解 + 下一步建议。
     领域知识的正式注入口是 research_msg.md（进 prompt），本文件不进 prompt，
     是给人看的。两者内容重叠时以 research_msg 的版本号为准。 -->

# modmul 实验

进化一个提交给 SAIR Modular Arithmetic Challenge 的 PyTorch 模型，计算
`(a * b) mod p`。截止 2026-08-12 23:59 AoE。

排名键是 `(highest_tier_above_90, overall_accuracy)`。官方三个参考基线全部
`h90 = 1`、`overall <= 0.127`。

## 目录边界

```text
modmul/
├── task.py             # ScorableTask 组装点：三岛异构种子 + 两个 prompt 注入
├── grade.py            # ASHA 三档评分、裁定、扰动闸、fitness
├── holdout.py          # 私有种子测试集生成（另一个 master seed，只做过拟合预警）
├── base_model.py       # 官方接口的本地镜像
├── sys_msg.md          # 注入每次变异的 system prompt（SearchConfig.task_sys_msg）
├── research_msg.md     # 注入所有岛的冻结领域简报（TaskBundle.research_brief）
├── seeds/
│   ├── limb_horner/    # 岛 0：位宽通用扫描 cell —— 通往 tier 4+ 的路线
│   ├── horner_cell/    # 岛 1：固定 16-bit cell，快而诚实的 tier 1-3 基线
│   └── serial_ar/      # 岛 2：串行 AR transformer，多样性/对照谱系
├── baselines/          # closed_genome.py，非进化对照
├── experiments/        # 运行配置 yaml（gpu_run3.yaml 是 Round-1 主跑）
└── 调研报告_神经Horner线_2026-07-25.md
```

基因组是三个文本文件 —— `model.py`（推理契约，合规关键面）、`arch.py`（架构）、
`train.py`（训练配方）。见 `task.py` 的 `_GENOME_FILES`。

主跑命令（在评测机上，in-process grader）：

```bash
python -m experiments.run_evolution --recipe e3r --task modmul --live --run-dir /root/userdata/modmul_r1 --config experiments/modmul/experiments/gpu_run3.yaml
```

## 当前状态（2026-07-26）

| 项 | 值 | 出处 |
|---|---|---|
| ASHA 三档追加训练 | R0 480s / R1 1320s / R2 3600s | `grade.py` 的 `RUNGS` |
| 每档评测层 | R0 = t1-3 / R1 = t1-6 / R2 = t1-10 + t0 诊断 | 同上 |
| 晋级判据 | R0: t3 ≥ 15% 或 t2 ≥ 60%；R1: h90 ≥ 3 或 t4 ≥ 10% | `_promotes` |
| fitness | 连续 overall + 软化跨阈奖励（跨阈约 2 倍权重，非 50 倍） | `_fitness` |
| 代数 / 并发 | 120 代 / `eval_batch_size: 3` | `experiments/gpu_run3.yaml` |
| 噪声实测 | QUICK 档三次重复 fitness 0.1018 / 0.1018 / 0.1036 | `results/modmul_noise.json` |
| limb_horner 参数量 | 91,841 | 实测 |

## 对照系：Lei285714/mod-arith-k2 拆解（2026-07-26 研究）

HuggingFace 上的公开提交，是目前这条路线跑到的最远处。公开基准 overall 98.9%、
`h90 = 10`、4090 上 129.7s / 300s 预算。它是 Robby Sneiderman 的 neural-horner v8
（radix-2，MIT）的 radix-4 分叉。

### 它的结构

一个约 471K 参数的 `p`-条件循环 cell（双向 3 层 GRU，d_model 96，hidden 128），
学单步转移 `s' = (4s + d*x) mod p`，`d ∈ {0,1,2,3}`；外面套一个手写的、
固定的、无反馈的 Horner 循环，三趟共享同一份权重：

1. 扫 `a` 的 radix-4 数字，`x = 1` → `a mod p`
2. 扫 `b` 的 radix-4 数字，`x = 1` → `b mod p`
3. 扫 `b mod p` 的 radix-4 数字，`x = a mod p` → 答案

`output_base = 2`，状态位直接作为答案数字交给官方 decoder。这和本实验
`seeds/limb_horner/model.py` 的三趟结构是同一个形状。

合规论证的搭法值得照抄：循环调度不接受模型任何反馈；包括条件减法在内的每一步
算术都出自训练参数；两处路由只以素数的 bit-length 为 key，不看数值；交付物里
带一个**权重置换消融脚本**（`build_and_verify.sh`），把参数 flatten 后随机重排，
断言准确率从 >80% 塌到 <5%。我们的 `grade.py` 已有 L3 扰动闸，但提交时也该附
这样一份自证。

### 它真正的收益来自哪里 —— 不是架构

作者自己写明：**最大单点收益是一个推理期设置，零重训。**

存 `s` 的寄存器开多宽（Δ = 状态位宽 − 素数位宽），数学上多出来的高位恒为零、
完全等价，但实测（576 题/档，四个 tier 带）：

| Δ | 0 | 1 | 2 | 3 | 4 | 8 | ≥16 |
|---|---|---|---|---|---|---|---|
| 错误数 | 27 | **516** | **470** | 116 | 13 | 13 | 3 |

Δ ∈ {1,2} 掉到 2–30% 准确率。原因不是数学，是**训练分布**：radix-4 重训时每个
素数尺寸都在恰好填满的位宽上训，只覆盖 Δ = 0。而 v8 的一步训练把状态宽固定在 L
并按 bit-length 分层，覆盖了全部 Δ，在同一批题上 **Δ ∈ {0,1,2,3,4,32} 全部零错误**
—— 平坦曲线。循环、推理代码、题目全同，差别纯粹在训练分布。

派生出的陷阱，作者说证伪它花了一周：官方 tier 的素数聚集在 32 的倍数下方一点点，
所以"向上取整到 32 的倍数"恰好把它们送进 Δ = 1~2 的死亡区；正确做法是"**加** 32
的固定余量"。两者听起来几乎一样，结果差几十倍。

### 已被证否的方向（别重复）

| 方向 | 结果 |
|---|---|
| 无锚点小素数微调 | 治好小素数，吃掉大尺度精度（T8 −8、T10 −13） |
| L2-SP 抗遗忘 | 无 λ 窗口；λ = 10/30/100 只是把干扰在位宽之间搬家 |
| 同配方再退火（单独用） | 零收益 |
| 集成（位宽委员会 / 跨 checkpoint 投票） | 两种形式都失败 |
| 加大容量 | 三方独立测量一致：不是瓶颈 |

机理作者量出来了：**单步验证的精度地板约 8e-6，而有效 ε 已经是 1.8e-6**。
目标函数和 checkpoint 选择都掉到量程以下 —— 继续训练是在噪声里走。
他的结论："覆盖度和精度在这里是零和的。"

### 有效的两个便宜手段

- **model soup**：交付权重是退火 checkpoint 与同配方再退火的均匀权重平均，
  在同题配对评测上打赢两个端点（单侧 McNemar p = 0.02，n = 1008）。
- **小素数专家**：`random_prime` 类构造器只产最高位置 1 的奇数，p = 2 永远采不到，
  Tier 1 卡在 61%。p ∈ {2,3,5,7} 的单步转移全空间只有 348 条，直接枚举，
  另训一份权重按 bit-length ≤ 3 路由。

### 标注为 OPEN 的一条

作者测得 radix-4 相对 radix-2 **只买到 2× 速度，精度反而略降**（同一探针集上 v8
在每个位宽都赢）。但那是在 **GRU cell** 上测的。本实验的 cell 是 log 深度的
Hillis-Steele 扫描，结构不同，这个结论是否迁移**未知** —— 留给搜索去答，
不要当定论写进 prompt。

## 对本实验的诊断

### 错配一：进化在搜"90 分钟内能跑多远"，目标却是"能不能到 tier 10"

R0 + R1 + R2 = 90 分钟，从随机初始化开始。mod-arith-k2 的结果之所以存在，是因为它
warm-start 自一个已训到位的 v8，再退火、再退火、做 soup、再单独退火一个小素数专家。
90 分钟从零训出来的候选，架构之间的差异大部分被"训练量不够"这个共同因素淹没。

`_fitness` 里的软化（`_soft_h90`）治的是地形悬崖；地形之所以平，根子在这里。

### 错配二：k2 收益最大的那条轴，在本实验里被封死了

- `seeds/limb_horner/train.py:98` — `p = rng.getrandbits(width) | (1 << (width - 1))`，
  恒满位宽，**训练只覆盖 Δ = 0**；
- `seeds/limb_horner/model.py:90` — `width = max(value.bit_length(), 2)`，推理也是 Δ = 0。

两边自洽，所以现在不崩（Δ = 0 在 k2 的表里是 27 错，第二好）。但要发现 Δ 这条轴，
必须**同时改 train.py 和 model.py**，而 `sys_msg.md:39` 写的是
"Prefer changing ONE file per mutation" —— 这条规则直接挡住了整条最高价值的路径。

即使某个变异碰巧只改推理端的 Δ，它仍要付满 90 分钟训练才能被评估。**最该密集探索
的轴上，采样成本最高。**

### 错配三：没有跨代记忆，每个候选都从随机初始化开始

`task.py` 的 `_GENOME_FILES` 只有三个文本文件；`GitWorkspace` 是文本树，
`materialize()`（`evoharness/evocore/workspace.py:217`）只写 `base_files` 加 patch。
权重不在基因组里，也没有别的通道 —— **120 代 × 90 分钟的训练全部各自扔掉**。

### 错配四：推理预算这个硬约束要到 R2 才被发现

实测 `limb_horner` 单步成本（**这台 Mac 的 MPS，设备相关；步数是结构性的**）：

```
参数 91,841   RADIX_BITS=1   ROUNDS=3
W= 128 N=32    0.225 ms/step/sample
W= 512 N=16    0.863 ms/step/sample
W=2048 N= 8    4.137 ms/step/sample
```

tier 10 在 `RADIX_BITS = 1` 下需要 4096 + 4096 + 2048 = **10240 步串行**。
对比 k2 的 radix-4 是 5136 步，而 k2 在 4090 上已经用掉 300 秒预算里的 130 秒。
所以本 seed 在 tier 9-10 大概率撞 `inference-budget-exceeded`，**与精度无关**，
而这件事目前要 90 分钟后才被告知。

一个反向的好消息：seed 的 Hillis-Steele 扫描是 log 深度（W = 2048 时 11 层，
全宽并行），k2 的 GRU 沿 2080 个位置串行。**在 GPU 上本 seed 的单步结构比 k2 更
友好** —— 问题出在 radix-1 的外层循环，不在 cell。这正是上面标 OPEN 那条的价值所在。

## 建议（按落地顺序）

前提：GPU 已就位，**算力不是约束**。真正的约束是到 8-12 的墙钟时间和**评估信号
的质量**。以下每条都是冲这两个去的。

### 一、seed 直接补丁（最先做，改变第一代看到的东西）

1. `train.py` 加 **Δ 分层**：素数按 `pbits` 采样，状态宽按 `pbits + margin`，
   `margin ∈ {0,1,2,3,4,8,16,32}`。k2 的 v8 覆盖全 Δ 得到平坦零错误曲线，
   这是免费的鲁棒性。
2. `sys_msg.md` 的"一次只改一个文件"开一个明确例外：**train.py 的数据分布与
   model.py 的推理策略是耦合的，允许同时改这一对**，并点名 Δ 是已知耦合轴。
3. 把推理策略提成显式变异面：`model.py` 顶部放一个 `POLICY` 字典，把旋钮命名出来
   —— 状态位宽余量、`max_batch_size`、推理期 radix、abstain 阈值、refinement rounds。
   现在这些散在代码里，LLM 要改就得重写整个方法。

**已覆盖、不要再花变异预算的**：`limb_horner` 的 `WIDTHS` 含 2 和 3，
`getrandbits(2) | 2` → p ∈ {2,3}，宽度 3 → {4..7}。**k2 踩的"小素数永远采不到"
那个坑，本 seed 天生没有。**

### 二、训练前成本探针（约 30 秒，第一代就杀掉走不通的架构）

随机权重下测 W ∈ {128, 512, 2048} 的单步耗时，乘候选自己声明的每层步数，投影
各层秒/题并与预算比。报成 `projected_infer_s_tier_N` 和 `budget_headroom_tier_10`。

把"要 90 分钟才能拿到的 `inference-budget-exceeded`"变成第一代就有的密集梯度。

### 三、免训练快通道

对 `arch.py` 与 `train.py` 做内容哈希；**两者都与父代逐字节相同 → 跳过
`_run_training`，继承父代权重，只评推理。** 90 分钟降到约 2 分钟。

算力免费不改变这条的价值：它买的是**同样墙钟时间内的采样密度**。Δ 那条轴只有在
这条落地后才谈得上真正可搜。

### 四、跨代权重继承（结构性的那条）

候选目录建好后、训练开始前，按 `arch.py` 哈希决定继承方式：

- 哈希相同 → 完整继承父代 `weights.pt` / `optimizer.pt` / `train_state.json`；
- 哈希不同 → **形状兼容的部分继承**（能对上的张量加载，对不上的留随机初始化）；
- 都不行 → 冷启动。

把 `warm_start: full|partial|cold` 与 `inherited_steps` 写进 `visible_metrics`。

k2 顺手示范了一个应当写进 `sys_msg` 的通用变异惯用法：**加新组件时把它零初始化，
让子代在旧设置下逐比特复现父代**（它给 op-embedding 零初始化，使 k = 1 时精确等于
v8）。这让"部分继承"从有损变成无损。

代价，写清楚：权重不在基因组里，适应度就变成路径依赖 —— 谱系的分数反映累积算力
而不只是基因，跨岛不可比，早期谱系容易锁死。缓解：定期把冠军**冷启动重测**，量它
基因本身的价值。这与仓库已有的 `_refuse_to_grade_the_seeds` 是同一种纪律。

### 五、把省下来的算力花在训练量上（针对错配一）

算力免费后，两个方向同时开：

- **提高并发**：cell 只有 92K 参数，`eval_batch_size: 3` 远未吃满 L40S 46GB。
  提并发保住代数。
- **提高每档训练**：R2 从 3600s 往上抬。抬多少要按抬完后的每代墙钟反推 ——
  120 代必须能在剩余天数内跑完，跑不完就降代数或降并发，不要两头都要。

具体数值需要按 gen-0 基线实测标定，不要拍脑袋写进 yaml。

### 六、代理适应度（先只做可见指标，别急着进 fitness）

端到端 tier 精度在当前训练量下对所有候选都接近 0，信息量极低。两个便宜、密集、
有依据的代理：

- **零样本位宽迁移**：只在宽度 8/12/16 上训，测 32/64/…/2048 的单步 exact-match。
  `arch.py` 的 docstring 里已经报过这张表（1.0 / .99 / .98 / .97 / .83 / .43 / .12）
  —— 它就是"位宽通用"与"位宽绑定"的分界线。
- **滚动存活深度**：固定宽度跑循环，记录到第一个 bit 错为止走了多少步。这是**直接
  决定能到第几层**的量（需要每步 1 − 1e-5）。

先塞进 `visible_metrics`（零风险，纯送信息），确认与 h90 在三个 seed 上相关之后，
再考虑折进 fitness。**测量协议必须写死在 grader 侧**，否则会被变异钻空子。

## research_msg 升版：v3 该加什么

`research_msg.md` 目前是 v2，写在 k2 模型卡之前（k2 建于 2026-07-25、改于 07-26）。
以下几条能直接省掉几十代变异预算：

| 要加的内容 | 为什么值钱 |
|---|---|
| Δ 位宽余量表（576 题/档）+ ceil32 与 +32 的陷阱 | 最高价值的可迁移事实 |
| 仪器极限：验证地板 8e-6 vs 有效 ε 1.8e-6 | 解释**为什么**继续训练没用，把搜索从死路拉回 |
| 已证否清单：L2-SP、集成、同配方再退火、加容量 | 直接标 DEAD END |
| model soup 有效（McNemar p = 0.02, n = 1008） | 便宜的"最后一公里"算子 |
| radix-4 = 2× 速度、精度略降 | **必须标 OPEN**：在 GRU cell 上测的，本 cell 未知 |
| 小素数 348 条枚举 | 标 already-covered（见上），别再修一遍 |

每条都要带 provenance 和样本量，标明"他人在 GRU cell 上的测量"。

**需要拍板的 confound**：`research_msg` 是所有岛共享的冻结简报，按其头部注释
"不是消融变量"。把 k2 的答案灌进去会提高本轮夺分概率，但削弱"框架能不能自己发现"
这个论证。建议折中 —— **注入"现象与已被证否的方向"（省时间），不注入 `+32` 这个
具体数值**（留给搜索定），并在 run manifest 里记版本号。

## 出处

- 模型卡与代码：`https://huggingface.co/Lei285714/mod-arith-k2`
  （commit `f11f10e7ed5295417330a6c6af4693d0dbef33e4`，2026-07-26 拉取）
- 上游：`https://github.com/Robby955/neural-horner`（Robby Sneiderman，MIT）
- 官方规则与评测：`third_party/modular-arithmetic-challenge/rules/`
- 本项目前序调研：`调研报告_神经Horner线_2026-07-25.md`
- 单步耗时与参数量：本文件写作时在 macOS / MPS 上实测，**非评测设备**，
  只用于相对判断

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-07-26 | v1：建立本文件；加入 mod-arith-k2 拆解、四项诊断、六条建议、research_msg v3 清单 |
