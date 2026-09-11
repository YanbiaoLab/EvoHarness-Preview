# 手动跑 5 道 IMO-Bench(证明模式,2026-09-03 起)

PBBasic002 于 2026-09-01 晚经证明模式全链证毕(open → sketch → attack ×2 → assemble,
公理集 `[propext, Classical.choice, Quot.sound]`)。那是 n=1。这一轮把它变成一个比率:
**5 道手动、一道一个会话、不做批量驱动器。**

题目全部取自 LEAP 已公布解答的那 42 道 —— 失败时归因才干净(题目已知可解,
所以失败一定是我们的管线)。这不是攻克 IMO,是证明模式的体检。

**这份文件刻意不放在任何一个会话工作区里。** 下面的难度标签、题目出处、
LEAP 参照长度都是关于题目难度的提示,会话工作区里只该有题面本身。

## 阶梯

选题标准不是「最容易」,是**每道换一个形状**——再做 5 道 PBBasic002 那样的不等式
测不出新东西。5 道覆盖 4 个类别、3 个难度,LEAP 参照证明都很短,失败便宜且刺眼。

⚠️ **「参照短」这个信号,对「测分解」这个目的是反的(2026-09-03 记)。** 参照短
往往意味着求解器一次就能做完,于是图只剩包装作用,PB-Basic-001 已经中过一次。
这 5 道仍然照跑——改成先直接 attack 之后,一次尝试就能把「这是道 solver 题」
问出来,便宜且是它自己的基线。但**下一批选题要往参照长的那一端挪**:非几何的
候选有 PB-Advanced-012(4033 行)、PB-Basic-012(2091)、PB-Basic-006(1523)、
PB-Advanced-020(1188)、PB-Basic-011(1184)、PB-Advanced-008(1129)、
PB-Advanced-006(1041)。几何那两道长的先不碰——Mathlib 几何的工具短板会和
「分解有没有用」混成两个变量。参照长度混着 LEAP 自己的风格,只是最便宜的代理量;
真正的判据仍是**一次直接 attack 做不动**。

| # | 题号 | 工作区 | 难度 / 类别 | 出处 | LEAP 参照 | 挑它测什么 |
|---|---|---|---|---|---|---|
| 1 | PB-Basic-008 | `~/proofs/pbbasic008` | pre-IMO / 代数 | (Modified) All-Russia MO 2002 | 78 行 | 对照组。与已证的 PBBasic002 同形(带根号的不等式) |
| 2 | PB-Basic-017 | `~/proofs/pbbasic017` | pre-IMO / 数论 | folklore | 104 行 | 换域。`nlinarith` 那条路不通,得走因式分解 |
| 3 | PB-Basic-010 | `~/proofs/pbbasic010` | IMO-easy / 组合 | (Modified) Tournament of Towns 2022 Senior O-level P3 | 118 行 | 组合是 LEAP 未公布的两个残差类别之一,先探底 |
| 4 | PB-Basic-001 | `~/proofs/pbbasic001` | IMO-easy / 代数 | (Modified) IMO 2019 P1 | 138 行 | 换**形状**:结论是集合等式,分解要拆两个包含方向 |
| 5 | PB-Basic-024 | `~/proofs/pbbasic024` | IMO-medium / 数论 | (Modified) Serbia MO 2008 P1 | 79 行 | 难度标签跳一级但参照只有 79 行,测成败跟的是标签还是长度 |
| 6 | PB-Advanced-006 | `~/proofs/pbadvanced006` | **IMO-hard** / 代数 | Novel Problem | **1041 行** | 换轴。前五道按「参照短」选,而那个信号对「测分解」是反的 |

两处口径要记在前面,写结论时不能含糊:

- **PB-Basic-010 的官方题面在结论两行中间夹着一句注释**,说明这里用的是自然数减法
  (`a > b` 时 `b - a = 0`)。工作区里的 `statement.txt` 按删掉处理——留着等于替模型
  想掉一步建模,而这一步正是组合题在 Lean 里最容易翻车的地方。要严格照抄官方题面
  就加回去,但**两种口径不能混着比**。
- **PB-Basic-001 与 PB-Basic-024 的答案写在结论里**(`{0} ∪ {2x+c}`、`{(0,0,0)}`)。
  这是「求所有满足…」形式化后的固有形态,LEAP 面对同一份题面,所以不算我们泄题
  ——但也**不能说成「从零推出了答案」**。

## 每道题的流程

一道题一个会话,cwd 设成上表那一列。图落在该目录的 `.evo/graph.db`,跨会话存活。

```
proof_open                    (preamble + statement 从 statement.txt 粘)
   ↓
模型读题、几句话说清形状与难点所在,给出推荐 —— 然后停下来等你选
   ├ L1  proof_attack level=L1   一次模型调用,不带 Lean 循环
   ├ L2  proof_attack level=L2   agent 会话,反复编辑再编译,分钟级、花钱
   └ 分解 proof_sketch            一次 Lean 编译(30 秒起),不花**求解器**预算
   ↓
证出来 → proof_assemble;task-failed → 回到岔口重选
```

三档回答的是三个不同问题:**模型本来就会**(L1)/ **求解器要靠 Lean 循环**(L2)/
**整块够不着,得拆**(分解)。用 L2 去问第一个问题,是拿最贵的工具问最便宜的问题。

**模型在花钱前的那段判断是一个可证伪的预测**,不是提案说明。它说「这道题一次就能出」
然后 L1 失败,那条记录本身就是关于模型自我认知的数据——所以要它写短,并且写下来。

⚠️ **「分解不花钱」是错的说法(2026-09-03 更正)。** `proof_sketch` 里没有任何模型调用
——它拿工具参数里那份 proposal 去让 Lean 编译,`solver_needed=False`。所以它不花的是
**求解器预算**,而那份 proposal(引理签名加 `parent_body`)是**会话里的模型写的**,
那些 token 是真花的钱,只是**账记在 dsh 会话那一侧,板子上的 `spent` 永远看不到**。

于是板子上的 `0.0` 有两个互不相干的来源,读的时候要分清:①这一步没请求求解器
(sketch 就是这种);②请求了但没配价(`cost_priced: false`,PB-Basic-001 那次的 0
是这种)。两个都显示成 0,而**「没花」和「没记」是完全不同的两件事**。

实际后果:岔口上「分解最便宜」并不自动成立。给一道硬题想出一组站得住的引理,
可能比一次 L1 attack 贵——只是贵在一个没人看的账本上。

**读题、给判断、问人选档**(2026-09-03 改)。原来的顺序是「便宜的先做」——
先 sketch 验路线再花钱 attack。PB-Basic-001 就是那么跑的,结果分解只产出一条
和原命题等价的引理,全部内容由一次 attack 完成:**图看起来转了一圈,实际没参与**。

默认倾向仍然是「先整块试,打不动再拆」,理由是**一次没成功的整体尝试是这道题的基线**,
也是后面那个 `certified` 唯一能说明问题的前提。但它是**推荐,不是强制**:模型可以
建议直接分解,而 persona 要求它在这么建议时说清代价——没人试过整块,事后板子上
就看不出分解到底需不需要。

只有 `task-failed`(试过了做不到)才触发分解。`infra-failed` / `budget-exhausted` /
`interrupted` **不说明目标难**,遇到这三个是去修或重试,不是绕着它做分解。

**开跑前的岔口由人拍板**,会话必须停下来问、拿到答复才花钱,沉默不算同意。
代价要记在账上:**这一步之后,「分解有没有用」就掺进了人的判断**。所以下面记录表里
那一列填的是「谁选的、理由是什么」——人选了分解而它奏效了,是一条比系统自选更弱的证据,
写结论时不能混作同一件事。

- `proof_open` / `proof_status` 不花钱;`proof_sketch` 花一次 Lean 编译;
  `proof_attack` 花真钱、以分钟计。
- **一次带 Mathlib 的编译约 28 秒起**,编译类调用上限 360 秒。工具返回慢是正常的。
  超时消息会说「被切断的是这次调用,它启动的东西可能还在后台跑」——那不是 Lean
  给出的数学判定,别当成反例,也别因此重交同一份分解。
- **`proved` ≠ `certified`。** `proved` 只说明路线闭合;`certified` 才是 `proof_assemble`
  重编译整篇、报出公理集之后的结论。看板子认后面那个字段。
- `proof_attack` 经这条通道只跑成过一次(PBBasic002,两条引理各一行 `nlinarith`)。
  它是这 5 道里最可能出新缺陷的地方。

## 记录(每道跑完填,不填就只是 5 段各自漂亮的会话)

| 题号 | 模型开跑前的判断 | 选了哪档 / 谁选的 | 判断对了吗 | 停在哪一步 | Lean 拒绝次数 / 理由类别 | 墙钟 | 污染检查 |
|---|---|---|---|---|---|---|---|
| PB-Basic-008 | 未记(旧流程) | 分解(无岔口) | — | 组装失败 → **已修,待重新认证** | 见下 | — | 待查 |
| PB-Basic-017 | | | | | | | |
| PB-Basic-010 | | | | | | | |
| PB-Basic-001 | 未记(旧流程) | 分解(无岔口) | — | assemble 认证 | 1 次 / sketch 编译不过:化简后类型不匹配 | 39 分钟 | 干净 |
| PB-Basic-024 | | | | | | | |
| PB-Advanced-006 | | | | | | | |

「停在哪一步」取值:`open` / `sketch 通过` / `attack 通过` / `assemble 认证`。

**PB-Basic-008 是这条链上第一张真正的两层图**:`PBBasic008` → `PBBasic008_core` →
{`_core_low`, `_core_high`},三条引理全部证出。**数学那一半成功了,塌的是管道**——
组装报 `invalid 'import' command`,拆出来是三个缺陷摞在同一处接缝上(preamble 每层
各发一次、子树的引理被声明两次、父层引用了一个谁也没定义的 `_assembled` 名字),
外加组装器**永远取最早那条路线**、会话另提的扁平路线用不上。四处都已修,见
[todo/dsh_integration.md](../../todo/dsh_integration.md) 的 DH-3.10。**根目标还没
重新认证**——修复只保证拼得出来,还没再跑一次 Lean。

重新认证时建议走**原来那条嵌套路线** `dec_8621dcd08720`(而不是扁平的
`dec_49bc2aaf6e4a`):两条现在都拼得出干净文件,但嵌套那条才是这道题真正的价值
所在——**换扁平那条等于绕开嵌套出证明,而嵌套正是这张图存在的理由**。

```bash
.venv/bin/python -m evoharness.proof.cli --work ~/proofs/pbbasic008/.evo \
  --lean-project tasks/lean_env assemble --goal goal_8e3ed83e765c \
  --route dec_8621dcd08720
```

**PB-Basic-001 的结论只能写成「求解器一次做完,分解是空操作」,不能写成「框架证明了它」。**
成品 `PBBasic001.lean` 71 行(LEAP 参照 138 行),`certified.ok: true`,公理集
`[Classical.choice, Quot.sound, propext]` 干净。但根定理的证明只有 `ext f` 加一句
`simpa … using PBBasic001_solution_iff f`,而那条唯一的引理就是同一命题的逐点形式
——56 行数学全在引理里,由一次 attack 产出。这一道验的是 attack 与 assemble 的管道
连通,不是分解。它也是「先直接打」这条顺序的由来。

**第二列最值钱。** PBBasic002 那次的拒绝理由是「实数幂消去的 elaboration 问题」,
读了理由改用 `nlinarith` 就过了。5 道的拒绝理由若集中在同一类,那是管线的下一个
改进点;若每道都不一样,说明瓶颈还在模型而不在管线。

## 污染检查(每道跑完做一次)

LEAP 的 42 份成品与带 `Solution` 列的 `lean_proof_bench.csv` 都在本机可读路径上,
而会话沙箱是 `workspace-write`——**只 confine 写,不 confine 读**。挪目录挡不住
(`superhuman` 是 git 仓库,解答被 tracked,`git show HEAD:...` 原样吐出来)。
所以这一层是**事后检测,不是预防**:它决定这个数据点该不该扔。

```bash
zstdcat ~/.dsh/sessions/--Users-zhangkang-proofs-pbbasic008--/session-*/session.jsonl.zstd \
  | grep -o -E "LEAN-IMO-Bench|_solution\.lean|lean_proof_bench|Grading guidelines|git show HEAD" \
  | sort | uniq -c
```

**别 grep 题号。** `PBBasic008` 在一次正常会话里出现几十次,查出来全是噪声。
要查的是**取答案这个动作的痕迹**,不是题目本身。

命中即作废该道数据点,换一道或换个环境重跑。5 道的样本量下扔掉一道承受得起,
「不知道有没有被污染」承受不起。
