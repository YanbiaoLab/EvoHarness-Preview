# 种群可见性:让 agent 能看见选择器没选中的候选

> 状态:设计稿,未实现。默认关的 `ProposalConfig` 开关,先在 ETP 上量一轮再决定
> 要不要默认开。与 `inspect_candidate` 互补而非替代 —— 那个工具能力已经是全的,
> 缺的是索引。

## 1. 问题

**agent 对种群的全部认知,是选择器的副产品。**

提案提示词里关于种群的内容只有两类:父代(全量渲染),和灵感(`num_archive_inspirations`
+ `num_top_k_inspirations` 各取 N)。而灵感的挑法是:

```python
# core/selection.py
best = max(archive_scope, key=lambda c: c.fitness)      # 归档灵感
ranked = sorted(scope, key=lambda c: -c.fitness)        # top-k 灵感
```

纯按 `fitness` 排名。于是**任何被闸门清零的候选,无论信息量多大,都不可能出现在
提示词里**。

`inspect_candidate` 的实现是 `self.store.get(candidate_id)`,没有任何范围限制 ——
它读得到种群里的任何一个候选。但它的工具描述明写:

> candidate_id must be copied verbatim from an 'id=...' shown in the
> reference-program list of your prompt … If your prompt listed no reference
> programs, there is nothing to inspect.

**能力是全的,索引是缺的。**

## 2. 这不是假想的失败

ETP Stage 2 第九轮,种群 30 个候选,`archive_size=40` —— 归档从没满过,所有候选
一直在归档里。其中 6 个是字节回收尝试:

```
    467,958 字节   解出 380   fitness 0.0000   raw_fitness 0.8422
    483,028 字节   解出 380   fitness 0.0000   raw_fitness 0.8422
    493,257 字节   解出 380   fitness 0.0000   raw_fitness 0.8422
    499,921 字节   解出 383   fitness 0.8481   ← 冠军
```

同样解出 380 行,可以花 467,958 字节,也可以花 493,257。冠军多的 3 行花了 32,042
字节。全局零回归把前几个判 0 分,于是**连续八轮里,提案 agent 一次都没见过它们**,
每一轮都从贴着字节天花板的冠军重新出发去"腾空间",八次全失败。

诊断的第一版归咎于归档策略,是错的(归档没满)。真凶是灵感选择的排序键。

## 3. 现状盘点(已经有的,不用重做)

```
_render_candidate()   父代:change_title/summary + report.render_for_prompt()
                      (fitness + 全部 visible_metrics 逐条 + notes) + 代码
MutationContext       parent / archive_inspirations / top_k_inspirations /
                      operator / generation / inspiration_notes
PromptContributor     brief · directives · experience · feedback · island_brief ·
                      human_directive · resource_ledger
agent 工具            inspect_candidate(读任意候选源码)· inspect_eval(父代轨迹)
```

`ResourceLedgerContributor` 已经是本问题在**单个成本轴**上的特例:它绕开
`ctx.archive_inspirations`,直接读 store 算分桶精英。通用版应当把这个模式抽出来,
而不是让每个域各写一个 contributor。

## 4. 提案:一行清单 + 一个工具

不每次渲染大表。固定成本压到一行,内容按需拉 —— 这是仓库已有的做法,
`inspect_candidate.py` 的注释原话:"progressive disclosure instead: the prompt
carries an inventory, and the agent expands only what it decides to read"。参考程序
早就这么做了,**只是清单本身仍然由选择器给**。

```
提示词固定成本    一行:「种群有 N 个已评估候选,list_candidates 可查」
list_candidates   agent 自查,返回 id + 指标摘要
inspect_candidate 现成的,用查到的 id 展开源码
```

工具签名草案:

```
list_candidates(order_by, limit, island=null, spread_by=null)
  order_by    必填,无默认值(见第 5 节)
  spread_by   可选:沿某个 visible_metrics 键分桶,每桶取一个,而不是取前 N
  返回        [{id, generation, operator, change_title, <声明的指标子集>}]
```

## 5. 一条硬约束

**`order_by` 不能有默认值,更不能默认 `fitness`。**

给它一个默认值就等于把同一个 bug 换个地方再犯一遍:调用方会继承一个自己没做过的
假设,而"按什么排"恰恰是这里唯一重要的决定。强制显式指定,让它成为一次决策而不是
一次继承。

`spread_by` 是给"我不知道该按什么排"准备的正解:沿声明的特征轴分散取样,拿到的是
种群的**跨度**而不是**头部**。

## 6. 代价

这是给所有域的 agent 加工具,不是局部改动:

- 多一段系统提示词(工具定义)
- 多一次工具调用往返
- agent 可能把提案预算花在翻种群而不是改代码上

所以做成 `ProposalConfig` 开关,默认关。先在 ETP 上开一轮,对比"agent 是否真的
用了它、用了之后提案质量有没有变化",再决定默认值。

## 7. 验收

- [ ] 开关默认关时,提示词与工具集与现在**逐字节相同**
- [ ] `order_by` 缺省时报错,不静默回退到 fitness
- [ ] 一个测试:种群里放一个 `fitness=0` 但 `raw_fitness` 很高的候选,
      断言选择器给的灵感里没有它、而 `list_candidates` 能返回它
- [ ] `ResourceLedgerContributor` 的 store 读取逻辑抽成共用件,不再各写一份
