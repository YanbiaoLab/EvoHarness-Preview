# 待你过目:给组织者的披露草稿(**未发送**)

我不替你对外发言。下面是草稿、发送前该确认的几件事、以及不发的后果。

## 该确认的

1. **发不发。** 我的建议是发,理由在下面"为什么建议发"。
2. **发到哪。** 规则指向 SAIR Zulip(<https://zulip.sair.foundation/>),那里有
   "community feedback shapes the rules"的明确邀请。也可以直接邮件组织者。
3. **署名。** 草稿里没写队名/个人身份,你补。
4. **要不要附我们的数字。** 草稿里附了(788 秒那组)。附上更有说服力,
   但也等于告诉所有人我们的模型到了 tier 10。**如果你不想暴露进度,
   可以把第二段的具体数字换成"a tier that takes several minutes"。**

## 为什么建议发

- 截止是 2026-08-12,还有两周。现在报,不管裁决如何都来得及应对;
  提交后被发现,整年工作归零。
- 规则自己写着预算 "May be tuned before the official runs",且计划中的
  行为信号明确列了 **"latency-vs-operand-size profile"** —— 一个 788 秒的
  单层是他们最容易发现的形状。
- 不对称:报告的代价是一天;不报的风险是反作弊条款下的取消资格。
- 这个问题影响**每一个**参赛者的分数,不只是我们的。他们会想知道。

## 不发的后果(诚实列出)

- 我们的 h90=10 在**现行 published harness 下是成立的**,机制上不违规。
- 但它依赖的性质随时可能在正式评测前被调整,而我们不会提前得到通知。
- 若被认定为规避意图,反作弊条款的措辞是"disqualified",没有中间档。

---

## 草稿正文

> **Subject: Inference timer is checked per batch — a tier can overrun the
> whole budget and still be scored**
>
> While calibrating a local copy of the official pipeline we ran into a
> behaviour in `evaluation/pipeline.py::run_inference` that we think is worth
> flagging, because it affects how the 5-minute budget applies to any model
> that batches.
>
> The timer is cooperative and checked in two places: before each tier, and
> before each batch inside a tier. A model declares its own batch width via
> `max_batch_size()`, and the rules recommend batching ("Use
> `predict_digits_batch()` and GPU batching to amortize"). With 100 problems
> per tier, any model declaring `max_batch_size() >= 100` runs each tier as a
> single batch — so the inner check fires exactly once per tier, at its start,
> and never during it.
>
> The consequence is that the effective rule becomes "each tier must *start*
> within the budget" rather than "the run must *finish* within it". A tier
> that begins at, say, 190 s and then runs for 13 minutes completes, keeps
> `tier_complete = True`, and is scored in full.
>
> We hit this concretely. Our current model reaches tier 10 with the
> cumulative clock at roughly 190 s, well inside 300 s. Tier 10 then takes
> 788 s as one batch. Every tier is scored, the leaderboard keys come out at
> `highest_tier_above_90 = 10`, and the total inference wall-clock is about
> 978 s — 3.3x the stated budget.
>
> We have not submitted anything relying on this and we are not asking for a
> ruling in our favour. We would rather know the intended reading before the
> deadline than after. Some options we can see, in case they are useful:
>
> - check the elapsed time inside the batch loop as well as before it, so an
>   overrun is caught mid-tier;
> - cap the per-batch size used by the pipeline independently of the model's
>   declared width, so every tier has several checkpoints;
> - or state explicitly that "starts within the budget" is the intended
>   semantics, in which case contestants should know to optimise for it.
>
> Happy to share the exact timings or a reproduction if that helps.

---

## 无论裁决如何,不变的技术目标

诚实地把 tiers 1–10 塞进 300 秒。现在第一次知道确切要多少:

```
tier 10 单层 788 秒 / 100 题
tiers 0–9 累计  ≈ 190 秒
总计           ≈ 978 秒     需要约 3.3 倍加速
```

r9 的 fitness 已经把这条做成连续梯度(见 `_time_factor`),所以进化会一直
朝这个方向推,不依赖上面的裁决。杠杆在简报 §8.3:每步成本的第二项
(每步内部的传播深度与功复杂度),至今无人攻过。
