# 交互式控制台设计(evoweb)

> 状态:设计稿,P2 之后的可选项。与 viz_design.md 的静态报告互补而非替代:
> 静态 HTML 管归档/博客/面试,evoweb 管运行中监控 + HITL 点按操作。
> 结论:**前后端分离在代码层,合并在部署层**("分离但不分家")。

## 1. 架构

```
evoweb/
├── server/            # FastAPI,薄读层(~300 行)
│   ├── api.py         # 只读 API:复用 evoviz.extract 的 run_summary 契约
│   ├── control.py     # 写路径:仅写 control/directives.yaml 与 decision.json
│   └── events.py      # SSE:轮询 run.db mtime → 推送"新一代完成"
└── ui/                # Vite + React + TS,构建产物随 Python 包分发
```

三条决策:
1. **引擎不知道 web 存在**:SearchLoop 照旧读控制文件、写 SQLite;web 挂了实验照跑。
   读路径只读打开 run.db(mode=ro);写路径只落 directives.yaml(HITL 的留痕/TTL
   纪律全部继承,见 hitl_design.md)。
2. **单进程部署**:`evoweb serve results/ --port 7860`,FastAPI 直接 serve SPA
   构建产物;开发时 Vite proxy /api。零 CORS、零 nginx。
3. **SSE 不用 WebSocket**:数据单向(引擎→UI)、每代一事件;HITL 写走普通 POST
   (本来就是"下一代生效"的异步语义)。

## 2. API 草案

```
GET  /api/runs                          # 列表 + 状态(运行中/完成/熔断)
GET  /api/runs/{id}/summary             # = evoviz run_summary.json
GET  /api/runs/{id}/lineage             # 谱系 DAG 节点/边
GET  /api/runs/{id}/candidates/{cid}    # code、diff vs 父代、EvalReport、feedback
GET  /api/runs/{id}/prompts/{gen}       # 某代 system/user dump(调试注入)
GET  /api/runs/{id}/events              # SSE
POST /api/runs/{id}/directives          # 追加/更新指令 → directives.yaml
POST /api/runs/{id}/decision            # 审阅闸门 continue/stop
```

## 3. 技术栈

React 18 + Vite + TypeScript(API 类型由 pydantic schema 经 openapi-typescript
生成,契约不漂移);Tailwind + shadcn/ui;ECharts(曲线/热图/错误流);
React Flow(谱系 DAG 交互);Monaco diff editor(候选 vs 父代)。

## 4. 信息架构

左栏 run 列表(状态点)+ 多选进对比视图 → 主区 tabs:
- **总览**:实时 fitness 曲线(事件标注)+ 事件流 + 预算进度条
- **谱系**:React Flow DAG(色=operator,径=fitness),点节点开右侧抽屉
- **行为**:C2 行为覆盖热图 + C1 错误类别流
- **prompt**:任一代的完整 system/user dump
- **控制**:HITL 指令编辑器 + 审阅闸门
右侧候选抽屉 = HITL 落点:diff、失败类别、三个按钮(从此分支 / 否决谱系 /
冻结区域)→ 直接 POST 成 directive,对应 hitl_design 的三级指令。

## 5. 设计语言("好看"的纪律)

- 中性底色 + 单一强调色(紫);暗色模式一等公民(shadcn 令牌)
- 算子颜色全站唯一映射:紫 rewrite / 青 revise / 珊瑚 repair / 灰 seed
  (与静态报告一致);fitness 用同一连续色标
- 等宽字体仅用于 id/代码/数字;信息密度向 Linear 看齐,细节进抽屉不弹窗
- StagedGrader 早退候选空心点、veto/迁移候选有视觉标记(与 viz 纪律一致)

## 6. 优先级(诚实版)

静态 report.html 覆盖 80% 需求;SPA 真实增量 = 实时监控 + HITL 点按。
顺序:evoviz extract(必做)→ FastAPI 读 API + SSE(0.5 天)→ SPA 总览+谱系
(2–3 天)→ 控制 tab 随 HITL 实装。时间紧则止步于"FastAPI + 单页 ECharts"。
