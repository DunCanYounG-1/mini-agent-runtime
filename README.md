# mini-agent-runtime

从零实现的最小可用 Agent Runtime。**运行时只用 Python 标准库**，没有 langgraph / openhands / langchain，
也没有用厂商的 function calling —— 工具调用协议、解析、循环、session、context 压缩、trace 全部自己写。

- 真实 LLM：任意 OpenAI-compatible 接口（默认 DeepSeek，换 GLM / 通义 / Kimi / 豆包只改 `.env`）
- 4 个工具：`calculator`、`search`（mock）、`weather`（mock）、`todo`
- 测试：72 个离线用例（假 LLM，确定性） + 8 个真实 LLM 端到端用例（DeepSeek 实测 8/8 通过，见 `docs/demo_transcript.md`）

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                    # 只有 pytest，运行时零依赖
cp .env.example .env                                   # 填 LLM_API_KEY

# 对话：同一个用户开两个窗口（两个终端）
python -m mini_agent --user A --session w1 -v          # -v 打印思考与工具调用
python -m mini_agent --user A --session w2 -v

# 单条消息模式（随时接着某个窗口聊）
python -m mini_agent --user A --session w1 -m "那上海呢？"

# 测试
pytest                    # 离线用例
pytest -m live -v         # 真实 LLM 用例（需要 .env）
python scripts/demo.py    # 跑题目场景，生成 docs/demo_transcript.md
```

会话内命令：`/sessions` 列出会话 · `/switch <id>` 切换 · `/new` 新建 · `/todo` 看待办 · `/summary` 看压缩摘要 · `/trace` 看上一轮执行轨迹。

## 需求对照

| 题目要求 | 实现位置 | 测试 |
|---|---|---|
| 基本循环：输入 → 决策 → 调工具 → 继续/返回 | `agent.py` `Agent._run` | `test_agent_loop.py` |
| 工具注册（名称/描述/参数 Schema），LLM 自主决策 | `tools/base.py` `ToolRegistry` | `test_tools.py` |
| 解析 LLM 输出：思考 / 工具调用 / 最终答案 | `protocol.py` `parse_output` | `test_parser.py`（14 种格式） |
| 多窗口独立 session，可随时续聊 | `session.py` | `test_session.py` |
| 最大轮次限制 | `max_steps`（单轮）+ `max_history_turns`（历史） | `test_max_steps_*`、`test_max_history_turns_*` |
| 记住之前的状态、纯对话追问、带工具追问 | `context.py` `render_history` | `test_context.py` |
| context 过长的基础压缩 | `context.py` `maybe_compress` | `test_compression_*` |
| 基本异常处理 | 见下文「异常处理」 | `test_agent_loop.py` 后半 |
| trace / 执行日志 | `trace.py` → `data/traces/*.jsonl` | `test_trace_log_written` |
| 真实 LLM | `llm.py` `OpenAICompatLLM` | `test_live_llm.py`、`docs/demo_transcript.md` |

## 系统设计

```
           ┌──────────── cli.py / 任意调用方 ────────────┐
           │       agent.chat(user_id, session_id, text)   │
           └───────────────────────┬──────────────────────┘
                                   ▼
 SessionManager ──加锁──▶ Agent._run ─────────────────────────────── Tracer (jsonl)
 (每个 user×session      │ 1. ContextManager.maybe_compress()   压缩旧轮次
  一个 JSON 文件)         │ 2. ContextManager.recall()           从归档召回
                         │ 3. loop ≤ max_steps:
                         │      build messages → LLM.chat()
                         │      parse_output() → thought / tool_calls / final
                         │      tool_calls → ToolRegistry.execute() → observation → 继续
                         │      final      → 跳出
                         │      格式错误    → 把错误回传给模型重试
                         │ 4. 超过 max_steps → 禁用工具，逼模型基于已有信息作答
                         │ 5. 写入 Turn，落盘
```

### 1. 输出协议与解析

system prompt 要求模型这样输出：

```
<thought>思考</thought>
<tool_call>{"name": "weather", "arguments": {"city": "北京"}}</tool_call>   ← 可多个
或
<thought>思考</thought>
<final_answer>回答</final_answer>
```

**为什么不用厂商的 function calling**：一是题目要求自己写解析；二是与厂商解耦，任何能输出文本的模型都能跑；
三是思考、调用、答案三段边界清楚，能逐一容错。调用时传 `stop=["<observation"]`，防止模型自己编工具结果。

解析器（`protocol.py`）处理了真实模型常见的毛病：标签没闭合、JSON 外包了 \`\`\`json、`arguments` 写成字符串、
字段名写成 `tool/args`、JSON 尾逗号、字符串里含 `{}`、推理模型的 `<think>` 和 `reasoning_content`、
自己编造 `<observation>`（丢弃）、同时给 tool_call 和 final_answer（以 tool_call 为准——没看到结果就下结论是错的）、
完全没写标签（当作直接回复）。修不了的才算格式错误，错误信息回传给模型让它重写，连续 2 次失败就降级回复。

### 2. 工具注册

```python
@registry.tool(name="echo", description="回显", parameters={"type": "object", "properties": {...}})
def echo(args, ctx): ...
```

每个工具 = 名称 + 描述 + JSON Schema + 函数。Registry 负责：把 Schema 渲染进 system prompt；执行前按 Schema 校验
（必填、类型、enum、多余字段）；在线程池里执行并设超时；所有异常包装成 `ToolResult(ok=False)`。
校验失败、工具不存在、超时、工具抛异常，都**作为 observation 回传给模型**，让它自己修正参数——而不是打断循环。

`todo` 是有状态工具，通过 `ToolContext.session` 读写 `session.state["todos"]`，所以天然按 session 隔离。

### 3. Session

- 一个 session 由 `(user_id, session_id)` 唯一确定，存成 `data/sessions/{user}/{session}.json`，原子写入（先写临时文件再 `os.replace`）。
- 窗口 1 和窗口 2 是两个文件，历史、待办、摘要完全独立；进程重启后自动从磁盘恢复，可以接着聊。
- 同一 session 同时只跑一个请求（`SessionBusyError`），不同 session 可以并发（有测试验证并行耗时）。
- id 有白名单校验，防止 `../` 路径穿越；文件损坏时备份成 `.corrupt.*` 而不是静默覆盖。

### 4. Context：每次调 LLM 时塞了什么

| 位置 | 内容 | 理由 |
|---|---|---|
| system | 角色、输出协议、决策规则、工具 Schema | 固定部分 |
| system | 当前时间 | 让“明天”“后天”能解析 |
| system | **待办快照**（每轮开始时从 `session.state` 现读，本轮内冻结） | 状态的“唯一事实来源”，历史里的工具结果可能已过期（比如后来删了）。本轮内不刷新：否则模型会在快照里看到自己刚加的条目，误以为“原本就有”（DeepSeek 实测踩到，见 `docs/AI_PROMPTS.md`） |
| system | 早前对话摘要（压缩产物） | 有压缩时才出现 |
| system | 召回的更早片段 | 命中时才出现 |
| 历史 | 每轮的用户输入、**工具调用 + 结果**、最终答案 | 带工具的追问（“那上海呢”）要靠上一轮的调用来补全意图和参数 |
| 历史 | ~~历史轮次的思考过程~~（不放） | 只对当轮推理有用；留着浪费 token，还会诱导模型复读旧思路 |
| 历史 | 较早轮次的工具结果截短到 300 字，最近 2 轮保留 1500 字 | 越近越可能被追问 |
| 历史 | ~~失败的轮次~~（不放，但保留在 session 里审计） | 避免“抱歉服务不可用”这种回复污染上下文 |
| 当轮 | 用户输入 + 本轮的思考/调用/结果 | 当轮思考保留，保证多步推理连贯 |

历史轮次也用同样的 `<tool_call>` / `<observation>` / `<final_answer>` 格式回放，相当于给模型持续的格式示范，格式错误率明显更低。

### 5. 压缩

- **触发**：每轮开始前估算 token（中文 1 字≈1 token，其他 4 字符≈1 token），超过 `max_context_tokens × 0.75`；或活跃历史超过 `max_history_turns`（默认 20）轮。
- **做法**：保留最近 `keep_recent_turns`（默认 4）轮原文，更早的轮次交给 LLM 和旧摘要**合并成滚动摘要**（要求保留用户身份/偏好、已确认结论、工具查到的关键数据、未完成事项），原文移入 `archive`。
- **兜底**：摘要的 LLM 调用失败时，退化成规则摘要（每轮一行“用户问了什么 → 答了什么”），不阻塞对话。
- **当轮兜底**：一轮里连续调很多次工具时，把较早的 observation 截短（`shrink_scratch`）。
- 所有压缩都记录 trace：压缩了几轮、前后 token 数、用的哪种方法。

### 6. 异常处理

| 场景 | 处理 |
|---|---|
| LLM 网络错误 / 429 / 5xx / 空回复 | 指数退避重试 3 次；仍失败返回友好提示，该轮标记 `llm_error` 并排除出 context |
| LLM 401 / 400 | 不重试，直接报错（重试没意义） |
| 输出格式错误 | 回传错误让模型重写；连续 2 次降级回复 |
| 工具不存在 / 参数不合 Schema / 工具异常 / 超时 | 包装成 error observation 回传，模型可自行修正 |
| 模型原地打转（同参数重复调用） | 直接复用结果并提示“不要重复调用” |
| 超过 `max_steps` | 追加“禁止再用工具，基于已有信息作答”再问一次；还不行就固定话术收尾 |
| 摘要失败 | 规则摘要兜底 |
| session 忙 / 非法 id / 文件损坏 | `SessionBusyError` / `ValueError` / 备份损坏文件 |

### 7. Trace

每个事件一行 JSON，写入 `data/traces/{user}/{session}.jsonl`，同一次请求共享 `run_id`：

```
user_input → [context_compressed] → [memory_recalled] → llm_request(消息数, 估算 token)
→ llm_response(耗时, usage, 思考, 原始输出, 解析告警) → tool_call → tool_result(耗时, ok, 结果) → … → final
```

CLI 里 `-v` 实时打印，`/trace` 回看上一轮。

## Memory：召回时机与放置方式

这个项目里的 memory 分四层，原则是“**状态靠结构化存储实时注入，对话靠窗口 + 摘要，旧细节靠检索**”：

| 层 | 存什么 | 什么时候写 | 什么时候召回 | 放在哪 |
|---|---|---|---|---|
| 工作记忆 | 本轮的思考、工具调用、结果 | loop 每一步 | 本轮每次调 LLM | 当轮消息末尾 |
| 短期记忆 | 最近 N 轮原文（含工具调用与结果，不含思考） | 每轮结束 | 每轮都带 | 历史消息，按真实顺序 |
| 结构化状态 | 待办列表 `session.state` | 工具执行时 | 每轮都带，现读现渲染 | system prompt「当前会话状态」 |
| 摘要记忆 | 被压缩轮次的滚动摘要 | 触发压缩时 | 有就每轮都带 | system prompt「早前对话摘要」 |
| 归档检索 | 被压缩轮次的原文 | 触发压缩时 | 每次用户输入后、调 LLM 前，用输入去检索；命中（重叠度 ≥ 0.2）才带 top-2 | system prompt「可能相关的更早片段」 |

几个取舍：

- **为什么召回结果放 system 而不是插进历史**：插进历史会打乱对话的时间顺序，模型会以为这是刚说过的话；放在 system 里并标注“仅供参考、来自更早的对话”，模型能区分。
- **为什么待办每轮现读而不是靠历史**：历史里“添加成功”的 observation 可能已经过期（后来被完成或删除了），状态只认一个来源。
- **为什么只在有归档时才检索**：没被压缩的内容本来就在 context 里，再检索是重复。
- 检索用的是字符二元组重叠（无依赖、对中文够用）；换成向量检索只需替换 `ContextManager.recall`。

## 测试

```
tests/
  test_parser.py      14 种输出格式：标准、多调用、代码块、别名字段、未闭合、编造 observation、纯文本……
  test_tools.py       注册、Schema 校验、超时、异常、计算器安全性（拒绝 __import__）、天气、待办增删改查
  test_agent_loop.py  直接回复、单/多步工具、并行调用、max_steps、工具报错自修正、格式错误、LLM 故障、重复调用、trace
  test_session.py     双窗口隔离（题目场景）、跨用户隔离、重启恢复、忙状态、并发、路径穿越、文件损坏
  test_context.py     纯对话追问、带工具追问、旧思考不入历史、待办快照、截断、压缩触发/降级、召回
  test_live_llm.py    真实 LLM：题目里的双窗口场景、追问、计算、压缩后仍记得早期事实
```

离线测试用 `ScriptedLLM`：按脚本回复，同时**记录每次收到的完整 messages**，所以能精确断言“窗口 2 的内容没出现在窗口 1 的 context 里”“上一轮的工具结果出现在追问的 context 里”这类 context 层面的行为。

## 目录

```
mini_agent/
  agent.py      主循环            protocol.py  输出协议与解析
  context.py    组装/压缩/召回     session.py   session 与持久化
  llm.py        LLM 客户端         trace.py     执行日志
  config.py     配置               cli.py       命令行
  tools/        base.py（注册机制）calculator / search / weather / todo
tests/          离线 + live 测试
scripts/demo.py 真实场景演示
docs/           AI_PROMPTS.md（AI 使用与问题记录）、demo_transcript.md
```

## 已知限制

- token 是估算值，没接 tokenizer；对阈值判断够用。
- 不支持流式输出；工具是同步执行（有超时），没有异步工具与完成通知。
- 同一 session 的互斥锁是进程内的；多进程同时写同一个 session 需要换文件锁或数据库。
- search / weather 是 mock，接口形状和真实服务一致，替换函数即可。
