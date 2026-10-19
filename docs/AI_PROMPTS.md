# AI Prompt 与问题解决记录

## 使用的 AI 工具

- Claude Code（Claude Opus）：架构讨论、代码生成、测试编写、真实模型联调
- 被测模型：DeepSeek `deepseek-chat`，走 Anthropic Messages 兼容接口（`https://api.deepseek.com/anthropic`）；OpenAI-compatible 接口也保留并测过

## 1. 给 AI 的 Prompt

### 1.1 初始 prompt

把笔试题原文完整贴给 Claude Code（含全部要求：循环、工具注册与 Schema、解析、session、context、压缩、异常、trace、测试、README）。

### 1.2 过程中的关键约束 / 决策（人定）

| 问题 | 决定 |
|---|---|
| 用哪家 LLM | DeepSeek（`deepseek-chat`），key 只放本地 `.env`，`.gitignore` 排除 |
| 接口格式 | 先做了 OpenAI-compatible；后来要求“把 DeepSeek 换成 Messages 渠道”，于是加了 `AnthropicMessagesLLM` 并设为默认，两种格式用 `LLM_PROVIDER` 切换 |
| 是否用厂商 function calling | 不用。自定义 `<thought>/<tool_call>/<final_answer>` 文本协议，自己写解析 —— 题目明确要求“实现 LLM 输出的解析逻辑” |
| 依赖 | 运行时只用标准库，HTTP 用 `urllib`，避免被质疑“依赖框架” |
| 待办的作用域 | 按 session 隔离（题目场景：窗口 1、窗口 2 互不影响） |
| 测试策略 | 离线用“脚本化假 LLM”保证确定性，并记录每次收到的 messages 以断言 context；另有 live 用例只断言行为，不断言措辞 |

### 1.3 Agent 运行时内部使用的 prompt

- 主 system prompt：`mini_agent/context.py` 中的 `SYSTEM_TEMPLATE`（协议、决策规则、工具 Schema、时间、实时状态、摘要、召回片段）
- 压缩摘要 prompt：`mini_agent/context.py` 中的 `SUMMARY_PROMPT`
- 格式错误纠正：`mini_agent/protocol.py` 中的 `FORMAT_ERROR_HINT`
- 超步数强制作答：`mini_agent/agent.py` 中的 `FORCE_FINAL_PROMPT`

## 2. 问题解决记录

### 问题 1（真实模型暴露）：模型把自己刚加的待办当成“原本就有”

- **现象**：demo 中窗口 2 第一轮“写周报提纲并记待办”，DeepSeek 先 `todo add`，又 `todo list` 确认，最终回答里出现“会话里原本就有这条，内容一致，没有重复添加”。离线测试全绿，没发现这个问题。
- **定位**：看 trace（`data/demo/traces/A/w2.jsonl`）。system prompt 里的“当前会话状态（实时）”每个 step 都重新渲染，第二步时刚加的条目已经出现在“实时状态”里，模型把它理解为轮次开始前就存在。
- **修复**：system prompt 在每轮开始时渲染一次并在整轮内冻结；本轮内的状态变化只通过 observation 呈现，下一轮再刷新快照（`agent.py` 中的 `system = self.ctx.render_system(...)`）。附带收益：同一轮内 prompt 前缀不变，对服务端的前缀缓存（KV cache）更友好。
- **回归测试**：`test_context.py::test_system_prompt_frozen_within_turn`。
- **收获**：“状态每轮注入”这个设计本身没错，但注入的**时间点**有讲究：对模型来说，同一份状态出现在 system 里和出现在 observation 里，含义不同。

### 问题 2：双窗口隔离测试误报

- **现象**：断言“窗口 1 的 context 里没有‘周报’”失败。
- **原因**：不是串话，而是 system prompt 中 `search` 工具的描述本身就有“周报写法”“北京气候”等词。
- **修复**：只对 system 以外的历史消息做“不包含”断言；对 system 中的待办快照用精确条目（`#1 明天带伞`）断言。
- **收获**：做隔离测试，要先排除“共享的固定内容”带来的干扰，否则测出来的结论不可信。

### 问题 3：压缩测试中摘要请求“吃掉”了脚本回复

- **现象**：压缩相关测试的 summary 内容变成了某一轮的 `<final_answer>`。
- **原因**：token 预算设得小，压缩在前几轮就提前触发了（而且每轮都可能触发）。摘要也是一次 LLM 调用，按顺序消耗了脚本里本该给对话的回复。
- **修复**：测试里加 `RoutingLLM`：识别摘要请求并单独应答，不消耗对话脚本。断言改为看结构（保留轮数、归档轮数、摘要注入位置、原文已移出活跃历史），不依赖“第几轮触发”。

### 问题 4：Windows 下 trace 分析脚本报 JSONDecodeError

- **现象**：用 `cat *.jsonl | python -c ...` 分析 trace 时 JSON 解析失败。
- **原因**：Windows 管道的 stdin 默认按 GBK 解码，UTF-8 中文被读坏。trace 文件本身是好的。
- **修复**：分析脚本直接用 `open(..., encoding="utf-8")` 读文件；CLI 和 demo 启动时 `reconfigure` stdout 为 UTF-8。

### 问题 5：切换到 Messages 渠道时要处理的格式差异

- **需求**：把 DeepSeek 从 `/chat/completions` 换到 Anthropic 兼容的 `/v1/messages`。
- **先手工验证端点**：Git Bash 里用 curl 发中文请求体，DeepSeek 返回 `invalid unicode code point`——是 Windows 终端把命令行参数里的中文编码坏了，不是接口问题；改用 Python 发请求就正常了。
- **适配要点**：system 从消息列表挪到顶层字段；Messages 要求 user/assistant 严格交替，而本项目的“格式错误纠正”流程可能产生连续两条 user 消息，所以适配器会自动合并；`max_tokens` 必填；`stop` 改名为 `stop_sequences`；返回的 content 是块数组，要拼接 `text` 块并把 `thinking` 块当作思考；529（过载）也要重试。
- **重构**：把 HTTP、重试、错误分类抽到公共基类，两个适配器只实现“拼请求 / 拆响应”。新增 `test_llm_client.py`（8 个用例，mock HTTP）。
- **结果**：Messages 渠道上 live 8/8 通过；demo 共 15 次 LLM 调用，没有一次格式告警。

## 3. 真实模型运行结果

- `pytest -m live`：OpenAI-compatible 与 Messages 两种渠道各跑一遍，均 8/8 通过（双窗口场景、带工具追问“那上海呢”、状态追问、纯对话追问、计算、直接回复、压缩后仍记得早期事实）。
- 单次 LLM 调用耗时约 0.5–2.0 秒，prompt 约 900–1900 tokens（场景 8 轮）。
- 完整过程见 `docs/demo_transcript.md`。
