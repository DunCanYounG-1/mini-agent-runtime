from mini_agent.protocol import parse_output


def test_standard_tool_call():
    p = parse_output('<thought>查天气</thought>\n<tool_call>{"name": "weather", "arguments": {"city": "北京"}}</tool_call>')
    assert p.kind == "tool_call"
    assert p.thought == "查天气"
    assert p.tool_calls[0].name == "weather"
    assert p.tool_calls[0].arguments == {"city": "北京"}


def test_final_answer():
    p = parse_output("<thought>打个招呼</thought><final_answer>你好！</final_answer>")
    assert p.kind == "final" and p.final_answer == "你好！" and p.thought == "打个招呼"


def test_multiple_tool_calls():
    text = ('<thought>两件事</thought><tool_call>{"name":"weather","arguments":{"city":"北京"}}</tool_call>'
            '<tool_call>{"name":"todo","arguments":{"action":"add","content":"带伞"}}</tool_call>')
    p = parse_output(text)
    assert [c.name for c in p.tool_calls] == ["weather", "todo"]
    assert [c.id for c in p.tool_calls] == ["call_1", "call_2"]


def test_code_fence_and_string_arguments():
    text = '<tool_call>\n```json\n{"name": "calculator", "arguments": "{\\"expression\\": \\"1+1\\"}"}\n```\n</tool_call>'
    p = parse_output(text)
    assert p.tool_calls[0].arguments == {"expression": "1+1"}


def test_alias_keys_and_trailing_comma():
    p = parse_output('<tool_call>{"tool": "search", "args": {"query": "ReAct",},}</tool_call>')
    assert p.tool_calls[0].name == "search" and p.tool_calls[0].arguments == {"query": "ReAct"}


def test_unclosed_tags():
    p = parse_output('<thought>算一下\n<tool_call>{"name": "calculator", "arguments": {"expression": "2*3"}}')
    assert p.kind == "tool_call" and p.thought == "算一下"
    p2 = parse_output("<thought>好</thought><final_answer>结果是 6")
    assert p2.final_answer == "结果是 6"


def test_braces_inside_json_string():
    p = parse_output('<tool_call>{"name": "todo", "arguments": {"action": "add", "content": "写 {周报}"}}</tool_call>')
    assert p.tool_calls[0].arguments["content"] == "写 {周报}"


def test_hallucinated_observation_is_dropped():
    text = ('<tool_call>{"name":"weather","arguments":{"city":"北京"}}</tool_call>\n'
            '<observation>晴 30 度</observation>\n<final_answer>北京晴 30 度</final_answer>')
    p = parse_output(text)
    assert p.kind == "tool_call" and p.final_answer is None
    assert any("observation" in w for w in p.warnings)


def test_tool_call_wins_over_premature_final():
    p = parse_output('<tool_call>{"name":"weather","arguments":{"city":"北京"}}</tool_call><final_answer>晴</final_answer>')
    assert p.kind == "tool_call" and p.final_answer is None


def test_plain_text_treated_as_final():
    p = parse_output("你好，我是助手。")
    assert p.kind == "final" and p.final_answer == "你好，我是助手。"


def test_invalid_json_is_error():
    p = parse_output('<tool_call>{"name": "weather", "arguments": {city: 北京}}</tool_call>')
    assert p.kind == "format_error" and p.errors


def test_partial_bad_tool_call_keeps_good_one():
    p = parse_output('<tool_call>{"name":"weather","arguments":{"city":"北京"}}</tool_call><tool_call>{bad}</tool_call>')
    assert p.kind == "tool_call" and len(p.tool_calls) == 1 and p.warnings


def test_empty_or_thought_only_is_error():
    assert parse_output("").kind == "format_error"
    assert parse_output("<thought>我想想</thought>").kind == "format_error"


def test_reasoning_content_and_think_tag():
    p = parse_output("<final_answer>好的</final_answer>", reasoning="模型内部推理")
    assert p.thought.startswith("模型内部推理")
    p2 = parse_output("<think>深度思考</think><final_answer>答案</final_answer>")
    assert p2.thought == "深度思考" and p2.final_answer == "答案"
