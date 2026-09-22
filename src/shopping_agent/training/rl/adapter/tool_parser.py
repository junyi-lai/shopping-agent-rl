"""veRL tool parser for MiniCPM5's native XML tool-call format.

veRL 0.8.0 ships built-in parsers (``hermes`` / ``gpt-oss`` / ``gemma4``) but not
``minicpm5``. MiniCPM5-2B emits tool calls as

    <function name="search_products"><param name="query">乳胶枕</param></function>

with optional ``<![CDATA[ ... ]]>`` wrappers and a ``</think>`` reasoning prefix.
This module registers a ``minicpm5`` parser into veRL's ``ToolParser._registry``
so ``configs/grpo.yaml`` can select it via ``rollout.multi_turn.format``.
"""

from __future__ import annotations

import json
import logging
import re

try:  # 本地单测不安装 veRL；部署时由 veRL 注入真实类型。
    from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
    from verl.utils.rollout_trace import rollout_trace_op
except ImportError:  # pragma: no cover - 仅轻量开发环境使用
    class FunctionCall:
        def __init__(self, name, arguments):
            self.name, self.arguments = name, arguments

    class ToolParser:
        _registry: dict = {}

        @classmethod
        def register(cls, name):
            def decorator(subclass):
                cls._registry[name] = subclass
                return subclass
            return decorator

    def rollout_trace_op(function):
        return function


logger = logging.getLogger(__name__)

_FUNC_BLOCK_RE = re.compile(
    r"<function\s+name=['\"]([^'\"]+)['\"][^>]*>(.*?)</function>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<param\s+name=['\"]([^'\"]+)['\"][^>]*>(.*?)</param>",
    re.DOTALL,
)
_CDATA_RE = re.compile(r"^<!\[CDATA\[(.*)\]\]>$", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Return only user-visible content after MiniCPM5's ``</think>`` marker."""
    if "</think>" not in text:
        return text
    visible = text.rsplit("</think>", 1)[-1]
    return visible if visible.strip() else text


def _param_value(raw: str) -> str:
    raw = raw.strip()
    match = _CDATA_RE.match(raw)
    if match:
        return match.group(1)
    return raw


@ToolParser.register("minicpm5")
class MiniCPM5ToolParser(ToolParser):
    """Decode MiniCPM5 ``<function>`` blocks into veRL ``FunctionCall`` objects."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.tool_call_start_token = "<function"
        self.tool_call_end_token = "</function>"

    @property
    def stop_token_ids(self) -> list[int]:
        """Stop after the closing function tag when it is a real special token."""
        try:
            token_id = self.tokenizer.convert_tokens_to_ids(self.tool_call_end_token)
        except Exception:
            return []
        if token_id is None or isinstance(token_id, list):
            return []
        if token_id == self.tokenizer.unk_token_id:
            return []
        return [int(token_id)]

    @rollout_trace_op
    async def extract_tool_calls(self, responses_ids, tools=None):
        loop = __import__("verl.utils.ray_utils", fromlist=["get_event_loop"]).get_event_loop()
        try:
            decode = lambda: self.tokenizer.decode(responses_ids, skip_special_tokens=False)
        except TypeError:  # 某些最小 tokenizer 不支持该参数
            decode = lambda: self.tokenizer.decode(responses_ids)
        text = await loop.run_in_executor(None, decode)

        pad_token = getattr(self.tokenizer, "pad_token", None)
        if pad_token:
            text = text.replace(str(pad_token), "")

        if self.tool_call_start_token not in text:
            return _strip_thinking(text), []

        function_calls = []
        consumed_spans = []
        for match in _FUNC_BLOCK_RE.finditer(text):
            name = match.group(1).strip()
            block = match.group(2)
            params = {}
            for param in _PARAM_RE.finditer(block):
                params[param.group(1).strip()] = _param_value(param.group(2))
            function_calls.append(
                FunctionCall(name=name, arguments=json.dumps(params, ensure_ascii=False))
            )
            consumed_spans.append(match.span())

        content = _strip_thinking(_remove_spans(text, consumed_spans))
        return content, function_calls


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    parts = []
    cursor = 0
    for start, end in spans:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)
