"""Pure-CPU smoke checks for the repository's stable public contracts."""

from __future__ import annotations

from shopping_agent.evaluation.summary import summarize_trajectories
from shopping_agent.training.sft.dataset import IGNORE_INDEX, build_supervised_example
from shopping_agent.environment.tools import SHOP_TOOL_SCHEMAS
from shopping_agent.training.rl.dynamic_sampling import select_reward_varying_groups


class _CharacterTemplate:
    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
    ):
        del tools, tokenize
        text = ""
        for message in messages:
            text += f"<{message['role']}>"
            text += message.get("content") or ""
            for call in message.get("tool_calls") or []:
                text += f"[tool={call['function']['name']}]"
            text += f"</{message['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


def _reward_sample() -> dict:
    """One minimal Reward v4 gold purchase, used by the aggregation check."""

    return {
        "trajectory_id": "cpu-smoke-trajectory",
        "task_id": 900001,
        "status": "done",
        "done": True,
        "final_reward": 1.0,
        "steps": [
            {
                "tool_name": "search_products",
                "parameters": {"query": "白色保温杯"},
                "observation": "1|12345678|39.0|白色保温杯",
            },
            {
                "tool_name": "open_product",
                "parameters": {"asin": "12345678"},
                "observation": "product detail",
            },
            {
                "tool_name": "view_description",
                "parameters": {},
                "observation": "description",
            },
            {
                "tool_name": "buy_now",
                "parameters": {},
                "observation": "Environment terminated.",
            },
        ],
        "terminal_result": {
            "done": True,
            "over": True,
            "reward": 1.0,
            "reward_detail": {
                "reward_version": "shopping-reward-v4",
                "reward_type": "gold_purchase",
                "reward_valid": True,
                "purchase_success": True,
                "termination_reason": "gold_purchase",
                "terminal_utility": 1.0,
                "weighted_score": 1.0,
            },
        },
    }


def run_cpu_smoke() -> dict:
    checks = []
    tool_names = {
        schema["function"]["name"] for schema in SHOP_TOOL_SCHEMAS
    }
    required_tools = {
        "search_products",
        "open_product",
        "select_option",
        "buy_now",
        "finish_without_purchase",
    }
    if not required_tools <= tool_names:
        raise AssertionError("shopping tool schema is incomplete")
    checks.append("action_schema")

    summary = summarize_trajectories([900001], [_reward_sample()])
    if summary["strict_successes"] != 1 or summary["strict_success_rate"] != 1.0:
        raise AssertionError("Reward v4 strict-success aggregation failed")
    if summary["expected_tasks"] != 1 or summary["purchase_successes"] != 1:
        raise AssertionError("Reward v4 fixed-denominator aggregation failed")
    checks.append("reward_summary")

    tokenizer = _CharacterTemplate()
    supervised = build_supervised_example(
        messages=[
            {"role": "user", "content": "buy a cup"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_products",
                            "arguments": '{"query":"cup"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "private observation"},
        ],
        tools=[],
        tokenizer=tokenizer,
        max_length=1000,
    )
    if supervised is None:
        raise AssertionError("SFT label-mask sample was rejected")
    labeled = [
        token for token in supervised["labels"] if token != IGNORE_INDEX
    ]
    if not labeled or len(labeled) == len(supervised["labels"]):
        raise AssertionError("SFT labels are not assistant-only")
    checks.append("sft_label_mask")

    selected, diagnostics = select_reward_varying_groups(
        ["a", "a", "b", "b"],
        [0.0, 1.0, 0.5, 0.5],
    )
    if selected != [0, 1] or diagnostics["kept_group_count"] != 1:
        raise AssertionError("dynamic sampling grouping changed")
    checks.append("dynamic_sampling_grouping")

    return {
        "schema_version": "shopping-cpu-smoke-v1",
        "checks": checks,
    }
