import json
import tempfile
import unittest
from http.client import RemoteDisconnected
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch

from shopping_agent.environment.actions import action_guard_tool_message
from shopping_agent.environment.client import ShopEnvironmentError
from shopping_agent.evaluation.rollout import (
    CollectionInfrastructureError,
    OpenAIChatClient,
    SYSTEM_PROMPT,
    collect_tasks,
    collect_for_task,
    completed_task_attempts,
    load_tasks,
    rollout_interrupted,
)


OBSERVATION_VERSION = "shopping-observation-v2"
TERMINAL_ASIN = "100000000001"
PRODUCT = {
    "asin": TERMINAL_ASIN,
    "title": "乳胶枕 家用",
    "brand": "测试品牌",
    "shop_name": "测试品牌旗舰店",
    "category": "家纺",
    "attribute": ["乳胶"],
    "pricing": [39.0],
    "small_description": ["乳胶枕"],
    "full_description": "乳胶枕，家用。",
    "customization_options": {},
}
GOAL = {
    "asin": TERMINAL_ASIN,
    "category": "家纺",
    "instruction_text": "我想买一个乳胶枕，预算100元以内。",
    "attributes": [],
    "goal_options": [],
}


def observation_state(page_type, *, actions=(), search_available=True, **extra):
    """Build one Environment v2 observation state for the fake environment."""
    return {
        "observation_version": OBSERVATION_VERSION,
        "page_type": page_type,
        "actions": list(actions),
        "search_available": search_available,
        **extra,
    }


def search_results_state(asin=TERMINAL_ASIN, *, actions=None):
    return observation_state(
        "search_results",
        actions=[asin] if actions is None else actions,
        query="乳胶枕",
        normalized_query="乳胶枕",
        page=1,
        total_pages=1,
        total_results=1,
        rank_start=1,
        rank_end=1,
        products=[
            {
                "rank": 1,
                "asin": asin,
                "price": "39.0",
                "brand": "测试品牌",
                "category": "家纺",
                "key_attributes": ["乳胶"],
                "title": "乳胶枕",
            }
        ],
    )


def product_state(*, actions=("Buy Now",)):
    return observation_state(
        "product_detail",
        actions=list(actions),
        product={
            "asin": TERMINAL_ASIN,
            "title": "乳胶枕",
            "brand": "测试品牌",
            "category": "家纺",
            "key_attributes": ["乳胶"],
            "price": "39.0",
        },
        selected_options={},
        available_options={"颜色": ["白色"]},
    )


def subpage_state(subpage, *, actions=("< Prev",)):
    return observation_state(
        "information_subpage",
        actions=list(actions),
        subpage=subpage,
        content=f"{subpage} 内容",
        product={
            "asin": TERMINAL_ASIN,
            "title": "乳胶枕",
            "brand": "测试品牌",
            "category": "家纺",
            "key_attributes": ["乳胶"],
            "price": "39.0",
        },
        selected_options={},
        available_options={},
    )


def terminal_result():
    """Raw terminal facts of a gold purchase, as the pure simulator returns them."""
    return {
        "done": True,
        "over": True,
        "reward": 1.0,
        "termination_reason": "purchase",
        "purchase": {"asin": TERMINAL_ASIN},
        "purchased_product": dict(PRODUCT),
        "goal": dict(GOAL),
        "target_product": dict(PRODUCT),
        "options": {},
        "price_resolution": {"status": "pass", "price": 39.0},
        "observation_state": observation_state("terminal"),
    }


class FakeEnv:
    """Synthetic leased environment: v2 observations plus raw terminal facts.

    ``reset`` opens on a product page so a single ``buy_now`` is legal; the guard
    only ever needs the buttons and ASINs that the current page exposes.
    """

    def __init__(self, **kwargs):
        self.actions = []
        self.released = False

    def reset(self, task_id):
        return {
            "env_idx": 0,
            "instruction": f"Instruction: task {task_id}",
            "observation_state": product_state(),
        }

    def step(self, action):
        self.actions.append(action)
        if action == "search[乳胶枕]":
            return {
                "observation_state": search_results_state(),
                "reward": 0.0,
                "done": False,
            }
        if action == f"click[{TERMINAL_ASIN}]":
            return {
                "observation_state": product_state(),
                "reward": 0.0,
                "done": False,
            }
        return terminal_result()

    def release(self):
        self.released = True


class FailingEnv(FakeEnv):
    def step(self, action):
        self.actions.append(action)
        raise RuntimeError("env exploded")


class NonTerminalEnv(FakeEnv):
    def step(self, action):
        self.actions.append(action)
        return {"instruction": "keep going", "reward": 0.0, "done": False}


class ReleaseFailingEnv(FakeEnv):
    def release(self):
        raise OSError("ShopSimulator unavailable during release")


class UnavailableEnv(FakeEnv):
    def reset(self, task_id):
        raise ShopEnvironmentError("Unable to get available environment resource, please try again later")


class GuardRecoveryEnv(FakeEnv):
    """用于验证非法尝试被合法工具调用隔开时仍可恢复。"""

    def step(self, action):
        self.actions.append(action)
        if action == "search[乳胶枕]":
            return {
                "observation_state": search_results_state(),
                "reward": 0.0,
                "done": False,
            }
        if action == f"click[{TERMINAL_ASIN}]":
            return {
                "observation_state": product_state(actions=("Features", "Buy Now")),
                "reward": 0.0,
                "done": False,
            }
        if action == "click[Features]":
            return {
                "observation_state": subpage_state("Features"),
                "reward": 0.0,
                "done": False,
            }
        if action == "click[< Prev]":
            return {
                "observation_state": product_state(actions=("Features", "Buy Now")),
                "reward": 0.0,
                "done": False,
            }
        if action == "click[Buy Now]":
            return terminal_result()
        raise AssertionError(f"unexpected action: {action}")


class MockClient:
    def __init__(self, messages):
        self.messages = list(messages)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append({"messages": messages, "tools": tools})
        return self.messages.pop(0)


def assistant_tool(name, arguments, call_id="call_1"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        ],
    }


class RolloutTest(unittest.TestCase):
    def test_default_prompt_matches_reward_policy(self):
        """默认提示词应表达 Reward v4 的购买优先级和停止门槛。"""
        self.assertIn("单轮购物任务", SYSTEM_PROMPT)
        self.assertIn("不得向用户追问", SYSTEM_PROMPT)
        self.assertIn("buy_now", SYSTEM_PROMPT)
        self.assertIn("不要在任务结束前输出最终答复", SYSTEM_PROMPT)
        self.assertIn("当前页面是动作合法性的唯一依据", SYSTEM_PROMPT)
        self.assertIn("信息子页", SYSTEM_PROMPT)
        self.assertIn("必须先调用当前页面可见的 `prev_page` 或 `back_to_search` 返回", SYSTEM_PROMPT)
        self.assertIn("无参数工具", SYSTEM_PROMPT)
        self.assertIn("历史 observation 可以用于记住和比较候选", SYSTEM_PROMPT)
        self.assertIn("不能直接点击历史页面中的 ASIN", SYSTEM_PROMPT)
        self.assertIn("品类必须正确", SYSTEM_PROMPT)
        self.assertIn("不得超过用户明确预算", SYSTEM_PROMPT)
        self.assertIn("品类 > 预算 > 品牌 > 型号与核心功能 > 规格属性", SYSTEM_PROMPT)
        self.assertIn("Reviews 只用于辅助判断使用体验", SYSTEM_PROMPT)
        self.assertIn("不能用于确认型号、官方功能、规格或价格", SYSTEM_PROMPT)
        self.assertIn("所有影响可购买 variant 的必要规格轴", SYSTEM_PROMPT)
        self.assertIn("finish_without_purchase", SYSTEM_PROMPT)
        self.assertIn("多次有实质差异的搜索和多个候选核验", SYSTEM_PROMPT)
        self.assertIn("没有明显值得继续核验的候选", SYSTEM_PROMPT)
        self.assertIn("是否达到结束资格由环境判断", SYSTEM_PROMPT)
        self.assertIn("不要连续重复同一动作", SYSTEM_PROMPT)
        self.assertIn("不要调用 `think` 工具", SYSTEM_PROMPT)

    def test_guard_gives_a_return_only_instruction_on_information_subpage(self):
        """子页误操作后，守卫应明确引导模型先返回，不重复猜测按钮。"""
        message = action_guard_tool_message(
            assistant_tool("view_attributes", {}, "call_attributes"),
            "click_not_in_previous_observation",
            '详情页内容\n\n可点击的按钮: ["< Prev"]',
        )

        self.assertIn("你处于信息子页", message["content"])
        self.assertIn("prev_page", message["content"])
        self.assertIn("不要重复使用历史页面目标", message["content"])

    def test_collect_for_task_executes_openai_tool_calls_until_done(self):
        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_open"),
                assistant_tool("buy_now", {}, "call_buy"),
            ]
        )
        env = FakeEnv()

        traj = collect_for_task(
            {"task_id": 7},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=4,
        )

        self.assertEqual(traj["status"], "done")
        self.assertTrue(traj["trajectory_id"])
        self.assertEqual(env.actions, ["search[乳胶枕]", "click[100000000001]", "click[Buy Now]"])
        self.assertTrue(env.released)
        self.assertEqual(traj["steps"][0]["tool_call"]["function"]["name"], "search_products")
        self.assertEqual(traj["steps"][2]["env_action"], "click[Buy Now]")
        self.assertEqual(traj["terminal_result"]["purchase"]["asin"], TERMINAL_ASIN)
        self.assertTrue(any(message["role"] == "tool" for message in traj["messages"]))

    def test_environment_exposes_finish_without_purchase_to_agent(self):
        class EnvironmentV21(FakeEnv):
            def reset(self, task_id):
                result = super().reset(task_id)
                result["environment_version"] = "shopsimulator-environment-v2.1"
                return result

        client = MockClient([{"role": "assistant", "content": "stop"}])
        env = EnvironmentV21()

        collect_for_task(
            {"task_id": 7},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=1,
        )

        tool_names = [
            schema["function"]["name"]
            for schema in client.requests[0]["tools"]
        ]
        self.assertIn("finish_without_purchase", tool_names)
        self.assertIn("finish_without_purchase", client.requests[0]["messages"][0]["content"])
        self.assertTrue(env.released)

    def test_collect_for_task_blocks_invalid_click_then_keeps_clean_recovery(self):
        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_open"),
                assistant_tool("view_features", {}, "call_invalid"),
                assistant_tool("buy_now", {}, "call_buy"),
            ]
        )
        env = FakeEnv()

        traj = collect_for_task(
            {"task_id": 10},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=5,
        )

        self.assertEqual(traj["status"], "done")
        self.assertEqual(env.actions, ["search[乳胶枕]", "click[100000000001]", "click[Buy Now]"])
        self.assertEqual(len(traj["blocked_tool_calls"]), 1)
        self.assertEqual(traj["blocked_tool_calls"][0]["reason"], "click_not_in_previous_observation")
        self.assertNotIn(
            "view_features",
            [step["tool_name"] for step in traj["steps"]],
        )
        self.assertTrue(
            any(
                message.get("role") == "tool"
                and message.get("tool_call_id") == "call_invalid"
                and message.get("runtime_action_guard") is True
                for message in traj["messages"]
            )
        )

    def test_collect_for_task_allows_current_page_navigation_after_option_selection(self):
        """选择规格后仍由环境当前页面决定能否浏览，采集器不另造状态机。"""
        class OptionEnv(FakeEnv):
            def step(self, action):
                self.actions.append(action)
                if action == "search[乳胶枕]":
                    return {
                        "observation_state": search_results_state(),
                        "reward": 0.0,
                        "done": False,
                    }
                if action == f"click[{TERMINAL_ASIN}]":
                    return {
                        "observation_state": product_state(
                            actions=("满天星", "Description", "Buy Now")
                        ),
                        "reward": 0.0,
                        "done": False,
                    }
                if action == "click[满天星]":
                    return {
                        "observation_state": product_state(
                            actions=("Description", "Buy Now")
                        ),
                        "reward": 0.0,
                        "done": False,
                    }
                if action == "click[Description]":
                    return {
                        "observation_state": subpage_state(
                            "Description", actions=("Buy Now",)
                        ),
                        "reward": 0.0,
                        "done": False,
                    }
                if action == "click[Buy Now]":
                    return terminal_result()
                raise AssertionError(action)

        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_open"),
                assistant_tool("select_option", {"value": "满天星"}, "call_option"),
                assistant_tool("view_description", {}, "call_wrong_navigation"),
                assistant_tool("buy_now", {}, "call_buy"),
            ]
        )
        env = OptionEnv()

        traj = collect_for_task({"task_id": 12}, client=client, env_factory=lambda **kwargs: env)

        self.assertEqual(traj["status"], "done")
        self.assertEqual([step["env_action"] for step in traj["steps"]], [
            "search[乳胶枕]",
            "click[100000000001]",
            "click[满天星]",
            "click[Description]",
            "click[Buy Now]",
        ])
        self.assertEqual(traj["blocked_tool_calls"], [])

    def test_collect_for_task_keeps_one_tool_schema_after_option_selection(self):
        """所有阶段使用同一工具 schema，环境 observation 是唯一动作边界。"""
        class OptionEnv(FakeEnv):
            def step(self, action):
                self.actions.append(action)
                if action == "search[乳胶枕]":
                    return {
                        "observation_state": search_results_state(),
                        "reward": 0.0,
                        "done": False,
                    }
                if action == f"click[{TERMINAL_ASIN}]":
                    return {
                        "observation_state": product_state(
                            actions=("满天星", "Buy Now")
                        ),
                        "reward": 0.0,
                        "done": False,
                    }
                if action == "click[满天星]":
                    return {
                        "observation_state": product_state(actions=("Buy Now",)),
                        "reward": 0.0,
                        "done": False,
                    }
                if action == "click[Buy Now]":
                    return terminal_result()
                raise AssertionError(action)

        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_open"),
                assistant_tool("select_option", {"value": "满天星"}, "call_option"),
                assistant_tool("buy_now", {}, "call_buy"),
            ]
        )

        trajectory = collect_for_task(
            {"task_id": 13}, client=client, env_factory=OptionEnv, base_url="http://shop.test"
        )

        self.assertEqual(trajectory["status"], "done")
        exposed_after_selection = [
            schema["function"]["name"] for schema in client.requests[3]["tools"]
        ]
        self.assertIn("view_description", exposed_after_selection)
        self.assertIn("search_products", exposed_after_selection)

    def test_collect_for_task_allows_recovery_after_separated_guard_rejections(self):
        """合法动作应重置守卫计数，避免累计三次历史点击提前中止。"""
        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("open_product", {"asin": "999999999999"}, "call_old_asin"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_open"),
                assistant_tool("view_attributes", {}, "call_missing_attributes"),
                assistant_tool("view_features", {}, "call_features"),
                assistant_tool("buy_now", {}, "call_buy_on_subpage"),
                assistant_tool("prev_page", {}, "call_return"),
                assistant_tool("buy_now", {}, "call_buy"),
            ]
        )
        env = GuardRecoveryEnv()

        traj = collect_for_task(
            {"task_id": 11},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=8,
        )

        self.assertEqual(traj["status"], "done")
        self.assertEqual(len(traj["blocked_tool_calls"]), 3)
        self.assertEqual(env.actions, [
            "search[乳胶枕]",
            "click[100000000001]",
            "click[Features]",
            "click[< Prev]",
            "click[Buy Now]",
        ])

    def test_collect_for_task_keeps_exception_trajectory_and_releases_env(self):
        client = MockClient([assistant_tool("search_products", {"query": "乳胶枕"})])
        env = FailingEnv()

        traj = collect_for_task(
            {"task_id": 8},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=2,
        )

        self.assertEqual(traj["status"], "error")
        self.assertIn("env exploded", traj["error"]["message"])
        self.assertEqual(traj["steps"][0]["env_action"], "search[乳胶枕]")
        self.assertTrue(env.released)

    def test_rollout_interrupted_raises_keyboard_interrupt_for_finally_release(self):
        with self.assertRaises(KeyboardInterrupt):
            rollout_interrupted(None, None)

    def test_completed_task_attempts_defaults_missing_attempt_index_to_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            output.write_text(
                json.dumps({"task_id": 1, "trajectory_id": "old"}) + "\n"
                + json.dumps({"task_id": 3, "trajectory_id": "old2"}) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(completed_task_attempts(output), {(1, 0), (3, 0)})

    def test_collect_tasks_skips_existing_output_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            output.write_text(json.dumps({"task_id": 1, "trajectory_id": "old"}) + "\n")
            client = MockClient([assistant_tool("buy_now", {}, "call_buy")])

            written = collect_tasks(
                [{"task_id": 1}, {"task_id": 2}],
                client=client,
                output_path=output,
                base_url="http://shop.test",
                max_steps=1,
                env_factory=FakeEnv,
            )
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([row["task_id"] for row in rows], [1, 2])
        self.assertEqual(len(written), 1)

    def test_collect_tasks_resumes_missing_attempts_for_each_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            output.write_text(
                json.dumps({"task_id": 1, "attempt_index": 0, "trajectory_id": "old"}) + "\n",
                encoding="utf-8",
            )
            client = MockClient(
                [
                    assistant_tool("buy_now", {}, "call_1"),
                    assistant_tool("buy_now", {}, "call_2"),
                    assistant_tool("buy_now", {}, "call_3"),
                ]
            )

            written = collect_tasks(
                [{"task_id": 1}, {"task_id": 2}],
                client=client,
                output_path=output,
                base_url="http://shop.test",
                max_steps=1,
                env_factory=FakeEnv,
                attempts_per_task=2,
            )

        self.assertEqual(
            [(row["task_id"], row["attempt_index"]) for row in written],
            [(1, 1), (2, 0), (2, 1)],
        )

    def test_collect_tasks_stops_after_a_release_failure(self):
        """环境租约未释放时，不能继续消耗后续 task。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            client = MockClient([{"role": "assistant", "content": "stop"}])

            with self.assertRaises(CollectionInfrastructureError):
                collect_tasks(
                    [{"task_id": 1}, {"task_id": 2}],
                    client=client,
                    output_path=output,
                    base_url="http://shop.test",
                    env_factory=ReleaseFailingEnv,
                )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([row["task_id"] for row in rows], [1])
        self.assertEqual(rows[0]["status"], "environment_release_failed")
        self.assertEqual(rows[0]["release_error"]["type"], "OSError")

    def test_collect_tasks_stops_after_environment_resource_is_unavailable(self):
        """服务报告无可用环境时，不能把其余 task 误记成失败轨迹。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            client = MockClient([])

            with self.assertRaises(CollectionInfrastructureError):
                collect_tasks(
                    [{"task_id": 1}, {"task_id": 2}],
                    client=client,
                    output_path=output,
                    base_url="http://shop.test",
                    env_factory=UnavailableEnv,
                )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([row["task_id"] for row in rows], [1])
        self.assertEqual(rows[0]["error"]["type"], "ShopEnvironmentError")

    def test_collect_tasks_stops_after_model_api_is_unreachable(self):
        """模型 API 重试后仍断线时，不能把后续 task 批量记成失败。"""
        class DisconnectedClient:
            def complete(self, messages, tools):
                raise URLError("Remote end closed connection without response")

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"

            with self.assertRaises(CollectionInfrastructureError):
                collect_tasks(
                    [{"task_id": 1}, {"task_id": 2}],
                    client=DisconnectedClient(),
                    output_path=output,
                    base_url="http://shop.test",
                    env_factory=FakeEnv,
                )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([row["task_id"] for row in rows], [1])
        self.assertEqual(rows[0]["error"]["type"], "URLError")

    def test_collect_for_task_serializes_multiple_tool_calls_before_execution(self):
        client = MockClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": "search_products",
                                "arguments": json.dumps({"query": f"乳胶枕{index}"}, ensure_ascii=False),
                            },
                        }
                        for index in range(3)
                    ],
                },
                assistant_tool("search_products", {"query": "第二次观察后搜索"}, "call_after_observation"),
            ]
        )
        env = NonTerminalEnv()

        traj = collect_for_task(
            {"task_id": 9},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=2,
        )

        self.assertEqual(traj["status"], "max_steps")
        self.assertEqual(len(traj["steps"]), 2)
        self.assertEqual(env.actions, ["search[乳胶枕0]", "search[第二次观察后搜索]"])
        self.assertEqual(len(traj["messages"][2]["tool_calls"]), 1)
        self.assertEqual(
            [call["id"] for call in traj["tool_call_truncations"][0]["dropped_tool_calls"]],
            ["call_1", "call_2"],
        )

    def test_load_tasks_reads_interaction_task_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = Path(tmpdir) / "tasks.jsonl"
            tasks.write_text(
                json.dumps(
                    {
                        "prompt": [{"role": "user", "content": "hello"}],
                        "extra_info": {"interaction_kwargs": {"task_id": 42}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = load_tasks(tasks)

        self.assertEqual(rows[0]["task_id"], 42)
        self.assertEqual(rows[0]["prompt"][0]["content"], "hello")

    def test_openai_client_sends_standard_tool_payload(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"url": url, "payload": payload, "headers": headers, "timeout": timeout})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="deepseek-chat",
            base_url="https://api.example.test/v1",
            api_key="secret",
            temperature=0.2,
            top_p=0.9,
            timeout=12,
            transport=transport,
        )

        message = client.complete([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

        self.assertEqual(message["content"], "ok")
        self.assertEqual(captured["url"], "https://api.example.test/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], "deepseek-chat")
        self.assertEqual(captured["payload"]["tools"], [{"type": "function"}])
        self.assertEqual(captured["payload"]["max_tokens"], 512)
        self.assertEqual(captured["payload"]["temperature"], 0.2)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(captured["headers"]["User-Agent"], "shopping-agent/0.1")

    def test_openai_client_supports_responses_tool_payload_and_result(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"url": url, "payload": payload})
            return {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_search",
                        "name": "search_products",
                        "arguments": '{"query":"乳胶枕"}',
                    }
                ]
            }

        client = OpenAIChatClient(
            model="gpt-5.6-luna",
            base_url="https://opencode.ai/zen/go/v1/responses",
            api_key="secret",
            transport=transport,
        )
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "买乳胶枕"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {
                            "name": "search_products",
                            "arguments": '{"query":"枕头"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_old",
                "name": "search_products",
                "content": "search result",
            },
        ]
        message = client.complete(
            messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search_products",
                        "description": "search",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )

        self.assertEqual(captured["url"], "https://opencode.ai/zen/go/v1/responses")
        self.assertEqual(captured["payload"]["max_output_tokens"], 512)
        self.assertEqual(captured["payload"]["tools"][0]["name"], "search_products")
        self.assertEqual(captured["payload"]["input"][-1]["type"], "function_call_output")
        self.assertEqual(captured["payload"]["input"][-1]["call_id"], "call_old")
        self.assertEqual(message["tool_calls"][0]["id"], "call_search")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "search_products")

    def test_openai_client_allows_bounded_completion_override(self):
        """本地推理服务必须收到单次生成上限，避免无工具文本耗尽上下文。"""
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="openbmb/MiniCPM5-2B",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            max_tokens=256,
            transport=transport,
        )

        client.complete([{"role": "user", "content": "hi"}], tools=[])

        self.assertEqual(captured["payload"]["max_tokens"], 256)

    def test_openai_client_compacts_old_complete_tool_groups_before_request(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "task"},
            assistant_tool("search_products", {"query": "old"}, "old"),
            {"role": "tool", "tool_call_id": "old", "name": "search_products", "content": "old page"},
            assistant_tool("search_products", {"query": "middle"}, "middle"),
            {"role": "tool", "tool_call_id": "middle", "name": "search_products", "content": "middle page"},
            assistant_tool("search_products", {"query": "latest"}, "latest"),
            {"role": "tool", "tool_call_id": "latest", "name": "search_products", "content": "latest page"},
        ]
        client = OpenAIChatClient(
            model="shopping",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            max_tokens=2,
            context_window=9,
            context_safety_margin=1,
            context_compaction_enable=True,
            token_counter=lambda candidate, tools: len(candidate),
            transport=transport,
        )

        client.complete(messages, tools=[])

        self.assertNotIn("old page", str(captured["payload"]["messages"]))
        self.assertIn("middle page", str(captured["payload"]["messages"]))
        self.assertIn("latest page", str(captured["payload"]["messages"]))
        self.assertEqual(client.last_context_event["removed_groups"], 1)
        self.assertEqual(messages[3]["content"], "old page")

    def test_openai_client_projects_tool_observation_with_serving_tokenizer(self):
        client = OpenAIChatClient(
            model="shopping",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            observation_token_budget=128,
            observation_generic_token_budget=128,
            observation_token_counter=len,
        )
        raw = (
            "Description " + "x" * 200
            + "\n\n搜索功能是否可用: False"
            + '\n\n可点击的按钮: ["back to search", "< prev"]'
        )

        visible, meta = client.project_observation("view_description", raw, {})

        self.assertLessEqual(len(visible), 128)
        self.assertTrue(meta["truncated"])
        self.assertTrue(meta["critical_footer_preserved"])

    def test_openai_client_thinking_mode_keeps_reasoning_for_tool_follow_up(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "先核对规格，再搜索。",
                            "tool_calls": [
                                {
                                    "id": "call_search",
                                    "type": "function",
                                    "function": {"name": "search_products", "arguments": '{"query":"乳胶枕"}'},
                                }
                            ],
                        }
                    }
                ]
            }

        client = OpenAIChatClient(
            model="deepseek-v4-flash",
            base_url="https://api.example.test/v1",
            api_key="secret",
            thinking=True,
            reasoning_effort="high",
            transport=transport,
        )

        message = client.complete([{"role": "user", "content": "买乳胶枕"}], tools=[{"type": "function"}])

        self.assertEqual(captured["payload"]["thinking"], {"type": "enabled"})
        self.assertEqual(captured["payload"]["reasoning_effort"], "high")
        self.assertNotIn("temperature", captured["payload"])
        self.assertNotIn("top_p", captured["payload"])
        self.assertEqual(message["reasoning_content"], "先核对规格，再搜索。")

    def test_openai_client_explicitly_disables_deepseek_v4_thinking(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="deepseek-v4-flash",
            base_url="https://opencode.ai/zen/go/v1",
            api_key="secret",
            thinking=False,
            reasoning_effort="max",
            transport=transport,
        )

        client.complete([{"role": "user", "content": "继续"}], tools=[])

        self.assertEqual(captured["payload"]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", captured["payload"])
        self.assertEqual(captured["payload"]["temperature"], 0.0)
        self.assertEqual(captured["payload"]["top_p"], 1.0)

    def test_openai_client_disables_thinking_for_plain_deepseek_ids(self):
        """真实 DeepSeek 模型 id 是 ``deepseek-flash``(无 v4 前缀),同样必须关 thinking。

        回归测试:判定曾写成 ``startswith("deepseek-v4")``,于是 ``deepseek-flash``
        的 thinking 关不掉——模型把计划写进 ``reasoning_content``、正文留空且不发
        tool call,整条轨迹被判成 ``assistant_final`` 而拒绝(实测验收率 12.5% → 60%)。
        """
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="deepseek-flash",
            base_url="https://api.deepseek.com/v1",
            api_key="secret",
            thinking=False,
            transport=transport,
        )

        client.complete([{"role": "user", "content": "继续"}], tools=[])

        self.assertEqual(captured["payload"]["thinking"], {"type": "disabled"})

    def test_openai_client_does_not_send_thinking_to_local_non_deepseek_model(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="openbmb/MiniCPM5-2B",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            thinking=False,
            transport=transport,
        )

        client.complete([{"role": "user", "content": "继续"}], tools=[])

        self.assertNotIn("thinking", captured["payload"])
        self.assertNotIn("reasoning_effort", captured["payload"])

    def test_openai_client_retries_transient_disconnect_without_replaying_tools(self):
        attempts = []

        def transport(url, payload, headers, timeout):
            attempts.append(payload)
            if len(attempts) == 1:
                raise RemoteDisconnected("connection closed")
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="deepseek-v4-pro",
            base_url="https://api.example.test/v1",
            api_key="secret",
            transport=transport,
        )

        with patch("shopping_agent.evaluation.rollout.time.sleep") as sleep:
            message = client.complete([{"role": "user", "content": "继续"}], tools=[])

        self.assertEqual(message["content"], "ok")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], attempts[1])
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
