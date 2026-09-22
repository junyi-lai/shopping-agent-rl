"""ShopSimulator 的最小 HTTP 客户端。

每个 ``ShopAgentEnv`` 对象只负责一条 trajectory：先租用一个环境实例，
反复执行动作，最后释放租约。训练和评测都通过这个生命周期访问商店。
"""

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from shopping_agent import __version__


class ShopHttpError(RuntimeError):
    """The HTTP request did not reach a usable ShopSimulator response."""


class ShopEnvironmentError(RuntimeError):
    """ShopSimulator accepted HTTP request but reported an environment error."""


class ShopProtocolError(RuntimeError):
    """ShopSimulator response did not match its structured API contract."""


class ShopEnvironmentStateError(RuntimeError):
    """The client lifecycle was used out of order."""


# 环境槽位池是"显式释放"语义(environments/ShopSimulator/shop_env/shop_env/slot_lease_pool.py),
# 默认只有 8 个槽位(可用 SHOPSIM_ENV_SLOTS 调大),而 rollout.agent.num_workers 也是 8 ——
# 没有任何余量:一次瞬时重叠或一条漏释放就会让服务端返回
# "Unable to get available environment resource, please try again later",直接打挂整轮训练
# (实测在 13 步后连续复现两次)。这里对"取租约"这一步做有界退避重试:等到有空槽就继续,
# 超过窗口仍如实抛出,不会无限等待。
LEASE_BUSY_MARKER = "Unable to get available environment resource"
LEASE_RETRY_WINDOW = 120.0
LEASE_RETRY_INTERVAL = 1.0


class ShopAgentEnv:
    """一条 trajectory 独占的 ShopSimulator API 租约。"""

    def __init__(self, base_url="http://127.0.0.1:5700", timeout=60, transport=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport
        self.env_idx = None
        self.done = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.release()
        except Exception:
            if exc_type is None:
                raise
        return False

    def reset(self, task_id):
        """为任务申请环境实例，并保存服务端返回的 ``env_idx``。"""
        if self.env_idx is not None:
            raise ShopEnvironmentStateError("Environment is already leased; release it before reset")

        # reset 只负责建立租约；真正的购物动作统一走 step，便于上层记录轨迹。
        payload = {"action": "reset", "idx": int(task_id)}
        result = self._call_reset_with_retry(payload)
        env_idx = result.get("env_idx")
        if not isinstance(env_idx, int):
            raise ShopProtocolError("reset response is missing integer env_idx")
        self.env_idx = env_idx
        self.done = False
        return result

    def step(self, action):
        """执行一个已经转换好的环境动作，并更新终局状态。"""
        if not isinstance(action, str) or not action:
            raise ValueError("action must be a non-empty string")
        if self.done:
            raise ShopEnvironmentStateError("Environment is already done; release it before reset")
        result = self._call(
            {"action": "interact", "env_idx": self._leased_env_idx(), "response": action}
        )
        self.done = bool(result.get("done", False))
        return result

    def release(self):
        """释放当前租约；重复 release 是安全的空操作。"""
        if self.env_idx is None:
            return None

        env_idx = self.env_idx
        result = self._call({"action": "release_one", "env_idx": env_idx})
        self.env_idx = None
        self.done = False
        return result

    def _leased_env_idx(self):
        if self.env_idx is None:
            raise ShopEnvironmentStateError("reset must succeed before step")
        return self.env_idx

    def _call(self, payload):
        """统一处理 HTTP、环境错误和结构化协议错误。"""
        try:
            response = self._send(payload)
        except (HTTPError, URLError, OSError) as exc:
            raise ShopHttpError(f"ShopSimulator HTTP request failed: {exc}") from exc

        if not isinstance(response, dict):
            raise ShopProtocolError("ShopSimulator response must be a JSON object")
        result = response.get("result")
        if not isinstance(result, dict):
            raise ShopProtocolError("ShopSimulator response is missing object result")
        if result.get("error"):
            raise ShopEnvironmentError(str(result["error"]))
        return result

    def _call_reset_with_retry(self, payload):
        """取租约遇到"槽位池打满"时有界退避重试；其它错误立即抛出。"""

        deadline = time.monotonic() + LEASE_RETRY_WINDOW
        while True:
            try:
                return self._call(payload)
            except ShopEnvironmentError as exc:
                if LEASE_BUSY_MARKER not in str(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(LEASE_RETRY_INTERVAL)

    def _send(self, payload):
        endpoint = f"{self.base_url}/api/shop_agent"
        if self.transport is not None:
            return self.transport(endpoint, payload, self.timeout)

        body = json.dumps(payload).encode("utf-8")
        request = Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"shopping-agent/{__version__}",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
