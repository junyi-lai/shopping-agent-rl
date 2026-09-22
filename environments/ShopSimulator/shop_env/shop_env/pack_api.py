"""HTTP service exposing ShopSimulator Environment v2.1 over ``/api/shop_agent``.

One leased slot per trajectory: ``reset`` acquires a slot, ``interact`` applies
one action, ``release_one`` frees it.  The service only ever returns raw
environment facts; scoring lives in the project's ``shopping_agent.reward``.
"""
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from flask import Flask, request, jsonify, Response

SHOP_ENV_ROOT = Path(__file__).resolve().parents[1]
if str(SHOP_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(SHOP_ENV_ROOT))

from shop_agent import shop_agent  # noqa: E402
from slot_lease_pool import SlotLeasePool  # noqa: E402
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv  # noqa: E402

# Constants
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 5
DEFAULT_ENV_MAX_NUM = int(os.environ.get("SHOPSIM_ENV_SLOTS", "8"))
SERVER_HOST = os.environ.get("SHOPSIM_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SHOPSIM_PORT", "5700"))

# Global variables
envs: List[Any] = []
env_max_num: int = DEFAULT_ENV_MAX_NUM
slot_pool = SlotLeasePool(env_max_num)

logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/api/shop_agent', methods=['POST'])
def shop_agent_api() -> Response:
    """
    API endpoint for shop agent operations.

    Handles three types of actions:
    - release_all: Release all environments
    - release_one: Release a specific environment
    - reset/interact: Process shop agent actions

    Returns:
        JSON response with result or error message
    """
    data = request.json
    if data is None:
        logger.error("[Error] No JSON data provided in request")
        return jsonify({'result': {"error": "No JSON data provided"}})

    action = data.get('action')
    env_idx = data.get('env_idx', None)
    response = data.get('response', None)
    idx = data.get('idx', None)
    try:
        # Release all environments
        if action == 'release_all':
            slot_pool.reset(env_max_num)
            logger.info("[Init] All environments have been initialized")
            return jsonify({'result': {"message": "All environments have been initialized"}})

        # Release one environment
        if action == 'release_one':
            if env_idx is not None and isinstance(env_idx, int):
                was_leased = slot_pool.release(env_idx)
                if was_leased:
                    logger.info(f"[Release] Environment {env_idx} has been released")
                    return jsonify({'result': {"message": f"Environment {env_idx} has been released"}})
                else:
                    logger.warning(f"[Release] Environment {env_idx} is already free, no need to release again")
                    return jsonify({'result': {"message": f"Environment {env_idx} is already free"}})
            else:
                logger.error("[Error] No valid environment index provided")
                return jsonify({'result': {"error": "No valid environment index provided"}})

        # If env_idx is not provided, assign an available env_idx
        if env_idx is None:
            retry_count = 0
            while retry_count < MAX_RETRIES:
                env_idx = slot_pool.acquire()
                if env_idx is not None:
                    break
                retry_count += 1
                logger.info(f"[Retry {retry_count}/{MAX_RETRIES}] No available environment index, retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                # Reached max retries but still couldn't get env_idx
                logger.error("[Error] Reached max retries, unable to get available environment")
                return jsonify({'result': {'error': 'Unable to get available environment resource, please try again later'}})

        # Call shop_agent function
        result = shop_agent(
            envs[env_idx],
            env_idx,
            action,
            idx,
            response,
        )

        # The caller owns the lease until release_one.  Auto-releasing here
        # races with the caller's finally-release: another worker can lease
        # this slot between the two releases and then have its active lease
        # accidentally freed by the previous worker.
        if result.get('over'):
            logger.info(
                f"[Task Over] Environment {env_idx} is awaiting explicit release"
            )

    except Exception as e:
        logger.exception(f"[Exception] Exception occurred while processing request: {str(e)}")
        return jsonify({'result': {'error': str(e)}})

    return jsonify({'result': result})


def initialize_environments() -> None:
    """
    Initialize all environments and add them to the free environment index.
    """
    global envs, env_max_num

    envs = []
    slot_pool.reset(env_max_num)

    shared_server = None
    for i in range(env_max_num):
        logger.info(f"Environment {i} is being initialized")
        env = WebAgentTextEnv(
            observation_mode='text',
            server=shared_server,
            session_prefix=f"slot-{i}",
        )
        if shared_server is None:
            shared_server = env.server
        envs.append(env)


if __name__ == '__main__':
    initialize_environments()
    app.run(host=SERVER_HOST, port=SERVER_PORT)
