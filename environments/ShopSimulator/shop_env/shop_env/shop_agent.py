import json
import logging
from typing import Dict, Any, Optional

# The service redirects stdout to its own log file, so one stream handler is
# enough for the whole process.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

logger = logging.getLogger(__name__)


def _handle_reset_action(
    env: Any,
    env_idx: int,
    task_idx: Optional[int],
) -> Dict[str, Any]:
    """
    Handle reset action

    Args:
        env: Environment object
        env_idx: Environment index
        task_idx: Task index

    Returns:
        Dictionary containing reset information
    """
    logger.info(f"[Reset] Starting task {task_idx}, environment index: {env_idx}")
    message = f"Task {task_idx} started"
    env.reset(idx=task_idx)
    return {
        'instruction': env.instruction_text,
        'message': message,
        'env_idx': env_idx,
        'idx': task_idx,
        'environment_version': getattr(
            env.server,
            "environment_version",
            "shopsimulator-environment-v2.1",
        ),
        "observation_state": env.structured_observation(),
    }


def _format_available_actions(available_actions: Dict[str, Any]) -> str:
    """
    Format available actions information

    Args:
        available_actions: Dictionary of available actions

    Returns:
        Formatted action text
    """
    clickables = available_actions['clickables'].copy()
    if "search" in clickables:
        clickables.remove("search")

    return (
        f"\n\n搜索功能是否可用: {available_actions['has_search_bar']}"
        f"\n\n可点击的按钮: {json.dumps(clickables, ensure_ascii=False)}"
    )


def _handle_interact_action(
    env: Any,
    env_idx: int,
    response: str
) -> Dict[str, Any]:
    """
    Handle interact action

    Args:
        env: Environment object
        env_idx: Environment index
        response: Response string

    Returns:
        Dictionary containing interaction information
    """
    logger.info(
        f"[Interact] Received response: {response}, "
        f"environment index: {env_idx}, Session ID: {env.session}"
    )

    # The caller always sends one already-normalized action string.
    action_str = response.replace("\\n", "\n")

    # Execute environment step
    observation, status, info = env.step(action_str)
    done = status.get('done', False)
    reward = float(status.get('reward', 0.0))

    # Get and format available actions
    available_actions = env.get_available_actions()
    action_text = _format_available_actions(available_actions)
    observation = observation + action_text

    # Extract raw terminal facts (no reward computation here; scoring lives in
    # the project's shopping_agent.reward package).
    if done:
        purchase = status.get('purchase', {})
        purchased_product = status.get('purchased_product', {})
        goal = status.get('goal', {})
        target_product = status.get('target_product', {})
        options = status.get('options', {})
        price = status.get('price')
        price_resolution = status.get('price_resolution')
        abstain_facts = status.get('abstain_facts')
    else:
        purchase = {}
        purchased_product = {}
        goal = {}
        target_product = {}
        options = {}
        price = None
        price_resolution = None
        abstain_facts = None

    # Build return information
    return_info = {
        "done": done,
        "reward": reward,
        "instruction": observation,
        "message": "Continue interaction",
        "env_idx": env_idx,
        "idx": env.session,
        "purchase": purchase,
        "purchased_product": purchased_product,
        "goal": goal,
        "target_product": target_product,
        "options": options,
        "price": price,
        "price_resolution": price_resolution,
        # `over` mirrors `done`: the caller owns the lease until release_one.
        "over": done,
        "observation_state": env.structured_observation(),
    }
    if done:
        return_info["termination_reason"] = status.get(
            "termination_reason", "environment_done"
        )
    if abstain_facts is not None:
        return_info["abstain_facts"] = abstain_facts
    if status.get("progress") is not None:
        return_info["progress"] = status["progress"]

    return return_info


def shop_agent(
    env: Any,
    env_idx: int,
    action: str,
    idx: Optional[int] = None,
    response: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main shop agent function that handles environment reset and interaction actions

    Args:
        env: Environment object
        env_idx: Environment index
        action: Action type ("reset" or "interact")
        idx: Task index, only used for reset action
        response: Response string, only used for interact action

    Returns:
        Dictionary containing action processing results

    Raises:
        ValueError: When action is not "reset" or "interact"
    """
    if action == "reset":
        if idx is None:
            raise ValueError("reset action requires idx parameter")
        return _handle_reset_action(
            env,
            env_idx,
            idx,
        )

    elif action == "interact":
        if response is None:
            raise ValueError("interact action requires response parameter")
        return _handle_interact_action(env, env_idx, response)

    else:
        raise ValueError(f"Unknown action type: {action}, supported actions: 'reset', 'interact'")
