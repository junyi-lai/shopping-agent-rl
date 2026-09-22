"""Build task goals (instruction + gold metadata) for ShopSimulator sessions.

The environment is a pure simulator: goals carry raw task facts only. All
scoring (terminal classification + Reward v4 shaping) lives in the project's
``shopping_agent.reward`` package.

The goal list is positional: a task id indexes this list, so the iteration order
and the set of emitted goals are part of the frozen Environment v2.1 contract.
"""


def get_goals(all_products):
    """Return one goal per instruction that declares at least one attribute."""
    goals = []
    for item in all_products:
        if 'instructions' not in item:
            continue
        for product in item['instructions']:
            attributes = product.get('attributes', [])
            if len(attributes) == 0:
                continue
            goals.append(
                {
                    'asin': item['asin'],
                    'category': item['category'],
                    'query': item['query'],
                    'name': item['title'],
                    'instruction_text': product['instruction'],
                    'attributes': attributes,
                    'goal_options': product['instruction_options'],
                    'weight': 1,
                }
            )
    return goals
