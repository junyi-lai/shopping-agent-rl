import unittest

from web_agent_site.engine.observation import build_observation_state


class ObservationV2Test(unittest.TestCase):
    def test_search_page_preserves_all_twenty_products_and_ranks(self):
        products = {
            f"{index:012d}": {
                "asin": f"{index:012d}",
                "title": f"商品 {index}",
                "category": "测试",
                "pricing": [index],
            }
            for index in range(1, 41)
        }
        state = build_observation_state(
            page_type="search_results",
            session={
                "keywords": ["商品"],
                "normalized_query": "商品",
                "page": 2,
                "total_pages": 2,
                "total_results": 40,
                "search_result_asins": list(products),
                "current_page_asins": list(products)[20:40],
            },
            product_item_dict=products,
            available_actions={
                "has_search_bar": False,
                "clickables": ["< prev", *list(products)[20:40]],
            },
        )
        self.assertEqual(len(state["products"]), 20)
        self.assertEqual(state["rank_start"], 21)
        self.assertEqual(state["rank_end"], 40)
        self.assertEqual(
            {product["asin"] for product in state["products"]},
            set(state["actions"]) - {"< prev"},
        )

    def test_builder_has_no_goal_parameter_or_hidden_answer(self):
        state = build_observation_state(
            page_type="search_home",
            session={},
            product_item_dict={},
            available_actions={"has_search_bar": True, "clickables": []},
        )
        self.assertNotIn("goal", state)
        self.assertNotIn("reward", state)


if __name__ == "__main__":
    unittest.main()
