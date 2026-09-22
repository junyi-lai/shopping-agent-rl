import unittest

from shopping_agent.training.sft.dataset import select_training_examples


class SftAblationTest(unittest.TestCase):
    def setUp(self):
        self.examples = [
            {"task_id": index, "trajectory_id": f"trajectory-{index}"}
            for index in range(20)
        ]

    def test_fixed_count_is_stable_for_the_same_seed(self):
        first = select_training_examples(self.examples, count=5, seed=17)
        second = select_training_examples(list(reversed(self.examples)), count=5, seed=17)

        self.assertEqual({item["task_id"] for item in first}, {item["task_id"] for item in second})
        self.assertEqual(len(first), 5)

    def test_ratio_selects_a_fixed_fraction(self):
        selected = select_training_examples(self.examples, ratio=0.25, seed=42)

        self.assertEqual(len(selected), 5)

    def test_count_and_ratio_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "count.*ratio"):
            select_training_examples(self.examples, count=5, ratio=0.5)


if __name__ == "__main__":
    unittest.main()
