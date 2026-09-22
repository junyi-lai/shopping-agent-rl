import unittest

from shopping_agent.environment.manifest import (
    MANIFEST_VERSION,
    validate_manifest,
)


class EnvironmentManifestTest(unittest.TestCase):
    def test_current_environment_contract_is_validated(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "environment_version": "shopsimulator-environment-v2.1",
            "shopsimulator_commit": "a" * 40,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopping-reward-v4"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "shopping-tools-v2",
            "max_steps": 35,
            "seed": 20260726,
        }
        self.assertIs(validate_manifest(manifest), manifest)

    def test_page_size_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_manifest({})

    def test_current_environment_requires_reward_v4(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "environment_version": "shopsimulator-environment-v2.1",
            "shopsimulator_commit": "a" * 40,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopping-reward-v4"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "shopping-tools-v2",
            "max_steps": 35,
            "seed": 20260726,
        }
        self.assertIs(validate_manifest(manifest), manifest)
        manifest["reward"] = {"version": "unsupported-reward"}
        with self.assertRaisesRegex(ValueError, "requires shopping-reward-v4"):
            validate_manifest(manifest)

    def test_wrong_tool_contract_is_rejected(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "shopsimulator_commit": "a" * 40,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopping-reward-v4"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "unsupported-tools",
            "max_steps": 35,
            "seed": 20260726,
        }
        with self.assertRaisesRegex(ValueError, "Tool v2"):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
