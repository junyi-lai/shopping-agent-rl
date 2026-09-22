"""Public CPU and parameterized GRPO entry-point tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shopping_agent.training.rl.train import build_command, parse_args
from shopping_agent.smoke import run_cpu_smoke

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "00_environment.py",
    "01_collect.py",
    "02_filter.py",
    "03_sft.py",
    "04_rl.py",
    "05_evaluate.py",
)


class PublicEntrypointTest(unittest.TestCase):
    def test_cpu_smoke_covers_public_contracts(self):
        result = run_cpu_smoke()

        self.assertEqual(
            result["checks"],
            [
                "action_schema",
                "reward_summary",
                "sft_label_mask",
                "dynamic_sampling_grouping",
            ],
        )

    def test_pipeline_entrypoints_are_runnable_and_documented(self):
        """The five stages plus the environment step each expose a runnable entry."""

        for name in ENTRYPOINTS:
            script = ROOT / "scripts" / name
            self.assertTrue(script.is_file(), f"missing entry point: {name}")
            completed = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("usage:", completed.stdout)

    def test_stage_configs_are_complete_for_every_entry(self):
        from shopping_agent import config

        settings = {name: config.load_config(name) for name in ("collect", "filter", "sft", "eval")}
        for algo in ("grpo", "rloo"):
            settings[algo] = config.load_config(algo)

        # 采集配置必须自洽:任务池容量要撑得起 target-accepted,否则会在达标前
        # 就采完(静默少采)。这里约束"关系"而不是钉死某个数值。
        from shopping_agent import pipeline

        collect = settings["collect"]
        pool_tasks = int(collect["task_pool"]["target_tasks"])
        attempts = int(collect["args"]["attempts-per-task"])
        target = int(collect["args"]["target-accepted"])
        self.assertGreater(pool_tasks, 0)
        self.assertGreaterEqual(
            pool_tasks * attempts * pipeline.EXPECTED_ACCEPTANCE,
            target,
            f"采集配置不自洽:{pool_tasks} 题 × {attempts} 次 × "
            f"{pipeline.EXPECTED_ACCEPTANCE} < target-accepted={target}",
        )
        self.assertEqual(settings["filter"]["output"]["dataset"], "data/sft/all.jsonl")
        self.assertEqual(settings["sft"]["final_model_dir"], "models/sft")
        self.assertEqual(settings["grpo"]["algorithm"]["adv_estimator"], "grpo")
        self.assertEqual(settings["rloo"]["algorithm"]["adv_estimator"], "rloo")
        self.assertEqual(settings["eval"]["args"]["benchmark"], "data/evaluation/tasks.jsonl")

    def test_patch_file_matches_the_configured_patch(self):
        """The vendored patch must be the one the RL entry point applies."""

        from shopping_agent.training.rl import patch

        self.assertTrue(patch.PATCH_FILE.is_file(), patch.PATCH_FILE)
        text = patch.PATCH_FILE.read_text(encoding="utf-8")
        self.assertIn(patch.PATCH_MARKER, text)
        self.assertIn("shopping_agent.training.rl.dynamic_sampling", text)
        self.assertNotIn("shopping_grpo", text)
        self.assertEqual(patch.PROJECT_ROOT, ROOT)

    def test_public_grpo_launcher_accepts_sharded_weights_and_console(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text(
                "{}",
                encoding="utf-8",
            )
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            output = temporary / "output"
            with patch.object(
                sys,
                "argv",
                [
                    "lora.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(output),
                    "--config",
                    str(root / "configs/grpo.yaml"),
                    "--logger",
                    "console",
                    "--dry-run",
                ],
            ):
                args = parse_args()
            command, environment = build_command(args)

        self.assertIn("verl.trainer.main_ppo", command)
        self.assertEqual(environment["RL_MODEL_PATH"], str(model.resolve()))
        self.assertEqual(environment["RL_TRAIN_FILE"], str(train.resolve()))
        self.assertEqual(environment["RL_VAL_FILE"], str(validation.resolve()))
        self.assertEqual(
            environment["SHOPPING_AGENT_DIAGNOSTICS_PATH"],
            str(output.resolve() / "training_diagnostics.jsonl"),
        )
        self.assertIn("trainer.logger=[console]", command)

    def test_rl_keeps_hydra_out_of_the_repository_root(self):
        """Hydra 不许往仓库根写 outputs/,也不许在 run 目录里留任何自己的文件。

        四条 override 各管一件事:run.dir 把快照归到本次 run、output_subdir=null 不写 .hydra/、
        job_logging=stdout 不写 main_ppo.log(仍保留控制台 INFO 日志)、job.chdir=false 不切 CWD
        (否则相对路径全移位)。这样 run 目录里只剩真产物,「产物目录必须为空」的判定不会被
        Hydra 自己的残留误触发。
        """
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            train = temporary / "train.parquet"
            val = temporary / "validation.parquet"
            train.write_bytes(b"example")
            val.write_bytes(b"example")
            output = temporary / "works" / "rl" / "demo"
            args = parse_args(
                [
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(val),
                    "--output",
                    str(output),
                    "--dry-run",
                ]
            )
            command, _ = build_command(args)

            for override in (
                f"hydra.run.dir={output.resolve()}",
                "hydra.job.chdir=false",
                "hydra.output_subdir=null",
                "hydra/job_logging=stdout",
            ):
                self.assertIn(override, command)

            # 目录里只有 Hydra 的产物时,仍然要求 --resume(它们已经不写了,所以真产物才算)。
            output.mkdir(parents=True)
            (output / "training_diagnostics.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(SystemExit):
                build_command(args)

    def test_rl_dry_run_leaves_no_empty_run_directory(self):
        """`--dry-run` 只打印命令,不许在 works/rl 下留下空目录。

        回归:run_rl 过去会先把产物目录建出来,于是任何一次 dry-run 都会在
        works/rl/<label>/ 留一个空壳,让人误以为跑过一次。
        """
        from shopping_agent import config as project_config
        from shopping_agent import pipeline

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            settings = project_config.load_config("grpo")
            settings["pipeline"]["paths"]["works"] = str(temporary / "works")
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")

            captured = {}

            def fake_call(func, argv):
                captured["argv"] = argv
                return 0

            with patch.object(pipeline, "_call", fake_call):
                status = pipeline.run_rl(
                    settings, algo="rloo", model=str(model), label="demo", dry_run=True
                )

            self.assertEqual(status, 0)
            self.assertIn("--dry-run", captured["argv"])
            self.assertFalse((temporary / "works" / "rl" / "demo").exists())

    def test_eval_run_dir_mirrors_the_model_path(self):
        """评测产物按模型路径镜像:models/rl/<label> → works/eval/rl/<label>。

        这样 base/sft 与 RL 模型在 works/eval 下层次一致,也避免所有评测挤在同一层。
        """
        from shopping_agent import config as project_config
        from shopping_agent import pipeline

        with tempfile.TemporaryDirectory() as tmp:
            settings = project_config.load_config("eval")
            settings["pipeline"]["paths"]["works"] = tmp
            works = Path(tmp).resolve()

            for model, tail in (
                ("models/rl/rloo_v6_lr3e-5_200step", "eval/rl/rloo_v6_lr3e-5_200step"),
                ("models/sft", "eval/sft"),
                ("models/base", "eval/base"),
            ):
                self.assertEqual(pipeline.eval_run_dir(settings, model), works / tail)

            # 显式 --label 优先,用于 models/ 之外的模型
            outside = pipeline.eval_run_dir(settings, "/tmp/somewhere/m", label="custom")
            self.assertEqual(outside, works / "eval" / "custom")

    def test_rl_run_name_keeps_works_and_models_in_sync(self):
        """一次 run 的名字必须让 works/rl/<名> 与 models/rl/<名> 同名。

        回归测试:models 侧的导出目录曾经写死成 ``<algo>``,于是每次 RL run 都覆盖
        上一次,只能事后手动改名——这正是 works/rl 与 works/eval 命名混乱的来源。
        """
        from shopping_agent import pipeline

        self.assertEqual(
            pipeline.rl_run_name(algo="rloo", label="rloo_v6_lr3e-5_200step"),
            "rloo_v6_lr3e-5_200step",
        )
        self.assertEqual(
            pipeline.rl_run_name(algo="rloo", output="works/rl/my_run"),
            "my_run",
        )
        self.assertEqual(
            pipeline.rl_run_name(algo="rloo", output="works/rl/ignored", label="explicit"),
            "explicit",
        )
        self.assertEqual(pipeline.rl_run_name(algo="grpo"), "grpo")

    def test_rl_resume_allows_non_empty_output_and_pins_resume_mode(self):
        """非空产物目录默认拒绝;显式 --resume 时放开,并把 resume_mode 写进命令。

        回归测试:该检查曾经既挡误覆盖、也挡掉了 veRL 的续训能力(--resume 之前不存在,
        只能改代码绕过)。
        """
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            output = temporary / "output"
            (output / "global_step_50").mkdir(parents=True)  # 非空 + 已有 checkpoint

            def command_for(*extra):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "04_rl.py",
                        "--model",
                        str(model),
                        "--train-data",
                        str(train),
                        "--val-data",
                        str(validation),
                        "--output",
                        str(output),
                        "--config",
                        str(root / "configs/rloo.yaml"),
                        "--logger",
                        "console",
                        *extra,
                    ],
                ):
                    return build_command(parse_args())

            with self.assertRaises(SystemExit) as caught:
                command_for()
            self.assertIn("--resume", str(caught.exception))

            command, _ = command_for("--resume")
            self.assertIn("verl.trainer.main_ppo", command)
            self.assertIn("trainer.resume_mode=auto", command)


if __name__ == "__main__":
    unittest.main()
