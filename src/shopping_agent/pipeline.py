"""Stage orchestration for the five pipeline entry points.

``scripts/NN_*.py`` only parse a handful of overrides and call one function here.
Everything that glues the pipeline together lives in this module: task-pool
maintenance for incremental collection, filter/curriculum/RL-dataset assembly,
SFT archiving, RL patch + export, and evaluation serving + reporting.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from shopping_agent import config as project_config
from shopping_agent.collection import collect as collect_stage
from shopping_agent.collection import curriculum as curriculum_stage
from shopping_agent.collection import filter as filter_stage
from shopping_agent.collection import labels as labels_stage
from shopping_agent.collection import rl_dataset
from shopping_agent.collection import task_pool as task_pool_module
from shopping_agent.environment import service as environment_service
from shopping_agent.evaluation import benchmark as benchmark_stage
from shopping_agent.evaluation import comparison_report, recurate, report
from shopping_agent.training.rl import export as grpo_export
from shopping_agent.training.rl import patch as grpo_patch
from shopping_agent.training.rl import train as grpo_train
from shopping_agent.training.sft import train as sft_train

section = project_config.section
resolve_path = project_config.resolve_path

# 采集的经验验收率(只接受 Reward v4 gold_purchase 的完整轨迹,历史约 0.41)。
# 仅用于在任务池容量撑不起 target-accepted 时给出警告,不参与任何打分。
EXPECTED_ACCEPTANCE = 0.4


def _call(module_main, argv: Sequence[str]) -> int:
    """Run a stage ``main`` and normalize its exit code (some raise SystemExit)."""

    try:
        result = module_main(list(argv))
    except SystemExit as exc:  # argparse/SystemExit 风格的历史入口
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        # 字符串形式的 SystemExit 是环节给出的**人类可读报错**,必须原样打印:
        # 直接返回 1 会让入口脚本"静默失败"(只退出、不解释原因)。
        print(code, file=sys.stderr)
        return 1
    return int(result or 0)


def _require_file(path: Path, hint: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"缺少 {path};{hint}")
    return path


def export_reward_config(
    config: Mapping[str, Any], *, version: str | None = None
) -> dict[str, Any]:
    """Publish ``pipeline.yaml``'s reward section to every reward consumer.

    Filtering, RL training and evaluation all score trajectories with the same
    reward package; exporting the shaping parameters once keeps them on one set
    of numbers. ``version`` pins one reward contract ("v4"/"v5"); when omitted
    the pipeline's ``reward.version`` decides, and only RL honours that knob —
    filtering and evaluation pass "v4" explicitly so their frozen metrics never
    drift with an RL experiment.
    """

    active = version or project_config.reward_version(config)
    shaping = project_config.reward_shaping(config, version=active)
    if shaping:
        os.environ["SHOPPING_AGENT_REWARD_CONFIG"] = json.dumps(shaping, sort_keys=True)
    os.environ["SHOPPING_AGENT_REWARD_VERSION"] = active
    terminal = project_config.reward_terminal_overrides(config, version=active)
    if terminal:
        os.environ["SHOPPING_AGENT_REWARD_TERMINAL"] = json.dumps(terminal, sort_keys=True)
    else:
        os.environ.pop("SHOPPING_AGENT_REWARD_TERMINAL", None)
    return shaping


def ensure_task_pool(
    path: Path,
    *,
    target_tasks: int,
    seed: int,
    exclude_paths: Iterable[Path],
    stratify_by: str | None = None,
) -> list[int]:
    """Grow the deterministic task pool without ever dropping existing tasks.

    ``stratify_by="category"`` allocates slots uniformly across top-level product
    categories instead of inheriting the catalog's natural distribution.
    """

    excluded: set[int] = set()
    for exclude in exclude_paths:
        excluded |= task_pool_module.jsonl_task_ids(exclude)
    existing = task_pool_module.jsonl_task_ids(path) if path.is_file() else set()
    if len(existing) >= int(target_tasks):
        return sorted(existing)
    if stratify_by == "category":
        chosen = task_pool_module.sample_pool_stratified(
            int(target_tasks), seed=seed, excluded=excluded
        )
    else:
        chosen = task_pool_module.sample_pool(int(target_tasks), seed=seed, excluded=excluded)
    selected = set(chosen)
    selected |= existing
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task_id in sorted(selected):
            handle.write(json.dumps({"task_id": int(task_id)}) + "\n")
    return sorted(selected)


def select_dataset_rows(
    rows: Sequence[Mapping[str, Any]],
    quality: Sequence[Mapping[str, Any]],
    *,
    min_total: float | None,
    keep_top: int | None,
) -> list[dict]:
    """Keep the best V4-scored trajectory per task, honouring the quality knobs."""

    best: dict[int, tuple[float, Mapping[str, Any]]] = {}
    for item in quality:
        task_id = int(item["task_id"])
        score = float(item["v4_total"])
        if task_id not in best or score > best[task_id][0]:
            best[task_id] = (score, item)

    ranked = sorted(best.values(), key=lambda pair: pair[0], reverse=True)
    if keep_top is not None and int(keep_top) > 0:
        ranked = ranked[: int(keep_top)]
    elif min_total is not None:
        ranked = [pair for pair in ranked if pair[0] >= float(min_total)]

    keep_keys = {
        (int(item["task_id"]), item.get("trajectory_id")) for _, item in ranked
    }
    selected = [
        dict(row)
        for row in rows
        if (int(row["task_id"]), row.get("trajectory_id")) in keep_keys
    ]
    selected.sort(key=lambda row: int(row["task_id"]))
    return selected


# --------------------------------------------------------------------------- 01 采集


def run_collect(
    config: Mapping[str, Any],
    *,
    target_accepted: int | None = None,
    target_tasks: int | None = None,
    workers: int | None = None,
    attempts_per_task: int | None = None,
    build_only: bool = False,
    dry_run: bool = False,
) -> int:
    pool = section(config, "task_pool", default={}) or {}
    pool_path = resolve_path(pool.get("path", "works/collect/task_pool.jsonl"))
    exclude_paths = [resolve_path(item) for item in pool.get("exclude", []) or []]
    seed = int(pool.get("seed", 20260814))
    environment_url = project_config.environment_url(config)

    if not build_only:
        if not dry_run:
            environment_service.ensure_environment(environment_url)
        task_ids = ensure_task_pool(
            pool_path,
            target_tasks=int(target_tasks or pool.get("target_tasks", 600)),
            seed=seed,
            exclude_paths=exclude_paths,
            stratify_by=pool.get("stratify_by"),
        )
        print(f"任务池:{pool_path}({len(task_ids)} 个任务,已采任务会自动跳过)")
        # 池子容量必须撑得起目标条数,否则会在达到 target-accepted 前就采完(静默少采)。
        stage_args = section(config, "args", default={}) or {}
        planned_attempts = len(task_ids) * int(
            attempts_per_task or stage_args.get("attempts-per-task", 1) or 1
        )
        target = int(target_accepted or stage_args.get("target-accepted", 0) or 0)
        if target and planned_attempts * EXPECTED_ACCEPTANCE < target:
            print(
                f"警告:任务池容量可能不足——最多 {planned_attempts} 次尝试 × 经验验收率 "
                f"{EXPECTED_ACCEPTANCE:.0%} ≈ {int(planned_attempts * EXPECTED_ACCEPTANCE)} 条,"
                f"低于 target-accepted={target}。请调大 task_pool.target_tasks "
                f"或 args.attempts-per-task,否则会在达标前就采完。"
            )

    inject = {
        "tasks": str(pool_path),
        "held-out-tasks": str(exclude_paths[0]) if exclude_paths else None,
        "base-url": environment_url,
        "max-steps": project_config.max_steps(config),
    }
    overrides = {
        "target-accepted": target_accepted,
        "workers": workers,
        "attempts-per-task": attempts_per_task,
        "build-only": True if build_only else None,
    }
    argv = project_config.stage_argv(config, inject=inject, overrides=overrides)
    if dry_run:
        print("采集命令参数:" + " ".join(argv))
        return 0
    return _call(collect_stage.main, argv)


# --------------------------------------------------------------------------- 02 筛选


def run_filter(config: Mapping[str, Any], *, skip_labels: bool = False, skip_rl_data: bool = False) -> int:
    # 筛选重打分永远用 v4 冻结契约;RL 的 v5 实验不影响已冻结数据。
    export_reward_config(config, version="v4")
    rescoring = section(config, "rescoring", default={}) or {}
    output = section(config, "output", default={}) or {}
    work_dir = resolve_path(output.get("work_dir", "works/filter"))
    work_dir.mkdir(parents=True, exist_ok=True)

    source = _require_file(
        resolve_path(rescoring.get("input", "works/collect/accepted.jsonl")),
        "请先运行 01 采集",
    )
    rescore_argv = [
        "--input",
        str(source),
        "--output-dir",
        str(work_dir),
        "--max-steps",
        str(int(rescoring.get("max-steps", project_config.max_steps(config)))),
    ]
    if rescoring.get("min-total") is not None:
        rescore_argv += ["--min-total", str(rescoring["min-total"])]
    if rescoring.get("keep-top") is not None:
        rescore_argv += ["--keep-top", str(rescoring["keep-top"])]
    print("== 1/5 V4 重打分 ==")
    status = _call(filter_stage.main, rescore_argv)
    if status != 0:
        return status

    rows = project_config.read_jsonl(source)
    quality = project_config.read_jsonl(work_dir / "quality.jsonl")
    selected = select_dataset_rows(
        rows,
        quality,
        min_total=rescoring.get("min-total"),
        keep_top=rescoring.get("keep-top"),
    )
    dataset_path = resolve_path(output.get("dataset", "data/sft/all.jsonl"))
    project_config.write_jsonl(dataset_path, selected)
    print(f"== 2/5 SFT 数据:{dataset_path}({len(selected)} 条)")

    # 冻结盲测集路径固定(data/evaluation/tasks.jsonl),既不入 SFT 也不入 RL 任务池
    evaluation_tasks = resolve_path("data/evaluation/tasks.jsonl")

    labels_path = resolve_path(output.get("labels", "data/sft/difficulty_labels.jsonl"))
    label_settings = section(config, "labels", default={}) or {}
    if label_settings.get("enabled", True) and not skip_labels:
        print("== 3/5 难度标签(只补缺失任务)==")
        status = _call(
            labels_stage.main,
            [
                "--input",
                str(dataset_path),
                "--output",
                str(labels_path),
                "--model",
                str(label_settings.get("model", "deepseek-v4-flash")),
                "--workers",
                str(int(label_settings.get("workers", 3))),
                "--batch-size",
                str(int(label_settings.get("batch-size", 5))),
                "--max-batch-chars",
                str(int(label_settings.get("max-batch-chars", 60000))),
            ],
        )
        if status != 0:
            return status
    elif not labels_path.is_file():
        raise SystemExit(f"缺少难度标签 {labels_path};请开启 labels.enabled 或先补齐标签")

    curriculum_path = resolve_path(output.get("curriculum", "data/sft/curriculum.json"))
    print("== 4/5 课程清单 ==")
    status = _call(
        curriculum_stage.main,
        [
            "--source",
            str(dataset_path),
            "--labels",
            str(labels_path),
            "--evaluation",
            str(evaluation_tasks),
            "--output",
            str(curriculum_path),
            "--seed",
            str(int(section(config, "pipeline", "seed", default=20260814) or 20260814)),
        ],
    )
    if status != 0:
        return status

    rl_settings = section(config, "rl_dataset", default={}) or {}
    if rl_settings.get("enabled", True) and not skip_rl_data:
        print("== 5/5 RL 任务集 ==")
        environment_url = project_config.environment_url(config)
        environment_service.ensure_environment(environment_url)
        rl_dir = resolve_path(rl_settings.get("dir", "data/rl"))
        status = _call(
            rl_dataset.main,
            [
                "--output-dir",
                str(rl_dir),
                "--train-count",
                str(int(rl_settings.get("train_count", 1000))),
                "--validation-count",
                str(int(rl_settings.get("validation_count", 50))),
                "--seed",
                str(int(rl_settings.get("seed", 20260808))),
                "--base-url",
                environment_url,
                "--source",
                str(dataset_path),
                "--exclude",
                str(evaluation_tasks),
            ],
        )
        if status != 0:
            return status

    target = int(section(config, "targets", "min_accepted", default=0) or 0)
    dataset_ids = {int(row["task_id"]) for row in selected}
    pool_path = resolve_path("works/collect/task_pool.jsonl")
    pool_ids = task_pool_module.jsonl_task_ids(pool_path) if pool_path.is_file() else set()
    missing_ids = sorted(pool_ids - dataset_ids)
    if missing_ids:
        project_config.write_jsonl(
            work_dir / "missing_tasks.jsonl",
            [{"task_id": int(task_id)} for task_id in missing_ids],
        )
    scoring_totals = sorted(float(item.get("v4_total", 0.0)) for item in quality)
    mean_shaping = {
        key: (
            sum(float((item.get("shaping") or {}).get(key, 0.0)) for item in quality)
            / len(quality)
            if quality
            else 0.0
        )
        for key in ("efficiency_bonus", "evidence_bonus", "repeat_penalty")
    }
    summary = {
        "stage": "filter",
        "source": str(source),
        "candidates": len(rows),
        "dataset": {"path": str(dataset_path), "rows": len(selected)},
        "task_pool": {
            "path": str(pool_path),
            "tasks": len(pool_ids),
            "missing_task_count": len(missing_ids),
            "missing_tasks": missing_ids[:50],
        },
        "targets": {"min_accepted": target, "shortfall": max(target - len(selected), 0)},
        "scoring": {
            "min_total": rescoring.get("min-total"),
            "keep_top": rescoring.get("keep-top"),
            "quality_report": str(work_dir / "quality.jsonl"),
            "v4_total": (
                {
                    "min": scoring_totals[0],
                    "median": scoring_totals[len(scoring_totals) // 2],
                    "max": scoring_totals[-1],
                    "mean": sum(scoring_totals) / len(scoring_totals),
                }
                if scoring_totals
                else None
            ),
            "mean_shaping": mean_shaping,
        },
    }
    project_config.write_json(work_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if target and len(selected) < target:
        print(
            f"\n合格数据不足:当前 {len(selected)} 条,目标 {target} 条,还差 {target - len(selected)} 条。\n"
            "请调大 configs/collect.yaml 的 target_tasks / target-accepted(或 attempts-per-task),"
            "再运行 01 采集(增量补采)与 02 筛选。"
        )
    return 0


# --------------------------------------------------------------------------- 03 SFT


def _link_or_copy_tree(source: Path, target: Path) -> Path:
    """Archive a merged model with hard links when possible (same filesystem)."""

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            import os

            os.link(path, destination)
        except OSError:
            shutil.copy2(path, destination)
    return target


def run_sft(
    config: Mapping[str, Any],
    *,
    start_stage: str | None = None,
    stop_after_stage: str | None = None,
    swanlab: bool | None = None,
    dry_run: bool = False,
) -> int:
    base_model = project_config.base_model_reference(config)
    print(f"SFT 起点模型:{base_model}")
    overrides: dict[str, Any] = {
        "start-stage": start_stage,
        "stop-after-stage": stop_after_stage,
        "dry-run": True if dry_run else None,
    }
    if swanlab is not None:
        overrides["swanlab"] = bool(swanlab)
    argv = project_config.stage_argv(config, inject={"base-model": base_model}, overrides=overrides)
    status = _call(sft_train.main, argv)
    if status != 0 or dry_run:
        return status

    final_stage = str(stop_after_stage or (config.get("args") or {}).get("stop-after-stage") or "c")
    merged = resolve_path((config.get("args") or {}).get("output-root", "works/sft")) / f"stage-{final_stage}" / "merged"
    if not merged.is_dir():
        raise SystemExit(f"课程结束后未找到合并模型:{merged}")
    target = resolve_path(config.get("final_model_dir", "models/sft"))
    _link_or_copy_tree(merged, target)
    print(f"SFT 模型已归档:{target}(源:{merged})")
    return 0


# --------------------------------------------------------------------------- 04 RL


def rl_run_name(*, algo: str, output: str | None = None, label: str | None = None) -> str:
    """Return the name shared by ``works/rl/<name>`` and ``models/rl/<name>``.

    Priority: explicit ``--label`` → the ``--output`` directory name → the
    algorithm name. Deriving from ``--output`` keeps the training directory and
    the exported model directory in sync, so archived runs never collide.
    """

    name = (label or "").strip()
    if not name and output:
        name = Path(str(output)).name
    return name or algo


def run_rl(
    config: Mapping[str, Any],
    *,
    algo: str = "grpo",
    model: str | None = None,
    output: str | None = None,
    logger: str = "console",
    extra_args: Sequence[str] = (),
    dry_run: bool = False,
    resume: bool = False,
    label: str | None = None,
) -> int:
    """Run one RL job.

    ``label`` names the run and keeps the three artifact trees in sync:

        ``works/rl/<label>``  ≡  ``models/rl/<label>``  ≡  ``works/eval/rl/<label>``

    (the eval directory mirrors the model path, so evaluating
    ``models/rl/<label>`` writes ``works/eval/rl/<label>`` automatically). Without
    ``--label`` the name falls back to the output directory's name and finally to
    the algorithm name, so a plain ``04_rl.py rloo`` keeps its historical
    ``works/rl/rloo`` + ``models/rl/rloo`` layout.
    """
    if algo not in {"grpo", "rloo"}:
        raise SystemExit(f"不支持的算法:{algo}(可选 grpo、rloo)")

    export_reward_config(config)
    environment_url = project_config.environment_url(config)
    if not dry_run:
        environment_service.ensure_environment(environment_url)
        print("== 应用/校验 veRL 动态采样补丁 ==")
        grpo_patch.main([])

    model_dir = model or str(project_config.models_dir(config, "sft"))
    if not (Path(model_dir) / "config.json").is_file():
        if dry_run:
            print(f"提示:RL 起点模型尚不存在({model_dir});请先运行 03 SFT。")
            return 0
        raise SystemExit(f"RL 起点模型不存在或未归档:{model_dir}(请先运行 03 SFT)")

    data_dir = project_config.pipeline_paths(config)["data"] / "rl"
    # 一次 run 一个名字:works/rl/<name> 与 models/rl/<name> 必须同名,
    # 否则多次实验会互相覆盖(models 侧以前写死成 <algo>,只能事后手动改名)。
    run_name = rl_run_name(algo=algo, output=output, label=label)
    # create=False:目录由真正开跑的那一方(train.py)创建,否则 `--dry-run` 这种只打印命令的调用
    # 也会在 works/rl 下留下一个空目录。
    run_dir = (
        resolve_path(output)
        if output
        else project_config.works_dir(config, "rl", run_name, create=False)
    )
    argv = [
        "--model",
        str(model_dir),
        "--train-data",
        str(data_dir / "train.parquet"),
        "--val-data",
        str(data_dir / "validation.parquet"),
        "--env-url",
        environment_url,
        "--output",
        str(run_dir),
        "--config",
        str(config.get("config_path") or (project_config.CONFIG_DIR / f"{algo}.yaml")),
        "--experiment-name",
        f"shopping-agent-{algo}",
        "--logger",
        logger,
    ]
    if dry_run:
        argv.append("--dry-run")
    if resume:
        argv.append("--resume")
    argv += list(extra_args)

    status = _call(grpo_train.main, argv)
    if status != 0 or dry_run:
        return status

    target = project_config.models_dir(config, "rl", run_name, create=True)
    checkpoint = grpo_export.latest_actor_checkpoint(run_dir)
    print(f"== 合并 RL 检查点 {checkpoint} → {target} ==")
    return grpo_export.export_checkpoint(checkpoint, target)


# --------------------------------------------------------------------------- 05 测评


def eval_run_dir(
    config: Mapping[str, Any], model_dir: str | Path, *, label: str | None = None
) -> Path:
    """Return the evaluation output directory for one model.

    The model's path is **mirrored** under ``works/eval`` so the three trees line
    up one-to-one::

        models/base          → works/eval/base
        models/sft           → works/eval/sft
        models/rl/<label>    → works/eval/rl/<label>

    An explicit ``--label`` overrides the mirroring (and is the only way to name
    a model that lives outside ``models/``).
    """

    if label:
        return project_config.works_dir(config, "eval", label)
    model_path = Path(str(model_dir))
    try:
        parts = model_path.resolve().relative_to(
            project_config.models_dir(config).resolve()
        ).parts
    except ValueError:
        parts = ()
    return project_config.works_dir(
        config, "eval", *(parts or (model_path.name or "model",))
    )


def run_evaluate(
    config: Mapping[str, Any],
    model_dir: str | Path,
    *,
    label: str | None = None,
    no_serve: bool = False,
    extra_args: Sequence[str] = (),
    recurate_only: bool = False,
) -> int:
    # 测评口径永远锁 v4,保证 v4/v5 训出的模型用同一把尺子比 strict success。
    export_reward_config(config, version="v4")
    environment_url = project_config.environment_url(config)
    if not recurate_only:
        environment_service.ensure_environment(environment_url)
    else:
        return _call(
            recurate.main,
            [
                "--tasks",
                str(resolve_path((config.get("args") or {}).get("benchmark", "data/evaluation/tasks.jsonl"))),
                "--output",
                str(project_config.works_dir(config, "eval", "recuration") / "reward_v4_audit.jsonl"),
            ],
        )

    run_dir = eval_run_dir(config, model_dir, label=label)
    print(f"被测模型:{model_dir}(输出 {run_dir})")

    serving_settings = section(config, "pipeline", "serving", default={}) or {}
    base_argv = project_config.stage_argv(
        config,
        inject={
            "base-url": environment_url,
            "output": str(run_dir / "trajectories.jsonl"),
            "summary": str(run_dir / "summary.json"),
        },
        overrides=None,
    )

    def _evaluate(endpoint: str, served_name: str, api_key: str) -> int:
        argv = [
            *base_argv,
            "--llm-base-url",
            endpoint,
            "--model",
            served_name,
            "--api-key",
            api_key,
            *extra_args,
        ]
        return _call(benchmark_stage.main, argv)

    if no_serve:
        endpoint = str(
            section(config, "args", "llm-base-url", default=None)
            or os.environ.get("LLM_BASE_URL")
            or "http://127.0.0.1:8000/v1"
        )
        served_name = str(
            section(config, "args", "model", default=None)
            or os.environ.get("SERVED_MODEL_NAME")
            or serving_settings.get("served_model_name", "shopping-agent")
        )
        api_key = str(
            section(config, "args", "api-key", default=None)
            or os.environ.get("LLM_API_KEY")
            or "EMPTY"
        )
        status = _evaluate(endpoint, served_name, api_key)
    else:
        from shopping_agent import serving

        with serving.served_model(
            model_dir,
            port=int(serving_settings.get("port", 8000)),
            host=str(serving_settings.get("host", "127.0.0.1")),
            served_model_name=str(serving_settings.get("served_model_name", "shopping-agent")),
            max_model_len=int(serving_settings.get("max_model_len", 24576)),
            tensor_parallel_size=int(serving_settings.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=serving_settings.get("gpu_memory_utilization"),
            log_path=run_dir / "serving.log",
        ) as endpoint:
            status = _evaluate(
                endpoint, str(serving_settings.get("served_model_name", "shopping-agent")), "EMPTY"
            )
    if status != 0:
        return status

    if section(config, "report", default=True):
        _call(report.main, ["--run-dir", str(run_dir)])
    if section(config, "compare", default=True):
        _call(comparison_report.main, ["--evaluation-dir", str(run_dir.parent)])
    print(f"测评完成:{run_dir}")
    return 0
