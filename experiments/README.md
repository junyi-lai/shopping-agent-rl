# 实验结果快照

`works/` 是运行目录、不入库；这里放 8 个模型在同一套 Final-200 Clean 上的测评产物，直接看数字不必重跑。

## 目录

| 路径 | 内容 |
| --- | --- |
| `comparison.md` | 8 个模型的横向对比：总体指标、终局类型分布、跨模型共识 |
| `comparison-report.html` | 同一批结果的可视化报告，浏览器打开 |
| `base/`、`sft/`、`rl/<label>/` | 每个模型一组：`summary.json`、`report.html`、`run_config.json` |

结构与 `works/eval/` 一致（`05_evaluate` 按模型路径镜像产出），所以报告里的相对链接照旧可用。

## 评测口径

- 数据集 `data/evaluation/tasks.jsonl`，200 题，分母固定 200，失败轨迹不剔除。
- 每题 1 条轨迹、贪心解码（temperature 0.0）、并发 8：数字是确定值，不反映采样方差。
- 判分只用冻结契约 Reward v4 的终局判定，与训练用哪个 reward 版本无关，全程没有 LLM 裁判。
- 严格成功率 = `gold_purchase` 题数 ÷ 200。`unknown` 不是一种判定结果，而是该题没有形成可判定的终局（例如 `status=error`）。

## run_config.json 的字段来源

| 字段 | 来源 |
| --- | --- |
| `verified_from_training_log` | 训练日志 `works/rl/<label>/training_diagnostics.jsonl` |
| `eval` | 该次测评 `summary.json` 的 `protocol`，轨迹条数取自同目录的 `trajectories.jsonl` |
| `train_data`、`validation_data`、`benchmark` | 对数据集现场计算 `sha256` 与行数，行数与 `data/*/metadata.json` 一致 |
| `curriculum` | `data/sft/curriculum.json` 的三个阶段 |
| `base_model` | `models/base/config.json` |

训练侧的采样温度、top_p、LoRA 秩等超参没有写进日志，见 `configs/grpo.yaml` 与 `configs/rloo.yaml`（`config_file` 指向它们）。这两份配置是按 RLOO v6 与 GRPO 500 步两轮写死的，其余轮次的差异只在学习率、步数、reward 版本上，标签和 `run_config.json` 里都能读到。

## 重新生成

`summary.json`、`report.html`、`comparison-report.html` 都是 `works/eval/` 产物的原样拷贝，由 `05_evaluate` 生成；`comparison.md` 与 `run_config.json` 是导出时按上表整理出来的。重跑测评后做两步：

```bash
python scripts/05_evaluate.py models/rl/<label>
python -m shopping_agent.evaluation.comparison_report --evaluation-dir works/eval --output experiments/comparison-report.html
```

## 没有收录什么

`trajectories.jsonl` 与 `serving.log`（每个模型 26～51 MB，合计 241 MB），以及模型权重（约 34 GB）。需要逐条轨迹时重跑 `05_evaluate`，或在本机 `works/eval/` 里看。
