# 仓库约定

本仓库只支持一条工作流：`采集 → 筛选 → SFT → RL（GRPO / RLOO） → 测评`。
每个环节恰好一个入口，入口只解析少量覆盖项，逻辑全部在 `src/shopping_agent/`。

| 环节 | 入口 | 说明 |
| --- | --- | --- |
| 前置 | `python scripts/00_environment.py setup\|start\|status\|smoke\|stop` | 安装 / 启停 ShopSimulator |
| 采集 | `python scripts/01_collect.py` | 教师模型（API）生成轨迹，支持增量补采 |
| 筛选 | `python scripts/02_filter.py` | 重打分（锁 Reward v4）→ 过滤 → 难度标签 → 课程清单 → RL 任务集 |
| SFT | `python scripts/03_sft.py` | 课程 a/b/c 训练并归档到 `models/sft` |
| RL | `python scripts/04_rl.py grpo` / `rloo` | 训练并导出到 `models/rl/<label>`（用 `--label` 命名） |
| 测评 | `python scripts/05_evaluate.py <模型目录>` | Final-200 Clean 盲测（同样锁 v4）+ HTML 报告 |

## 目录约定

| 位置 | 放什么 |
| --- | --- |
| `configs/` | 参数的唯一来源：`pipeline.yaml`（全局）+ 每个环节一个文件；RL 超参（lr / temperature / steps）在 `configs/<algo>.yaml` |
| `src/shopping_agent/` | 全部逻辑：`collection/`、`environment/`、`reward/`、`training/sft/`、`training/rl/`、`evaluation/`，外加 `pipeline.py`（环节编排）、`config.py`、`serving.py`、`smoke.py`、`resources/` |
| `environments/ShopSimulator/` | 纯模拟器（Environment v2.1 / observation v2 / tools v2）：只返回原始事实，不含 reward 与终止策略 |
| `data/` | 随仓库提交的小数据集：`sft/`、`rl/`、`evaluation/` |
| `models/` | 下载 / 训练出的模型：`base/`、`sft/`、`rl/<label>/`；不入库 |
| `works/` | 运行产物：`collect/`、`filter/`、`sft/`、`rl/<label>/`、`eval/`、`environment/`；不入库 |
| `experiments/` | 从 `works/eval/` 导出的测评结果快照，唯一入库的结果目录；只放 `summary.json` / `report.html` / `run_config.json` 与两份对比报告 |
| `patches/` | veRL 0.8 动态采样补丁，由 RL 入口自动应用 |

`reward/` 的内部分工：`contract.py` 是与版本无关的冻结契约（`REWARD_VERSION`、过程塑造、strict
闸门、步数记账、类型表）；`terminal.py` 是终局判定骨架（硬闸门、特征比对、**唯一一处**
`reward_type` 分类）；`v4.py` / `v5.py` / `v6.py` 各一份规格，接口统一为 `NAME`、`CONTRACT`、
`terminal_value(reward_type, values, *, asin_match, match_score)`；`versions.py` 管版本名与环境变量；
`score.py` 组装消费方读到的 reward 明细。

**一次 run 三处同名**：`works/rl/<label>` ≡ `models/rl/<label>` ≡ `works/eval/rl/<label>`
（`05_evaluate` 按模型路径镜像：`models/base` → `works/eval/base`、`models/sft` → `works/eval/sft`）。
`--label` 不填时回落到 `--output` 的目录名、再回落到算法名。标签格式
`<algo>_<reward>_lr<lr>_<steps>step`（例：`rloo_v6_lr3e-5_200step`），非 RL 模型用 `base` / `sft`；
学习率指数不补零（`lr1e-6`，不是 `lr1e-06`）。`works/` 下每个子目录都对应一个环节或一次 run，
run 目录里只有真产物。

## Reward 版本

一次只启用一个契约，由 `configs/pipeline.yaml` 的 `reward.version` 选择，当前是 `v6`。三个版本跑
**同一条代码路径**，差异只有两处：`pipeline.yaml` 里它完整的 `reward.<version>` 块，以及它的
`terminal_value` 映射。

| 版本 | 终局取值 | 过程塑造 | 状态 |
| --- | --- | --- | --- |
| `v4` | 冻结离散桶（1.0 / 0.55 / partial≤0.25 …） | flat 默认值 | **冻结基线**：采集/筛选/SFT/测评固定用它 |
| `v5` | 同 v4 | 证据 0.15 + 重复 0.03 + 探索 0.10 + `reward_unverifiable=-0.40` | RL 实验，实测无增益；保留可复跑 |
| `v6` | 连续 `0.5·ASIN命中 + match_score − 0.5` | 同 v4 | **RL 冠军**（比 v4 +5.5pp） |

- `reward_type`（驱动 strict success）**与版本无关**：分类只有一处，在 `terminal.py`；按版本分派的只有取值。
- **RL 训练**按 `reward.version` 选版本（训练要更细的梯度）；**采集 / 筛选 / SFT / 测评固定 v4**——
  它们是量具与数据源，不能跟着 RL 实验漂移（筛选与测评在导出配置时显式锁 v4，采集与 SFT 落到 v4
  默认值）。锁 v4 不损失判定能力，因为分类与版本无关。
- v4 是**参考契约**、不是"最优版本"，不允许悄悄换：迁移要显式做——新增版本、重建数据、用新契约
  重跑全部基线，并说明新旧数字不可比。
- 调数值改 `pipeline.yaml` 的对应块；改取值逻辑改版本文件的 `terminal_value`，共享行为放 `contract.py`；
  行为变更用三个版本全组合的差分检查验证，不只靠单元测试。

