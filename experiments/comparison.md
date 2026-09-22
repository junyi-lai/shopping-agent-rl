# Final-200 横向对比

8 个模型跑同一套 Final-200 Clean 盲测集、用同一把 Reward v4 量具判定（每题 1 条贪心轨迹）。
数字由各模型 `summary.json` 汇总，未经手工修改。

## 总体指标

| 模型 | 训练 Reward 版本 | 严格成功 | 购买成功 | Reward 有效 | 平均 Reward | 平均步数 | 最多的失败类型 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rl/rloo_v6_lr3e-5_200step | v6 | 153/200（76.5%） | 76.5% | 98.0% | 0.834 | 10.54 | partial_alternative_purchase（24 条） |
| rl/rloo_v4_lr1e-5_200step | v4 | 142/200（71.0%） | 71.0% | 97.5% | 0.753 | 9.72 | partial_alternative_purchase（32 条） |
| rl/grpo_v4_lr1e-5_500step | v4 | 140/200（70.0%） | 70.0% | 97.5% | 0.733 | 11.09 | partial_alternative_purchase（29 条） |
| rl/grpo_v4_lr1e-5_200step | v4 | 139/200（69.5%） | 69.5% | 95.5% | 0.746 | 10.38 | partial_alternative_purchase（33 条） |
| rl/grpo_v5_lr1e-5_200step | v5 | 138/200（69.0%） | 69.0% | 96.5% | 0.721 | 9.85 | partial_alternative_purchase（32 条） |
| rl/grpo_v4_lr1e-6_200step | v4 | 134/200（67.0%） | 67.0% | 97.5% | 0.698 | 10.34 | partial_alternative_purchase（38 条） |
| sft | — | 134/200（67.0%） | 67.0% | 97.5% | 0.698 | 10.34 | partial_alternative_purchase（38 条） |
| base | — | 6/200（3.0%） | 3.0% | 59.0% | -0.253 | 23.89 | max_steps（94 条） |

「训练 Reward 版本」是训练时用的奖励契约；评测量具对所有模型都固定为 Reward v4（即各 `summary.json` 里的 `shaped_reward_version`）。base / sft 没有训练 Reward 版本，记为 `—`。

`unknown` 不是一种判定结果，而是该题没有形成可判定的 v4 终局（例如 `status=error`）。

## 终局类型分布

| 模型 | 正确购买 | partial_alternative_purchase | wrong_purchase | repeat_loop | max_steps | unknown |
| --- | --- | --- | --- | --- | --- | --- |
| rl/rloo_v6_lr3e-5_200step | 153 | 24 | 9 | 1 | 9 | 4 |
| rl/rloo_v4_lr1e-5_200step | 142 | 32 | 11 | 4 | 6 | 5 |
| rl/grpo_v4_lr1e-5_500step | 140 | 29 | 11 | 2 | 13 | 5 |
| rl/grpo_v4_lr1e-5_200step | 139 | 33 | 8 | 3 | 8 | 9 |
| rl/grpo_v5_lr1e-5_200step | 138 | 32 | 12 | 3 | 8 | 7 |
| rl/grpo_v4_lr1e-6_200step | 134 | 38 | 12 | 2 | 9 | 5 |
| sft | 134 | 38 | 12 | 2 | 9 | 5 |
| base | 6 | 3 | 0 | 15 | 94 | 82 |

其余终局类型（early_abstain、graceful_stop、reward_unverifiable、valid_alternative_purchase）在所有模型上都是 0 条，故未列成列。

## 跨模型共识

| 做对该题的模型数 | 题目数 |
| --- | --- |
| 8 | 6 |
| 7 | 117 |
| 6 | 4 |
| 5 | 6 |
| 4 | 6 |
| 3 | 7 |
| 2 | 4 |
| 1 | 12 |
| 0 | 38 |

全模型共同失败 38 题、全模型共同成功 6 题。共同失败的题目通常有大量近似商品，要靠详情页或精确规格轴区分。

