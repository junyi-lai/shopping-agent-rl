#!/usr/bin/env python3
"""Build a self-contained comparison report for Final-200 evaluation runs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from shopping_agent.evaluation.artifacts import read_jsonl


ROOT = Path(__file__).resolve().parents[3]
HISTOGRAM_EDGES = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.000001]


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _histogram(values: list[float]) -> list[int]:
    counts = [0] * (len(HISTOGRAM_EDGES) - 1)
    for value in values:
        for index, (lower, upper) in enumerate(zip(HISTOGRAM_EDGES, HISTOGRAM_EDGES[1:])):
            if lower <= value < upper:
                counts[index] += 1
                break
    return counts


def _reward_type(row: dict) -> str:
    terminal = row.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    return detail.get("reward_type") or terminal.get("reward_type") or "unknown"


def _model_data(run_dir: Path, *, root: Path | None = None) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_jsonl(run_dir / "trajectories.jsonl")
    rewards = [float(row.get("final_reward") or 0.0) for row in rows]
    protocol = summary.get("protocol") or {}
    outcome_counts = summary.get("reward_type_counts") or Counter(_reward_type(row) for row in rows)
    success_ids = summary.get("strict_success_task_ids") or [
        int(row["task_id"]) for row in rows if _reward_type(row) == "gold_purchase"
    ]
    # 评测产物按模型路径镜像存放,所以报告链接必须相对评估根目录计算(可能深一层)。
    slug = (run_dir.relative_to(root) if root else Path(run_dir.name)).as_posix()
    return {
        "key": slug,
        # 展示名用 run 标签:协议里的 ``model`` 是 vLLM 服务名,对所有被测模型都一样
        # (都是 configs/pipeline.yaml 的 served_model_name),拿来当名字会让整张表全是同一行。
        "name": run_dir.name or protocol.get("model") or "model",
        "report": f"{slug}/report.html",
        "tasks": len(rows),
        "successes": int(summary.get("strict_successes", len(success_ids))),
        "success_rate": float(summary.get("strict_success_rate", len(success_ids) / len(rows))),
        "purchase_rate": float(summary.get("purchase_success_rate", 0.0)),
        "reward_valid_rate": float(summary.get("reward_valid_rate", 0.0)),
        "average_steps": float(summary.get("average_steps", 0.0)),
        "outcomes": dict(outcome_counts),
        "statuses": summary.get("status_counts") or dict(Counter(row.get("status", "unknown") for row in rows)),
        "guards": summary.get("guard_reason_counts") or {},
        "success_ids": success_ids,
        "reward": {
            "mean": statistics.fmean(rewards),
            "median": statistics.median(rewards),
            "stddev": statistics.pstdev(rewards),
            "min": min(rewards),
            "q1": _quantile(rewards, 0.25),
            "q3": _quantile(rewards, 0.75),
            "max": max(rewards),
            "histogram": _histogram(rewards),
        },
    }


def build_comparison_data(evaluation_dir: Path) -> dict:
    # 递归查找:评测产物按模型路径镜像存放(models/rl/<label> → works/eval/rl/<label>),
    # 因此 RL 的评测目录比 base/sft 深一层,不能只扫顶层。
    run_dirs = sorted(
        path.parent
        for path in evaluation_dir.rglob("summary.json")
        if (path.parent / "trajectories.jsonl").is_file()
    )
    models = [_model_data(path, root=evaluation_dir) for path in run_dirs]
    if not models:
        raise ValueError(f"no evaluation runs found under {evaluation_dir}")
    all_task_ids = set().union(*(set(model["success_ids"]) for model in models))
    for path in run_dirs:
        all_task_ids.update(int(row["task_id"]) for row in read_jsonl(path / "trajectories.jsonl"))
    solved_by = Counter(
        sum(task_id in set(model["success_ids"]) for model in models) for task_id in all_task_ids
    )
    best = max(models, key=lambda model: model["success_rate"])
    return {
        "models": models,
        "outcome_order": [
            "gold_purchase",
            "valid_alternative_purchase",
            "partial_alternative_purchase",
            "wrong_purchase",
            "repeat_loop",
            "max_steps",
            "early_abstain",
            "graceful_stop",
            "reward_unverifiable",
            "unknown",
        ],
        "histogram_labels": [
            f"{HISTOGRAM_EDGES[index]:.1f}～{min(HISTOGRAM_EDGES[index + 1], 1.0):.1f}"
            for index in range(len(HISTOGRAM_EDGES) - 1)
        ],
        "agreement": [{"models": count, "tasks": solved_by.get(count, 0)} for count in range(len(models) + 1)],
        "all_failed_task_ids": sorted(
            task_id
            for task_id in all_task_ids
            if not any(task_id in set(model["success_ids"]) for model in models)
        ),
        "all_succeeded_task_ids": sorted(
            task_id
            for task_id in all_task_ids
            if all(task_id in set(model["success_ids"]) for model in models)
        ),
        "best_model": best["name"],
    }


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Final-200 多模型轨迹与 Bad Case 综合报告</title>
<style>
:root{--ink:#142033;--muted:#64748b;--line:#dbe4ef;--bg:#f3f6fa;--blue:#2563eb;--green:#16a34a;--amber:#d97706;--red:#dc2626;--violet:#7c3aed}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"PingFang SC",sans-serif}
main{max-width:1440px;margin:auto;padding:28px}.hero,.card{background:#fff;border:1px solid var(--line);border-radius:16px;box-shadow:0 8px 28px #0f172a0a}
.hero{padding:28px;margin-bottom:18px;background:linear-gradient(135deg,#fff 55%,#e8f0ff)}h1{margin:0 0 8px;font-size:30px}h2{font-size:20px;margin:0 0 14px}h3{margin:0 0 8px}.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:18px 0}.card{padding:20px;overflow:auto}.wide{grid-column:1/-1}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:12px;margin-top:18px}.kpi{padding:14px;border-radius:12px;background:#f8fafc;border:1px solid var(--line)}.kpi b{display:block;font-size:24px}
table{border-collapse:collapse;width:100%;min-width:900px}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:#fff}th{color:var(--muted);font-size:12px}
.bar-row{display:grid;grid-template-columns:150px 1fr 70px;gap:10px;align-items:center;margin:9px 0}.track{height:12px;background:#eef2f7;border-radius:99px;overflow:hidden}.fill{height:100%;background:var(--blue);border-radius:99px}.small{font-size:12px}.hist{display:grid;grid-template-columns:160px repeat(10,minmax(24px,1fr));gap:5px;align-items:end;margin:12px 0}.hist-name{align-self:center}.hist-bin{height:92px;background:#f1f5f9;display:flex;align-items:end;border-radius:5px 5px 0 0;overflow:hidden}.hist-bin i{display:block;width:100%;background:var(--violet);min-height:2px}.hist-labels{display:grid;grid-template-columns:160px repeat(10,minmax(24px,1fr));gap:5px;color:var(--muted);font-size:10px}.hist-labels span{writing-mode:vertical-rl;height:58px}
.stack{display:flex;height:18px;border-radius:99px;overflow:hidden;background:#eef2f7}.seg{height:100%}.outcome-row{display:grid;grid-template-columns:150px 1fr;gap:10px;align-items:center;margin:12px 0}.legend{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}
.notes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.note{padding:14px;border:1px solid var(--line);border-radius:12px;background:#fbfdff}.note ul{padding-left:20px;margin:7px 0}.task-ids{word-break:break-all;padding:12px;background:#f8fafc;border-radius:10px;color:#475569}.links a{display:inline-block;margin:5px 10px 5px 0;padding:7px 11px;border-radius:9px;background:#eff6ff;color:#1d4ed8;text-decoration:none}
@media(max-width:900px){main{padding:14px}.grid{grid-template-columns:1fr}.kpis,.notes{grid-template-columns:1fr 1fr}.hist{grid-template-columns:100px repeat(10,minmax(15px,1fr))}.hist-labels{grid-template-columns:100px repeat(10,minmax(15px,1fr))}}
</style></head><body><main>
<section class="hero"><div class="muted">Shopping Agent · Final-200</div><h1>多模型轨迹与 Bad Case 综合报告</h1><p>同一套 Final-200、${D.models.length} 个模型。先看成功率和 Reward 分布，再看失败轨迹为什么会错、强模型强在哪里。</p><div class="kpis" id="kpis"></div><div class="links" id="links"></div></section>
<div class="grid">
<section class="card"><h2>严格成功率</h2><div id="success-bars"></div></section>
<section class="card"><h2>任务共识难度</h2><p class="muted">横轴含义：一道题被多少个模型做对。</p><div id="agreement-bars"></div></section>
<section class="card wide"><h2>描述性统计</h2><p class="muted">Reward 统计包含全部 200 条；未形成可验证终局的轨迹按其记录值（通常为 0）计入。</p><div id="reward-reading"></div><table><thead><tr><th>模型</th><th>严格成功</th><th>购买成功率</th><th>Reward 有效率</th><th>均值</th><th>中位数</th><th>标准差</th><th>最小</th><th>P25</th><th>P75</th><th>最大</th><th>平均步数</th></tr></thead><tbody id="stats"></tbody></table></section>
<section class="card wide"><h2>Reward 分布</h2><p class="muted">每一小柱是一个 0.2 宽区间；紫柱越高，落在该 Reward 区间的任务越多。</p><div id="histograms"></div><div class="hist-labels" id="hist-labels"></div></section>
<section class="card wide"><h2>终局类型分布</h2><div class="legend" id="legend"></div><div id="outcomes"></div></section>
<section class="card wide"><h2>核心结论：强模型强在哪</h2><div id="strengths"></div></section>
<section class="card wide"><h2>各模型主要 Bad Case</h2><div class="notes" id="model-notes"></div></section>
<section class="card wide"><h2>跨模型共性</h2><div id="common-notes"></div><h3>所有模型都没做对的任务（<span id="all-failed-count"></span>）</h3><div class="task-ids" id="all-failed"></div><h3 style="margin-top:16px">所有模型都做对的任务（<span id="all-succeeded-count"></span>）</h3><div class="task-ids" id="all-succeeded"></div></section>
<section class="card wide"><h2>需要谨慎解读的评测数据</h2><div id="data-notes"></div></section>
</div></main>
<script>
const D=__REPORT_DATA__;
const COLORS={gold_purchase:'#16a34a',valid_alternative_purchase:'#4ade80',partial_alternative_purchase:'#f59e0b',wrong_purchase:'#ef4444',repeat_loop:'#7c3aed',max_steps:'#db2777',early_abstain:'#94a3b8',graceful_stop:'#38bdf8',reward_unverifiable:'#a16207',unknown:'#cbd5e1'};
const LABELS={gold_purchase:'正确购买',valid_alternative_purchase:'有效替代品',partial_alternative_purchase:'部分匹配',wrong_purchase:'买错商品',repeat_loop:'重复循环',max_steps:'步数耗尽',early_abstain:'过早放弃',graceful_stop:'主动停止',reward_unverifiable:'无法核验',unknown:'未形成终局'};
const pct=x=>(x*100).toFixed(1)+'%'; const n=x=>Number(x).toFixed(3); const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sorted=[...D.models].sort((a,b)=>b.success_rate-a.success_rate); const total=D.models[0].tasks;
document.querySelector('#kpis').innerHTML=`<div class="kpi"><span>模型</span><b>${D.models.length}</b></div><div class="kpi"><span>每模型任务</span><b>${total}</b></div><div class="kpi"><span>最高严格成功率</span><b>${pct(sorted[0].success_rate)}</b><small>${esc(sorted[0].name)}</small></div><div class="kpi"><span>全模型共同失败</span><b>${D.all_failed_task_ids.length}</b></div>`;
document.querySelector('#links').innerHTML=sorted.map(m=>`<a href="${encodeURI(m.report)}">${esc(m.name)} 单模型报告</a>`).join('');
function bars(target,items,max,label){document.querySelector(target).innerHTML=items.map(x=>`<div class="bar-row"><span>${esc(x.name)}</span><div class="track"><div class="fill" style="width:${x.value/max*100}%"></div></div><b>${label(x)}</b></div>`).join('')}
bars('#success-bars',sorted.map(m=>({name:m.name,value:m.success_rate,count:m.successes})),1,x=>`${x.count}/${total}`);
bars('#agreement-bars',D.agreement.map(x=>({name:`${x.models} 个模型`,value:x.tasks})),Math.max(...D.agreement.map(x=>x.tasks)),x=>x.value);
document.querySelector('#stats').innerHTML=sorted.map(m=>`<tr><td><a href="${encodeURI(m.report)}">${esc(m.name)}</a></td><td>${m.successes}/${m.tasks}（${pct(m.success_rate)}）</td><td>${pct(m.purchase_rate)}</td><td>${pct(m.reward_valid_rate)}</td><td>${n(m.reward.mean)}</td><td>${n(m.reward.median)}</td><td>${n(m.reward.stddev)}</td><td>${n(m.reward.min)}</td><td>${n(m.reward.q1)}</td><td>${n(m.reward.q3)}</td><td>${n(m.reward.max)}</td><td>${m.average_steps.toFixed(2)}</td></tr>`).join('');
document.querySelector('#reward-reading').innerHTML=`<p>分布很明显是“两头多、中间少”：正确购买直接落在 1.0，循环、步数耗尽和买错则集中在负分。${esc(sorted[0].name)} 的 P25 仍有 ${n(sorted[0].reward.q1)}，说明至少四分之三任务没有掉到低分区；${esc(sorted[sorted.length-1].name)} 的中位数是 ${n(sorted[sorted.length-1].reward.median)}、P25 是 ${n(sorted[sorted.length-1].reward.q1)}，一半任务连正向终局都没有。标准差越大表示越不稳：本批最高是 ${esc([...sorted].sort((a,b)=>b.reward.stddev-a.reward.stddev)[0].name)}（${n([...sorted].sort((a,b)=>b.reward.stddev-a.reward.stddev)[0].reward.stddev)}）。</p>`;
document.querySelector('#histograms').innerHTML=sorted.map(m=>{const max=Math.max(...m.reward.histogram);return `<div class="hist"><b class="hist-name">${esc(m.name)}</b>${m.reward.histogram.map(v=>`<div class="hist-bin" title="${v} 条"><i style="height:${v/max*100}%"></i></div>`).join('')}</div>`}).join('');
document.querySelector('#hist-labels').innerHTML='<b></b>'+D.histogram_labels.map(x=>`<span>${x}</span>`).join('');
document.querySelector('#legend').innerHTML=D.outcome_order.map(k=>`<span><i style="background:${COLORS[k]}"></i>${LABELS[k]}</span>`).join('');
document.querySelector('#outcomes').innerHTML=sorted.map(m=>`<div class="outcome-row"><b>${esc(m.name)}</b><div class="stack">${D.outcome_order.map(k=>`<div class="seg" title="${LABELS[k]}：${m.outcomes[k]||0}" style="width:${(m.outcomes[k]||0)/m.tasks*100}%;background:${COLORS[k]}"></div>`).join('')}</div></div>`).join('');
const best=sorted[0], weakest=sorted[sorted.length-1];
document.querySelector('#strengths').innerHTML=`<p><b>${esc(best.name)}</b> 是这批里最稳的：严格成功 ${best.successes}/${best.tasks}（${pct(best.success_rate)}），平均 ${best.average_steps.toFixed(2)} 步。和最低的 ${esc(weakest.name)}（${pct(weakest.success_rate)}）相比，它重复循环 ${best.outcomes.repeat_loop||0} 条（对方 ${weakest.outcomes.repeat_loop||0} 条）、未形成终局 ${best.outcomes.unknown||0} 条（对方 ${weakest.outcomes.unknown||0} 条）。差距主要落在“能不能收尾”：强模型少走回头路，也更少在接近答案时卡住。</p>`;
document.querySelector('#model-notes').innerHTML=sorted.map(m=>{const g=Object.values(m.guards||{}).reduce((a,b)=>a+b,0);return `<article class="note"><h3>${esc(m.name)}</h3><ul><li>正确购买 ${m.outcomes.gold_purchase||0} 条，有效替代 ${m.outcomes.valid_alternative_purchase||0} 条，部分匹配 ${m.outcomes.partial_alternative_purchase||0} 条。</li><li>买错 ${m.outcomes.wrong_purchase||0} 条，重复循环 ${m.outcomes.repeat_loop||0} 条，步数耗尽 ${m.outcomes.max_steps||0} 条。</li><li>过早放弃 ${m.outcomes.early_abstain||0} 条，主动停止 ${m.outcomes.graceful_stop||0} 条，无法核验 ${m.outcomes.reward_unverifiable||0} 条，未形成终局 ${m.outcomes.unknown||0} 条。</li><li>守卫拦截 ${g} 次。</li></ul></article>`}).join('');
document.querySelector('#common-notes').innerHTML=`<p>失败通常集中在四类，可按轨迹逐条核对：</p><ul><li><b>看标题就退：</b>候选标题不够像时，打开后立即返回，没继续看属性、规格轴和变体价。</li><li><b>找到后不收口：</b>正确候选已出现，仍换同义搜索词、重复打开商品或来回切规格。</li><li><b>规格轴没管住：</b>型号、颜色、尺码、容量、数量只选一部分，或点过但购买记录没保留；预算应按最终变体价核对。</li><li><b>页面状态没跟上：</b>拿旧页面的 ASIN/按钮继续点，或给无参工具乱传参数。</li></ul><p>${D.all_failed_task_ids.length} 道题所有模型都失败，说明这部分通常有大量近似品、要靠详情或精确规格区分；${D.all_succeeded_task_ids.length} 道题所有模型都做对，说明基础搜索和明显匹配项不是主要瓶颈。</p>`;
document.querySelector('#data-notes').innerHTML=`<p>严格成功率是官方口径。部分边界案例不宜直接归因于模型推理：口语预算被当硬上限（“60 元出头”买 62 会被判失败）、字符归一化（➕/爱心/大小写）、Query 与 gold 冲突或 gold 偷加约束。完整清单与筛选依据见 data/evaluation/metadata.json。</p>`;
document.querySelector('#all-failed-count').textContent=D.all_failed_task_ids.length;document.querySelector('#all-failed').textContent=D.all_failed_task_ids.join(', ');document.querySelector('#all-succeeded-count').textContent=D.all_succeeded_task_ids.length;document.querySelector('#all-succeeded').textContent=D.all_succeeded_task_ids.join(', ');
</script></body></html>'''


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="生成 Final-200 多模型综合 HTML 报告")
    parser.add_argument(
        "--evaluation-dir", type=Path, default=ROOT / "works" / "eval"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output or args.evaluation_dir / "comparison-report.html"
    data = json.dumps(build_comparison_data(args.evaluation_dir), ensure_ascii=False, separators=(",", ":"))
    output.write_text(HTML.replace("__REPORT_DATA__", data.replace("</", "<\\/")), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
