"""同一份本地追踪的终端摘要和独立 HTML 瀑布图，不依赖服务器或 CDN。"""

import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from rich.table import Table
from rich.text import Text
from rich.tree import Tree


STATUS = {"completed": "正常结束", "cancelled": "已取消", "loop_stopped": "循环保护",
          "plan_incomplete": "计划未完成",
          "round_limit": "轮数上限", "context_limit": "上下文超限", "model_error": "模型错误",
          "internal_error": "程序异常", "interrupted": "未闭合／原因未知", "ok": "成功",
          "denied": "已拒绝", "approval_required": "尚未审批", "not_run": "未执行",
          "error": "失败", "timeout": "超时", "unknown": "结果未知"}
COLORS = {"model": "#7c8cff", "execution": "#38bca3", "approval": "#e9ad51", "other": "#9ca8ba"}
LABELS = {"model": "模型请求", "execution": "工具执行", "approval": "审批等待", "other": "其他开销"}


def milliseconds(value):
    if not isinstance(value, (int, float)):
        return "未知"
    if value >= 60000:
        return f"{value / 60000:.1f}m"
    if value >= 1000:
        return f"{value / 1000:.2f}s"
    return f"{value:.1f}ms"


def _elapsed(node):
    if node["duration_ms"] is None:
        return None
    if node["name"] == "model_request" and type(node["attrs"].get("active_duration_ms")) in (int, float):
        return max(0, node["attrs"]["active_duration_ms"])
    return node["duration_ms"]


def _duration(node):
    return max(0, _elapsed(node) or 0)


def _union(intervals):
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def summarize(trace):
    nodes, root = trace["nodes"], trace["root"]
    models = [node for node in nodes if node["name"] == "model_request"]
    tools = [node for node in nodes if node["name"].startswith("tool:")]
    rounds = [node for node in nodes if node["name"] == "round"]
    known_models = [node for node in models if all(type(node["attrs"].get(key)) is int for key in ("input_tokens", "output_tokens"))]
    buckets = {key: 0.0 for key in COLORS}
    intervals = defaultdict(list)
    for node in nodes:
        bucket = "model" if node["name"] == "model_request" else node["attrs"].get("bucket")
        if bucket in {"model", "execution", "approval"} and node["start_ms"] is not None and node["end_ms"] is not None:
            intervals[bucket].append((node["start_ms"], node["end_ms"]))
    for key, ranges in intervals.items():
        buckets[key] = _union(ranges)
    consumer_ms = sum(min(node["duration_ms"] or 0, node["attrs"].get("client_consumer_ms") or 0)
                      for node in models if node["end_ms"] is not None)
    buckets["model"] = max(0.0, buckets["model"] - consumer_ms)
    duration = _duration(root) if root else 0
    buckets["other"] = max(0.0, duration - sum(buckets.values()))
    issues = list(trace["issues"])
    if sum(buckets.values()) > duration + 0.01:
        issues.append("存在重叠计时，分类占比不能直接加总")
    if any(node["end_ms"] is None for node in models):
        issues.append("模型用量或耗时可能尚未完整返回")
    counts = Counter(node["status"] for node in tools)
    calls = Counter((node["attrs"].get("arguments_sha256"), node["attrs"].get("result", {}).get("sha256"))
                    for node in tools if node["attrs"].get("arguments_sha256") and node["attrs"].get("result", {}).get("sha256"))
    aggregate = defaultdict(lambda: {"calls": 0, "executed": 0, "success": 0, "denied": 0, "error": 0, "timeout": 0, "times": []})
    for node in tools:
        row = aggregate[node["attrs"].get("tool", node["name"][5:])]
        row["calls"] += 1
        row["denied"] += node["status"] == "denied"
        row["error"] += node["status"] == "error"
        row["timeout"] += node["status"] == "timeout"
        row["executed"] += node["attrs"].get("executed") is True
        row["success"] += node["status"] == "ok" and node["attrs"].get("executed") is True
        if node["attrs"].get("exec_ms") is not None:
            row["times"].append(node["attrs"]["exec_ms"])
    for row in aggregate.values():
        values = sorted(row.pop("times"))
        row["total_ms"] = sum(values)
        row["p50_ms"] = values[max(0, math.ceil(len(values) * .5) - 1)] if values else None
        row["p95_ms"] = values[max(0, math.ceil(len(values) * .95) - 1)] if values else None
    root_attrs = root["attrs"] if root else {}
    return {"duration_ms": root["duration_ms"] if root else None, "status": root["status"] if root else "unknown",
            "stop_detail": root_attrs.get("stop_detail"), "error_type": root_attrs.get("error_type"),
            "http_status": root_attrs.get("http_status"), "plan_state": root_attrs.get("plan_state"),
            "client_consumer_ms": consumer_ms,
            "legacy_timing": sum("client_consumer_ms" not in node["attrs"] for node in models),
            "rounds": len(rounds), "models": len(models), "tools": len(tools),
            "input_tokens": sum(node["attrs"]["input_tokens"] for node in models if type(node["attrs"].get("input_tokens")) is int),
            "output_tokens": sum(node["attrs"]["output_tokens"] for node in models if type(node["attrs"].get("output_tokens")) is int),
            "cached_tokens": sum(node["attrs"]["cached_tokens"] for node in models if type(node["attrs"].get("cached_tokens")) is int),
            "reasoning_tokens": sum(node["attrs"]["reasoning_tokens"] for node in models if type(node["attrs"].get("reasoning_tokens")) is int),
            "unknown_usage": len(models) - len(known_models), "counts": dict(counts),
            "repeated": sum(count - 1 for count in calls.values()), "buckets": buckets,
            "issues": sorted(set(issues)), "tool_stats": dict(aggregate)}


def stop_description(summary):
    detail = summary["stop_detail"]
    if not isinstance(detail, str) or not detail:
        detail = {"completed": "本轮执行流程正常结束。", "cancelled": "用户取消了当前任务。",
                  "interrupted": "记录未闭合，无法确认是强制结束还是尾部记录丢失。"}.get(
                      summary["status"], STATUS.get(summary["status"], summary["status"]))
    extra = [str(value) for value in (summary["error_type"],
             f"HTTP {summary['http_status']}" if summary["http_status"] is not None else None) if value]
    return detail + ("（" + " · ".join(extra) + "）" if extra else "")


def plan_description(summary):
    state = summary["plan_state"]
    if not isinstance(state, dict):
        return "未记录计划进度（旧记录或未接入计划）。"
    if state.get("total") == 0:
        return "无分阶段计划。"
    number = lambda key: str(state[key]) if type(state.get(key)) is int else "未知"
    return (f"{number('completed')}/{number('total')} 阶段完成 · {number('evidence_count')} 条依据 · {number('blocked')} 阶段受阻。"
            + ("当前阶段依据待修复，允许申请审批重验。" if state.get("repair_required") else ""))


def show_trace(trace, console, *, full=False):
    summary = summarize(trace)
    console.print(f"trace {trace['run'][:8]} · {STATUS.get(summary['status'], summary['status'])} · "
                  f"{milliseconds(summary['duration_ms'])} · {summary['rounds']} 轮 · {summary['tools']} 次工具调用",
                  style="bold #c4b5fd", markup=False)
    if summary["issues"]:
        console.print("记录不完整：" + "；".join(summary["issues"]), style="yellow", markup=False)
    console.print(f"token 已知合计：输入 {summary['input_tokens']:,} · 输出 {summary['output_tokens']:,}"
                  f" · 未知请求 {summary['unknown_usage']} 次", markup=False)
    console.print(" · ".join(f"{LABELS[key]} {milliseconds(value)}" for key, value in summary["buckets"].items()), markup=False)
    console.print(f"拒绝 {summary['counts'].get('denied', 0)} · 失败 {summary['counts'].get('error', 0)} · 超时 {summary['counts'].get('timeout', 0)}"
                  f" · 结果未知 {summary['counts'].get('unknown', 0)} · 未执行 {summary['counts'].get('not_run', 0)}", markup=False)
    console.print("停止详情：" + stop_description(summary), markup=False)
    console.print("计划状态（模型记录）：" + plan_description(summary), markup=False)
    console.print(f"客户端处理 {milliseconds(summary['client_consumer_ms'])}，已归入其他开销。" +
                  (f"另有 {summary['legacy_timing']} 次旧请求无法拆分客户端耗时。" if summary["legacy_timing"] else ""), style="#8c8c8c", markup=False)
    if not full:
        _show_highlights(trace, console)
        console.print("/trace tree [编号前缀] 查看完整时间线；计划进度不等同独立验收。", style="#8c8c8c", markup=False)
        return
    by_parent = defaultdict(list)
    for node in trace["nodes"]:
        by_parent[node["parent"]].append(node)
    tree = Tree(Text("执行时间线", style="bold #67d8ef"))

    def append(parent, branch, visited):
        for node in by_parent[parent]:
            if node["id"] in visited:
                continue
            label = f"{node['name']}  {milliseconds(_elapsed(node))} · {STATUS.get(node['status'], node['status'])}"
            if node["name"] == "model_request":
                label += f" · 生命周期 {milliseconds(node['duration_ms'])}"
                label += f" · 首事件 {milliseconds(node['attrs'].get('first_event_ms'))} · 首正文 {milliseconds(node['attrs'].get('first_text_ms'))}"
            child = branch.add(Text(label, style="#8c8c8c" if node["status"] in {"ok", "completed"} else "yellow"))
            append(node["id"], child, visited | {node["id"]})
    append(None, tree, set())
    console.print(tree)


def _show_highlights(trace, console):
    """长任务默认只给最慢五轮和前五条异常，完整明细按需查询。"""
    by_id = {node["id"]: node for node in trace["nodes"]}

    def in_round(node, identifier):
        seen = set()
        while node["parent"] in by_id and node["parent"] not in seen:
            if node["parent"] == identifier:
                return True
            seen.add(node["parent"])
            node = by_id[node["parent"]]
        return False

    rounds = sorted((node for node in trace["nodes"] if node["name"] == "round"), key=_duration, reverse=True)
    if rounds:
        table = Table(title="最慢的轮次（最多 5 轮）", box=None, padding=(0, 1))
        for label in ("轮次", "总耗时", "模型", "工具执行", "审批"):
            table.add_column(label)
        for node in rounds[:5]:
            children = [item for item in trace["nodes"] if in_round(item, node["id"])]
            values = [sum(_duration(item) for item in children if predicate(item)) for predicate in (
                lambda item: item["name"] == "model_request",
                lambda item: item["attrs"].get("bucket") == "execution",
                lambda item: item["attrs"].get("bucket") == "approval")]
            table.add_row(str(node["attrs"].get("round", "?")), milliseconds(node["duration_ms"]),
                          *(milliseconds(value) for value in values))
        console.print(table)
    problems = [node for node in trace["nodes"] if node["name"].startswith("tool:") and node["status"] not in {"ok", "not_run"}]
    for node in problems[:5]:
        console.print(f"{node['name']} · {STATUS.get(node['status'], node['status'])} · {node['attrs'].get('call_id', '编号未知')}", style="yellow", markup=False)
    if len(problems) > 5:
        console.print(f"其余 {len(problems) - 5} 条异常或拒绝请查看完整时间线或 HTML 报告。", style="#8c8c8c", markup=False)


def list_traces(reader, console):
    table = Table(box=None, padding=(0, 1), expand=False)
    for label in ("编号", "类型", "开始时间 UTC", "状态", "耗时", "轮／工具"):
        table.add_column(label, overflow="fold")
    paths = reader.list()
    if not paths:
        console.print("当前会话暂无追踪记录。", style="#8c8c8c", markup=False)
        return
    for path in paths[:100]:
        trace = reader.read(path)
        summary = summarize(trace)
        kind = trace["root"]["attrs"].get("kind") if trace["root"] else None
        table.add_row(trace["run"][:8], "任务" if kind == "task" else "命令", (trace["started_at"] or "未知")[:19].replace("T", " "),
                      STATUS.get(summary["status"], summary["status"]) + (" · 不完整" if summary["issues"] else ""),
                      milliseconds(summary["duration_ms"]), f"{summary['rounds']} / {summary['tools']}")
    console.print(table)


def export_html(trace, destination=None):
    summary = summarize(trace)
    destination = Path(destination) if destination else trace["path"].with_suffix(".html")
    if destination.is_symlink():
        raise ValueError("报告目标是文件链接，未覆盖。")
    esc = lambda value: html.escape(str(value), quote=True)
    root = trace["root"]
    duration = max(1, summary["duration_ms"] or max((node["end_ms"] or 0 for node in trace["nodes"]), default=1))
    cards = [("总耗时", milliseconds(summary["duration_ms"])), ("模型请求", summary["models"]),
             ("工具调用", summary["tools"]), ("输入 / 输出 token", f"{summary['input_tokens']:,} / {summary['output_tokens']:,}"),
             ("未知用量请求", summary["unknown_usage"]), ("拒绝 / 失败 / 超时", f"{summary['counts'].get('denied', 0)} / {summary['counts'].get('error', 0)} / {summary['counts'].get('timeout', 0)}")]
    card_html = "".join(f'<div class="card"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>' for label, value in cards)
    segments = "".join(f'<div style="width:{max(0, value / duration * 100):.4f}%;background:{COLORS[key]}" title="{LABELS[key]} {milliseconds(value)}"></div>'
                       for key, value in summary["buckets"].items())
    legend = "".join(f'<span><i style="background:{COLORS[key]}"></i>{LABELS[key]} <b>{milliseconds(value)}</b></span>' for key, value in summary["buckets"].items())
    parents = {node["id"]: node for node in trace["nodes"]}

    def depth(node):
        seen, level = set(), 0
        while node["parent"] in parents and node["parent"] not in seen and level < 12:
            seen.add(node["parent"])
            node, level = parents[node["parent"]], level + 1
        return level

    rows = []
    for node in trace["nodes"]:
        key = "model" if node["name"] == "model_request" else node["attrs"].get("bucket", "other")
        color = COLORS.get(key, COLORS["other"])
        start, elapsed = node["start_ms"] or 0, node["duration_ms"] or 0
        level = depth(node)
        info = {"status": node["status"], "span": node["id"], "parent": node["parent"],
                "start_ms": node["start_ms"], "duration_ms": node["duration_ms"], **node["attrs"]}
        rows.append(f'''<details class="entry" data-search="{esc(node['name'].casefold())}" data-status="{esc(node['status'])}">
<summary><span class="operation" style="padding-left:{level * 14}px">{esc(node['name'])}<small>{esc(STATUS.get(node['status'], node['status']))}</small></span>
<span class="rail"><i style="left:{start / duration * 100:.4f}%;width:{max(.18, elapsed / duration * 100):.4f}%;background:{color}"></i></span>
<span class="elapsed">{milliseconds(_elapsed(node))}{" 净" if node["name"] == "model_request" and "active_duration_ms" in node["attrs"] else ""}</span></summary><pre>{esc(json.dumps(info, ensure_ascii=False, indent=2))}</pre></details>''')
    stats = "".join(f'<tr><td>{esc(name)}</td><td>{row["calls"]}</td><td>{row["executed"]}</td><td>{row["success"]}</td><td>{row["denied"]}</td><td>{row["error"]} / {row["timeout"]}</td><td>{milliseconds(row["p50_ms"])}</td><td>{milliseconds(row["p95_ms"])}</td><td>{milliseconds(row["total_ms"])}</td></tr>'
                    for name, row in sorted(summary["tool_stats"].items(), key=lambda pair: pair[1]["total_ms"], reverse=True))
    model_rows = []
    for node in trace["nodes"]:
        if node["name"] != "model_request":
            continue
        data = node["attrs"]
        parent = parents.get(node["parent"], {})
        label = parent.get("attrs", {}).get("round", data.get("purpose", "—"))
        value = lambda key: f"{data[key]:,}" if type(data.get(key)) is int else "未知"
        model_rows.append(f'<tr><td>{esc(label)}</td><td>{milliseconds(_elapsed(node))}</td><td>{milliseconds(node["duration_ms"])}</td><td>{milliseconds(data.get("client_consumer_ms"))}</td><td>{milliseconds(data.get("first_event_ms"))}</td><td>{milliseconds(data.get("first_text_ms"))}</td><td>{value("input_tokens")} / {value("output_tokens")}</td><td>{value("cached_tokens")} / {value("reasoning_tokens")}</td><td>{esc(data.get("finish_reason") or "未知")}</td></tr>')
    slow = sorted((node for node in trace["nodes"] if node["name"] == "model_request" or node["attrs"].get("bucket") in {"execution", "approval"}), key=_duration, reverse=True)[:8]
    slow_html = "".join(f'<li><span>{esc(node["name"])}<small>{esc(node["attrs"].get("call_id") or node["attrs"].get("request_id") or "")}</small></span><b>{milliseconds(node["duration_ms"])}</b></li>' for node in slow)
    notices = "；".join(summary["issues"]) if summary["issues"] else "记录结构完整"
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>zzcode · 本地执行报告</title><style>
:root{color-scheme:dark;--bg:#0d1118;--panel:#151b25;--line:#283242;--text:#e5ebf3;--muted:#94a1b3}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.6 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1320px;margin:auto;padding:44px 30px 70px}header{border-bottom:1px solid var(--line);padding-bottom:24px;margin-bottom:24px}.brand{letter-spacing:3px;color:#8c99ff;font-size:12px;font-weight:700}h1{font-size:30px;letter-spacing:-1px;margin:10px 0 8px;font-weight:650}h2{font-size:17px;margin:0 0 18px;font-weight:600}.meta,small{color:var(--muted);font-size:12px}.meta code{color:#c5ceec}.badge{display:inline-block;padding:3px 10px;border:1px solid #365846;border-radius:20px;background:#142b25;color:#92dcb9;margin-left:12px;font-size:12px}.notice{padding:12px 16px;border-left:3px solid #e9ad51;background:#292318;color:#eed3a8;margin-bottom:24px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:9px;overflow:hidden;margin-bottom:24px}.card{background:var(--panel);padding:20px 22px}.card span{display:block;color:var(--muted);font-size:12px}.card strong{display:block;font-size:25px;font-weight:600;margin-top:5px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:24px;margin-bottom:22px}.stack{display:flex;height:18px;border-radius:5px;overflow:hidden;background:#263141}.legend{display:flex;gap:20px;flex-wrap:wrap;margin-top:16px;color:var(--muted);font-size:12px}.legend i{display:inline-block;width:8px;height:8px;margin-right:7px;border-radius:2px}.legend b{color:var(--text);margin-left:8px;font-weight:500}.columns{display:grid;grid-template-columns:1.8fr 1fr;gap:22px}.columns .panel{min-width:0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:12px}th{text-align:left;font-weight:500;color:var(--muted)}td,th{padding:10px 12px;border-bottom:1px solid var(--line)}td:first-child,th:first-child{padding-left:0}ol{list-style:none;margin:0;padding:0}li{display:flex;justify-content:space-between;gap:16px;padding:10px 0;border-bottom:1px solid var(--line)}li small{display:block;max-width:220px;overflow:hidden;text-overflow:ellipsis}li b{font-size:13px;font-weight:500;white-space:nowrap}.filters{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:18px}input,select{font:inherit;background:#0e141f;color:var(--text);border:1px solid #39465b;border-radius:5px;padding:8px 12px}input{flex:1;min-width:160px}.axis{display:flex;justify-content:space-between;margin-left:36%;margin-right:84px;color:var(--muted);font-size:11px;padding-bottom:8px}.entry{border-top:1px solid var(--line)}summary{display:grid;grid-template-columns:36% 1fr 78px;gap:14px;align-items:center;cursor:pointer;padding:11px 0;list-style:none}summary::-webkit-details-marker{display:none}.operation{font-family:Consolas,monospace;font-size:12px;overflow-wrap:anywhere}.operation:before{content:'▸';margin-right:7px;color:#8c99ff}details[open] .operation:before{content:'▾'}.operation small{display:block;font-family:"Microsoft YaHei",sans-serif;padding-left:15px;font-size:10px}.rail{position:relative;height:22px;background:repeating-linear-gradient(to right,transparent 0,transparent calc(25% - 1px),#283242 calc(25% - 1px),#283242 25%)}.rail i{display:block;position:absolute;top:6px;height:9px;border-radius:2px;min-width:2px;max-width:100%}.elapsed{text-align:right;font-variant-numeric:tabular-nums;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#0e141f;color:#aabbd3;padding:18px;border-radius:6px;font-size:12px;margin:0 0 14px}.foot{color:var(--muted);font-size:12px;margin-top:16px}footer{color:var(--muted);font-size:11px;text-align:right}button{cursor:pointer;background:transparent;border:1px solid #39465b;color:#b7c4db;border-radius:5px;padding:8px 12px}@media(max-width:800px){main{padding:24px 14px}.columns{grid-template-columns:1fr;gap:0}.cards{grid-template-columns:repeat(2,1fr)}.panel{padding:16px}summary{grid-template-columns:46% 1fr 60px;gap:8px}.axis{margin-left:46%;margin-right:68px}.legend{gap:10px}h1{font-size:25px}}
</style></head><body><main>
<header><div class="brand">ZZCODE / LOCAL TRACE</div><h1>任务执行报告<span class="badge" style="__BADGESTYLE__">__STATUS__</span></h1><div class="meta">__DATE__ · run <code>__RUN__</code><br>会话 <code>__SESSION__</code></div></header>
__NOTICE__<section class="cards">__CARDS__</section><section class="panel"><h2>停止详情与计划进度</h2>__DETAILS__</section><section class="panel"><h2>时间花在哪里</h2><div class="stack">__SEGMENTS__</div><div class="legend">__LEGEND__</div><div class="foot">父子 span 不重复累加。模型净耗时扣除 yield 交回客户端期间的处理，包含网络等待、SDK 与解析，并非服务端纯推理时间。其他开销包括客户端处理 __CONSUMER__、准备、快照、保存记录与调度。</div></section>
<div class="columns"><section class="panel"><h2>工具执行分布</h2><div class="scroll"><table><thead><tr><th>工具</th><th>提出</th><th>执行</th><th>成功</th><th>拒绝</th><th>失败 / 超时</th><th>P50</th><th>P95</th><th>执行合计</th></tr></thead><tbody>__STATS__</tbody></table></div><div class="foot">P50 / P95 是本次已测执行耗时的分位数；样本少时仅供定位。未执行不算执行成功率。</div></section><section class="panel"><h2>最耗时的操作</h2><ol>__SLOW__</ol></section></div>
<section class="panel"><h2>模型请求延迟与用量</h2><div class="scroll"><table><thead><tr><th>轮次</th><th>请求净耗时</th><th>生命周期</th><th>客户端处理</th><th>首有效事件</th><th>首正文</th><th>输入 / 输出</th><th>其中缓存 / 思考</th><th>接口结束原因</th></tr></thead><tbody>__MODELS__</tbody></table></div><div class="foot">生命周期为请求真实起止跨度，含客户端处理。首事件 / 首正文保留从请求开始计算的实际延迟，首正文可能包含此前客户端处理；非流式压缩无法测量首段时间。缓存和思考属于输入、输出的细分，不重复计入总量。未知值保留未知。</div></section>
<section class="panel"><h2>执行瀑布图</h2><div class="filters"><input id="search" aria-label="按操作名筛选" placeholder="筛选 model_request / tool / snapshot…"><select id="status" aria-label="按状态筛选"><option value="">全部状态</option><option value="ok">成功</option><option value="error">失败</option><option value="denied">拒绝</option><option value="timeout">超时</option><option value="unknown">未知</option><option value="not_run">未执行</option></select><button id="expand">展开全部</button><button id="collapse">收起全部</button></div><div class="axis"><span>0</span><span>25%</span><span>50%</span><span>75%</span><span>__TOTAL__</span></div>__ROWS__<div class="foot">点击动作查看编号、首事件 / 首正文延迟、token 和状态。位置是相对任务开始的实际时间；未闭合动作不推断成功。相同调用与结果重复 __REPEATED__ 次，合理重测也可能产生重复。</div></section>
<footer>离线报告 · 无远程资源 · 正常结束不等于业务验收通过 · 不包含完整对话正文</footer></main><script>
const entries=[...document.querySelectorAll('.entry')];function filter(){const q=document.getElementById('search').value.toLowerCase(),s=document.getElementById('status').value;entries.forEach(e=>e.hidden=!(e.dataset.search.includes(q)&&(!s||e.dataset.status===s)))}document.getElementById('search').addEventListener('input',filter);document.getElementById('status').addEventListener('change',filter);document.getElementById('expand').onclick=()=>entries.filter(e=>!e.hidden).forEach(e=>e.open=true);document.getElementById('collapse').onclick=()=>entries.forEach(e=>e.open=false);
</script></body></html>'''
    replacements = {"STATUS": esc(STATUS.get(summary["status"], summary["status"])), "DATE": esc(trace["started_at"] or "开始时间未知"),
        "BADGESTYLE": "" if summary["status"] == "completed" else "background:#30241c;border-color:#805c35;color:#f0ce98",
        "RUN": esc(trace["run"]), "SESSION": esc(trace["session"] or "未知"),
        "NOTICE": ('<div class="notice">演示样本：模拟任务，仅用于检查报告布局，不是真实模型性能测量。</div>' if root and root["attrs"].get("demo") else "")
                  + (f'<div class="notice">旧记录中 {summary["legacy_timing"]} 次请求无法拆分客户端处理，耗时仍含客户端处理。</div>' if summary["legacy_timing"] else "")
                  + (f'<div class="notice">记录不完整：{esc(notices)}</div>' if summary["issues"] else ""),
        "DETAILS": f'<p>停止详情：{esc(stop_description(summary))}</p><p>计划状态（模型记录）：{esc(plan_description(summary))}</p><div class="foot">阶段进度来自计划状态及工具结果引用，不代表独立验收通过。</div>',
        "CONSUMER": milliseconds(summary["client_consumer_ms"]),
        "CARDS": card_html, "SEGMENTS": segments, "LEGEND": legend, "STATS": stats,
        "SLOW": slow_html, "ROWS": "".join(rows), "MODELS": "".join(model_rows),
        "TOTAL": milliseconds(summary["duration_ms"]), "REPEATED": str(summary["repeated"])}
    # 单次替换，日志文本中即使含相同占位符也不会再次被解释。
    import re
    page = re.sub(r"__([A-Z]+)__", lambda match: replacements[match.group(1)], page)
    destination.write_text(page, encoding="utf-8")
    return destination
