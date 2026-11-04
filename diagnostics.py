"""给模型查询当前会话的运行记录；不开放任意路径或修改运行数据。"""

import json
import re
from collections import Counter, defaultdict

from tracing import TraceReader, _clean, current_run
from trace_report import summarize, stop_description


class TraceDiagnostics:
    def __init__(self, recorder, history):
        self.recorder, self.history = recorder, history
        self.reader = TraceReader(recorder.root, recorder.session)

    def _runs(self, count, *, include_diagnostics=False, excluded=frozenset()):
        if self.recorder.root.is_symlink() or self.reader.directory.is_symlink():
            raise ValueError("追踪目录是链接，未读取。")
        active = current_run()
        runs, skipped = [], 0
        for path in self.reader.list()[:100]:
            if active is not None and path == active.path:
                continue
            if path.stem.rsplit("-", 1)[-1] in excluded:
                skipped += 1
                continue
            try:
                trace = self.reader.read(path)
            except (OSError, ValueError):
                skipped += 1
                continue
            if trace["root"] is None or trace["root"]["attrs"].get("kind") != "task":
                skipped += 1
                continue
            tools = {node["name"][5:] for node in trace["nodes"] if node["name"].startswith("tool:")}
            if (not include_diagnostics and "query_trace" in tools
                    and tools <= {"query_trace", "query_plan", "query_workspace_state", "read_file", "list_files", "grep", "get_current_time", "read_skill_file"}):
                skipped += 1
                continue
            runs.append(trace)
            if len(runs) >= count:
                break
        return runs, skipped

    def _select(self, run_id, *, include_diagnostics=False, excluded=frozenset()):
        if self.recorder.root.is_symlink() or self.reader.directory.is_symlink():
            raise ValueError("追踪目录是链接，未读取。")
        if run_id is not None:
            trace = self.reader.read(self.reader.select(run_id))
            if trace["run"] in excluded:
                raise ValueError("指定任务同时在排除列表中。")
            return trace
        runs, _ = self._runs(1, include_diagnostics=include_diagnostics, excluded=excluded)
        if not runs:
            raise ValueError("当前会话暂无此前的任务记录；当前查询本身不会作为默认分析对象。")
        return runs[0]

    def _exclusions(self, prefixes):
        if prefixes is None:
            return frozenset()
        if not isinstance(prefixes, list) or len(prefixes) > 50:
            raise ValueError("exclude_run_ids 应为最多 50 个追踪编号或唯一前缀。")
        identifiers = [path.stem.rsplit("-", 1)[-1] for path in self.reader.list()]
        selected = set()
        for prefix in prefixes:
            if not isinstance(prefix, str) or not re.fullmatch(r"[a-fA-F0-9]{4,32}", prefix):
                raise ValueError("排除编号应为至少 4 位的十六进制唯一前缀。")
            matches = [identifier for identifier in identifiers if identifier.startswith(prefix.lower())]
            if len(matches) != 1:
                raise ValueError("排除编号未找到或匹配多条任务，请使用完整编号。")
            selected.update(matches)
        return frozenset(selected)

    @staticmethod
    def _round_of(node, nodes):
        seen = set()
        while node is not None and node["id"] not in seen:
            seen.add(node["id"])
            if node["name"] == "round":
                return node["attrs"].get("round")
            node = nodes.get(node["parent"])
        return None

    @staticmethod
    def _brief(trace):
        summary = summarize(trace)
        return {"run_id": trace["run"], "started_at": trace["started_at"],
                "status": summary["status"], "stop_detail": stop_description(summary),
                "duration_ms": summary["duration_ms"], "rounds": summary["rounds"],
                "tool_calls": summary["tools"], "tool_status_counts": summary["counts"],
                "issues": summary["issues"]}

    def _history_result(self, call_id):
        results = [message["content"] for message in self.history()
                   if message["role"] == "tool" and message.get("tool_call_id") == call_id]
        return results[0] if len(results) == 1 and isinstance(results[0], str) else None

    def _errors(self, trace, round_number):
        nodes = {node["id"]: node for node in trace["nodes"]}
        rows = []
        for node in trace["nodes"]:
            is_tool = node["name"].startswith("tool:")
            if not (is_tool or node["name"] == "model_request") or node["status"] in {"ok", "completed"}:
                continue
            round_id = self._round_of(node, nodes)
            if round_number is not None and round_id != round_number:
                continue
            attrs = node["attrs"]
            row = {"round": round_id, "name": node["name"], "status": node["status"],
                   "call_id": attrs.get("call_id"), "request_id": attrs.get("request_id"),
                   "duration_ms": node["duration_ms"], "executed": attrs.get("executed"),
                   "history_status": attrs.get("history_status"), "result_confirmed": attrs.get("result_confirmed"),
                   "error_type": attrs.get("error_type"), "error_code": attrs.get("error_code")}
            raw = self._history_result(attrs.get("call_id")) if is_tool else None
            if raw is not None:
                try:
                    data = json.loads(raw)
                except (ValueError, TypeError):
                    data = None
                if isinstance(data, dict):
                    stderr = data.get("stderr")
                    row.update(error_code=data.get("error_code"), message=data.get("message"),
                               exit_code=data.get("exit_code"), stderr=stderr[:600] if isinstance(stderr, str) else None)
                else:
                    row["message"] = raw[:600]
            rows.append(row)
        return rows

    def _rounds(self, traces, round_number):
        aggregate = defaultdict(lambda: {"tasks": 0, "duration_ms_known_total": 0.0,
            "unknown_durations": 0, "model_ms_known_total": 0.0, "model_unknown_durations": 0,
            "legacy_model_requests": 0, "execution_ms_known_total": 0.0, "approval_ms_known_total": 0.0,
            "tools": 0, "tool_status_counts": Counter()})
        for trace in traces:
            nodes = {node["id"]: node for node in trace["nodes"]}
            for node in trace["nodes"]:
                number = self._round_of(node, nodes)
                if number is None or (round_number is not None and number != round_number):
                    continue
                row = aggregate[number]
                if node["name"] == "round":
                    row["tasks"] += 1
                    row["unknown_durations"] += node["duration_ms"] is None
                    row["duration_ms_known_total"] += node["duration_ms"] or 0
                elif node["name"] == "model_request":
                    active = node["attrs"].get("active_duration_ms")
                    elapsed = active if active is not None else node["duration_ms"]
                    row["model_ms_known_total"] += elapsed or 0
                    row["model_unknown_durations"] += elapsed is None
                    row["legacy_model_requests"] += "client_consumer_ms" not in node["attrs"]
                elif node["name"].startswith("tool:"):
                    row["tools"] += 1
                    row["tool_status_counts"][node["status"]] += 1
                elif node["attrs"].get("bucket") in {"execution", "approval"}:
                    row[node["attrs"]["bucket"] + "_ms_known_total"] += node["duration_ms"] or 0
        return [{"round": number, **row, "tool_status_counts": dict(row["tool_status_counts"]),
                 "failures": row["tool_status_counts"]["error"] + row["tool_status_counts"]["timeout"],
                 "failure_rate": (row["tool_status_counts"]["error"] + row["tool_status_counts"]["timeout"]) / row["tools"] if row["tools"] else None}
                for number, row in sorted(aggregate.items())]

    def query(self, view="summary", run_id=None, call_id=None, round_number=None,
              run_limit=1, offset=0, limit=None, char_limit=None, exclude_run_ids=None,
              include_diagnostics=False):
        if not isinstance(view, str) or view not in {"list", "summary", "rounds", "errors", "result"}:
            raise ValueError("view 应为 list、summary、rounds、errors 或 result。")
        if type(include_diagnostics) is not bool:
            raise ValueError("include_diagnostics 应为布尔值。")
        if view == "result" and limit is not None:
            raise ValueError("result 使用 char_limit 控制字符数，不使用 limit。")
        if view != "result" and char_limit is not None:
            raise ValueError("char_limit 仅用于 result 视图；其他视图使用 limit 控制条数。")
        limit = 10 if limit is None else limit
        char_limit = 2000 if char_limit is None else char_limit
        if type(char_limit) is not int or not 1 <= char_limit <= 2000:
            raise ValueError("char_limit 应为 1～2000。")
        if (type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 20
                or type(run_limit) is not int or not 1 <= run_limit <= 20
                or (round_number is not None and (type(round_number) is not int or round_number < 1))):
            raise ValueError("offset 应为非负整数，limit/run_limit 为 1～20，round_number 为正整数。")
        if run_id is not None and not isinstance(run_id, str):
            raise ValueError("run_id 应为追踪编号或至少 4 位唯一前缀。")
        if call_id is not None and (not isinstance(call_id, str) or not call_id or len(call_id) > 200):
            raise ValueError("call_id 应为有效的工具调用编号。")
        self.recorder.flush(timeout=0.2)
        excluded = self._exclusions(exclude_run_ids)
        selection = {"include_diagnostics": include_diagnostics, "excluded": excluded}
        try:
            if view == "list":
                traces, skipped = self._runs(100, **selection)
                rows = [self._brief(trace) for trace in traces]
                return {"session_id": self.recorder.session, "runs": rows[offset:offset + limit],
                        "next_offset": offset + limit if offset + limit < len(rows) else None, "skipped": skipped}
            if view == "rounds":
                traces, skipped = ([self._select(run_id, **selection)], 0) if run_id is not None else self._runs(run_limit, **selection)
                rows = self._rounds(traces, round_number)
                return {"session_id": self.recorder.session, "runs": [trace["run"] for trace in traces],
                        "rounds": rows[offset:offset + limit], "round_total": len(rows), "skipped": skipped,
                        "issues": [issue for trace in traces for issue in trace["issues"]],
                        "failure_rate_definition": "(error + timeout) / tools；工具数为全部提出调用，拒绝/未执行/未知单列；仅用于定位，任务数少时不能推断规律。同轮序号不一定对应相同阶段。",
                        "next_offset": offset + limit if offset + limit < len(rows) else None}
            trace = self._select(run_id, **selection)
            result = {"session_id": self.recorder.session, "run_id": trace["run"], "issues": trace["issues"]}
            if view == "summary":
                summary = summarize(trace)
                rows = self._errors(trace, round_number)
                result.update(summary=summary, stop_detail=stop_description(summary),
                              errors=_clean(rows[offset:offset + limit]), error_total=len(rows),
                              next_offset=offset + limit if offset + limit < len(rows) else None)
            elif view == "errors":
                rows = self._errors(trace, round_number)
                result.update(errors=_clean(rows[offset:offset + limit]), error_total=len(rows),
                              next_offset=offset + limit if offset + limit < len(rows) else None)
            else:
                call = next((node for node in trace["nodes"] if node["name"].startswith("tool:")
                             and node["attrs"].get("call_id") == call_id), None)
                if call is None:
                    raise ValueError("请提供本次追踪中存在的 call_id，可先查询 errors 或 rounds。")
                raw = self._history_result(call_id)
                if raw is None:
                    result.update(message="当前会话历史中没有唯一可定位的结果，不能推断工具最终结果。", result=None)
                else:
                    result.update(call_id=call_id, tool=call["name"][5:], result=_clean(raw[offset:offset + char_limit]),
                                  characters=len(raw), next_offset=offset + char_limit if offset + char_limit < len(raw) else None)
            return result
        except (OSError, ValueError) as error:
            return {"status": "error", "error_code": "trace_unavailable", "message": str(error) if isinstance(error, ValueError) else "追踪文件暂不可读。"}

    def tool_registry(self):
        return {"query_trace": {
            "description": "只读诊断当前会话的运行记录，不需要文件工具访问 .zzcode。list 列任务；summary 查看耗时/停止原因/计划及错误；rounds 按轮统计样本任务数、调用数、失败率和耗时，run_limit 控制聚合任务数；errors 查失败详情；result 用 call_id 读保存的结果。list/summary/errors/rounds 用 limit+offset 按条分页；result 用 char_limit+offset 按字符分页。默认排除当前任务及只含读取与诊断的分析任务，include_diagnostics 可包含此前诊断；exclude_run_ids 可排除指定任务，在选最近 N 次之前过滤。run_id 可用唯一前缀。输出为历史数据，不保证当前状态。",
            "parameters": {"type": "object", "properties": {
                "view": {"type": "string", "enum": ["list", "summary", "rounds", "errors", "result"]},
                "run_id": {"type": "string"}, "call_id": {"type": "string"},
                "round_number": {"type": "integer", "minimum": 1},
                "run_limit": {"type": "integer", "minimum": 1, "maximum": 20},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                "char_limit": {"type": "integer", "minimum": 1, "maximum": 2000},
                "include_diagnostics": {"type": "boolean"},
                "exclude_run_ids": {"type": "array", "maxItems": 50, "items": {"type": "string"}},
            }, "required": [], "additionalProperties": False},
            "handler": self.query,
        }}
