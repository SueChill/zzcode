"""执行一次用户任务，维护模型循环、工具结果和任务收尾。"""

import json
import sqlite3
import time
from contextlib import closing

from httpx import HTTPError
from openai import APIError

from loop_guard import LoopGuard, LoopStopped, WorkspaceVersion
from mcp_client import MCPCallError
from model import ContextLimitError, ModelReplyError
from reply_display import ReplyDisplay
from terminal_input import confirm_tool_execution
from tool_result import ToolResult
from tracing import classify_tool, current_run, fingerprint, set_run_status, span
from tools import (
    command_summary, execute_tool, file_batch_preview, prepare_tool_groups,
    resolve_project_path,
)


# 暂定值：限制单次请求的模型调用次数，合适的阈值仍需实际使用评估。
MAX_MODEL_ROUNDS = 75
# 到达该比例时提醒模型收尾；硬上限只作为保护，不承担“正常预算”。
SOFT_BUDGET_RATIO = 0.8


def soft_round_limit():
    return max(1, int(MAX_MODEL_ROUNDS * SOFT_BUDGET_RATIO))


def run_turn(text, *, model, skills, snapshots, workspace_state, workspace_root,
             tool_registry, input_session, console, permissions, show_reasoning):
    """处理本次需求；结束、取消或失败后将控制权交回输入循环。"""
    try:
        with span("skills.prepare"):
            explicit = [skills.load_skill(name) for name in skills.explicit_names(text)]
            model.set_active_skills(explicit)
    except (OSError, ValueError) as error:
        set_run_status("internal_error", preparation_error="skill", error_type=type(error).__name__,
                       stop_detail="Skill 加载失败，本次请求未执行。")
        console.print(f"Skill 加载失败：{type(error).__name__}" if isinstance(error, OSError)
                      else str(error), style="red", markup=False)
        return

    try:
        with span("snapshot.start"):
            snapshots.start()
    except (OSError, ValueError) as error:
        set_run_status("internal_error", preparation_error="snapshot", error_type=type(error).__name__,
                       stop_detail="保存初始快照失败，本次请求未执行。")
        model.set_active_skills([])
        console.print(f"保存快照失败，本次请求未执行：{error}", style="red", markup=False)
        return

    current_tool_id = None
    loop_guard = LoopGuard()
    workspace_version = WorkspaceVersion(workspace_root, resolve_project_path)
    activated = {data["name"] for data in explicit}
    for name in activated:
        console.print(f"已启用 skill：{name}（本次任务）", style="#8c8c8c", markup=False)

    def mark_started(tool_id):
        nonlocal current_tool_id
        current_tool_id = tool_id

    stats = {"rounds": 0, "tools": 0, "files": [], "commands": 0,
             "failed_commands": 0, "status": "completed", "duration": 0.0}
    started = time.perf_counter()
    try:
        for round_index in range(MAX_MODEL_ROUNDS):
            stats["rounds"] = round_index + 1
            if round_index + 1 == soft_round_limit():
                notice = f"已执行 {round_index + 1} 轮，接近本次请求的调用预算，请总结进度并收尾。"
                console.print("\n" + notice, style="#8794a5", markup=False)
                model.set_budget_notice(
                    f"本次请求已进行 {round_index + 1} 次模型调用，接近 {MAX_MODEL_ROUNDS} 次上限。"
                    "请尽快完成关键验证、总结进度并收尾，不要开启新的大范围改动。"
                )
            with span("round", {"round": round_index + 1}):
                called_tool = False
                with ReplyDisplay(console, show_reasoning=show_reasoning) as display:
                    tool_calls = _receive_reply(model, text, display)
                    for event, result in _execute_tools(
                        tool_calls, registry=tool_registry, permissions=permissions,
                        input_session=input_session, console=console, display=display,
                        loop_guard=loop_guard, on_started=mark_started, stats=stats,
                    ):
                        with span("tool.result_save", {"call_id": event["id"]}):
                            model.add_tool_result(event["id"], result)
                        _apply_tool_feedback(event, result, activated=activated, display=display,
                                             console=console, workspace_state=workspace_state)
                        current_tool_id = None
                        called_tool = True
                        with span("workspace.scan"):
                            version = workspace_version.current()
                        if current_run() is not None:
                            current_run().root.event("workspace.version", {"call_id": event["id"], "version": version.hex()})
                        loop_guard.record(event, result, version)
                if not called_tool:
                    break
                text = None
        else:
            notice = (f"已达到本次请求的 {MAX_MODEL_ROUNDS} 轮上限，停止继续调用模型。任务可能尚未完成；"
                      "输入 /continue 可继续未完成的工作。")
            set_run_status("round_limit", stop_detail=notice, round_limit=MAX_MODEL_ROUNDS)
            console.print("\n" + notice, style="#8794a5", markup=False)
            workspace_state.record_task("round_limit", notice)
        console.print()
    except LoopStopped as error:
        stats["status"] = "loop_stopped"
        set_run_status("loop_stopped", stop_detail=str(error))
        model.stop_turn(str(error))
        console.print("\n" + str(error), style="#8794a5", markup=False)
    except KeyboardInterrupt:
        stats["status"] = "cancelled"
        set_run_status("cancelled", stop_detail="用户按 Ctrl+C 取消了当前任务；已执行的操作保留。")
        model.cancel_turn(current_tool_id)
        console.print("\n已取消", style="#8794a5", markup=False)
    except ContextLimitError as error:
        stats["status"] = "context_limit"
        set_run_status("context_limit", stop_detail=str(error))
        workspace_state.record_task("context_limit", "上下文超过输入预算，本轮已停止；已完成的操作保留。")
        console.print("\n" + str(error), style="red", markup=False)
    except (APIError, HTTPError, ConnectionError, TimeoutError, ModelReplyError) as error:
        stats["status"] = "model_error"
        model.fail_turn(current_tool_id)
        status = getattr(error, "status_code", None)
        message = str(error) if isinstance(error, ModelReplyError) else (
            f"调用失败（HTTP {status}）。" if status else "调用失败，请检查网络。"
        )
        set_run_status("model_error", error_type=type(error).__name__, http_status=status, stop_detail=message)
        console.print("\n" + message, style="red", markup=False)
    except MCPCallError as error:
        stats["status"] = "internal_error"
        set_run_status("internal_error", error_type=type(error).__name__, stop_detail="MCP 工具调用异常，本轮停止。")
        model.fail_turn(current_tool_id)
        console.print("\n" + str(error), style="red", markup=False)
    except sqlite3.Error:
        stats["status"] = "internal_error"
        set_run_status("internal_error", error_type="sqlite3.Error", stop_detail="会话数据库写入失败，停止后续操作。")
        raise
    except Exception as error:
        stats["status"] = "internal_error"
        set_run_status("internal_error", error_type=type(error).__name__, stop_detail="本轮发生程序异常，已执行的操作不会自动重跑。")
        model.fail_turn(current_tool_id)
        console.print(f"\n本轮异常（{type(error).__name__}），已停止；已执行的操作不会自动重跑。", style="red", markup=False)
    finally:
        stats["duration"] = time.perf_counter() - started
        model.set_active_skills([])
        try:
            with span("snapshot.finish"):
                snapshots.finish()
        except (OSError, ValueError) as error:
            stats["status"] = "internal_error"
            set_run_status("internal_error", cleanup_error="snapshot", stop_detail="保存撤销点失败，本次修改无法撤销。")
            console.print(f"保存撤销点失败，本次修改无法撤销：{error}", style="red", markup=False)
    return stats


def _receive_reply(model, text, display):
    """显示流式正文和思考，收集完整工具调用后再交给执行阶段。"""
    tool_calls = []
    with closing(model.reply(text)) as chunks:
        for event in chunks:
            if event["type"] == "reasoning":
                display.reasoning(event["content"])
            elif event["type"] == "text":
                display.text(event["content"])
            elif event["type"] == "tool_call":
                tool_calls.append(event)
    return tool_calls


def _execute_tools(tool_calls, *, registry, permissions, input_session, console,
                   display, loop_guard, on_started, stats):
    """提出的调用全部留痕；实际执行、审批和准备使用独立计时。"""
    run = current_run()
    operations_trace = []
    for event in tool_calls:
        arguments = event.get("arguments", "")
        try:
            normalized = json.loads(arguments)
        except (ValueError, TypeError):
            normalized = arguments
        attrs = {"call_id": event["id"], "tool": event["name"], "executed": False,
                 "exec_ms": None, "approval_ms": 0, "arguments_sha256": fingerprint([event["name"], normalized])}
        if run is not None:
            attrs["arguments"] = run.describe(arguments)
            if isinstance(normalized, dict):
                attrs.update({key: normalized[key] for key in ("path", "program", "cwd") if isinstance(normalized.get(key), str)})
        operation_trace = span("tool:" + event["name"], attrs)
        operation_trace.status = "not_run"
        operation_trace.start()
        operations_trace.append(operation_trace)

    def approve(operation, *, file_edit=False, batch_id=None):
        display.pause()
        with span("approval", {"batch_id": batch_id, "bucket": "approval"}) as approval:
            decision = confirm_tool_execution(
                input_session, console, operation,
                permissions=permissions if file_edit else None,
            )
            approval.status = "ok" if decision is True else "denied"
            approval.attrs["decision"] = "allowed" if decision is True else "denied"
        return decision, approval

    groups = iter(prepare_tool_groups(tool_calls, registry))
    offset = 0
    try:
        while True:
            with span("batch.prepare") as preparation:
                group = next(groups, None)
                preparation.attrs["calls"] = len(group[0]) if group is not None else 0
            if group is None:
                break
            calls, changes = group
            group_traces = operations_trace[offset:offset + len(calls)]
            offset += len(calls)
            for call in calls:
                loop_guard.check(call)
            shared_approval = None
            if changes is not None:
                preview = file_batch_preview(changes)
                display.tool(preview.title)
                if permissions.allows(registry[calls[0]["name"]]):
                    decision = True
                else:
                    decision, shared_approval = approve(preview, file_edit=True,
                                                        batch_id=group_traces[0].id)

            for index, event in enumerate(calls):
                active = group_traces[index]
                with active:
                    loop_guard.check(event)
                    if changes is None:
                        display.tool(summarize_tool_call(event))
                    started = None

                    def mark_started():
                        nonlocal started
                        on_started(event["id"])
                        active.attrs["executed"] = True
                        active.status = "unknown"
                        display.executing(event["name"])
                        started = time.perf_counter_ns()

                    def approve_call(operation):
                        if changes is not None:
                            if shared_approval is not None:
                                active.attrs["shared_approval_id"] = shared_approval.id
                                active.attrs["approval_ms"] = None
                            return decision
                        file_edit = registry.get(event["name"], {}).get("permission_group") == "file_edit"
                        selected, approval = approve(operation, file_edit=file_edit)
                        active.attrs["approval_id"] = approval.id
                        active.attrs["approval_ms"] = (time.perf_counter_ns() - approval.started_ns) / 1e6
                        return selected

                    prepared = {"prepared": changes[index]} if changes is not None else {}
                    try:
                        result = execute_tool(event, registry=registry, approve=approve_call,
                                              on_execute=mark_started, permissions=permissions, **prepared)
                    finally:
                        if started is not None:
                            active.attrs["exec_ms"] = (time.perf_counter_ns() - started) / 1e6
                    active.status = classify_tool(result)
                    if run is not None:
                        active.attrs["result"] = run.describe(result)
                    view = ToolResult.parse(result)
                    metadata = view.data
                    if metadata:
                        active.attrs.update({key: metadata[key] for key in
                            ("exit_code", "timed_out", "stdout_truncated", "stderr_truncated", "truncated", "error_code", "failure_scope", "revalidation") if key in metadata})
                    stats["tools"] += 1
                    if view.status == "ok" and event["name"] in {"write_file", "edit_file", "delete_file"}:
                        path = metadata.get("path")
                        if isinstance(path, str) and path not in stats["files"]:
                            stats["files"].append(path)
                    if event["name"] in {"run_command", "run_python"}:
                        stats["commands"] += 1
                        if metadata.get("timed_out") or metadata.get("exit_code") not in (None, 0):
                            stats["failed_commands"] += 1
                yield event, result
    finally:
        for operation_trace in operations_trace:
            operation_trace.end()


def _apply_tool_feedback(event, result, *, activated, display, console, workspace_state):
    """处理技能启用提示及状态查询回执，不负责执行或保存工具结果。"""
    if event["name"] not in {"load_skill", "query_workspace_state"}:
        return
    data = ToolResult.parse(result).data
    if not data:
        return
    if event["name"] == "load_skill":
        if isinstance(data.get("instructions"), str) and data.get("name") not in activated:
            activated.add(data["name"])
            display.pause()
            console.print(f"已启用 skill：{data['name']}（本次任务）", style="#8c8c8c", markup=False)
    else:
        workspace_state.acknowledge(data.get("revision"))


def summarize_tool_call(call):
    """只缩短终端日志，完整参数仍交给工具执行。"""
    def compact(value):
        text = " ".join(value.split())
        text = "".join(character for character in text if character.isprintable())
        return text if len(text) <= 80 else text[:77] + "..."

    name = compact(call["name"])
    try:
        arguments = json.loads(call["arguments"])
    except (ValueError, TypeError):
        return f"{name} · 参数无法解析"
    if not isinstance(arguments, dict):
        return f"{name} · 参数不是对象"

    parts = [name]
    if name == "run_command":
        if isinstance(arguments.get("program"), str):
            args = arguments.get("args", [])
            args = [arg for arg in args if isinstance(arg, str)] if isinstance(args, list) else []
            parts.append(compact(command_summary(arguments["program"], args, 80)))
        if isinstance(arguments.get("cwd"), str):
            parts.append(f"目录 {compact(arguments['cwd'])}")
    path = arguments.get("path", "." if name in {"list_files", "grep"} else None)
    if isinstance(path, str):
        parts.append(compact(path))
    if isinstance(arguments.get("owner"), str) and isinstance(arguments.get("repo"), str):
        parts.append(compact(f"{arguments['owner']}/{arguments['repo']}"))
    if isinstance(arguments.get("query"), str):
        parts.append(compact(arguments["query"]))
    if isinstance(arguments.get("content"), str):
        parts.append(f"{len(arguments['content'])} 字符")
    elif isinstance(arguments.get("old_text"), str) and isinstance(arguments.get("new_text"), str):
        parts.append(f"{len(arguments['old_text'])} → {len(arguments['new_text'])} 字符")
    if name == "list_files":
        depth = arguments.get("depth", 1)
        parts.append(f"深度 {depth}" if type(depth) is int and 1 <= depth <= 5 else "深度参数无效")
    if name == "read_file" and ("start_line" in arguments or "line_count" in arguments):
        parts.append(f"第 {arguments.get('start_line', 1)} 行起 · {arguments.get('line_count', 200)} 行")
    return " · ".join(parts)
