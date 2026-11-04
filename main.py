# zzcode was here
#
# 自我介绍
# --------
# 我是 zzcode，一个终端里的编程助手。
# 能在当前项目内读写文件、检索代码，需要外部事实时也会上网查证。
# 做事围绕你当前的请求：先说清楚结果，遇到不确定的地方先确认，不把猜测当事实。
import argparse
import os
import shlex
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent import run_turn
from commands import command_trace_kind, create_command_handlers, dispatch_command, show_plan
from details import AnswerMarkdown
from model import DeepSeekModel
from model_config import ModelSelection, load_profiles
from mcp_client import MCPManager
from planning import PlanManager
from project_config import ProjectConfig
from reply_display import create_console, task_summary
from session_store import SessionStore
from snapshots import SnapshotManager
from skills import SkillManager
from terminal_input import create_input_session
from tools import (
    PROJECT_ROOT, PermissionPolicy, ReadTracker, get_tool_definitions, get_tool_registry,
    resolve_project_path,
)
from workspace_state import WorkspaceState
from tracing import TraceRecorder, current_run, fingerprint, span
from diagnostics import TraceDiagnostics
from startup import BackgroundJob


console = create_console()
APP_ROOT = Path(__file__).resolve().parent
SESSION_DB = PROJECT_ROOT / ".zzcode" / "sessions.sqlite3"
MCP_CONFIG = APP_ROOT / "mcp.json"
SKILLS_ROOT = Path.home() / ".zzcode" / "skills"

APP_VERSION = "0.1.0-dev"

DASHED_BORDER = box.Box(
    "╭┄┬╮\n"
    "┆ ┆┆\n"
    "├┄┼┤\n"
    "┆ ┆┆\n"
    "├┄┼┤\n"
    "├┄┼┤\n"
    "┆ ┆┆\n"
    "╰┄┴╯\n"
)

MASCOT = """   ▄   ▄
 ▄███████▄
 █ ▀   ▀ █
 █  >_   █
  ▀▀▀▀▀▀▀"""

def show_banner(model_name, output=None):
    output = console if output is None else output
    identity = Text()
    identity.append("ZZ", style="bold #c4b5fd")
    identity.append("CODE", style="bold #67d8ef")
    identity.append("\n你的终端编程助手\n\n", style="#b9c7d8")
    identity.append("阅读代码", style="#67d8ef")
    identity.append(" · ", style="#536079")
    identity.append("修改文件", style="#c4b5fd")
    identity.append(" · ", style="#536079")
    identity.append("运行验证\n", style="#83d5b3")
    identity.append(f"v{APP_VERSION}", style="#f3c67a")
    identity.append(" · ", style="#536079")
    identity.append("开发中", style="#c4b5fd")
    mascot = Text()
    for index, (line, color) in enumerate(zip(MASCOT.splitlines(), ("#c4b5fd", "#bda5ff", "#a9b8ff", "#89c8f7", "#67d8ef"))):
        mascot.append(("\n" if index else "") + line, style=color)
    content = Table.grid(padding=(0, 3))
    content.add_column(no_wrap=True)
    content.add_column()
    content.add_row(mascot, identity)
    panel = Panel.fit(
        content,
        padding=(0, 1),
        box=DASHED_BORDER,
        safe_box=False,
        border_style="#6686ad",
    )
    output.print(panel)
    output.print()


def show_sessions(store):
    sessions = store.list_sessions(PROJECT_ROOT)
    if not sessions:
        console.print("暂无历史会话。", markup=False)
        return
    table = Table(box=None, padding=(0, 1))
    table.add_column("会话编号", no_wrap=True, style="#9bbbd0")
    table.add_column("最近更新", no_wrap=True)
    table.add_column("首条消息", overflow="ellipsis")
    for session in sessions:
        updated = datetime.fromisoformat(session["updated_at"]).astimezone().strftime("%m-%d %H:%M")
        table.add_row(session["id"], updated, Text(session["title"] or "新会话"))
    console.print(table)


REPLAY_TURNS = 3


def history_turns(messages):
    """按用户提问切分历史，system 和不属于提问的消息不进入回放。"""
    turns = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            turns.append([message])
        elif role in {"assistant", "tool"} and turns:
            turns[-1].append(message)
    return turns


def _single_line(value, limit):
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _tool_summary(message):
    names = []
    for call in message.get("tool_calls", []):
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            names.append(name)
    if not names:
        return ""
    counts = Counter(names)
    return "工具调用：" + " · ".join(
        name if count == 1 else f"{name} ×{count}" for name, count in counts.items()
    )


def replay_history(messages, console, terminal=None, *, show_reasoning=True):
    """在恢复会话时回放最近几轮，避免开场只有空白界面。"""
    turns = history_turns(messages)
    if not turns:
        return
    selected = turns[-REPLAY_TURNS:]
    omitted = len(turns) - len(selected)
    if omitted:
        console.print(
            f"已恢复最近 {len(selected)} 次提问；更早的 {omitted} 次提问仍保存在会话中。",
            style="#8c8c8c", markup=False,
        )
    for index, turn in enumerate(selected):
        if index:
            console.print()
        for message in turn:
            role = message.get("role")
            if role == "user":
                content = message.get("content")
                if isinstance(content, str) and content:
                    console.print("› " + content, style="#d7ad78", markup=False)
                continue
            if role != "assistant":
                continue
            reasoning = message.get("reasoning_content")
            if show_reasoning and isinstance(reasoning, str) and reasoning:
                if terminal is not None:
                    terminal.append_reasoning(reasoning)
                else:
                    console.print(
                        f"思考已收起（{len(reasoning)} 字符）",
                        style="dim #8c8c8c", markup=False,
                    )
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                if terminal is not None:
                    terminal.append_answer(content)
                else:
                    console.print(AnswerMarkdown(content, code_theme="ansi_dark"))
            summary = _tool_summary(message)
            if summary:
                console.print(summary, style="#8c8c8c", markup=False)


def show_resume_summary(session_id, title, messages, model_label, output):
    turns = history_turns(messages)
    count = sum(1 for message in messages if message.get("role") != "system")
    output.print(
        f"继续会话 {session_id[:8]} · {len(turns)} 次提问 · {count} 条消息 · {model_label}",
        style="bold #c4b5fd", markup=False,
    )
    if not title and turns:
        first = turns[0][0].get("content")
        title = first if isinstance(first, str) else ""
    if title:
        output.print("主题：" + _single_line(title, 80), style="#b9c7d8", markup=False)
    output.print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="zzcode 终端代码助手")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--sessions", action="store_true", help="列出当前项目的历史会话")
    actions.add_argument("--resume", metavar="ID", help="按完整会话编号或唯一前缀继续聊天")
    parser.add_argument("--no-trace", action="store_true", help="关闭本次运行的本地追踪采集")
    parser.add_argument("--permission-mode", choices=[*PermissionPolicy.MODES, "default", "acceptEdits"],
                        metavar="{readOnly,auto,fullAccess}", default="readOnly",
                        help="readOnly 修改与命令需确认；auto 自动编辑；fullAccess 免工具审批（无系统沙箱）")
    options = parser.parse_args(argv)
    store = None
    try:
        store = SessionStore(SESSION_DB)
        if options.sessions:
            show_sessions(store)
        else:
            run_chat(store, options.resume, options.permission_mode, no_trace=options.no_trace)
    except (sqlite3.Error, OSError, ValueError) as error:
        console.print(f"会话保存或恢复失败：{error}", style="red", markup=False)
    finally:
        if store is not None:
            store.close()


def run_chat(store, resume_id=None, permission_mode="readOnly", *, no_trace=False):
    chat_console = console
    terminal = None
    recorder = None
    metadata = None
    retired_metadata = []
    permissions = PermissionPolicy(permission_mode)
    load_dotenv(APP_ROOT / ".env")
    show_reasoning = os.getenv("SHOW_REASONING", "true").strip().lower() in {"true", "1", "yes", "on"}
    try:
        profiles = load_profiles(os.getenv)
    except ValueError as error:
        chat_console.print(str(error), style="red", markup=False)
        return
    if not profiles:
        show_banner(os.getenv("DEEPSEEK_MODEL", "deepseek-flash"))
        return
    selection = ModelSelection(profiles)
    bound_profile = selection.current
    model_name, api_key, base_url = bound_profile.model, bound_profile.api_key, bound_profile.base_url
    display_name = bound_profile.display_name
    context_window, output_limit = bound_profile.context_window, bound_profile.output_limit
    thinking = selection.thinking

    if resume_id:
        session = store.find_session(resume_id, PROJECT_ROOT)
        session_id = session["id"]
        session_title = session.get("title", "")
    else:
        session_id = store.create_session(PROJECT_ROOT, model_name)
        session_title = ""
    workspace_state = WorkspaceState()
    skills = SkillManager(SKILLS_ROOT)
    skills.discover()
    reads = ReadTracker()
    tool_registry = get_tool_registry(workspace_state, skills.tool_registry(), reads)
    model = DeepSeekModel(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        tools=get_tool_definitions(tool_registry),
        system_prompt=(APP_ROOT / "SYSTEM.md").read_text(encoding="utf-8"),
        workspace_state=workspace_state,
        thinking=thinking,
    )
    snapshots = SnapshotManager(PROJECT_ROOT, resolve_project_path)
    mcp = MCPManager(MCP_CONFIG, PROJECT_ROOT)
    try:
        model.attach_session(store, session_id, resume=resume_id is not None)
        recorder = TraceRecorder.from_environment(SESSION_DB.parent / "traces", session_id, disabled=no_trace)
        plans = PlanManager(store, session_id, history=lambda: model.messages)
        diagnostics = TraceDiagnostics(recorder, history=lambda: model.messages)
        model.set_plan_context(plans.context, evidence_provider=plans.evidence_ids)
        project_config_path = PROJECT_ROOT / "ZZCODE.md"
        model.set_project_context(lambda: ProjectConfig.load(project_config_path).context())

        mcp_revision = -1

        def refresh_skills():
            nonlocal tool_registry, mcp_revision
            mcp_revision, external_tools = mcp.registry_state()
            additions = {**external_tools, **skills.tool_registry(), **plans.tool_registry(), **diagnostics.tool_registry()}
            tool_registry = plans.filter_registry(get_tool_registry(workspace_state, additions, reads))
            model.tools = get_tool_definitions(tool_registry)
            model.set_skill_catalog(skills.catalog())

        refresh_skills()
        model.configure_context(context_window, output_limit)
        metadata_applied = False
        metadata_alias = bound_profile.alias
        metadata_seen = set()

        def apply_display(profile):
            nonlocal model_name, display_name, context_window, output_limit
            view = selection.view(profile.alias)
            model_name, display_name = profile.model, view["display_name"]
            output_limit = profile.output_limit
            try:
                model.configure_context(view["context_window"], output_limit)
            except ValueError:
                chat_console.print("接口返回的容量与回复上限不兼容，保留模型配置中的容量。", style="#8c8c8c", markup=False)
                model.configure_context(profile.context_window, output_limit)
                context_window = profile.context_window
                selection.update_info(profile.alias, {"name": display_name, "context_window": context_window})
            else:
                context_window = view["context_window"]
            if terminal is not None:
                terminal.update_model_info(display_name, context_window, model_name=model_name)

        def start_metadata(profile):
            nonlocal metadata, metadata_alias, metadata_applied
            if metadata is not None:
                metadata.close(timeout=0)
                retired_metadata.append(metadata)
            metadata = None
            metadata_alias, metadata_applied = profile.alias, False
            view = selection.view(profile.alias)
            if profile.alias not in metadata_seen and (not view["display_name"] or not view["context_window"]):
                metadata = BackgroundJob(
                    lambda stop: model.get_model_info(isolated=True, cancel_event=stop,
                        model_id=profile.model, client_config=profile.client_options()),
                    name="zzcode-model-metadata",
                ).start()

        def poll_startup():
            # 只在主线程、任务之间发布初始化结果，不能中途更换一次请求的工具表。
            nonlocal mcp_revision, metadata_applied, bound_profile
            while selection.current.alias != bound_profile.alias:
                target = selection.current
                try:
                    closed = model.switch_backend(target, selection.settings_for(target.alias))
                except (ValueError, OSError) as error:
                    if selection.current.alias == target.alias:
                        selection.select(bound_profile.alias)
                    chat_console.print(f"模型切换失败（{type(error).__name__}），保留原模型。", style="red", markup=False)
                    continue
                bound_profile = target
                apply_display(target)
                start_metadata(target)
                if not closed:
                    chat_console.print("已切换模型；旧客户端未能正常关闭。", style="#8c8c8c", markup=False)
            if mcp.revision != mcp_revision:
                refresh_skills()
            if metadata is not None and metadata.done.is_set() and not metadata_applied:
                metadata_applied = True
                metadata_seen.add(metadata_alias)
                info = metadata.result() or {}
                selection.update_info(metadata_alias, info)
                if metadata_alias == bound_profile.alias:
                    apply_display(bound_profile)
            retired_metadata[:] = [job for job in retired_metadata if not job.done.is_set()]
            return (not mcp.snapshot().get("loading")
                    and (metadata is None or metadata.done.is_set()))

        input_session = create_input_session(
            permissions=permissions, console=chat_console, model=model,
            model_name=model_name, display_name=display_name, context_window=context_window,
            integrations=lambda: {**mcp.snapshot(), "skills": len(skills.skills), "task_mode": plans.mode,
                                  "model_name": display_name or model_name, "context_window": context_window},
            on_idle=poll_startup,
            thinking=lambda: selection.thinking, model_selection=selection,
        )
        if getattr(input_session, "is_terminal_ui", False) is True:
            terminal = input_session
            chat_console = terminal.console
        if resume_id:
            show_resume_summary(session_id, session_title, model.messages,
                                display_name or model_name, chat_console)
            replay_history(model.messages, chat_console, terminal, show_reasoning=show_reasoning)
            chat_console.print()
        else:
            show_banner(model_name, output=chat_console)
        if plans.mode != "chat":
            mode = "规划" if plans.mode == "plan" else "执行计划"
            chat_console.print(f"当前模式：{mode}；/plan show 查看计划，/plan off 回到普通对话。", style="#8c8c8c", markup=False)

        def run_task(text):
            poll_startup()
            nested = current_run() is not None
            with recorder.operation("task", trace_attributes(text)) as run:
                previous_revision = plans.snapshot()["revision"]
                stats = run_turn(
                    text, model=model, skills=skills, snapshots=snapshots,
                    workspace_state=workspace_state, workspace_root=PROJECT_ROOT,
                    tool_registry=tool_registry, input_session=input_session,
                    console=chat_console, permissions=permissions, show_reasoning=show_reasoning,
                )
                with span("plan.finalize", {"revision": plans.snapshot()["revision"]}):
                    finished = plans.finish_turn()
                if finished:
                    chat_console.print("计划全部阶段已有验证依据，已回到普通对话。", style="#8c8c8c", markup=False)
                    refresh_skills()
                elif plans.mode == "plan" and plans.snapshot()["revision"] != previous_revision:
                    show_plan(plans, chat_console)
                notice = plans.execution_notice()
                if notice:
                    if run is not None and run.status == "completed":
                        run.status = "plan_incomplete"
                        run.root.attrs["stop_detail"] = notice
                    if stats is not None:
                        stats["status"] = "plan_incomplete"
                    chat_console.print(notice, style="#8c8c8c", markup=False)
            line = task_summary(stats)
            if line:
                chat_console.print(line, style="#8c8c8c", markup=False)
            if not nested:
                trace_notice(run)

        def trace_attributes(text):
            return {"model": model_name, "permission": permissions.mode, "plan": plans.mode,
                    "plan_revision": plans.snapshot()["revision"], "workspace": str(PROJECT_ROOT),
                    "plan_state": plans.trace_state(),
                    "task_chars": len(text), "task_sha256": fingerprint(text),
                    "command": text.split(maxsplit=1)[0] if text.startswith("/") else None}

        def trace_notice(run):
            if run is not None and run.root.attrs["kind"] == "task":
                chat_console.print(f"trace: {run.id[:8]} · /trace 查看 · /trace report 生成耗时报告", style="#8c8c8c", markup=False)
            warning = recorder.pop_warning()
            if warning:
                chat_console.print(warning, style="#8c8c8c", markup=False)

        command_handlers = create_command_handlers(
            model=model, console=chat_console, context_window=context_window,
            mcp=mcp, skills=skills, refresh_skills=refresh_skills,
            permissions=permissions, snapshots=snapshots, workspace_state=workspace_state,
            plans=plans, run_task=run_task, recorder=recorder,
            project_config=lambda: ProjectConfig.load(project_config_path), thinking=lambda: selection.thinking,
            model_selection=selection,
        )
        mcp.start_background()
        start_metadata(bound_profile)
        while True:
            poll_startup()
            text = input_session.prompt().strip()
            poll_startup()
            if text.lower() == "exit":
                break
            if not text:
                continue
            kind = command_trace_kind(text)
            if kind is not None:
                with recorder.operation(kind, trace_attributes(text)) as run:
                    handled = dispatch_command(text, command_handlers)
                trace_notice(run)
            else:
                handled = dispatch_command(text, command_handlers)
            if handled:
                continue
            run_task(text)

    finally:
        try:
            if recorder is not None:
                recorder.close()
                warning = recorder.pop_warning()
                if warning:
                    chat_console.print(warning, style="#8c8c8c", markup=False)
            try:
                for notice in mcp.close():
                    chat_console.print(notice, style="#8c8c8c", markup=False)
            finally:
                try:
                    jobs = [*retired_metadata, *([metadata] if metadata is not None else [])]
                    for job in jobs:
                        job.stop.set()
                    for job in jobs:
                        if not job.close():
                            chat_console.print("模型信息后台查询未及时结束。", style="#8c8c8c", markup=False)
                finally:
                    model.close()
            chat_console.print(f"\nsession: {session_id}", style="#8c8c8c", markup=False)
            resume_args = [sys.executable, str(APP_ROOT / "main.py"), "--resume", session_id]
            resume_command = (
                "& " + " ".join(f'"{argument}"' for argument in resume_args)
                if os.name == "nt" else shlex.join(resume_args)
            )
            chat_console.print(
                f"继续会话：{resume_command}",
                style="#8c8c8c", markup=False, soft_wrap=True,
            )
        finally:
            if terminal is not None:
                terminal.close(replay_to=console)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        pass
