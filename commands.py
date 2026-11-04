"""处理终端本地命令，不把命令写入模型对话历史。"""

from functools import partial

from httpx import HTTPError
from openai import APIError
from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from details import AnswerMarkdown
from model import ModelReplyError
from reply_display import REASONING_STYLE, ReplyDisplay
from skills import parse_command as parse_skill_command
from terminal_input import COMMANDS
from tracing import TraceReader, set_run_status, span
from trace_report import export_html, list_traces, show_trace


def create_command_handlers(*, model, console, context_window, mcp, skills,
                            refresh_skills, permissions, snapshots, workspace_state, plans,
                            run_task, recorder=None, project_config=None, thinking=None, model_selection=None):
    """在会话建立后绑定命令所需的对象，供输入循环直接分发。"""
    return {
        "/help": partial(_help, console=console, permissions=permissions, plans=plans),
        "/trace": partial(_trace, recorder=recorder, console=console),
        "/compact": partial(_compact, model=model, console=console),
        "/usage": partial(_usage, model=model, console=console, context_window=context_window),
        "/mcp": partial(_mcp, mcp=mcp, console=console),
        "/skills": partial(_skills, skills=skills, refresh_skills=refresh_skills, console=console),
        "/permissions": partial(_permissions, permissions=permissions, console=console),
        "/reasoning": partial(_reasoning, model=model, console=console),
        "/thinking": partial(_thinking, thinking=thinking, console=console),
        "/model": partial(_model, selection=model_selection, console=console),
        "/code": partial(_code, model=model, console=console),
        "/undo": partial(_undo, snapshots=snapshots, workspace_state=workspace_state, console=console,
                         plans=plans, refresh_skills=refresh_skills),
        "/plan": partial(_plan, plans=plans, refresh_skills=refresh_skills, run_task=run_task, console=console),
        "/execute": partial(_execute, plans=plans, refresh_skills=refresh_skills, run_task=run_task, console=console),
        "/continue": partial(_continue, run_task=run_task, console=console),
        "/project": partial(_project, project_config=project_config, console=console),
    }


def dispatch_command(text, handlers):
    """已处理返回 True；普通输入和不识别的命令交回聊天流程。"""
    parts = text.split(maxsplit=1)
    if not parts:
        return False
    command = parts[0].lower()
    if len(parts) > 1 and command not in {"/help", "/trace", "/skills", "/permissions", "/plan", "/thinking", "/model"}:
        return False
    handler = handlers.get(command)
    if handler is None:
        return False
    handler(text)
    return True


def command_trace_kind(text):
    parts = text.split(maxsplit=1)
    if not parts:
        return None
    command = parts[0].lower()
    argument = parts[1].strip().lower() if len(parts) > 1 else ""
    if command in {"/execute", "/continue"} and not argument:
        return "task"
    if command == "/plan" and argument not in {"", "show", "off"}:
        return "task"
    if ((command in {"/undo", "/compact"} and not argument) or command == "/permissions"
            or (command == "/thinking" and argument)
            or (command == "/model" and argument)
            or (command == "/plan" and argument in {"", "off"})
            or (command == "/skills" and argument.split(maxsplit=1)[0:1] in [["add"], ["reload"]])):
        return "command"
    return None


def _trace(text, *, recorder, console):
    if recorder is None:
        console.print("当前会话未初始化追踪器。", style="#8c8c8c", markup=False)
        return
    try:
        if not recorder.flush():
            console.print("追踪仍有待写入记录，以下结果可能不完整。", style="yellow", markup=False)
        parts = text.split()[1:]
        reader = TraceReader(recorder.root, recorder.session)
        if parts == ["list"]:
            list_traces(reader, console)
            return
        report = bool(parts and parts[0] == "report")
        full = bool(parts and parts[0] == "tree")
        if report or full:
            parts = parts[1:]
        if len(parts) > 1:
            raise ValueError("用法：/trace [编号前缀]；/trace list；/trace tree [编号前缀]；/trace report [编号前缀]。")
        trace = reader.read(reader.select(parts[0] if parts else None))
        if report:
            path = export_html(trace)
            console.print(f"离线耗时报告：{path}", style="#67d8ef", markup=False)
            console.print("用浏览器打开该 HTML 文件即可查看瀑布图与工具耗时分布。", style="#8c8c8c", markup=False)
        else:
            show_trace(trace, console, full=full)
            console.print("/trace report [编号前缀] 生成离线 HTML 报告", style="#8c8c8c", markup=False)
    except (OSError, ValueError) as error:
        console.print(str(error) if isinstance(error, ValueError) else "追踪文件无法读取或报告无法写入。", style="red", markup=False)


def _help(text, *, console, permissions, plans):
    parts = text.split(maxsplit=1)
    query = parts[1].strip().casefold() if len(parts) > 1 else ""
    groups = [
        ("会话与显示", ("/help", "/trace", "/undo", "/reasoning", "/code")),
        ("任务与权限", ("/plan", "/execute", "/continue", "/permissions", "/thinking", "/model")),
        ("上下文", ("/usage", "/compact")),
        ("工具与技能", ("/mcp", "/skills", "/project")),
    ]
    grouped = {command for _, commands in groups for command in commands}
    groups.append(("其他", tuple(command for command in COMMANDS if command not in grouped)))
    table = Table.grid(padding=(0, 2))
    table.add_column(style="#67d8ef", no_wrap=True)
    table.add_column(style="#b9c7d8", overflow="fold")
    matches = 0
    for title, commands in groups:
        rows = [(command, COMMANDS[command]) for command in commands
                if command in COMMANDS and (not query or query in f"{command} {COMMANDS[command]}".casefold())]
        if not rows:
            continue
        if matches:
            table.add_row("", "")
        table.add_row(Text(title, style="bold #c4b5fd"), "")
        for command, description in rows:
            table.add_row(command, description)
        matches += len(rows)
    mode = {"chat": "普通对话", "plan": "规划", "execute": "执行计划"}[plans.mode]
    content = [Text(f"模式：{mode}  ·  权限：{permissions.label}", style="#8c8c8c"), Text("")]
    if matches:
        content.append(table)
    else:
        content.append(Text(f"没有匹配的命令：{parts[1].strip()}", style="#8c8c8c"))
    content.append(Text(""))
    if not query:
        shortcuts = Table.grid(padding=(0, 2))
        shortcuts.add_column(style="#f3c67a", no_wrap=True)
        shortcuts.add_column(style="#b9c7d8", overflow="fold")
        shortcuts.add_row(Text("快捷键", style="bold #c4b5fd"), "")
        shortcuts.add_row("/ · ↑↓ · Tab", "查看、选择和补全命令，回车提交")
        shortcuts.add_row("Shift+Tab", "只读 → 自动 → 完全访问；完全访问需确认")
        shortcuts.add_row("Ctrl+T", "循环切换关闭思考及当前模型支持的力度；下次请求生效")
        shortcuts.add_row("Alt+M", "按配置顺序切换模型；少于两个可用模型时保持不变")
        shortcuts.add_row("Ctrl+C", "执行中取消本轮；等待输入时连续两次退出")
        shortcuts.add_row("Ctrl+D", "空输入时退出")
        shortcuts.add_row("y / n · 鼠标", "审批时允许或拒绝")
        shortcuts.add_row("Tab · 点击标题", "审批时展开 / 收起详情")
        shortcuts.add_row("点击思考标题", "展开 / 收起该轮思考")
        shortcuts.add_row("exit", "退出并显示会话编号")
        content.extend([shortcuts, Text("")])
    content.append(Text("/help 关键词 筛选命令 · /help 查看全部", style="#8c8c8c"))
    console.print(Panel.fit(Group(*content), title=Text("zzcode · 帮助", style="bold #c4b5fd"),
                            padding=(0, 1), border_style="#4b5b70", box=box.ROUNDED))
    console.print()


def _model(text, *, selection, console):
    if selection is None:
        console.print("当前没有可切换的模型配置。", style="#8c8c8c", markup=False)
        return
    parts = text.split()
    if len(parts) > 2:
        console.print("用法：/model [next|配置名称]。", style="red", markup=False)
        return
    if len(parts) == 2:
        requested = parts[1].lower()
        if requested != "next" and requested not in {profile.alias for profile in selection.profiles}:
            console.print("该模型没有完整配置，保持当前模型。", style="#8c8c8c", markup=False)
            return
        selection.select(None if requested == "next" else requested)
    view = selection.view()
    console.print(f"当前模型：{view['display_name'] or view['model']}（{view['alias']}）", style="#67d8ef", markup=False)
    for profile in selection.profiles:
        marker = "›" if profile.alias == view["alias"] else " "
        console.print(f"{marker} {profile.alias} · {profile.display_name or profile.model}", style="#8c8c8c", markup=False)
    console.print("Alt+M 循环切换；下一次请求生效。" if len(selection.profiles) > 1 else
                  "只有一个完整模型配置，切换时保持不变。", style="#8c8c8c", markup=False)


def _thinking(text, *, thinking, console):
    thinking = thinking() if callable(thinking) else thinking
    if thinking is None:
        console.print("当前模型没有思考控制配置。", style="#8c8c8c", markup=False)
        return
    parts = text.split()
    try:
        if len(parts) > 2:
            raise ValueError("用法：/thinking [off|on|auto|力度]。")
        if len(parts) == 2:
            choice = parts[1].lower()
            if choice in {"off", "disabled", "on", "enabled", "auto"}:
                thinking.configure(choice, "" if choice == "auto" else thinking.effort)
            else:
                thinking.configure("enabled", choice)
    except ValueError as error:
        console.print(str(error), style="red", markup=False)
        return
    state = thinking.snapshot()
    choices = " / ".join(thinking.efforts)
    suffix = (f"可选：off / {choices}；Ctrl+T 循环切换，下次请求生效。"
              if state["supported"] else "当前模型/接口尚未适配快捷切换，保留接口默认。")
    console.print(f"{state['label']} · {suffix}", style="#c4b5fd", markup=False)


def _compact(_text, *, model, console):
    try:
        with ReplyDisplay(console, show_reasoning=False) as display:
            display.compacting()
            result = model.compact()
        if result is None:
            console.print("没有可压缩的旧对话，最近两轮原文继续保留。", style="#8c8c8c", markup=False)
        else:
            console.print(
                f"上下文已压缩：≈{result['before_tokens']:,} → ≈{result['after_tokens']:,} token。"
                "保留最近两轮原文，完整历史仍在数据库中。", markup=False,
            )
    except KeyboardInterrupt:
        set_run_status("cancelled", stop_detail="用户取消了上下文压缩，完整历史保留。")
        console.print("已停止压缩，完整历史保留；/usage 可查看当前上下文用量。", style="#8c8c8c", markup=False)
    except (ModelReplyError, ValueError) as error:
        set_run_status("model_error", error_type=type(error).__name__, stop_detail="压缩回复未通过校验，未应用。")
        console.print(f"压缩未应用：{error}", style="red", markup=False)
    except (APIError, HTTPError, ConnectionError, TimeoutError) as error:
        set_run_status("model_error", error_type=type(error).__name__, http_status=getattr(error, "status_code", None),
                       stop_detail="压缩请求失败，原上下文保持不变。")
        console.print("压缩请求失败，原上下文保持不变。", style="red", markup=False)


def _usage(_text, *, model, console, context_window):
    stats = model.usage.snapshot()
    if type(model.context_window) is int:
        context_window = model.context_window
    capacity = f" / {context_window:,}" if context_window else "（容量上限未知）"
    console.print(f"上下文估算：≈{stats['context_tokens']:,} token{capacity}", markup=False)

    def number(value):
        return "未获取" if value is None else f"{value:,}"

    console.print(f"最近请求：输入 {number(stats['last_input'])} · 输出 {number(stats['last_output'])}", markup=False)
    console.print(f"本次运行已统计：输入 {stats['total_input']:,} · 输出 {stats['total_output']:,}", markup=False)
    console.print(f"请求 {stats['requests']} 次 · 缺少完整用量 {stats['unmeasured']} 次", markup=False)
    console.print("上下文始终为本地估算；输入、输出采用接口实际统计，缺失部分不计入累计。", style="#8794a5", markup=False)
    budget = f"输入预算 {model.input_budget:,}" if model.input_budget is not None else "容量未知，未启用窗口拦截"
    console.print(f"请求限制：{budget} · 回复最多 {model.output_reserve:,} token · /compact 手动压缩", markup=False)


def _mcp(_text, *, mcp, console):
    status = mcp.snapshot()
    console.print(
        f"MCP：已连接 {status['connected']} / 未连接 {status['disconnected']}"
        f" · 工具总数 {status['tools']} · 可用工具 {status['available_tools']}",
        style="#8c8c8c", markup=False,
    )
    if status.get("loading"):
        console.print("MCP 正在后台连接；工具就绪后会在后续任务中可用。", style="#8c8c8c", markup=False)
    if status["error"]:
        console.print(status["error"], style="#8c8c8c", markup=False)
    elif not status["servers"]:
        console.print("尚未配置 MCP 服务。", style="#8c8c8c", markup=False)
    for server in status["servers"]:
        detail = f" · {server['detail']}" if server["detail"] else ""
        console.print(
            f"  {server['name']} · {server['status']} · {server['tools']} 个工具{detail}",
            style="#8c8c8c", markup=False,
        )


def _skills(text, *, skills, refresh_skills, console):
    try:
        action, source, options = parse_skill_command(text)
        if action == "add":
            with ReplyDisplay(console, show_reasoning=False) as display:
                display.executing("安装 skill")
                installed = skills.install(source, **options)
            refresh_skills()
            console.print(f"已安装 skill：{installed}，本会话立即可用。", style="#8c8c8c", markup=False)
        else:
            if action == "reload":
                skills.discover()
                refresh_skills()
            console.print(f"Skills：可用 {len(skills.skills)} · {skills.root}", style="#8c8c8c", markup=False)
            for skill in skills.skills.values():
                console.print(f"  {skill.name} · {skill.description}", style="#8c8c8c", markup=False)
                if skill.compatibility:
                    console.print(f"    环境要求：{skill.compatibility}", style="#8c8c8c", markup=False)
            for notice in skills.notices:
                console.print(notice, style="#8c8c8c", markup=False)
            console.print('/skills add "本地目录或 GitHub 链接" 安装；/skills reload 重新扫描。', style="#8c8c8c", markup=False)
    except KeyboardInterrupt:
        skills.discover()
        refresh_skills()
        console.print("已停止安装；/skills 可查看当前安装结果。", style="#8c8c8c", markup=False)
    except (OSError, ValueError) as error:
        detail = str(error) if isinstance(error, ValueError) else type(error).__name__
        console.print(f"Skill 操作失败：{detail}", style="red", markup=False)


def _permissions(text, *, permissions, console):
    parts = text.split(maxsplit=1)
    try:
        if len(parts) == 1:
            permissions.toggle()
        else:
            permissions.set_mode(parts[1])
    except ValueError as error:
        console.print(str(error), style="red", markup=False)
    else:
        console.print(f"权限模式：{permissions.label} · {permissions.scope}", style="#8794a5", markup=False)


def _reasoning(_text, *, model, console):
    reasoning = model.get_last_reasoning()
    if reasoning:
        console.print("最近一段思考", style="dim " + REASONING_STYLE, markup=False)
        console.print(reasoning, style=REASONING_STYLE, markup=False, soft_wrap=True)
        console.print()
    else:
        console.print("暂无可查看的思考。", style=REASONING_STYLE, markup=False)


def _code(_text, *, model, console):
    answer = next((message["content"] for message in reversed(model.messages)
                   if message["role"] == "assistant" and isinstance(message.get("content"), str)
                   and message["content"]), "")
    if answer:
        console.print(AnswerMarkdown(answer, code_theme="ansi_dark"))
    else:
        console.print("暂无可查看的完整回复。", style="#8c8c8c", markup=False)


def _undo(_text, *, snapshots, workspace_state, console, plans, refresh_skills):
    try:
        with span("workspace.undo") as operation:
            paths = snapshots.undo()
            operation.attrs["changed_paths"] = len(paths)
    except (OSError, ValueError) as error:
        set_run_status("internal_error", error_type=type(error).__name__)
        console.print(f"撤销失败：{error}", style="red", markup=False)
    else:
        if paths:
            console.print("已撤销：" + "、".join(paths) + "。", markup=False)
            workspace_state.record_undo(paths)
            plans.invalidate("用户撤销了文件修改，原有计划进度及验证需要重新核实。")
            refresh_skills()
        else:
            console.print("没有可以撤销的修改。", markup=False)


def show_plan(plans, console):
    data = plans.snapshot()
    mode = {"chat": "普通对话", "plan": "规划", "execute": "执行"}[data["mode"]]
    console.print(f"计划 · {mode} · 版本 {data['revision']}", style="#c4b5fd", markup=False)
    if not data["steps"]:
        console.print("尚未保存计划。", style="#8c8c8c", markup=False)
        return
    console.print(data["goal"], markup=False)
    labels = {"pending": "待做", "in_progress": "进行中", "completed": "已完成", "blocked": "受阻"}
    for step in data["steps"]:
        console.print(f"  {step['id']}. [{labels[step['status']]}] {step['title']}", markup=False)
        console.print(f"     验收：{step['verification']}", style="#8c8c8c", markup=False)
        for evidence in step["evidence"]:
            console.print(f"     依据：{evidence['summary']} ({evidence['tool_call_id']})", style="#8c8c8c", markup=False)
        if step["note"]:
            console.print(f"     {step['note']}", style="#8c8c8c", markup=False)
    if data["note"]:
        console.print(data["note"], style="#8c8c8c", markup=False)


def _plan(text, *, plans, refresh_skills, run_task, console):
    parts = text.split(maxsplit=1)
    argument = parts[1].strip() if len(parts) > 1 else ""
    if argument.lower() == "show":
        show_plan(plans, console)
        return
    if argument.lower() == "off":
        plans.leave()
        refresh_skills()
        console.print("已回到普通对话，计划保留。", style="#8c8c8c", markup=False)
        return
    plans.enter(new_request=bool(argument))
    refresh_skills()
    console.print("已进入规划模式，只读项目与搜索资料；/execute 确认执行，/plan show 查看，/plan off 退出。", style="#8c8c8c", markup=False)
    if argument:
        run_task(text)
    else:
        show_plan(plans, console)


def _project(text, *, project_config, console):
    if project_config is None:
        console.print("项目配置不可用。", style="#8c8c8c", markup=False)
        return
    config = project_config()
    if not config.exists:
        console.print("未找到项目配置；可在项目根目录创建 ZZCODE.md。", style="#8c8c8c", markup=False)
        return
    console.print(f"项目配置：{config.path}", style="#67d8ef", markup=False)
    if config.truncated:
        console.print("文件超过 64 KiB，只读取开头部分。", style="#8c8c8c", markup=False)
    if config.commands:
        for name, command in config.commands.items():
            console.print(f"  {name}：{command}", style="#b9c7d8", markup=False)
    else:
        console.print("未解析到命令；在 ## 命令 小节使用「- 名称：命令」。", style="#8c8c8c", markup=False)
    console.print(f"说明 {len(config.text)} 字符，参与每次请求。", style="#8c8c8c", markup=False)


def _continue(text, *, run_task, console):
    console.print("继续未完成的任务。", style="#8c8c8c", markup=False)
    run_task("继续未完成的任务。先检查当前进度和已经完成的改动，不要重复已完成的操作；如任务已完成，请总结结果。")


def _execute(text, *, plans, refresh_skills, run_task, console):
    try:
        plans.approve()
    except ValueError as error:
        console.print(str(error), style="red", markup=False)
        return
    refresh_skills()
    console.print(f"开始执行计划 · 版本 {plans.snapshot()['revision']}", style="#c4b5fd", markup=False)
    run_task(text)
