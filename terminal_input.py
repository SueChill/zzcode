import sys
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app_session
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.filters import Always
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.mouse_events import MouseButton, MouseEventType
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.styles import Style
from rich.console import Console
from details import detail_label, detail_view


COMMANDS = {
    "/undo": "撤销最近一次修改",
    "/reasoning": "查看最近一段完整思考",
    "/thinking": "查看思考设置；off/on/auto 或具体力度，Ctrl+T 快速循环",
    "/model": "查看已配置模型；next 或名称切换，Alt+M 循环；仅一个时保持不变",
    "/permissions": "切换只读 / 自动；Shift+Tab 循环到完全访问需确认",
    "/usage": "查看上下文估算和实际 token 用量",
    "/compact": "压缩旧对话，保留最近两轮与完整历史存档",
    "/mcp": "查看 MCP 服务连接、工具数量和失败原因",
    "/skills": "查看技能；add 安装本地或 GitHub 技能，reload 重新扫描",
    "/project": "查看项目配置 ZZCODE.md 及解析出的命令",
    "/code": "查看最近一段模型回复的完整代码与正文",
    "/plan": "进入规划；show 查看计划，off 回到普通对话，也可直接附需求",
    "/execute": "确认执行已保存的计划，或继续未完成的计划",
    "/continue": "继续上次未完成的任务，例如达到轮数上限之后",
    "/help": "查看命令与快捷键；/help 关键词 筛选命令",
    "/trace": "查看追踪摘要；list 列表，tree [编号] 完整时间线，report [编号] 离线报告",
}


class CommandCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text
        if document.cursor_position != len(text) or not text.startswith("/"):
            return
        for command, description in COMMANDS.items():
            if command.startswith(text.lower()):
                yield Completion(command, start_position=-len(text), display_meta=description)


def create_input_session(*, input=None, output=None, permissions=None, console=None,
                         model=None, model_name="", display_name="", context_window=0,
                         integrations=None, on_idle=None, thinking=None, model_selection=None):
    """普通文字直接发送，斜杠命令在输入过程中补全。"""
    if console is not None and console.is_interactive and sys.stdin.isatty():
        from terminal_ui import TerminalSession
        devices = get_app_session()
        return TerminalSession(
            input=input or devices.input, output=output or devices.output,
            permissions=permissions, completer=CommandCompleter(), model_name=model_name,
            display_name=display_name, context_window=context_window, stats=model.usage.snapshot,
            integrations=integrations,
            on_idle=on_idle,
            thinking=thinking,
            model_selection=model_selection,
        ).start()
    if output is None and sys.platform == "win32":
        # Windows 使用库选出的控制台输出，避免 TERM=dumb 触发简化显示。
        output = get_app_session().output
    completer = CommandCompleter()
    bindings = KeyBindings()
    pending_full_access = {"mode": None, "until": 0.0}
    pending_exit = {"until": 0.0}

    def prompt():
        fragments = []
        if permissions is not None:
            if (pending_full_access["mode"] == permissions.mode
                    and time.monotonic() < pending_full_access["until"]):
                fragments.append(("class:mode", "再次 Shift+Tab 确认完全访问 · "))
            fragments.append(("class:mode", f"{permissions.label} · "))
        if time.monotonic() < pending_exit["until"]:
            fragments.append(("class:mode", "再按一次 Ctrl+C 退出 · "))
        if thinking is not None:
            active = thinking() if callable(thinking) else thinking
            fragments.append(("class:mode", active.snapshot()["label"] + " · "))
        if model_selection is not None:
            view = model_selection.view()
            fragments.append(("class:mode", (view["display_name"] or view["model"]) + " · "))
        return fragments + [("class:prompt", "› ")]

    if thinking is not None:
        @bindings.add("c-t")
        def switch_thinking(event):
            active = thinking() if callable(thinking) else thinking
            active.cycle()
            event.app.invalidate()

    if model_selection is not None:
        @bindings.add("escape", "m")
        def switch_model(event):
            model_selection.select()
            event.app.invalidate()

    if permissions is not None:
        @bindings.add("s-tab")
        def switch_mode(event):
            pending_exit["until"] = 0.0
            next_mode = permissions.next_mode()
            if next_mode == "fullAccess":
                now = time.monotonic()
                if (pending_full_access["mode"] == permissions.mode
                        and now < pending_full_access["until"]):
                    pending_full_access.update(mode=None, until=0.0)
                    permissions.set_mode("fullAccess")
                else:
                    pending_full_access.update(mode=permissions.mode, until=now + 5.0)
            else:
                pending_full_access.update(mode=None, until=0.0)
                permissions.set_mode(next_mode)
            event.app.invalidate()

    @bindings.add("enter")
    def accept_input(event):
        pending_full_access.update(mode=None, until=0.0)
        pending_exit["until"] = 0.0
        buffer = event.current_buffer
        state = buffer.complete_state
        if state:
            completion = state.current_completion or state.completions[0]
            buffer.apply_completion(completion)
        else:
            # 连续输入并立即回车时，候选菜单可能还没刷新。
            candidates = list(completer.get_completions(
                buffer.document, CompleteEvent(completion_requested=True)
            ))
            if candidates:
                buffer.apply_completion(candidates[0])
        buffer.validate_and_handle()

    @bindings.add("c-c")
    def interrupt(event):
        now = time.monotonic()
        if now < pending_exit["until"]:
            pending_exit["until"] = 0.0
            event.app.exit(exception=KeyboardInterrupt())
            return
        pending_exit["until"] = now + 2.0
        event.current_buffer.reset()
        event.app.invalidate()

    return PromptSession(
        prompt,
        completer=completer,
        complete_while_typing=True,
        key_bindings=bindings,
        reserve_space_for_menu=2,
        style=Style.from_dict({
            "prompt": "bold #d7ad78",
            "mode": "#8794a5",
            "completion-menu.completion": "#e1e7ed bg:default",
            "completion-menu.completion.current": "bold #d7ad78 bg:#263241",
            "completion-menu.meta.completion": "#8794a5 bg:default",
            "completion-menu.meta.completion.current": "#9bbbd0 bg:#263241",
        }),
        input=input,
        output=output,
    )


def create_approval_session(*, input, output, offer_accept_edits=False, operation=None, console=None):
    """只处理选项和快捷键，每次审批默认选中拒绝。"""
    options = [(True, "允许"), (False, "拒绝")]
    if offer_accept_edits:
        options.insert(1, ("acceptEdits", "本次运行自动编辑"))
    selected = len(options) - 1
    console = console or Console()
    expanded = False
    pressed = None
    bindings = KeyBindings()

    def finish(value):
        nonlocal expanded
        expanded = False
        get_app().exit(result=value)

    def toggle(event=None):
        nonlocal expanded
        if event is not None and not (event.button == MouseButton.LEFT and event.event_type == MouseEventType.MOUSE_UP):
            return NotImplemented
        expanded = not expanded
        get_app().invalidate()
        return None

    def choices():
        fragments = []
        if operation is not None and operation.content:
            fragments.append(("class:hint", ("▾ " if expanded else "▸ ") + detail_label(operation) + " · Tab 或点击展开/收起\n", toggle))
            if expanded:
                with console.capture() as capture:
                    console.print(detail_view(operation.content, operation.language))
                fragments.extend(to_formatted_text(ANSI(capture.get())))
        fragments.append(("class:hint", "允许执行？ "))
        for index, (_, label) in enumerate(options):
            def click(event, index=index):
                nonlocal pressed, selected
                if event.button != MouseButton.LEFT or get_app() is not approval_session.app or get_app().is_done:
                    return NotImplemented
                if event.event_type == MouseEventType.MOUSE_DOWN:
                    pressed = index
                    return None
                if event.event_type == MouseEventType.MOUSE_UP:
                    before, pressed = pressed, None
                    if before == index:
                        selected = index
                        finish(options[index][0])
                    return None
                return NotImplemented
            fragments.append(("class:selected" if index == selected else "class:choice",
                              f"[{label}]" if index == selected else label, click))
            fragments.append(("", "   "))
        hint = "\n← → 切换 · Enter 确认 · y 允许 / n 拒绝"
        if offer_accept_edits:
            hint += " / a 自动编辑"
        fragments.append(("class:hint", hint + " "))
        return fragments

    @bindings.add("tab")
    def details(event):
        if operation is not None and operation.content:
            toggle()

    @bindings.add("left")
    @bindings.add("right")
    def select(event):
        nonlocal selected
        step = -1 if event.key_sequence[-1].key == Keys.Left else 1
        selected = max(0, min(len(options) - 1, selected + step))
        event.app.invalidate()

    @bindings.add("y")
    @bindings.add("Y")
    @bindings.add("n")
    @bindings.add("N")
    def shortcut(event):
        nonlocal selected
        allowed = event.data.casefold() == "y"
        selected = 0 if allowed else len(options) - 1
        finish(allowed)

    if offer_accept_edits:
        @bindings.add("a")
        @bindings.add("A")
        def accept_edits(event):
            nonlocal selected
            selected = 1
            finish("acceptEdits")

    @bindings.add("enter")
    @bindings.add("c-j")
    def accept(event):
        finish(options[selected][0])

    @bindings.add("c-c")
    @bindings.add("c-d")
    def cancel(event):
        event.app.exit(exception=KeyboardInterrupt())

    @bindings.add(Keys.Any)
    @bindings.add(Keys.BracketedPaste)
    @bindings.add("escape", Keys.Any)
    def ignore(event):
        pass

    approval_session = PromptSession(
        choices, input=input, output=output, key_bindings=bindings,
        reserve_space_for_menu=0, mouse_support=True,
        style=Style.from_dict({
            "selected": "bold #d7ad78",
            "choice": "#8794a5",
            "hint": "#8794a5",
        }),
    )
    approval_session.default_buffer.read_only = Always()
    # PromptSession 会拆分多行提示并丢弃片段上的鼠标处理器。
    # 审批内容使用独立控件，底部保留只读输入缓冲区以兼容原键盘流程。
    approval_session.app.layout = Layout(HSplit([
        Window(FormattedTextControl(choices), wrap_lines=True, dont_extend_height=True),
        Window(BufferControl(buffer=approval_session.default_buffer), height=1),
    ]), focused_element=approval_session.default_buffer)
    return approval_session


def confirm_tool_execution(session, console, operation, permissions=None):
    """独立的按键审批，不改变聊天补全和输入历史。"""
    if getattr(session, "is_terminal_ui", False) is True:
        return session.confirm(operation, permissions=permissions)
    console.print("需要审批 · " + operation.title, style="bold #d7ad78", markup=False)
    if operation.description:
        console.print(operation.description, markup=False)
    if permissions is not None:
        console.print("自动编辑允许本次运行在工作区内新建、编辑和删除文件；运行命令仍需确认。", style="#8794a5", markup=False)
    approval_session = create_approval_session(
        input=session.app.input, output=session.app.output,
        offer_accept_edits=permissions is not None,
        operation=operation, console=console,
    )
    try:
        answer = approval_session.prompt()
        if answer == "acceptEdits" and permissions is not None:
            permissions.set_mode("acceptEdits")
            console.print("权限模式：自动 · Shift+Tab 或 /permissions readOnly 可切回只读审批。", style="#8794a5", markup=False)
            return True
        return answer is True
    except EOFError:
        raise KeyboardInterrupt from None
