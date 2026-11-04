"""持续运行的终端布局；主线程继续负责模型、工具和取消收尾。"""
import _thread
import threading
import time
from io import StringIO
from queue import Empty, Queue

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.containers import Float, FloatContainer
from prompt_toolkit.mouse_events import MouseButton, MouseEventType
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from rich.console import Console, ConsoleDimensions
from rich.text import Text

from details import AnswerMarkdown, answer_parts, detail_label, detail_view


def compact_tokens(number):
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}m"
    return f"{number / 1000:.1f}k" if number >= 1000 else str(number)


class ScreenConsole(Console):
    def __init__(self, terminal):
        self.terminal_ui = terminal
        # 这里捕获带色文本供界面解析，最终颜色输出由 prompt_toolkit 负责。
        super().__init__(file=StringIO(), force_terminal=True, force_interactive=False,
                         color_system="truecolor", no_color=False, highlight=False, legacy_windows=False)

    @property
    def size(self):
        size = self.terminal_ui.app.output.get_size()
        return ConsoleDimensions(max(1, size.columns), max(1, size.rows - 3))

    def print(self, *objects, **kwargs):
        with self.capture() as capture:
            super().print(*objects, **kwargs)
        self.terminal_ui.append(capture.get())

    def render_text(self, renderable):
        with self.capture() as capture:
            super().print(renderable)
        return capture.get()


class TerminalSession:
    is_terminal_ui = True

    def __init__(self, *, input, output, permissions, completer, model_name,
                 stats, context_window=0, display_name="", interrupt=None, integrations=None, on_idle=None,
                 thinking=None, model_selection=None):
        self.permissions = permissions
        self.model_name = model_name
        self.display_name = display_name
        self.stats = stats
        self.integrations = integrations or (lambda: {"connected": 0, "disconnected": 0})
        self.context_window = context_window
        self.on_idle = on_idle
        self.thinking = thinking
        self.model_selection = model_selection
        self._interrupt = interrupt or _thread.interrupt_main
        self._lock = threading.RLock()
        self._answers = Queue()
        self._ready = threading.Event()
        self._closed = False
        self._thread = None
        self._waiting = None
        self._phase = "就绪"
        self._started = None
        self._history = []
        self._details = {}
        self._approval_detail = None
        self._approval_prompt = None
        self._permission_mode_pending = None
        self._approval_generation = 0
        self._pressed_approval = None
        self._exit_pending_until = 0.0
        self._transient = None
        self._revision = 0
        self._cache_key = None
        self._lines = []
        self._scroll = 0
        self._approval_choices = []
        self._selected = 0
        self.buffer = Buffer(completer=completer, complete_while_typing=True,
                             read_only=Condition(lambda: self._waiting != "input"),
                             accept_handler=self._submit)
        self.default_buffer = self.buffer
        self.history = self.buffer.history
        self.completer = completer
        bindings = KeyBindings()

        @bindings.add("enter")
        @bindings.add("c-j")
        def enter(event):
            if self._waiting == "approval":
                self._answer(self._approval_choices[self._selected][0])
            elif self._waiting == "input":
                state = self.buffer.complete_state
                if state:
                    self.buffer.apply_completion(state.current_completion or state.completions[0])
                else:
                    candidates = list(completer.get_completions(self.buffer.document, CompleteEvent(completion_requested=True)))
                    if candidates:
                        self.buffer.apply_completion(candidates[0])
                self.buffer.validate_and_handle()

        @bindings.add("s-tab", filter=Condition(lambda: self._waiting == "input"))
        def mode(event):
            if permissions is None:
                return
            self._exit_pending_until = 0.0
            next_mode = permissions.next_mode()
            if next_mode == "fullAccess":
                self.begin_full_access_confirmation()
                return
            permissions.set_mode(next_mode)
            self.app.invalidate()

        @bindings.add("c-t", filter=Condition(lambda: self._waiting == "input"))
        def switch_thinking(event):
            if self.thinking is not None:
                active = self.thinking() if callable(self.thinking) else self.thinking
                active.cycle()
                self.app.invalidate()

        @bindings.add("escape", "m", filter=Condition(lambda: self._waiting == "input"))
        def switch_model(event):
            if self.model_selection is not None:
                self.model_selection.select()
                self.app.invalidate()

        @bindings.add("left", filter=Condition(lambda: self._waiting == "approval"))
        @bindings.add("right", filter=Condition(lambda: self._waiting == "approval"))
        def select(event):
            step = -1 if event.key_sequence[-1].key == Keys.Left else 1
            self._selected = max(0, min(len(self._approval_choices) - 1, self._selected + step))
            self.app.invalidate()

        @bindings.add("tab", filter=Condition(lambda: self._waiting == "approval"))
        def details(event):
            if self._approval_detail is not None:
                self.toggle_detail(self._approval_detail)

        for key in ("y", "Y", "n", "N", "a", "A"):
            @bindings.add(key, filter=Condition(lambda: self._waiting == "approval"))
            def choose(event):
                value = {"y": True, "n": False, "a": "acceptEdits"}[event.data.lower()]
                if any(type(choice) is type(value) and choice == value for choice, _ in self._approval_choices):
                    self._answer(value)

        @bindings.add(Keys.Any, filter=Condition(lambda: self._waiting != "input"))
        @bindings.add(Keys.BracketedPaste, filter=Condition(lambda: self._waiting != "input"))
        @bindings.add("escape", Keys.Any, filter=Condition(lambda: self._waiting != "input"))
        def ignore(event):
            pass

        @bindings.add("c-c")
        def cancel(event):
            if self._waiting == "input":
                now = time.monotonic()
                if now < self._exit_pending_until:
                    self._exit_pending_until = 0.0
                    self._answer(KeyboardInterrupt())
                    return
                self._exit_pending_until = now + 2.0
                self.buffer.reset()
                self.console.print("再按一次 Ctrl+C 退出", style="#8794a5", markup=False)
                self.app.invalidate()
            elif self._waiting is not None:
                self._answer(KeyboardInterrupt())
            else:
                self._interrupt()

        @bindings.add("c-d")
        def end(event):
            if self._waiting is not None:
                self._answer(KeyboardInterrupt() if self._waiting == "approval" else EOFError())

        @bindings.add("pageup")
        @bindings.add("pagedown")
        def page(event):
            amount = max(1, self.app.output.get_size().rows - 4)
            self.scroll(amount if event.key_sequence[-1].key == Keys.PageUp else -amount)

        @bindings.add("c-end")
        def tail(event):
            self._scroll = 0
            self.app.invalidate()

        log = FormattedTextControl(self._viewport, show_cursor=False)
        self._formatted_mouse = log.mouse_handler
        log.mouse_handler = self._mouse
        self.log_window = Window(log, wrap_lines=False, always_hide_cursor=True)
        approval = FormattedTextControl(self._approval_text, show_cursor=False)
        entry = HSplit([
            ConditionalContainer(Window(BufferControl(buffer=self.buffer), height=1,
                                        get_line_prefix=lambda line, wrap: [("class:phase", "› ")]),
                                 filter=Condition(lambda: self._waiting != "approval")),
            ConditionalContainer(Window(approval, height=1),
                                 filter=Condition(lambda: self._waiting == "approval")),
        ])
        body = HSplit([
            self.log_window,
            Window(height=1, char="┄", style="class:border"),
            entry,
            Window(FormattedTextControl(self._status), height=Dimension.exact(1)),
        ])
        layout = FloatContainer(body, floats=[Float(xcursor=True, ycursor=True,
                                                     content=CompletionsMenu(max_height=6))])
        self.app = Application(
            layout=Layout(layout, focused_element=self.buffer), key_bindings=bindings,
            input=input, output=output, full_screen=True, mouse_support=True,
            refresh_interval=0.1, min_redraw_interval=0.08,
            style=Style.from_dict({"border": "#4b5b70", "status": "#8794a5",
                                   "brand": "bold #c4b5fd", "separator": "#536079",
                                   "model": "bold #67d8ef", "context": "#83d5b3",
                                   "thinking": "#c4b5fd",
                                   "permissions": "#89b4fa", "permissions.auto": "#f3c67a",
                                   "permissions.full": "bold #f38ba8", "mcp": "#83d5b3", "skills": "#c4b5fd",
                                   "phase": "#d7ad78", "phase.ready": "#83d5b3",
                                   "phase.thinking": "#c4b5fd", "phase.reply": "#67d8ef",
                                   "phase.tool": "#8c8c8c", "phase.approval": "#f3c67a",
                                   "elapsed": "#8794a5", "selected": "bold #d7ad78",
                                   "warning": "bold #f3c67a", "danger": "bold #f38ba8",
                                   "completion-menu.completion": "bg:default #e1e7ed",
                                   "completion-menu.completion.current": "bg:#263241 #d7ad78"}),
        )
        self.console = ScreenConsole(self)

    def start(self):
        def run():
            try:
                self.app.run(pre_run=self._ready.set, handle_sigint=False)
            except BaseException as error:
                self._answers.put(error)
            finally:
                self._ready.set()
        self._thread = threading.Thread(target=run, name="zzcode-terminal", daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise RuntimeError("终端界面未能启动。")
        return self

    def append(self, text):
        with self._lock:
            self._history.append(text)
            self._revision += 1
        self.app.invalidate()

    def update_model_info(self, display_name, context_window, *, model_name=None):
        with self._lock:
            self.display_name = display_name
            self.context_window = context_window
            if model_name is not None:
                self.model_name = model_name
        self.app.invalidate()

    def add_detail(self, label, content, language):
        with self._lock:
            index = len(self._history)
            title = "▸ " + label + "（点击展开）"
            self._history.append(self.console.render_text(Text(title, style="#8c8c8c")))
            self._details[index] = {"label": label, "content": content, "language": language, "expanded": False}
            self._revision += 1
        self.app.invalidate()
        return index

    def toggle_detail(self, index):
        with self._lock:
            detail = self._details.get(index)
            if detail is None:
                return
            detail["expanded"] = not detail["expanded"]
            self._revision += 1
        self.app.invalidate()

    def append_answer(self, text):
        for kind, content, language in answer_parts(text):
            if kind == "code":
                self.add_detail(f"{language} 代码 · {len(content.splitlines())} 行", content, language)
            elif content.strip():
                self.console.print(AnswerMarkdown(content, code_theme="ansi_dark"))

    def append_reasoning(self, text):
        """一轮思考作为可点击详情加入历史，正文只在展开时渲染。"""
        return self.add_detail(f"思考 · {len(text)} 字符", text, "reasoning")

    def set_transient(self, renderable=None):
        with self._lock:
            self._transient = renderable
            self._revision += 1
        self.app.invalidate()

    def set_phase(self, phase):
        self._phase = phase
        self.app.invalidate()

    def begin_task(self):
        self._started = time.monotonic()
        self.set_phase("思考中")

    def _submit(self, buffer):
        text = buffer.text
        self._exit_pending_until = 0.0
        self.console.print("› " + text, style="#d7ad78", markup=False)
        self._scroll = 0
        self.begin_task()
        self._answer(text)
        return False

    def _answer(self, value):
        if self._permission_mode_pending is not None:
            self._finish_permission_confirmation(value)
            return
        self._waiting = None
        self._answers.put(value)
        self.app.invalidate()

    def begin_full_access_confirmation(self):
        """切换到完全访问前的二次确认；不阻塞 app 线程。"""
        if self._permission_mode_pending is not None or self._waiting != "input":
            return
        self.console.print("需要确认 · 切换到完全访问", style="bold #f38ba8", markup=False)
        self.console.print(
            "完全访问会跳过工具审批，命令以当前系统用户权限执行，没有操作系统沙箱。",
            style="#8794a5", markup=False,
        )
        self._permission_mode_pending = "fullAccess"
        self._approval_prompt = "切换到完全访问？"
        self._approval_choices = [(True, "确认切换"), (False, "取消")]
        self._approval_detail = None
        self._selected = 1
        self._approval_generation += 1
        self._pressed_approval = None
        self.set_phase("等待确认")
        self._waiting = "approval"
        self._scroll = 0
        self.app.invalidate()

    def _finish_permission_confirmation(self, value):
        pending, self._permission_mode_pending = self._permission_mode_pending, None
        confirmed = value is True
        self._approval_prompt = None
        self._approval_choices = []
        self._approval_detail = None
        self._selected = 0
        self._pressed_approval = None
        self._approval_generation += 1
        if confirmed and pending == "fullAccess":
            self.permissions.set_mode("fullAccess")
            self.console.print("已切换：完全访问", style="bold #f38ba8", markup=False)
        else:
            self.console.print("已取消切换", style="#8794a5", markup=False)
        self.set_phase("就绪")
        self._waiting = "input"
        self.app.invalidate()

    def _wait(self, on_idle=None):
        while not self._closed:
            if on_idle is not None:
                on_idle()
            try:
                answer = self._answers.get(timeout=0.1)
            except Empty:
                continue
            if isinstance(answer, BaseException):
                raise answer
            return answer
        raise EOFError()

    def prompt(self):
        self._started = None
        self._phase = "就绪"
        self._exit_pending_until = 0.0
        self._waiting = "input"
        self.app.invalidate()
        return self._wait(on_idle=self.on_idle)

    def confirm(self, operation, permissions=None):
        self._approval_prompt = None
        self.console.print("需要审批 · " + operation.title, style="bold #d7ad78", markup=False)
        if operation.description:
            self.console.print(operation.description, markup=False)
        self._approval_detail = (self.add_detail(detail_label(operation), operation.content, operation.language)
                                 if operation.content else None)
        self._approval_choices = [(True, "允许"), (False, "拒绝")]
        if permissions is not None:
            self.console.print("自动编辑允许本次运行在工作区内新建、编辑和删除；运行命令仍需确认。", style="#8794a5", markup=False)
            self._approval_choices.insert(1, ("acceptEdits", "本次运行自动编辑"))
        self._selected = len(self._approval_choices) - 1
        self._approval_generation += 1
        self._pressed_approval = None
        self.set_phase("等待审批")
        self._waiting = "approval"
        self.app.invalidate()
        try:
            answer = self._wait()
        finally:
            if self._approval_detail is not None and self._details[self._approval_detail]["expanded"]:
                self.toggle_detail(self._approval_detail)
            self._approval_detail = None
            self._pressed_approval = None
        if answer == "acceptEdits" and permissions is not None:
            permissions.set_mode("acceptEdits")
            answer = True
        self.console.print("审批：" + ("允许" if answer is True else "拒绝"), style="#8794a5", markup=False)
        return answer is True

    def _approval_text(self):
        result = [("class:status", (self._approval_prompt or "允许执行？") + " ")]
        generation = self._approval_generation
        for index, (_, label) in enumerate(self._approval_choices):
            def click(event, index=index):
                if self._waiting != "approval" or generation != self._approval_generation:
                    return NotImplemented
                if event.button != MouseButton.LEFT:
                    return NotImplemented
                if event.event_type == MouseEventType.MOUSE_DOWN:
                    self._pressed_approval = (generation, index)
                    return None
                if event.event_type == MouseEventType.MOUSE_UP:
                    pressed, self._pressed_approval = self._pressed_approval, None
                    if pressed == (generation, index):
                        self._selected = index
                        self._answer(self._approval_choices[index][0])
                    return None
                return NotImplemented
            result.append(("class:selected" if index == self._selected else "class:status",
                           f"[{label}]" if index == self._selected else label, click))
            result.append(("", "   "))
        if self._approval_detail is not None:
            result.append(("class:status", "Tab 展开/收起详情"))
        return result

    def _status(self):
        stats = self.stats()
        width = self.app.output.get_size().columns
        model = self.display_name or stats.get("backend_model") or self.model_name
        capacity = self.context_window
        if self.model_selection is not None:
            selected = self.model_selection.view()
            model = selected["display_name"] or selected["model"]
            capacity = selected["context_window"]
        context = "上下文 ≈" + compact_tokens(stats.get("context_tokens", 0))
        if capacity:
            context += f"/{compact_tokens(capacity)} · {stats.get('context_tokens', 0) / capacity:.0%}"
        elapsed = f" · {time.monotonic() - self._started:.1f}s" if self._started is not None else ""
        spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(time.monotonic() * 8) % 10] + " " if self._started is not None else ""
        context_style = "class:context"
        if capacity:
            ratio = stats.get("context_tokens", 0) / capacity
            if ratio >= 0.95:
                context_style = "class:danger"
            elif ratio >= 0.8:
                context_style = "class:warning"
        permission_style = {"auto": "class:permissions.auto", "fullAccess": "class:permissions.full"}.get(
            self.permissions.mode, "class:permissions")
        integrations = self.integrations()
        active = self.thinking() if callable(self.thinking) else self.thinking
        thinking = active.snapshot() if active is not None else {"label": "思考 默认", "mode": "auto"}
        thinking_style = "class:status" if thinking["mode"] == "disabled" else "class:thinking"
        fields = [("class:model", model), (context_style, context),
                  (thinking_style, thinking["label"]), (permission_style, self.permissions.label)]
        task_mode = {"plan": "规划", "execute": "执行计划"}.get(integrations.get("task_mode"))
        if task_mode:
            fields.insert(2, ("class:skills", task_mode))
        if width >= 120:
            fields.insert(0, ("class:brand", "zzcode"))
        if width < 100:
            fields[1] = (context_style, "≈" + compact_tokens(stats.get("context_tokens", 0)))
        if width < 76:
            spinner = elapsed = ""
        if width < 60:
            fields = [(thinking_style, thinking["label"]), (permission_style, self.permissions.label)]
            if task_mode:
                fields.insert(0, ("class:skills", task_mode))
        phase = {"就绪": "ready", "思考中": "thinking", "压缩中": "thinking", "回复中": "reply",
                 "等待审批": "approval", "等待确认": "approval"}.get(self._phase, "tool")
        right = spinner + self._phase + elapsed
        right_width = get_cwidth(right)
        available = max(0, width - right_width - 1)
        # 优先缩短较长的模型名，保留思考设置和当前权限。
        for index, (style, value) in enumerate(fields):
            if style == "class:model":
                rest = sum(get_cwidth(text) for _, text in fields) - get_cwidth(value) + 3 * (len(fields) - 1)
                budget = max(1, available - rest)
                if get_cwidth(value) > budget:
                    prefix, used = "", 0
                    for char in value:
                        if used + get_cwidth(char) > budget - 1:
                            break
                        prefix += char
                        used += get_cwidth(char)
                    fields[index] = (style, prefix + "…")
                break
        left = []
        for index, field in enumerate(fields):
            if index:
                left.append(("class:separator", " │ "))
            left.append(field)
        left_width = sum(get_cwidth(char) for _, text in left for char in text)
        if left_width > available:
            shortened, count = [], 0
            for style, text in left:
                for char in text:
                    if count + get_cwidth(char) > max(0, available - 1):
                        break
                    shortened.append((style, char))
                    count += get_cwidth(char)
                else:
                    continue
                break
            left = shortened + ([("class:separator", "…")] if available else [])
            left_width = count + bool(available)
        padding = max(1, width - left_width - right_width)
        return [*left, ("", " " * padding), ("class:phase." + phase, spinner + self._phase), ("class:elapsed", elapsed)]

    def _viewport(self):
        size = self.app.output.get_size()
        width, height = max(1, size.columns), max(1, size.rows - 3)
        with self._lock:
            key = (self._revision, width)
            history, transient = list(self._history), self._transient
            details = {index: dict(detail) for index, detail in self._details.items()}
        if key != self._cache_key:
            previous_count = len(self._lines)
            same_width = self._cache_key is not None and self._cache_key[1] == width
            blocks = []
            for index, text in enumerate(history):
                detail = details.get(index)
                if detail is None:
                    blocks.append((text, None))
                    continue
                def toggle(event, index=index):
                    if event.button == MouseButton.LEFT and event.event_type == MouseEventType.MOUSE_UP:
                        self.toggle_detail(index)
                        return None
                    return NotImplemented
                symbol, action = ("▾", "收起") if detail["expanded"] else ("▸", "展开")
                title = Text(f"{symbol} {detail['label']}（点击{action}）", style="#8c8c8c")
                blocks.append((self.console.render_text(title), toggle))
                if detail["expanded"]:
                    blocks.append((self.console.render_text(detail_view(detail["content"], detail["language"])), None))
            if transient is not None:
                blocks.append((self.console.render_text(transient), None))
            lines, line, column = [], [], 0
            for text, handler in blocks:
                for style, content, *_ in to_formatted_text(ANSI(text)):
                    for char in content:
                        if char == "\n":
                            lines.append(line)
                            line, column = [], 0
                        elif char != "\r":
                            cell = get_cwidth(char)
                            if column + cell > width and line:
                                lines.append(line)
                                line, column = [], 0
                            line.append((style, char, handler) if handler is not None else (style, char))
                            column += cell
            if line:
                lines.append(line)
            if self._scroll and same_width:
                self._scroll = max(0, self._scroll + len(lines) - previous_count)
            self._lines, self._cache_key = lines, key
        self._scroll = min(self._scroll, max(0, len(self._lines) - height))
        end = max(0, len(self._lines) - self._scroll)
        visible = self._lines[max(0, end - height):end]
        return [fragment for index, line in enumerate(visible)
                for fragment in ([('', '\n')] if index else []) + line]

    def scroll(self, amount):
        self._scroll = max(0, self._scroll + amount)
        self.app.invalidate()

    def _mouse(self, event):
        if event.event_type in (MouseEventType.SCROLL_UP, MouseEventType.SCROLL_DOWN):
            self.scroll(3 if event.event_type == MouseEventType.SCROLL_UP else -3)
            return None
        return self._formatted_mouse(event)

    def close(self, replay_to=None):
        self._closed = True
        if self.app.is_running and self.app.loop is not None:
            self.app.loop.call_soon_threadsafe(self.app.exit)
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("终端界面未能正常退出。")
        if replay_to is not None:
            from rich.text import Text
            replay_to.print(Text.from_ansi("".join(self._history)), end="", soft_wrap=True)
