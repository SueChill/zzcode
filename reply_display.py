import os
import sys
from collections import ChainMap

from rich.console import Console, Group
from rich.live import Live
from rich.segment import Segment, Segments
from rich.spinner import Spinner
from rich.text import Text

from details import AnswerMarkdown, folded_answer


REASONING_STYLE = "#8c8c8c"
TOOL_STYLE = "#8c8c8c"

TASK_STATUS_LABELS = {
    "completed": "完成",
    "cancelled": "取消",
    "round_limit": "达到轮数上限",
    "loop_stopped": "循环保护",
    "context_limit": "上下文超限",
    "model_error": "模型错误",
    "internal_error": "程序异常",
    "plan_incomplete": "计划未完成",
}
MAX_TASK_SUMMARY_CHARS = 140


def task_summary(stats):
    """任务结束后的一行摘要；没有工具调用的普通聊天不输出。"""
    if not stats or (not stats["tools"] and stats["status"] == "completed"):
        return ""
    status = TASK_STATUS_LABELS.get(stats["status"], stats["status"])
    parts = [status, f"{stats['duration']:.1f}s", f"{stats['rounds']} 轮", f"{stats['tools']} 次工具"]
    files = stats["files"]
    if files:
        names = "、".join(files[:3]) + (" 等" if len(files) > 3 else "")
        parts.append(f"改动 {len(files)} 个文件（{names}）")
    if stats["commands"]:
        failed = f"（{stats['failed_commands']} 次失败）" if stats["failed_commands"] else ""
        parts.append(f"运行 {stats['commands']} 次{failed}")
    line = " · ".join(parts)
    if len(line) > MAX_TASK_SUMMARY_CHARS:
        line = line[:MAX_TASK_SUMMARY_CHARS - 1] + "…"
    return line


def create_console():
    environment = os.environ
    if sys.platform == "win32" and sys.stdout.isatty():
        # 原生 Windows 终端具备重绘能力，避免继承 TERM=dumb 禁用动画和颜色。
        if os.getenv("TERM", "").lower() in {"dumb", "unknown"}:
            environment = ChainMap({"TERM": ""}, environment)
    return Console(highlight=False, _environ=environment)


class ReplyDisplay:
    """将临时状态、思考内容和正常回复分别渲染。"""

    def __init__(self, console, show_reasoning=True):
        self.console = console
        self._terminal_ui = getattr(console, "terminal_ui", None)
        self.show_reasoning = show_reasoning
        self._reasoning = ""
        self._answer = ""
        self._answer_live = None
        self._answer_cache_key = None
        self._answer_cache = None
        self._active = False
        self._label = ""
        self._spinner = Spinner("dots", style="#d7ad78")
        self._live = None

    def _reasoning_view(self):
        if not self.show_reasoning or not self._reasoning or self.console.height < 4:
            return self._spinner
        lines = self.console.render_lines(Text(self._reasoning), pad=False)
        limit = max(1, min(8, self.console.height - 4))
        visible = "\n".join("".join(segment.text for segment in line) for line in lines[-limit:])
        if len(lines) > limit:
            visible = "…\n" + visible
        content = [
            Text("思考", style="dim " + REASONING_STYLE),
            Text(visible, style=REASONING_STYLE),
        ]
        if self._terminal_ui is None:
            content.append(self._spinner)
        return Group(*content)

    def _start(self, label, refresh=False, style=None):
        if self._terminal_ui is not None:
            self._terminal_ui.set_phase(label.rstrip("…"))
            if self.show_reasoning and self._reasoning:
                self._terminal_ui.set_transient(self._reasoning_view())
            return
        changed = label != self._label
        self._label = label
        self._spinner.update(text=Text(label, style=style or "#8794a5"),
                             style=style or "#d7ad78")
        if not self._active:
            # 新实例不携带上一段动画的高度，避免重启时擦掉已打印的正文。
            self._live = Live(
                console=self.console,
                get_renderable=self._reasoning_view,
                transient=True,
                refresh_per_second=10,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live.start(refresh=True)
            self._active = True
        elif changed or refresh:
            self._live.refresh()
        # 同阶段的新片段由刷新线程统一绘制，避免每个 chunk 都强制重绘。

    def _stop(self):
        if self._terminal_ui is not None:
            self._terminal_ui.set_transient()
            return
        if self._active:
            self._live.stop()
            self._active = False
            self._live = None

    def _markdown(self):
        # ANSI 主题沿用终端背景，代码块保留语法高亮。
        return AnswerMarkdown(folded_answer(self._answer), code_theme="ansi_dark")

    def _answer_view(self):
        key = (self._answer, self.console.width, self.console.height)
        if key != self._answer_cache_key:
            lines = self.console.render_lines(AnswerMarkdown(folded_answer(key[0]), code_theme="ansi_dark"), pad=False)
            limit = max(1, self.console.height - 2)
            if len(lines) > limit:
                lines = [[Segment("…")], *lines[-(limit - 1):]] if limit > 1 else lines[-1:]
            self._answer_cache = Segments([
                segment for line in lines for segment in [*line, Segment.line()]
            ])
            self._answer_cache_key = key
        return self._answer_cache

    def _finish_answer(self):
        if self._terminal_ui is not None:
            self._terminal_ui.set_transient()
        if self._answer_live is not None:
            self._answer_live.stop()
            self._answer_live = None
        if self._answer:
            # 临时区域始终限制在终端高度内；清除后一次打印全文，避免长回复丢行。
            if self._terminal_ui is not None:
                self._terminal_ui.append_answer(self._answer)
            else:
                self.console.print(self._markdown())
            self._answer = ""
            self._answer_cache_key = None
            self._answer_cache = None

    def _flush_reasoning(self):
        if self.show_reasoning and self._reasoning:
            if self._terminal_ui is not None:
                # 全屏界面保留完整正文，标题可点击展开；非全屏仍用一行提示。
                self._terminal_ui.append_reasoning(self._reasoning)
            else:
                self.console.print(
                    f"思考已收起（{len(self._reasoning)} 字符） · /reasoning 查看最近一段",
                    style="dim " + REASONING_STYLE, markup=False,
                )
        self._reasoning = ""

    def reasoning(self, content):
        if not content:
            return
        if self._answer:
            self._finish_answer()
            self.console.print()
        first_chunk = not self._reasoning
        self._reasoning += content
        # 文案相同时，第一段思考也立即显示，后续片段继续按刷新频率绘制。
        self._start("思考中…", refresh=first_chunk)

    def text(self, content):
        if not content:
            return
        # Markdown 标记可能跨 chunk，始终基于已收到的完整正文解析。
        self._stop()
        self._flush_reasoning()
        self._answer += content
        if self._terminal_ui is not None:
            self._terminal_ui.set_phase("回复中")
            self._terminal_ui.set_transient(self._markdown())
            return
        if self.console.is_interactive and self._answer_live is None:
            self._answer_live = Live(
                console=self.console,
                get_renderable=self._answer_view,
                transient=True,
                vertical_overflow="crop",
                refresh_per_second=10,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._answer_live.start(refresh=True)

    def pause(self):
        self._stop()
        self._flush_reasoning()
        self._finish_answer()

    def executing(self, name):
        self._start("执行工具 · " + name, style=TOOL_STYLE)

    def compacting(self):
        self._start("压缩中…")

    def tool(self, summary):
        self.pause()
        self.console.print(summary, style=TOOL_STYLE, markup=False)
        self.executing(summary.split(" · ", 1)[0])

    def __enter__(self):
        self._start("思考中…")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop()
        self._flush_reasoning()
        self._finish_answer()
