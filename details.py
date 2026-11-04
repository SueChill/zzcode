"""显示用的详情摘要与长代码分段，不改变模型消息和执行内容。"""
from rich.markdown import ListItem, Markdown
from rich.segment import Segment
from rich.syntax import Syntax
from rich.text import Text


def _first_flags(items):
    for index, item in enumerate(items):
        yield index == 0, item


class _AlignedListItem(ListItem):
    """同 Rich 的列表项，但列表符号从第 0 列开始。

    Rich 默认渲染成 ``" • "`` 并预留 3 列，会让列表整体比正文右移一格；
    这里改为 ``"• "`` 和 2 列，项目符号与正文首字对齐。
    """

    def render_bullet(self, console, options):
        render_options = options.update(width=options.max_width - 2)
        lines = console.render_lines(self.elements, render_options, style=self.style)
        bullet_style = console.get_style("markdown.item.bullet", default="none")
        bullet = Segment("• ", bullet_style)
        padding = Segment("  ", bullet_style)
        for first, line in _first_flags(lines):
            yield bullet if first else padding
            yield from line
            yield Segment("\n")

    def render_number(self, console, options, number, last_number):
        number_width = len(str(last_number)) + 2
        render_options = options.update(width=options.max_width - number_width)
        lines = console.render_lines(self.elements, render_options, style=self.style)
        number_style = console.get_style("markdown.item.number", default="none")
        numeral = Segment(f"{number}. ".ljust(number_width), number_style)
        padding = Segment(" " * number_width, number_style)
        for first, line in _first_flags(lines):
            yield numeral if first else padding
            yield from line
            yield Segment("\n")


class AnswerMarkdown(Markdown):
    """正文用 Markdown：列表与正文左对齐，其余行为沿用 Rich。"""

    elements = {**Markdown.elements, "list_item_open": _AlignedListItem}


def detail_label(operation):
    kind = {"python": "脚本内容", "diff": "改动详情", "json": "参数详情"}.get(operation.language, "内容详情")
    return f"{kind} · {len(operation.content.splitlines())} 行 · {len(operation.content)} 字符"


def detail_view(content, language):
    if language == "reasoning":
        # 思考是自然语言，不加行号和语法高亮，保持和流式显示一致的灰色。
        return Text(content, style="#8c8c8c")
    return Syntax(content, language or "text", theme="ansi_dark", word_wrap=True,
                  line_numbers=language != "diff")


def answer_parts(text):
    """使用 Markdown 的实际代码范围，避免把正文中的反引号误作代码围栏。"""
    lines = text.splitlines(keepends=True)
    start = 0
    for token in AnswerMarkdown(text).parsed:
        if token.type not in {"fence", "code_block"} or token.level != 0 or token.map is None:
            continue
        if len(token.content.splitlines()) < 8 and len(token.content) < 400:
            continue
        first, last = token.map
        if first > start:
            yield "markdown", "".join(lines[start:first]), ""
        yield "code", token.content, token.info.split()[0] if token.info.strip() else "text"
        start = last
    if start < len(lines):
        yield "markdown", "".join(lines[start:]), ""


def folded_answer(text):
    parts = []
    for kind, content, language in answer_parts(text):
        if kind == "code":
            parts.append(f"\n> ▸ {language} 代码 · {len(content.splitlines())} 行（已收起，/code 查看完整回复）\n\n")
        else:
            parts.append(content)
    return "".join(parts)
