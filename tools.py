# zzcode 到此一游
import asyncio
import json
import locale
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tokenize
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from difflib import unified_diff
from inspect import signature
from io import BytesIO
from pathlib import Path

from httpx import HTTPError
from tavily import (
    AsyncTavilyClient,
    BadRequestError,
    InvalidAPIKeyError,
    TavilyKeylessLimitError,
    UsageLimitExceededError,
)
from tracing import span
from planning import PlanError
from tool_result import ToolResult, render_result


# 启动终端所在目录是本次工作区，独立于 zzcode 的安装位置。
PROJECT_ROOT = Path.cwd().resolve()


class PermissionPolicy:
    """当前运行的本地审批模式，不写入模型消息或数据库。"""

    MODES = {"readOnly": "只读", "auto": "自动", "fullAccess": "完全访问"}
    ORDER = ("readOnly", "auto", "fullAccess")
    ALIASES = {"default": "readOnly", "acceptedits": "auto"}
    SCOPES = {
        "readOnly": "读取正常放行，文件修改和命令运行需要确认；保留配置中明确预授权的 MCP 工具。",
        "auto": "工作区内新建、编辑和删除自动批准；命令和未预授权的 MCP 工具仍需确认。",
        "fullAccess": "工具调用免审批；已有路径与参数检查保留。命令仍以当前系统用户权限执行。",
    }

    def __init__(self, mode="readOnly"):
        self.set_mode(mode)

    def set_mode(self, mode):
        normalized = {name.casefold(): name for name in self.MODES}.get(mode.casefold())
        normalized = normalized or self.ALIASES.get(mode.casefold())
        if normalized is None:
            raise ValueError("权限模式应为 readOnly、auto 或 fullAccess。")
        self.mode = normalized

    def toggle(self):
        # /permissions 的简写只切换只读/自动；完全访问走 Shift+Tab 二次确认或显式参数。
        self.mode = "auto" if self.mode == "readOnly" else "readOnly"

    def next_mode(self):
        """Shift+Tab 的循环顺序：只读 → 自动 → 完全访问 → 只读。"""
        index = self.ORDER.index(self.mode)
        return self.ORDER[(index + 1) % len(self.ORDER)]

    @property
    def label(self):
        return self.MODES[self.mode]

    @property
    def scope(self):
        return self.SCOPES[self.mode]

    def allows(self, tool):
        return self.mode == "fullAccess" or (
            self.mode == "auto" and tool.get("permission_group") == "file_edit"
        )


@dataclass(frozen=True)
class ToolPreview:
    """仅用于本地审批显示，不作为 API 参数发送。"""

    title: str
    description: str
    content: str
    language: str


@dataclass(frozen=True)
class PreparedTool(ToolPreview):
    """审批预览与已准备好的执行操作。"""

    new_file: str | None = None

    def run(self):
        raise NotImplementedError


@dataclass(frozen=True)
class DeferredTool(PreparedTool):
    """用现成回调执行的准备操作，例如命令和脚本审批。"""

    action: Callable[[], object] = None

    def run(self):
        return self.action()


class FileToolError(ValueError):
    """可以直接返回给模型的结构化文件操作错误。"""

    def __init__(self, code, message, **details):
        super().__init__(message)
        self.result = {"status": "error", "error_code": code, "message": message, **details}


class ReadTracker:
    """记录本会话读过的文件，未读取就覆盖或编辑会被拒绝。"""

    def __init__(self):
        self._paths = set()

    def mark(self, file_path):
        self._paths.add(file_path)

    def forget(self, file_path):
        self._paths.discard(file_path)

    def require(self, file_path, relative):
        if file_path not in self._paths:
            raise FileToolError(
                "read_required",
                f"修改 {relative} 前需要先用 read_file 读取当前内容。",
                path=relative,
            )


_ENCODINGS = ("utf-8", "gbk", "gb18030")


def _normalize_newlines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _detect_newline(text):
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    cr = text.count("\r") - crlf
    counts = {"\r\n": crlf, "\n": lf, "\r": cr}
    present = [name for name, count in counts.items() if count]
    if not present:
        return "\n", False
    dominant = max(("\r\n", "\n", "\r"), key=lambda name: counts[name])
    return dominant, len(present) > 1


def _looks_binary(text):
    if not text:
        return False
    control = sum(1 for character in text if ord(character) < 32 and character not in "\t\n\r")
    return control > max(4, len(text) * 0.01)


def _decode_text(raw, file_path):
    name = file_path.name
    try:
        if raw.startswith(b"\xef\xbb\xbf"):
            return "utf-8", b"\xef\xbb\xbf", raw[3:].decode("utf-8")
        if raw.startswith(b"\xff\xfe"):
            return "utf-16-le", b"\xff\xfe", raw[2:].decode("utf-16-le")
        if raw.startswith(b"\xfe\xff"):
            return "utf-16-be", b"\xfe\xff", raw[2:].decode("utf-16-be")
    except UnicodeDecodeError as error:
        raise FileToolError("encoding_error", f"{name} 的 BOM 与内容不匹配，无法解码。") from error
    if b"\x00" in raw:
        raise FileToolError("binary_file", f"{name} 看起来是二进制文件，不能按文本处理。")
    for encoding in _ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if _looks_binary(text):
            break
        return encoding, b"", text
    raise FileToolError(
        "unsupported_encoding",
        f"{name} 无法按 UTF-8、GBK 或 UTF-16 解码，可能不是文本文件。",
    )


class TextFile:
    """文件的文本、编码、换行和 BOM；写回时保持原有表示。"""

    def __init__(self, file_path, raw):
        self.path = file_path
        self.raw = raw
        if raw is None:
            self.encoding = "utf-8"
            self.bom = b""
            self.text = ""
            self.newline = "\n"
            self.mixed_newlines = False
            return
        self.encoding, self.bom, decoded = _decode_text(raw, file_path)
        self.newline, self.mixed_newlines = _detect_newline(decoded)
        self.text = _normalize_newlines(decoded)

    @classmethod
    def load(cls, file_path):
        if not file_path.exists():
            return cls(file_path, None)
        if not file_path.is_file():
            raise ValueError("path 必须是文件路径。")
        return cls(file_path, file_path.read_bytes())

    @classmethod
    def for_new_file(cls, file_path, content):
        instance = cls(file_path, None)
        instance.text = _normalize_newlines(content)
        instance.newline, instance.mixed_newlines = _detect_newline(content)
        return instance

    @property
    def newline_label(self):
        return {"\r\n": "CRLF", "\n": "LF", "\r": "CR"}[self.newline]

    @property
    def has_bom(self):
        return bool(self.bom)

    def render(self, text):
        """按原编码、换行和 BOM 渲染新文本。"""
        normalized = _normalize_newlines(text)
        if self.newline != "\n":
            normalized = normalized.replace("\n", self.newline)
        try:
            data = normalized.encode(self.encoding)
        except UnicodeEncodeError as error:
            raise FileToolError(
                "encoding_error",
                f"{self.path.name} 使用 {self.encoding.upper()} 编码，新内容包含无法表示的字符；"
                "请调整内容，或先用 write_file 把文件转写为 UTF-8。",
            ) from error
        return self.bom + data


def _atomic_write(file_path, data):
    """同目录临时文件加替换，避免中途失败留下半截文件。"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    previous_mode = file_path.stat().st_mode if file_path.exists() else None
    handle, temporary = tempfile.mkstemp(prefix=f".{file_path.name}.", suffix=".tmp", dir=file_path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        if previous_mode is not None:
            os.chmod(temporary, previous_mode)
        os.replace(temporary, file_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


@dataclass(frozen=True, kw_only=True)
class FileChange(PreparedTool):
    """一次文件变更：审批预览、落盘和读取状态记录使用同一份数据。"""

    path: Path
    relative: str
    before_raw: bytes | None
    after_raw: bytes | None
    result: dict
    tracker: ReadTracker | None = None

    def run(self):
        _check_approved_file(self.relative, self.path, self.before_raw)
        if self.after_raw is None:
            self.path.unlink()
            if self.tracker is not None:
                self.tracker.forget(self.path)
        else:
            if self.after_raw != self.before_raw:
                _atomic_write(self.path, self.after_raw)
            if self.tracker is not None:
                self.tracker.mark(self.path)
        return self.result


def get_tool_registry(workspace_state, external_tools=None, reads=None):
    """合并内置工具、当前会话状态查询和已连接的外部工具。"""
    reads = ReadTracker() if reads is None else reads

    def tracked_read(path, start_line=None, line_count=None):
        result = read_file(path, start_line, line_count)
        reads.mark(resolve_project_path(path))
        return result

    def tracked_write(path, content):
        return prepare_write_file(path, content, reads=reads)

    def tracked_edit(path, old_text, new_text, replace_all=False):
        return prepare_edit_file(path, old_text, new_text, replace_all=replace_all, reads=reads)

    def tracked_delete(path):
        return prepare_delete_file(path, reads=reads)

    registry = {
        **TOOL_REGISTRY,
        "read_file": {**TOOL_REGISTRY["read_file"], "handler": tracked_read},
        "write_file": {**TOOL_REGISTRY["write_file"], "handler": tracked_write},
        "edit_file": {**TOOL_REGISTRY["edit_file"], "handler": tracked_edit},
        "delete_file": {**TOOL_REGISTRY["delete_file"], "handler": tracked_delete},
        "query_workspace_state": {
            "description": "查询当前会话记录的关键变化：撤销涉及的路径、任务取消或轮数上限等。收到工作区提示且当前请求需要这些详情时使用；问候或无关问题无需调用。仅返回状态，不返回文件正文；需要确认文件当前内容时再用 read_file。",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            "handler": workspace_state.query,
        },
    }
    if external_tools:
        if registry.keys() & external_tools.keys():
            raise ValueError("外部工具与现有工具重名，未覆盖。")
        registry.update(external_tools)
    return registry


def get_tool_definitions(registry=None):
    """从注册表生成 API 的工具声明，不把本地函数传给模型。"""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for name, tool in (TOOL_REGISTRY if registry is None else registry).items()
    ]


def _tool_arguments(call, registry):
    tool = registry.get(call["name"])
    if tool is None:
        raise ValueError("未知工具。")
    arguments = json.loads(call["arguments"])
    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是 JSON 对象。")
    if "validate_arguments" in tool:
        tool["validate_arguments"](arguments)
    else:
        signature(tool["handler"]).bind(**arguments)
    if tool.get("check_execution") is not None:
        tool["check_execution"](arguments)
    return tool, arguments


def prepare_tool_groups(calls, registry=None):
    """按原顺序准备连续的文件变更，合成一次审批；其他调用逐个处理。"""
    registry = TOOL_REGISTRY if registry is None else registry
    index = 0
    while index < len(calls):
        changes = []
        for call in calls[index:]:
            tool = registry.get(call["name"])
            if (not tool or tool.get("permission_group") != "file_edit"
                    or not tool.get("requires_approval")):
                break
            try:
                _, arguments = _tool_arguments(call, registry)
                operation = tool["handler"](**arguments)
            except (OSError, ValueError, TypeError):
                # 参数或前置条件错误留给统一执行入口报告，不加入这次审批。
                break
            if not isinstance(operation, FileChange) or any(
                    operation.path == previous.path for previous in changes):
                break
            changes.append(operation)
        if len(changes) > 1:
            yield calls[index:index + len(changes)], changes
            index += len(changes)
        else:
            yield [calls[index]], None
            index += 1


def file_batch_preview(changes):
    """合并预览：新建文件只列路径，其他变更附带完整差异。"""
    new_files = all(change.before_raw is None and change.after_raw is not None for change in changes)
    if new_files:
        title = f"新建 {len(changes)} 个文件"
        description = "\n".join(f"  {change.new_file}" for change in changes)
    else:
        title = f"文件变更 · {len(changes)} 个"
        description = "\n".join(f"  {change.relative}" for change in changes)
    content = "\n\n".join(change.content for change in changes if change.content)
    return ToolPreview(title, description, content, "diff")


def execute_tool(call, registry=None, approve=None, on_execute=None, prepared=None, permissions=None):
    """统一校验、准备和审批，获准后才执行需要审批的工具。"""
    try:
        with span("tool.prepare", {"call_id": call.get("id"), "prepared_in_batch": prepared is not None}):
            tool, arguments = _tool_arguments(call, TOOL_REGISTRY if registry is None else registry)
            handler = tool["handler"]
            operation = None
            if tool.get("requires_approval", False):
                operation = handler(**arguments) if prepared is None else prepared
                if not isinstance(operation, PreparedTool):
                    raise ValueError("需要审批的工具必须先返回可预览的操作。")
        if operation is not None:
            revalidation = call["name"] in {"run_command", "run_python"} and arguments.get("revalidate") is True
            if revalidation or permissions is None or not permissions.allows(tool):
                if approve is None:
                    return ToolResult(
                        status="approval_required", executed=False,
                        message="此工具需要用户审批，尚未执行。",
                    ).to_json()
                if approve(operation) is not True:
                    return ToolResult(
                        status="denied", executed=False,
                        message="用户拒绝了这次工具调用，未执行。请尊重该决定，不要自动重试相同操作。",
                    ).to_json()
        if on_execute is not None:
            on_execute()
        with span("tool.execute", {"call_id": call.get("id"), "bucket": "execution"}) as execution:
            try:
                result = operation.run() if operation is not None else handler(**arguments)
                return render_result(result)
            except (KeyboardInterrupt, GeneratorExit):
                execution.status = "unknown"
                execution.attrs.update(cancelled=True, result_confirmed=False)
                raise
    except FileToolError as error:
        return json.dumps(error.result, ensure_ascii=False)
    except PlanError as error:
        return json.dumps(error.result, ensure_ascii=False)
    except (OSError, ValueError, TypeError) as error:
        return ToolResult(status="error", error_code="tool_error",
                          message=f"工具执行失败：{error}").to_json()


def resolve_project_path(path):
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path 必须是非空字符串。")
    file_path = (PROJECT_ROOT / path).resolve()
    if not file_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("只能访问当前项目内的文件。")

    for part in file_path.relative_to(PROJECT_ROOT).parts:
        if ":" in part:
            raise ValueError("不支持 Windows 备用数据流路径。")
        name = part.rstrip(" .").lower()
        if name in {".git", ".zzcode"} or (
            name != ".env.example" and (name == ".env" or name.startswith(".env."))
        ):
            raise ValueError("不能访问 .env 配置、.git 或 .zzcode 内的文件。")

    return file_path


def read_file(path, start_line=None, line_count=None):
    file_path = resolve_project_path(path)
    if start_line is not None or line_count is not None:
        start_line = 1 if start_line is None else start_line
        line_count = 200 if line_count is None else line_count
        if type(start_line) is not int or start_line < 1:
            raise ValueError("start_line 必须是从 1 开始的正整数。")
        if type(line_count) is not int or not 1 <= line_count <= 500:
            raise ValueError("line_count 必须是 1 到 500 的整数。")
    source = TextFile.load(file_path)
    if source.raw is None:
        raise FileNotFoundError(f"文件不存在：{path}")
    if start_line is not None or line_count is not None:
        lines = source.text.splitlines(keepends=True)
        selected = []
        size = 0
        has_more = False
        for number in range(start_line, len(lines) + 1):
            content = lines[number - 1].rstrip("\n")
            numbered = f"{number}: {content}\n"
            if len(selected) == line_count or (selected and size + len(numbered) > 20000):
                has_more = True
                break
            if len(numbered) > 20000:
                raise ValueError(f"第 {number} 行超过单次 20000 字符上限，无法完整返回该行。")
            selected.append(numbered)
            size += len(numbered)
        end_line = start_line + len(selected) - 1 if selected else None
        return {
            "path": file_path.relative_to(PROJECT_ROOT).as_posix(),
            "start_line": start_line, "end_line": end_line,
            "content": "".join(selected), "has_more": has_more,
            "next_start_line": end_line + 1 if has_more else None,
            "encoding": source.encoding, "newline": source.newline_label,
            "bom": source.has_bom,
        }
    content = source.text[:20001]
    if len(content) > 20000:
        return content[:20000] + "\n[文件内容超过 20000 个字符，已截断]"
    return content


def _file_bytes(file_path):
    if not file_path.exists():
        return None
    if not file_path.is_file():
        raise ValueError("path 必须是文件路径。")
    return file_path.read_bytes()


def _check_approved_file(path, file_path, before):
    if resolve_project_path(path) != file_path or _file_bytes(file_path) != before:
        raise ValueError("目标文件或路径在审批期间发生变化，本次审批已失效，请重新读取后发起操作。")


def _text_details(source):
    if source is None:
        return "不存在"
    newline = source.newline_label + ("（混合）" if source.mixed_newlines else "")
    bom = "有 BOM" if source.has_bom else "无 BOM"
    ending = "有末尾换行" if source.text.endswith("\n") else "无末尾换行"
    return f"{len(source.raw)} 字节，{newline}，{bom}，{ending}，{source.encoding.upper()}"


def _make_file_change(*, file_path, relative, before, after, title, result,
                      tracker=None, new_file=None):
    if after is None:
        description = diff = ""
    else:
        diff = "\n".join(unified_diff(
            before.text.splitlines() if before.raw is not None else [],
            after.text.splitlines(),
            fromfile=f"a/{relative}" if before.raw is not None else "/dev/null",
            tofile=f"b/{relative}", lineterm="",
        ))
        description = (
            f"路径：{relative}\n"
            f"修改前：{_text_details(before if before.raw is not None else None)}\n"
            f"修改后：{_text_details(after)}"
        )
        if not diff:
            description += "\n文本行没有差异，请核对上方文件状态、编码与换行信息。"
    return FileChange(
        title=title, description=description, content=diff, language="diff",
        new_file=new_file, path=file_path, relative=relative,
        before_raw=before.raw, after_raw=after.raw if after is not None else None,
        result=result, tracker=tracker,
    )


def prepare_write_file(path, content, *, reads=None):
    """准备完整文件内容和差异；此阶段不创建目录或写文件。"""
    if not isinstance(content, str):
        raise ValueError("content 必须是字符串。")
    file_path = resolve_project_path(path)
    relative = file_path.relative_to(PROJECT_ROOT).as_posix()
    before = TextFile.load(file_path)
    if before.raw is not None and reads is not None:
        reads.require(file_path, relative)
    source = TextFile.for_new_file(file_path, content) if before.raw is None else before
    after_raw = source.render(content)
    after = TextFile(file_path, after_raw)
    result = {
        "path": relative,
        "characters_written": len(content),
        "encoding": source.encoding,
        "newline": source.newline_label,
        "bom": source.has_bom,
    }
    return _make_file_change(
        file_path=file_path, relative=relative, before=before, after=after,
        title="新建文件" if before.raw is None else "整文件写入",
        result=result, tracker=reads,
        new_file=relative if before.raw is None else None,
    )


def write_file(path, content):
    """执行本地写入；模型的调用由 execute_tool 统一审批。"""
    return prepare_write_file(path, content).run()


def prepare_edit_file(path, old_text, new_text, *, replace_all=False, reads=None):
    """在审批前完成精确匹配，预览和执行共用同一份替换结果。"""
    if not isinstance(old_text, str) or not old_text:
        raise ValueError("old_text 必须是非空字符串。")
    if not isinstance(new_text, str):
        raise ValueError("new_text 必须是字符串。")
    if not isinstance(replace_all, bool):
        raise ValueError("replace_all 必须是布尔值。")
    file_path = resolve_project_path(path)
    relative = file_path.relative_to(PROJECT_ROOT).as_posix()
    before = TextFile.load(file_path)
    if before.raw is None:
        raise FileToolError("file_not_found", f"{relative} 不存在；新建文件请使用 write_file。", path=relative)
    if reads is not None:
        reads.require(file_path, relative)
    old = _normalize_newlines(old_text)
    new = _normalize_newlines(new_text)
    occurrences = before.text.count(old)
    if occurrences == 0:
        raise ValueError("没有找到 old_text，请读取文件后重新定位。")
    if occurrences > 1 and not replace_all:
        raise ValueError(f"old_text 命中 {occurrences} 处，请提供更长的片段，或传 replace_all=true 全部替换。")
    updated = before.text.replace(old, new) if replace_all else before.text.replace(old, new, 1)
    after_raw = before.render(updated)
    after = TextFile(file_path, after_raw)
    result = {
        "path": relative,
        "replacements": occurrences if replace_all else 1,
        "encoding": before.encoding,
        "newline": before.newline_label,
        "bom": before.has_bom,
    }
    return _make_file_change(
        file_path=file_path, relative=relative, before=before, after=after,
        title="局部编辑文件", result=result, tracker=reads,
    )


def edit_file(path, old_text, new_text, replace_all=False):
    """执行本地精确替换；模型的调用由 execute_tool 统一审批。"""
    return prepare_edit_file(path, old_text, new_text, replace_all=replace_all).run()


def prepare_delete_file(path, *, reads=None):
    """准备单个文件的删除摘要，不递归或沿链接删除。"""
    file_path = resolve_project_path(path)
    if Path(os.path.abspath(PROJECT_ROOT / path)) != file_path:
        raise ValueError("删除文件不支持链接路径，请使用实际文件路径。")
    before_raw = _file_bytes(file_path)
    if before_raw is None:
        raise ValueError("要删除的文件不存在。")
    relative = file_path.relative_to(PROJECT_ROOT).as_posix()
    try:
        lines = len(_normalize_newlines(_decode_text(before_raw, file_path)[2]).splitlines())
        suffix = f" · {lines} 行"
    except FileToolError:
        suffix = " · 二进制或非文本"
    result = {"path": relative, "deleted": True}
    return FileChange(
        title=f"删除 {relative} · {len(before_raw)} 字节{suffix}",
        description="", content="", language="diff",
        new_file=None, path=file_path, relative=relative,
        before_raw=before_raw, after_raw=None, result=result, tracker=reads,
    )


def list_files(path=".", depth=1):
    if type(depth) is not int or not 1 <= depth <= 5:
        raise ValueError("depth 必须是 1 到 5 的整数。")
    directory = resolve_project_path(path)
    if not directory.is_dir():
        raise ValueError("path 必须是项目中的目录。")
    entries = []
    unreadable_directories = 0
    ignored_directories = {".venv", "venv", "__pycache__", ".previews", "node_modules", ".zzcode"}

    def visit(current, level):
        nonlocal unreadable_directories
        try:
            children = sorted(current.iterdir(), key=lambda child: (not child.is_dir(), child.name.lower()))
        except OSError:
            unreadable_directories += 1
            return False
        for child in children:
            if child.name.lower() in ignored_directories:
                continue
            relative_path = child.relative_to(PROJECT_ROOT).as_posix()
            try:
                resolved = resolve_project_path(relative_path)
                is_directory = child.is_dir()
            except (OSError, ValueError):
                continue
            if len(entries) == 200:
                return True
            entries.append({
                "path": relative_path,
                "type": "directory" if is_directory else "file",
            })
            # 目录链接只列出，不继续递归，避免循环及重复遍历。
            if is_directory and level < depth and resolved == child:
                if visit(child, level + 1):
                    return True
        return False

    truncated = visit(directory, 1)
    return {
        "directory": directory.relative_to(PROJECT_ROOT).as_posix(),
        "depth": depth,
        "entries": entries,
        "truncated": truncated,
        "unreadable_directories": unreadable_directories,
    }


def grep(query, path="."):
    """递归搜索 UTF-8 文本，按原样匹配关键词。"""
    if not isinstance(query, str) or not query:
        raise ValueError("query 必须是非空字符串。")
    directory = resolve_project_path(path)
    if not directory.is_dir():
        raise ValueError("path 必须是项目中的目录。")

    result = {
        "query": query,
        "directory": directory.relative_to(PROJECT_ROOT).as_posix(),
        "matches": [],
        "truncated": False,
        "skipped_files": 0,
    }
    errors = []
    ignored_directories = {".venv", "venv", "__pycache__", ".previews", "node_modules", ".zzcode"}
    for current, directories, filenames in os.walk(directory, onerror=errors.append):
        # 跳过缓存、受限目录和目录链接，避免沿链接走出项目或形成循环。
        allowed = []
        for name in sorted(directories):
            child = Path(current) / name
            try:
                resolved = resolve_project_path(child.relative_to(PROJECT_ROOT).as_posix())
            except (OSError, ValueError):
                continue
            if name.lower() not in ignored_directories and resolved == child:
                allowed.append(name)
        directories[:] = allowed

        for name in sorted(filenames):
            relative_path = (Path(current) / name).relative_to(PROJECT_ROOT).as_posix()
            try:
                file_path = resolve_project_path(relative_path)
            except (OSError, ValueError):
                continue
            try:
                if not file_path.is_file():
                    continue
                with file_path.open("rb") as file:
                    data = file.read(1024 * 1024 + 1)
                if len(data) > 1024 * 1024 or b"\x00" in data:
                    result["skipped_files"] += 1
                    continue
                content = data.decode("utf-8-sig")
            except (OSError, UnicodeError):
                result["skipped_files"] += 1
                continue

            for line_number, line in enumerate(content.splitlines(), start=1):
                if query not in line:
                    continue
                if len(result["matches"]) == 50:
                    result["truncated"] = True
                    result["unreadable_directories"] = len(errors)
                    return result
                result["matches"].append({
                    "path": relative_path,
                    "line": line_number,
                    "text": line[:500],
                    "line_truncated": len(line) > 500,
                })
    result["unreadable_directories"] = len(errors)
    return result


async def _search_web(query, max_results):
    client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY", "").strip())
    try:
        return await client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
            timeout=15,
        )
    except TavilyKeylessLimitError:
        raise ValueError("免密钥搜索额度已用完，可在 .env 配置 TAVILY_API_KEY 后继续。") from None
    except (BadRequestError, InvalidAPIKeyError, UsageLimitExceededError, HTTPError) as error:
        raise ValueError(f"联网搜索失败（{type(error).__name__}）。") from None
    finally:
        await client.close()


def web_search(query, max_results=5):
    """联网搜索，返回网页标题、链接和摘要。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串。")
    if type(max_results) is not int or not 1 <= max_results <= 10:
        raise ValueError("max_results 必须是 1 到 10 的整数。")

    response = asyncio.run(_search_web(query, max_results))

    return {
        "query": query,
        "results": [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", "")[:1200],
            }
            for item in response.get("results", [])[:max_results]
        ],
    }


def _stop_process_tree(process):
    """结束工具脚本及其后代，供超时和手动取消共用。"""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.TimeoutExpired):
        pass
    if process.poll() is None:
        process.kill()


def _python_path(path, timeout):
    if type(timeout) is not int or not 1 <= timeout <= 60:
        raise ValueError("timeout 必须是 1 到 60 的整数。")
    file_path = resolve_project_path(path)
    if not file_path.is_file() or file_path.suffix.lower() != ".py":
        raise ValueError("path 必须是项目中已有的 .py 文件。")
    return file_path


def prepare_run_python(path, timeout=10, revalidate=False):
    if type(revalidate) is not bool:
        raise ValueError("revalidate 必须是布尔值。")
    file_path = _python_path(path, timeout)
    before = file_path.read_bytes()
    try:
        encoding, _ = tokenize.detect_encoding(BytesIO(before).readline)
        content = before.decode(encoding)
    except (SyntaxError, LookupError, UnicodeError) as error:
        raise ValueError("无法识别脚本编码，未发起运行。") from error

    def run():
        _check_approved_file(path, file_path, before)
        result = run_python(path, timeout)
        if revalidate:
            result["revalidation"] = True
        return result

    description = (
        f"脚本：{file_path.relative_to(PROJECT_ROOT).as_posix()}\n"
        f"解释器：{sys.executable}\n工作目录：{PROJECT_ROOT}\n时限：{timeout} 秒"
    )
    return DeferredTool("审批重验脚本" if revalidate else "运行 Python 脚本", description, content, "python", action=run)


def _run_process(command, cwd, timeout, environment, encoding="utf-8"):
    """两个运行工具共用的等待、输出、超时及取消收尾。"""
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=encoding is not None,
        encoding=encoding,
        errors="replace" if encoding else None,
        env=environment,
        shell=False,
        creationflags=creationflags,
        start_new_session=os.name != "nt",
    )
    timed_out = False
    output_incomplete = False
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                # Windows 的长时间线程等待可能延迟键盘中断，分段等待同一个进程。
                stdout, stderr = process.communicate(timeout=min(0.2, max(0, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    raise
    except subprocess.TimeoutExpired:
        timed_out = True
        _stop_process_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired as error:
            stdout, stderr = error.stdout, error.stderr
            output_incomplete = True
    except BaseException:
        try:
            _stop_process_tree(process)
            process.communicate(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        # 清理后仍交给主循环处理，不把取消或读取失败当作正常工具结果。
        raise
    finally:
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()

    def decode(value):
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.decode(encoding or locale.getencoding(), errors="replace")
        return value or ""

    stdout, stderr = decode(stdout), decode(stderr)
    return {
        "exit_code": None if timed_out else process.returncode,
        "stdout": stdout[:10000],
        "stderr": stderr[:10000],
        "timed_out": timed_out,
        "output_incomplete": output_incomplete,
        "stdout_truncated": len(stdout) > 10000,
        "stderr_truncated": len(stderr) > 10000,
    }


def run_python(path, timeout=10):
    """保留原来的脚本入口、解释器选择和结果格式。"""
    file_path = _python_path(path, timeout)
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    return {
        "path": file_path.relative_to(PROJECT_ROOT).as_posix(),
        **_run_process([sys.executable, "-u", str(file_path)], PROJECT_ROOT, timeout, environment),
    }


def _command_environment(program):
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    scripts = PROJECT_ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    # 裸命令优先使用目标项目的环境；明确提供程序路径时不替用户换解释器。
    if not os.path.dirname(program) and (scripts / ("python.exe" if os.name == "nt" else "python")).is_file():
        environment["PATH"] = str(scripts) + os.pathsep + environment.get("PATH", "")
        environment["VIRTUAL_ENV"] = str(scripts.parent)
        environment.pop("PYTHONHOME", None)
    return environment


def _resolve_executable(program, cwd, environment):
    if os.path.dirname(program):
        candidates = [cwd / program]
    else:
        # 将 PATH 中的相对路径也按本次 cwd 解释，避免误查 zzcode 安装目录。
        candidates = [cwd / directory.strip('"') / program
                      for directory in environment.get("PATH", "").split(os.pathsep) if directory]
    for candidate in candidates:
        found = shutil.which(str(candidate.absolute()), path="")
        if found:
            # 保留虚拟环境里的逻辑入口；解析符号链接后再启动会丢失 venv 身份。
            executable = Path(found).absolute()
            if os.name == "nt" and executable.suffix.lower() not in {".exe", ".com", ".cmd", ".bat"}:
                raise ValueError("此文件不能直接启动，请明确提供解释器，例如 python 或 powershell.exe。")
            return executable
    raise ValueError(f"找不到程序：{program}。请检查目标项目环境、PATH，或提供程序路径。")


def _command_launch(executable, arguments):
    if os.name != "nt" or executable.suffix.lower() not in {".cmd", ".bat"}:
        return [str(executable), *arguments], None

    # npm/mvn 常用批处理启动器；不能让参数在 cmd 中意外变成另一条命令。
    values = [str(executable), *arguments]
    if any(any(character in '&|<>^%!"' or ord(character) < 32 for character in value) for value in values):
        raise ValueError("Windows 批处理参数不支持引号、控制符或 & | < > ^ % !；复杂命令请明确调用 shell 并审批。")
    shell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "cmd.exe"
    if not shell.is_file():
        raise ValueError("找不到 Windows cmd.exe，无法启动批处理。")
    # 使用 CRT 参数规则处理空格及末尾反斜杠；再用外层引号配合 cmd /s。
    quoted = ['"' + re.sub(r"(\\+)$", r"\1\1", value) + '"' for value in values]
    command = subprocess.list2cmdline([str(shell), "/d", "/v:off", "/s", "/c"])
    return command + ' "' + " ".join(quoted) + '"', shell.resolve()


def command_summary(program, arguments, limit=120):
    values, hide_next = [program], False
    for argument in arguments:
        values.append("<代码/命令详情已折叠>" if hide_next else argument)
        hide_next = argument.casefold() in {"-c", "/c", "-e", "--eval", "-command", "-encodedcommand"}
    summary = subprocess.list2cmdline(values).replace("\n", " ").replace("\r", " ")
    return summary if len(summary) <= limit else summary[:limit - 1] + "…"


def prepare_run_command(program, args=None, cwd=".", timeout=30, revalidate=False):
    """先冻结可审阅的命令与环境，批准后才启动。"""
    if type(revalidate) is not bool:
        raise ValueError("revalidate 必须是布尔值。")
    if not isinstance(program, str) or not program.strip() or any(ord(c) < 32 or c == '"' for c in program):
        raise ValueError("program 必须是非空的程序名或路径，不要把整条命令或引号放进 program。")
    if args is not None and (not isinstance(args, list) or any(not isinstance(arg, str) or "\0" in arg for arg in args)):
        raise ValueError("args 必须是字符串列表，不能包含空字符。")
    arguments = [] if args is None else args.copy()
    if type(timeout) is not int or not 1 <= timeout <= 300:
        raise ValueError("timeout 必须是 1 到 300 的整数。")
    directory = resolve_project_path(cwd)
    if not directory.is_dir():
        raise ValueError("cwd 必须是当前项目中已有的目录。")
    environment = _command_environment(program)
    executable = _resolve_executable(program, directory, environment)
    command, shell = _command_launch(executable, arguments)
    paths = [executable] + ([shell] if shell is not None else [])
    targets = [path.resolve() for path in paths]
    identities = [path.stat() for path in paths]

    def run():
        if resolve_project_path(cwd) != directory or not directory.is_dir():
            raise ValueError("工作目录已变化，审批已失效，请重新发起运行。")
        for path, target, before in zip(paths, targets, identities):
            after = path.stat()
            if path.resolve() != target or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("程序入口已变化，审批已失效，请重新发起运行。")
        result = _run_process(command, directory, timeout, environment, encoding=None)
        return {"program": str(executable), "args": arguments, "cwd": directory.relative_to(PROJECT_ROOT).as_posix(),
                **result, **({"revalidation": True} if revalidate else {})}

    preview = {"program": str(executable), "args": arguments}
    summary = command_summary(str(executable), arguments)
    description = f"命令：{summary}\n工作目录：{directory}\n时限：{timeout} 秒 · 无交互输入"
    if shell is not None:
        description += f"\n批处理启动器：{shell} /d /v:off /s /c"
    return DeferredTool("审批重验命令" if revalidate else "运行命令", description, json.dumps(preview, ensure_ascii=False, indent=2), "json", action=run)


def get_current_time():
    """读取运行程序的电脑当前本地时间。"""
    now = datetime.now().astimezone()
    return {
        "datetime": now.isoformat(timespec="seconds"),
        "timezone": now.tzname(),
    }


TOOL_REGISTRY = {
    "read_file": {
        "description": "读取当前项目内的文本文件，自动识别 UTF-8（含 BOM）、GBK 和 UTF-16。只传 path 时返回前 20000 字符；大文件或精确定位时传 start_line（从 1 开始）与 line_count（默认 200，最多 500），返回带行号的完整行、has_more、next_start_line 以及 encoding、newline、bom，可继续读取后续段落。每次最多 20000 字符，单行过长则报错，不悄悄跳过行尾。不能读取 .env 配置或 .git 内的文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对于项目根目录的文件路径，例如 main.py。",
                },
                "start_line": {"type": "integer", "minimum": 1, "description": "读取的起始行，从 1 开始。"},
                "line_count": {"type": "integer", "minimum": 1, "maximum": 500, "description": "读取的行数，分段读取时默认 200。"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "handler": read_file,
    },
    "write_file": {
        "description": "新建或整文件重写当前项目内的文本文件，自动创建缺失的父目录。新建时按给定内容写入；覆盖已有文件前必须先用 read_file 读取当前内容，写回时保持原编码（UTF-8/GBK/UTF-16）、换行和 BOM。局部修改优先使用 edit_file。不能写入 .env 配置或 .git 内的文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对于项目根目录的文件路径，例如 demo.py 或 src/demo.py。",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的完整文本内容，会覆盖文件原内容；空字符串会清空文件。",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "handler": prepare_write_file,
        "requires_approval": True,
        "permission_group": "file_edit",
    },
    "edit_file": {
        "description": "局部修改当前项目内的文本文件（UTF-8/GBK/UTF-16）。修改前先读取文件，提供旧文本和替换后的新文本；默认要求旧文本非空且唯一命中，命中多处时传 replace_all=true 全部替换。按原样匹配，不支持正则表达式，兼容 LF、CRLF 和 CR 换行；新文本可以为空以删除片段。写回保持原编码、换行和 BOM，保留其他内容；不能修改 .env 配置或 .git 内的文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对于项目根目录的文件路径。",
                },
                "old_text": {
                    "type": "string",
                    "description": "文件中待替换的原文片段，必须唯一命中；必要时带上相邻行定位。",
                },
                "new_text": {
                    "type": "string",
                    "description": "替换后的文本片段；空字符串表示删除旧片段。",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "old_text 命中多处时是否全部替换，默认 false；唯一命中时不需要传。",
                },
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
        "handler": prepare_edit_file,
        "requires_approval": True,
        "permission_group": "file_edit",
    },
    "delete_file": {
        "description": "删除当前项目中单个已有的 UTF-8 文本文件。仅在用户要求删除文件时使用；由本地审批模式决定是否需要确认，确认时展示文件路径、字节数和行数摘要。不能删除目录、通过链接删除文件，或访问 .env 配置、.git、.zzcode；不支持通配符或递归删除。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对于项目根目录的单个文件路径。"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "handler": prepare_delete_file,
        "requires_approval": True,
        "permission_group": "file_edit",
    },
    "list_files": {
        "description": "查看当前项目的目录结构，默认从项目根目录列出一层。depth 可指定 1 到 5 层，按目录层级顺序返回相对于项目根目录的路径和类型，最多 200 项。跳过 .env 配置、.git、虚拟环境、缓存和依赖目录；目录链接不继续递归。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对于项目根目录的目录路径，默认是 .。",
                },
                "depth": {
                    "type": "integer",
                    "description": "列出的层数，默认 1（仅当前目录），最多 5。",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        "handler": list_files,
    },
    "grep": {
        "description": "递归搜索项目内的 UTF-8 文本，返回命中的文件路径、行号和内容。区分大小写，按原样文本匹配，不使用正则表达式。最多返回 50 条，每条最多 500 个字符；跳过受限配置、Git、缓存、依赖目录，以及二进制或超过 1 MiB 的文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要搜索的非空文本，例如 execute_tool。",
                },
                "path": {
                    "type": "string",
                    "description": "相对于项目根目录的搜索目录，默认是 .。",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "handler": grep,
    },
    "web_search": {
        "description": "搜索互联网获取外部资料或最新信息，返回网页标题、URL 和摘要。默认返回 5 条，最多 10 条；摘要不是完整网页，回答时引用与结论对应的来源链接。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要搜索的关键词或问题。",
                },
                "max_results": {
                    "type": "integer",
                    "description": "返回条数，默认 5，范围为 1 到 10。",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "handler": web_search,
    },
    "run_command": {
        "description": "在当前项目内运行有限时、无交互输入的命令，例如 python 的 args=[\"-m\",\"pytest\",\"tests\"]、npm 的 args=[\"test\"] 或 mvn 的 args=[\"test\"]。program 只填程序名或路径，args 单独列出参数，不要添加外层引号；cwd 是项目内目录，默认根目录。裸命令优先查找项目 .venv，再查当前 PATH；明确提供程序路径则使用指定入口。Windows 兼容 .cmd/.bat 启动器，但拒绝其中带引号、控制符及 & | < > ^ % ! 的参数。默认不解释管道、重定向或环境变量；确需 shell 语法时明确调用 cmd.exe 或 powershell.exe 并提供对应参数。每次需要用户确认，默认 30 秒，最多 300 秒；返回 stdout、stderr、退出码和超时标记，两个输出各最多回传 10000 字符。退出码 0 不是功能正确的证明；不用于交互或长期服务。本机当前用户权限运行，不是沙箱，撤销只覆盖快照包含的工作区文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "program": {"type": "string", "description": "程序名或明确路径，例如 python、npm、mvn、.\\mvnw.cmd。"},
                "args": {"type": "array", "items": {"type": "string"}, "description": "逐个列出参数，默认空列表，例如 [\"-m\",\"pytest\",\"tests\"]。"},
                "cwd": {"type": "string", "description": "相对于项目根目录的已有工作目录，默认 .。"},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 300, "description": "运行时限，单位秒，默认 30，范围 1 到 300。"},
                "revalidate": {"type": "boolean", "description": "依据待修复时申请重新验证；必须逐次审批实际命令，完全访问模式也不跳过。默认 false。"},
            },
            "required": ["program"],
            "additionalProperties": False,
        },
        "handler": prepare_run_command,
        "requires_approval": True,
    },
    "run_python": {
        "description": "使用当前 Python 解释器在项目根目录执行已有的 .py 脚本，返回 stdout、stderr、退出码和超时标记，无交互输入。默认限时 10 秒，可设置 1 到 60 秒；输出和报错分别最多回传 10000 个字符，并标记截断。退出码 0 仅表示程序正常结束，是否符合要求还需依据输出或已有测试判断。这是在本机按当前用户权限执行，入口路径检查不是代码沙箱。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对于项目根目录的 Python 脚本路径，例如 test.py。",
                },
                "timeout": {
                    "type": "integer",
                    "description": "运行时限，单位秒，默认 10，范围 1 到 60。",
                    "minimum": 1,
                    "maximum": 60,
                },
                "revalidate": {"type": "boolean", "description": "申请逐次审批的重验脚本，依据待修复时也可使用。默认 false。"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "handler": prepare_run_python,
        "requires_approval": True,
    },
    "get_current_time": {
        "description": "读取运行程序的电脑当前本地日期、时间和时区。回答今天几月几日、现在几点等涉及当前日期或时间的问题时使用。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "handler": get_current_time,
    },
}
