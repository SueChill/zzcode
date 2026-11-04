"""单次任务的空转检测，不额外请求模型判断是否有进展。"""
import hashlib
import json
import os
from collections import deque
from pathlib import Path

from snapshots import IGNORED_DIRECTORIES


class LoopStopped(RuntimeError):
    """停止自动执行，交给主循环补齐未执行的工具结果。"""


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).digest()


def _parse(value):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return value


def _call_key(call):
    # JSON 字段顺序、空格和本次调用编号不影响是否为同一操作。
    return _digest(_canonical([call["name"], _parse(call["arguments"])]))


def _failure_kind(result):
    if isinstance(result, str) and result.startswith("工具执行失败："):
        return _digest(result)
    if isinstance(result, dict):
        if result.get("timed_out") is True:
            return "timeout"
        if "exit_code" in result and result["exit_code"] not in (None, 0):
            return f"exit_code:{result['exit_code']}"
        if result.get("is_error") is True:
            return "mcp_error"
        if result.get("status") == "error" and isinstance(result.get("error_code"), str):
            return result["error_code"]
    return None


class LoopGuard:
    def __init__(self, repeat_limit=3, failure_limit=3, max_cycle=4):
        if repeat_limit < 2 or failure_limit < 2 or max_cycle < 1:
            raise ValueError("循环检测阈值不合法。")
        self.repeat_limit = repeat_limit
        self.failure_limit = failure_limit
        self.max_cycle = max_cycle
        self._history = deque(maxlen=repeat_limit * max_cycle)
        self._failures = {}
        self._scoped_failures = {}
        self._denied = set()
        self._version = None

    def check(self, call):
        if _call_key(call) in self._denied:
            raise LoopStopped(f"循环保护：{call['name']} 已被拒绝，不再重复申请；本次任务已停止。")

    def record(self, call, result, workspace_version=None):
        if self._version != workspace_version:
            self._history.clear()
            self._failures.clear()
            self._version = workspace_version
        key = _call_key(call)
        data = _parse(result)
        # 计划依据写错时参数往往每次都变；按阶段归并，不依赖完整参数相同。
        scope = data.get("failure_scope") if isinstance(data, dict) else None
        if (isinstance(scope, dict) and scope.get("kind") == "plan_step"
                and data.get("status") == "error" and type(scope.get("step_id")) is int):
            scope_key = _digest(_canonical(scope))
            count = self._scoped_failures.get(scope_key, 0) + 1
            self._scoped_failures[scope_key] = count
            if count >= self.failure_limit:
                raise LoopStopped(f"循环保护：计划阶段 {scope['step_id']} 连续校验失败 {count} 次；已停止自动重试，请查询计划和可用依据后再继续。")
        elif (call["name"] == "update_plan_step" and isinstance(data, dict)
              and "steps" in data and not data.get("repair")):
            self._scoped_failures.clear()
        if isinstance(data, dict) and data.get("status") in ("denied", "approval_required"):
            self._denied.add(key)
        failure = _failure_kind(data)
        if failure is None:
            self._failures.pop(key, None)
        else:
            previous, count = self._failures.get(key, (None, 0))
            count = count + 1 if previous == failure else 1
            self._failures[key] = (failure, count)
            if count >= self.failure_limit:
                raise LoopStopped(
                    f"循环保护：{call['name']} 在工作区无变化时连续失败 {count} 次；本次任务已停止。"
                )
        self._history.append((key, _digest(_canonical(data))))
        history = list(self._history)
        for length in range(1, min(self.max_cycle, len(history) // self.repeat_limit) + 1):
            pattern = history[-length:]
            if history[-length * self.repeat_limit:] == pattern * self.repeat_limit:
                detail = "同一调用及结果" if length == 1 else f"长度为 {length} 的工具调用及结果序列"
                raise LoopStopped(
                    f"循环保护：工作区无变化，{detail}重复 {self.repeat_limit} 次；本次任务已停止。"
                )


class WorkspaceVersion:
    """按内容检测变化；缓存未变化文件的摘要，不把文件正文存进检测历史。"""

    def __init__(self, root, resolve_path):
        self.root = Path(root).resolve()
        self.resolve_path = resolve_path
        self._cache = {}

    def _allowed(self, path):
        try:
            return not path.is_symlink() and self.resolve_path(path.relative_to(self.root).as_posix()) == path
        except (ValueError, RuntimeError):
            return False

    def current(self):
        entries, cache = [], {}

        def fail(error):
            raise error

        try:
            if not self.root.is_dir():
                raise OSError("工作区不存在")
            for directory, names, files in os.walk(self.root, onerror=fail):
                current = Path(directory)
                names[:] = [name for name in names
                            if name.lower() not in IGNORED_DIRECTORIES and self._allowed(current / name)]
                entries.extend(("directory", (current / name).relative_to(self.root).as_posix()) for name in names)
                for name in files:
                    path = current / name
                    if not self._allowed(path):
                        continue
                    try:
                        stat = path.stat()
                        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                        previous = self._cache.get(path)
                        if previous is not None and previous[0] == identity:
                            digest = previous[1]
                        else:
                            with path.open("rb") as source:
                                digest = hashlib.file_digest(source, "sha256").hexdigest()
                    except FileNotFoundError:
                        # 编译器等进程可能在扫描过程中删除临时文件，下一次重新扫描。
                        continue
                    cache[path] = (identity, digest)
                    entries.append(("file", path.relative_to(self.root).as_posix(), digest))
        except OSError:
            raise LoopStopped("循环保护：无法检查工作区变化，本次任务已停止；已完成的操作保留。") from None
        self._cache = cache
        return _digest(_canonical(sorted(entries)))
