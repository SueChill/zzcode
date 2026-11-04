"""进程内的本地追踪：有界队列、JSONL、独立请求统计；不改变业务异常。"""

import hashlib
import json
import math
import os
import queue
import re
import threading
import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from tool_result import ToolResult


_active_run = ContextVar("zzcode_trace_run", default=None)
_active_span = ContextVar("zzcode_trace_span", default=None)
_secret = re.compile(r"(?i)(bearer\s+|(?:api[_-]?key|token|password|secret)[\"']?\s*[=:]\s*[\"']?)[^\s,;\"']+")


def fingerprint(value):
    try:
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
    except Exception:
        return None


def _clean(value, depth=0):
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _secret.sub(r"\1[隐藏]", value[:2000])
    if depth < 5 and isinstance(value, dict):
        return {str(key)[:100]: "[隐藏]" if str(key).lower() in {"api_key", "authorization", "password", "secret", "access_token"}
                else _clean(item, depth + 1) for key, item in list(value.items())[:64]}
    if depth < 5 and isinstance(value, (list, tuple)):
        return [_clean(item, depth + 1) for item in value[:64]]
    return f"<{type(value).__name__}>"


def current_run():
    return _active_run.get()


def current_span():
    return _active_span.get()


class NullSpan:
    def __init__(self, attrs=None):
        self.attrs, self.status, self.id = dict(attrs or {}), "ok", None
        self.started_ns = time.perf_counter_ns()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        return False

    def end(self, status=None):
        pass

    def start(self):
        return self

    def event(self, name, attrs=None, critical=False):
        pass


def span(name, attrs=None, *, parent=None):
    run = current_run()
    return Span(run, name, attrs or {}, parent=parent) if run is not None and run.recorder.enabled else NullSpan(attrs)


def set_run_status(status, **attrs):
    run = current_run()
    if run is not None:
        run.status = status
        run.root.attrs.update(attrs)


class Span:
    def __init__(self, run, name, attrs, *, parent=None):
        self.run, self.name, self.attrs = run, name, dict(attrs)
        self.id = uuid4().hex[:16]
        active = current_span() if parent is None else parent
        self.parent = active.id if active is not None else None
        self.status = "ok"
        self.started_ns = None
        self.closed = False
        self._token = None

    def start(self):
        if self.started_ns is None:
            self.started_ns = self.run.started_ns if self.parent is None else time.perf_counter_ns()
            self.run.open_spans[self.id] = self
            self.run.emit(self, "start", critical=True)
        return self

    def __enter__(self):
        self.start()
        self._token = _active_span.set(self)
        return self

    def end(self, status=None):
        if self.closed:
            return
        self.start()
        if status is not None:
            self.status = status
        # 包含尚未进入执行入口的工具，避免 guard/取消后丢失被提出的调用。
        for child in list(self.run.open_spans.values()):
            if child is not self and child.parent == self.id:
                child.end("not_run" if child.name.startswith("tool:") and not child.attrs.get("executed") else "unknown")
        self.closed = True
        self.run.emit(self, "end", critical=True)
        self.run.open_spans.pop(self.id, None)

    def __exit__(self, error_type, error, traceback):
        try:
            if isinstance(error, (KeyboardInterrupt, GeneratorExit)) and self.name.startswith("tool:") and not self.attrs.get("executed"):
                self.status = "cancelled"
            if error is not None and self.status == "ok":
                self.status = "cancelled" if isinstance(error, (KeyboardInterrupt, GeneratorExit)) else "error"
                self.attrs["error_type"] = type(error).__name__
            self.end()
        finally:
            if self._token is not None:
                _active_span.reset(self._token)
        return False

    def event(self, name, attrs=None, critical=False):
        self.run.emit(self, "event", extra={"event_name": name, **(attrs or {})}, critical=critical)


class Run:
    def __init__(self, recorder, kind, attrs):
        self.recorder = recorder
        self.id = uuid4().hex
        self.started_at = datetime.now(timezone.utc)
        self.started_ns = time.perf_counter_ns()
        stamp = self.started_at.strftime("%Y%m%dT%H%M%S%fZ")
        self.path = recorder.root / recorder.session / f"{stamp}-{self.id}.jsonl"
        self.seq = 0
        self.open_spans = {}
        self.dropped = Counter()
        self.reasons = set()
        self.root = Span(self, "turn" if kind == "task" else "command", {
            "kind": kind, "process_id": recorder.process_id, "pid": os.getpid(),
            **attrs,
        }, parent=None)
        self.root.parent = None
        self.root.status = "completed"

    @property
    def status(self):
        return self.root.status

    @status.setter
    def status(self, value):
        self.root.status = value

    def emit(self, target, event, extra=None, critical=False):
        now = time.perf_counter_ns()
        duration = (now - target.started_ns) / 1e6 if event == "end" else None
        try:
            if event == "end" and target.name == "model_request" and "client_consumer_ms" in target.attrs:
                # span 保留真实起止时间；请求净耗时另存，避免把 yield 后的客户端处理算给模型。
                target.attrs["active_duration_ms"] = max(0.0, duration - target.attrs["client_consumer_ms"])
            attributes = _clean({**target.attrs, **(extra or {})})
        except Exception:
            attributes = {"metadata_error": True}
            self.reasons.add("metadata_error")
            self.recorder._warn()
        record = {
            "v": 1, "seq": None, "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run": self.id, "session": self.recorder.session, "span": target.id,
            "parent": target.parent, "name": target.name, "event": event,
            "status": target.status if event == "end" else None,
            "duration_ms": round(duration, 6) if duration is not None else None,
            "offset_ms": round(((target.started_ns if event == "start" else now) - self.started_ns) / 1e6, 6),
            "attrs": attributes,
        }
        try:
            self.recorder.emit(self, record, critical)
        except Exception:
            self.reasons.add("recorder_error")
            self.recorder._warn()

    def describe(self, value):
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        data = {"sha256": fingerprint(content), "characters": len(content)}
        limit = self.recorder.preview_chars
        if limit:
            data.update(preview=_secret.sub(r"\1[隐藏]", content[:limit]), truncated=len(content) > limit)
        return data


class TraceRecorder:
    def __init__(self, root, session, *, enabled=True, preview_chars=0, max_mb=32, capacity=2048):
        self.root, self.session = Path(root), session
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", session):
            raise ValueError("追踪会话编号无效。")
        self.enabled = enabled
        self.preview_chars = max(0, min(2000, preview_chars))
        self.max_bytes = max(1, max_mb) * 1024 * 1024
        self.process_id = uuid4().hex
        self.capacity = max(1, capacity)
        self.queue = queue.Queue(maxsize=self.capacity + 128)
        self._lock = threading.Lock()
        self._thread = None
        self._closing = threading.Event()
        self._failed = False
        self._warning = None
        self._warned = False
        self._soft_limit = set()

    @classmethod
    def from_environment(cls, root, session, *, disabled=False):
        def integer(name, default):
            try:
                return max(0, int(os.getenv(name, str(default))))
            except ValueError:
                return default
        enabled = not disabled and os.getenv("TRACE_ENABLED", "true").strip().lower() not in {"0", "false", "off", "no"}
        return cls(root, session, enabled=enabled, preview_chars=integer("TRACE_PREVIEW_CHARS", 0),
                   max_mb=integer("TRACE_MAX_MB", 32))

    def _warn(self):
        if not self._warned:
            self._warning = "追踪记录不完整；任务继续执行。可用 /trace 检查记录状态。"
            self._warned = True

    def pop_warning(self):
        message, self._warning = self._warning, None
        return message

    def emit(self, run, record, critical=False):
        if not self.enabled:
            return
        try:
            with self._lock:
                run.seq += 1
                record["seq"] = run.seq
                if self._thread is None:
                    self._thread = threading.Thread(target=self._writer, daemon=True, name="zzcode-trace")
                    self._thread.start()
                reason = "write_failed" if self._failed else "shutdown" if self._closing.is_set() else None
                if reason is None and self.queue.qsize() >= self.capacity // 2:
                    stripped = self._without_previews(record["attrs"])
                    if stripped != record["attrs"]:
                        run.dropped["preview_fields"] += 1
                        run.reasons.add("preview_fields_dropped")
                        self._warn()
                    record["attrs"] = stripped
                if reason is None and not critical and (self.queue.qsize() >= self.capacity or run.id in self._soft_limit):
                    reason = "optional_events_dropped"
                if reason is None:
                    try:
                        self.queue.put_nowait((run, record))
                        return
                    except queue.Full:
                        reason = "queue_full"
                run.dropped[reason] += 1
                run.reasons.add(reason)
                self._warn()
        except Exception:
            # 只隔离追踪自己的处理故障，业务异常由 Span.__exit__ 原样传播。
            self._failed = True
            run.reasons.add("recorder_error")
            self._warn()

    @staticmethod
    def _without_previews(value):
        if isinstance(value, dict):
            return {key: TraceRecorder._without_previews(item) for key, item in value.items() if key != "preview"}
        if isinstance(value, list):
            return [TraceRecorder._without_previews(item) for item in value]
        return value

    def _writer(self):
        handles, sizes = {}, Counter()
        try:
            while not self._closing.is_set() or not self.queue.empty():
                try:
                    item = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    if isinstance(item, threading.Event):
                        item.set()
                        continue
                    run, record = item
                    if self._failed:
                        run.reasons.add("write_failed")
                        continue
                    try:
                        if run.id not in handles:
                            run.path.parent.mkdir(parents=True, exist_ok=True)
                            handles[run.id] = run.path.open("x", encoding="utf-8", newline="\n")
                        if run.id in self._soft_limit:
                            record["attrs"] = self._without_previews(record["attrs"])
                        line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
                        handles[run.id].write(line)
                        handles[run.id].flush()
                        sizes[run.id] += len(line.encode("utf-8"))
                        if sizes[run.id] > self.max_bytes:
                            self._soft_limit.add(run.id)
                            run.reasons.add("size_soft_limit")
                            self._warn()
                        if record["parent"] is None and record["event"] == "end":
                            if run.reasons:
                                tail = {**record, "seq": record["seq"] + 1, "event": "event", "status": None,
                                        "duration_ms": None, "attrs": {"event_name": "run.incomplete",
                                        "dropped": dict(run.dropped), "reasons": sorted(run.reasons)}}
                                handles[run.id].write(json.dumps(tail, ensure_ascii=False) + "\n")
                                handles[run.id].flush()
                            handles.pop(run.id).close()
                            sizes.pop(run.id, None)
                    except Exception:
                        self._failed = True
                        run.reasons.add("write_failed")
                        self._warn()
                finally:
                    self.queue.task_done()
        finally:
            for handle in handles.values():
                try:
                    handle.close()
                except Exception:
                    self._warn()

    @contextmanager
    def run(self, kind="task", attrs=None):
        if not self.enabled:
            yield None
            return
        run = Run(self, kind, attrs or {})
        token = _active_run.set(run)
        try:
            with run.root:
                try:
                    yield run
                except BaseException as error:
                    if run.status == "completed":
                        run.status = "cancelled" if isinstance(error, KeyboardInterrupt) else "internal_error"
                        run.root.attrs.update(error_type=type(error).__name__, stop_detail=(
                            "用户取消了当前任务。" if isinstance(error, KeyboardInterrupt) else "任务因程序异常停止。"))
                    raise
                finally:
                    if run.reasons:
                        run.root.event("run.incomplete", {"dropped": dict(run.dropped), "reasons": sorted(run.reasons)}, critical=True)
        finally:
            _active_run.reset(token)

    @contextmanager
    def operation(self, kind="task", attrs=None):
        active = current_run()
        if active is not None and active.recorder is self:
            yield active
        else:
            with self.run(kind, attrs) as run:
                yield run

    def flush(self, timeout=1.0):
        if self._thread is None:
            return True
        barrier = threading.Event()
        try:
            self.queue.put_nowait(barrier)
        except queue.Full:
            self._warn()
            return False
        complete = barrier.wait(timeout)
        if not complete:
            self._warn()
        return complete

    def close(self, timeout=2.0):
        self._closing.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                self._warn()


def classify_tool(result):
    view = ToolResult.parse(result)
    if view.status == "interrupted":
        return "unknown"
    if view.status in {"error", "timeout", "not_run", "denied", "approval_required",
                       "cancelled", "unknown"}:
        return view.status
    data = view.data
    if data.get("timed_out"):
        return "timeout"
    if data.get("is_error") or data.get("result_confirmed") is False:
        return "error" if data.get("is_error") else "unknown"
    if data.get("exit_code") not in (None, 0):
        return "error"
    return "ok"


def usage_fields(usage):
    def read(item, key):
        return item.get(key) if isinstance(item, dict) else getattr(item, key, None)
    def number(value):
        return value if type(value) is int and value >= 0 else None
    details = read(usage, "completion_tokens_details")
    prompt_details = read(usage, "prompt_tokens_details")
    return {
        "input_tokens": number(read(usage, "prompt_tokens")),
        "output_tokens": number(read(usage, "completion_tokens")),
        "cached_tokens": number(read(usage, "prompt_cache_hit_tokens")) if number(read(usage, "prompt_cache_hit_tokens")) is not None else number(read(prompt_details, "cached_tokens")),
        "cache_miss_tokens": number(read(usage, "prompt_cache_miss_tokens")),
        "reasoning_tokens": number(read(details, "reasoning_tokens")),
    }


class TraceReader:
    def __init__(self, root, session):
        self.directory = Path(root) / session

    def list(self):
        if not self.directory.is_dir():
            return []
        return sorted((path for path in self.directory.glob("*.jsonl") if not path.is_symlink()), reverse=True)

    def select(self, prefix=None):
        paths = self.list()
        if prefix:
            if not re.fullmatch(r"[a-fA-F0-9]{4,32}", prefix):
                raise ValueError("请输入至少 4 位的追踪编号前缀。")
            paths = [path for path in paths if path.stem.rsplit("-", 1)[-1].startswith(prefix.lower())]
        if not paths:
            raise ValueError("没有找到追踪记录。")
        if prefix and len(paths) > 1:
            raise ValueError("编号前缀匹配多条记录，请输入更长的前缀。")
        if not prefix:
            for path in paths:
                try:
                    with path.open("r", encoding="utf-8") as source:
                        first = json.loads(source.readline())
                    if isinstance(first, dict) and isinstance(first.get("attrs"), dict) and first["attrs"].get("kind") == "task":
                        return path
                except (OSError, ValueError, TypeError):
                    continue
        return paths[0]

    def read(self, path):
        path = Path(path)
        if path.is_symlink() or path.parent.resolve() != self.directory.resolve():
            raise ValueError("追踪文件路径无效。")
        events, issues, expected = [], set(), 1
        with path.open("rb") as handle:
            for line in handle:
                try:
                    if not line.endswith(b"\n"):
                        issues.add("尾行未完整写入")
                    record = json.loads(line)
                    if (not isinstance(record, dict) or record.get("v") != 1
                            or type(record.get("seq")) is not int or record["seq"] < 1
                            or record.get("event") not in {"start", "end", "event", "error"}
                            or not isinstance(record.get("attrs"), dict)):
                        raise ValueError()
                    for key in ("offset_ms", "duration_ms"):
                        value = record.get(key)
                        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                            raise ValueError()
                    if (not isinstance(record.get("span"), str) or not isinstance(record.get("name"), str)
                            or (record.get("parent") is not None and not isinstance(record["parent"], str))
                            or (record.get("status") is not None and not isinstance(record["status"], str))):
                        raise ValueError()
                    attrs = record["attrs"]
                    for key in ("exec_ms", "active_duration_ms", "client_consumer_ms", "input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens", "first_event_ms", "first_text_ms"):
                        value = attrs.get(key)
                        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                            attrs[key] = None
                            issues.add("部分属性格式无效")
                    for key in ("bucket", "arguments_sha256", "tool"):
                        if attrs.get(key) is not None and not isinstance(attrs[key], str):
                            attrs.pop(key)
                            issues.add("部分属性格式无效")
                    if "result" in attrs and not isinstance(attrs["result"], dict):
                        attrs.pop("result")
                        issues.add("部分属性格式无效")
                    if record["seq"] != expected:
                        issues.add("事件序号缺失或重复")
                    expected = record["seq"] + 1
                    events.append(record)
                except (ValueError, TypeError, UnicodeError):
                    issues.add("包含损坏或不支持的事件")
        nodes = {}
        for event in events:
            identifier = event.get("span")
            if not isinstance(identifier, str):
                issues.add("缺少 span 编号")
                continue
            node = nodes.setdefault(identifier, {"id": identifier, "name": event.get("name", "unknown"),
                "parent": event.get("parent"), "start_ms": None, "end_ms": None,
                "duration_ms": None, "status": "unknown", "attrs": {}})
            if event["event"] == "start":
                node["attrs"].update(event["attrs"])
                node["start_ms"] = event.get("offset_ms")
            elif event["event"] == "end":
                node["attrs"].update(event["attrs"])
                node.update(end_ms=event.get("offset_ms"), duration_ms=event.get("duration_ms"), status=event.get("status") or "unknown")
            if event["attrs"].get("event_name") == "run.incomplete":
                issues.add("采集器报告事件丢失")
        # 结果落库成功后才发回执；取消/异常补齐的结果也与原工具编号关联。
        results = {event["attrs"].get("call_id"): event["attrs"] for event in events
                   if event["event"] == "event" and event["attrs"].get("event_name") == "tool.result_recorded"
                   and isinstance(event["attrs"].get("call_id"), str)}
        for node in nodes.values():
            call_id = node["attrs"].get("call_id")
            result = results.get(call_id) if isinstance(call_id, str) else None
            if result is not None and node["name"].startswith("tool:"):
                node["attrs"].update({key: result.get(key) for key in
                    ("history_status", "result_confirmed", "history_recorded")})
        roots = [node for node in nodes.values() if node["parent"] is None]
        root = roots[0] if roots else None
        if root is None:
            issues.add("没有任务起点")
        for node in nodes.values():
            if node["start_ms"] is None or node["end_ms"] is None:
                issues.add("存在未闭合动作，结束原因未知")
        if root is not None and root["end_ms"] is None:
            root["status"] = "interrupted"
            root["attrs"]["inferred"] = True
        return {"path": path, "run": events[0].get("run", path.stem.rsplit("-", 1)[-1]) if events else path.stem.rsplit("-", 1)[-1],
                "session": events[0].get("session") if events else None,
                "started_at": events[0].get("ts") if events else None,
                "nodes": list(nodes.values()), "root": root, "issues": sorted(issues)}
