"""把 MCP 服务的工具接入现有注册表，模型仍使用普通 function calling。"""

import asyncio
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from referencing import Registry

from tools import DeferredTool
from startup import BackgroundJob


class MCPCallError(RuntimeError):
    """调用结果未确认，交给主循环停止当前工具批次，不自动重试。"""


def _error_kind(error):
    """保留可操作的 HTTP 状态，丢弃异常正文和响应体。"""
    pending, seen = [error], set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        status = getattr(getattr(item, "response", None), "status_code", None)
        if type(status) is int:
            return f"HTTP {status}"
        if isinstance(item, BaseExceptionGroup):
            pending.extend(item.exceptions)
        pending.extend(cause for cause in (item.__cause__, item.__context__) if cause is not None)
    return type(error).__name__


@dataclass(frozen=True)
class ServerConfig:
    name: str
    type: str
    target: str
    args: list[str]
    cwd: Path
    tools: set[str] | None
    auto_approve: set[str]
    description: str
    timeout: int
    connect_timeout: int
    headers: dict = field(repr=False)
    env: dict = field(repr=False)
    secrets: tuple[str, ...] = field(repr=False)

    def redact(self, text):
        for secret in sorted(self.secrets, key=len, reverse=True):
            text = text.replace(secret, "[凭据已隐藏]")
        return text


def _strings(value, label):
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} 必须是非空字符串组成的列表。")
    return value


def _mapping(value, label):
    if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()):
        raise ValueError(f"{label} 必须是字符串键值对象。")
    return value.copy()


def parse_server(name, data, workspace):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", name):
        raise ValueError("服务名仅支持 1～32 位字母、数字、下划线和短横线。")
    if not isinstance(data, dict):
        raise ValueError("服务配置必须是对象。")
    if type(data.get("enabled", True)) is not bool:
        raise ValueError("enabled 必须是 true 或 false。")
    if not data.get("enabled", True):
        return None
    allowed = {"enabled", "type", "url", "command", "args", "cwd", "headers", "headers_from_env",
               "bearer_token_env", "env", "env_from_env", "tools", "auto_approve", "description",
               "timeout", "connect_timeout"}
    if data.keys() - allowed:
        raise ValueError("配置包含不支持的字段，请检查字段名。")
    transport = data.get("type", "http")
    if transport not in {"http", "stdio"}:
        raise ValueError("type 仅支持 http 或 stdio。")
    target = data.get("url" if transport == "http" else "command")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("请填写服务 url 或启动 command。")
    if transport == "http":
        url = urlsplit(target)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("url 必须是 HTTP(S) 地址，凭据请通过环境变量提供。")
    args = _strings(data.get("args", []), "args")
    cwd = data.get("cwd", ".")
    if not isinstance(cwd, str):
        raise ValueError("cwd 必须是目录路径。")
    directory = (Path(workspace) / cwd).resolve()
    if not directory.is_dir():
        raise ValueError("MCP 服务的 cwd 目录不存在。")
    tools = set(_strings(data["tools"], "tools")) if "tools" in data else None
    auto = set(_strings(data.get("auto_approve", []), "auto_approve"))
    if tools is not None and not auto.issubset(tools):
        raise ValueError("auto_approve 必须属于 tools 允许清单。")
    description = data.get("description", "")
    if not isinstance(description, str):
        raise ValueError("description 必须是字符串。")
    timeout, connect_timeout = data.get("timeout", 30), data.get("connect_timeout", 10)
    if any(type(value) is not int or not 1 <= value <= 300 for value in (timeout, connect_timeout)):
        raise ValueError("timeout、connect_timeout 必须是 1～300 秒的整数。")
    headers = _mapping(data.get("headers", {}), "headers")
    if any(key.lower() == "authorization" for key in headers):
        raise ValueError("Authorization 请用 bearer_token_env 或 headers_from_env 配置。")
    environment = _mapping(data.get("env", {}), "env")
    secrets = []

    def credential(variable):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable):
            raise ValueError("凭据环境变量名无效。")
        value = os.getenv(variable, "").strip()
        if not value:
            raise ValueError(f"缺少环境变量 {variable}，请在 zzcode 的 .env 或终端环境中填写。")
        secrets.append(value)
        return value

    for key, variable in _mapping(data.get("headers_from_env", {}), "headers_from_env").items():
        headers[key] = credential(variable)
    for key, variable in _mapping(data.get("env_from_env", {}), "env_from_env").items():
        environment[key] = credential(variable)
    if "bearer_token_env" in data:
        variable = data["bearer_token_env"]
        if not isinstance(variable, str):
            raise ValueError("bearer_token_env 必须是环境变量名。")
        headers["Authorization"] = "Bearer " + credential(variable)
    return ServerConfig(name, transport, target, args, directory, tools, auto, description, timeout,
                        connect_timeout, headers, environment, tuple(secrets))


class MCPConnection:
    """连接的创建和关闭留在同一异步任务中，主线程短间隔等待以响应 Ctrl+C。"""

    def __init__(self, config):
        self.config = config
        self._ready = concurrent.futures.Future()
        self._loop = self._owner = self._client = self._stop = None
        self._http_error_status = None
        self._pending = set()
        self._closing = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"mcp-{config.name}", daemon=True)

    def _run(self):
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            if not self._ready.done():
                self._ready.set_exception(error)
        finally:
            if not self._ready.done():
                self._ready.set_exception(MCPCallError("MCP 连接已关闭。"))

    async def _serve(self):
        if self._closing.is_set():
            return
        from mcp import Client, StdioServerParameters, stdio_client

        if self._closing.is_set():
            return
        self._owner = asyncio.current_task()
        self._stop = asyncio.Event()
        # 最后发布 loop，close() 看见它时，owner/stop 必须已经可用。
        self._loop = asyncio.get_running_loop()
        if self._closing.is_set():
            return
        async with AsyncExitStack() as stack:
            try:
                async with asyncio.timeout(self.config.connect_timeout):
                    if self.config.type == "http":
                        import httpx2
                        from mcp.client.streamable_http import streamable_http_client

                        async def remember_status(response):
                            if response.request.method == "POST":
                                self._http_error_status = response.status_code if response.status_code >= 400 else None

                        http = await stack.enter_async_context(httpx2.AsyncClient(
                            headers=self.config.headers,
                            event_hooks={"response": [remember_status]},
                            # 由单次调用的总时限先取消；避免 HTTP 读超时同时触发并拆掉整个连接。
                            timeout=httpx2.Timeout(self.config.connect_timeout, read=max(300, self.config.timeout + 5)),
                        ))
                        transport = streamable_http_client(self.config.target, http_client=http)
                    else:
                        error_log = stack.enter_context(open(os.devnull, "w"))
                        params = StdioServerParameters(command=self.config.target, args=self.config.args,
                                                       cwd=str(self.config.cwd), env=self.config.env)
                        transport = stdio_client(params, errlog=error_log)
                    self._client = await stack.enter_async_context(Client(transport))
                    tools, cursors, cursor = [], set(), None
                    while True:
                        page = await self._client.list_tools(cursor=cursor)
                        tools.extend(page.tools)
                        if len(tools) > 512:
                            raise ValueError("单个服务工具超过 512 个，请缩小服务端工具集。")
                        cursor = page.next_cursor
                        if cursor is None:
                            break
                        if cursor in cursors:
                            raise ValueError("服务重复返回分页游标。")
                        cursors.add(cursor)
                    self._ready.set_result(tools)
                await self._stop.wait()
            finally:
                pending = list(self._pending)
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

    def start(self):
        if self._closing.is_set():
            raise MCPCallError("MCP 连接已关闭。")
        self._thread.start()
        try:
            return self._wait(self._ready)
        except BaseException:
            self.close()
            raise

    @property
    def connected(self):
        """只查看本地连接状态，不为状态栏发起网络探测。"""
        return (not self._closing.is_set() and self._thread.is_alive() and self._client is not None
                and self._ready.done() and not self._ready.cancelled()
                and self._ready.exception() is None
                and self._stop is not None and not self._stop.is_set())

    def _wait(self, future):
        while True:
            try:
                return future.result(timeout=0.1)
            except concurrent.futures.TimeoutError:
                if future.done():
                    raise MCPCallError(f"MCP {self.config.name} 请求超时，结果未确认。") from None
                if not self._thread.is_alive():
                    raise MCPCallError(f"MCP {self.config.name} 连接已关闭。") from None
            except Exception as error:
                # 不将 SDK 异常正文中的请求头、凭据或服务响应直接打印/送给模型。
                kind = f"HTTP {self._http_error_status}" if self._http_error_status else _error_kind(error)
                raise MCPCallError(f"MCP {self.config.name} 请求失败（{kind}），结果未确认。") from None

    def call(self, name, arguments):
        if not self._thread.is_alive() or self._client is None or self._stop.is_set():
            raise MCPCallError(f"MCP {self.config.name} 连接不可用，请重启后重试。")
        finished = threading.Event()

        async def invoke():
            task = asyncio.current_task()
            self._pending.add(task)
            self._http_error_status = None
            try:
                async with asyncio.timeout(self.config.timeout):
                    return await self._client.call_tool(name, arguments)
            finally:
                self._pending.discard(task)
                finished.set()

        operation = invoke()
        try:
            future = asyncio.run_coroutine_threadsafe(operation, self._loop)
        except RuntimeError:
            operation.close()
            raise MCPCallError(f"MCP {self.config.name} 连接已关闭。") from None
        try:
            return self._wait(future)
        except KeyboardInterrupt:
            future.cancel()
            if not finished.wait(timeout=1):
                self.close()
            raise

    def close(self):
        self._closing.set()
        if self._loop is not None and not self._loop.is_closed():
            try:
                if self._ready.done():
                    self._loop.call_soon_threadsafe(self._stop.set)
                else:
                    self._loop.call_soon_threadsafe(self._owner.cancel)
            except RuntimeError:
                pass
        if self._thread.ident is not None:
            deadline = time.monotonic() + 8
            while self._thread.is_alive() and time.monotonic() < deadline:
                self._thread.join(timeout=0.1)
        return not self._thread.is_alive()


def tool_alias(server, name):
    original = f"{server}__{name}"
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", original)
    if normalized != original or len(normalized) > 64:
        normalized = normalized[:53] + "__" + hashlib.sha256(original.encode()).hexdigest()[:8]
    return normalized


def tool_result(config, name, result):
    parts = []
    for block in result.content:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "resource" and getattr(block.resource, "text", None) is not None:
            parts.append(f"资源 {block.resource.uri}\n{block.resource.text}")
        elif block.type == "resource_link":
            parts.append(f"资源链接：{block.uri}")
        else:
            parts.append(f"[{block.type} 内容未展示；当前工具结果仅支持文本和资源链接。]")
    if result.structured_content is not None:
        parts.append("结构化结果：\n" + json.dumps(result.structured_content, ensure_ascii=False))
    text = config.redact("\n".join(parts))
    return {"server": config.name, "tool": name, "is_error": result.is_error,
            "content": text[:20000], "truncated": len(text) > 20000}


def make_tool(connection, tool):
    config = connection.config
    schema = json.loads(config.redact(json.dumps(tool.input_schema, ensure_ascii=False)))
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("MCP 工具参数 Schema 必须是 object。")
    try:
        validator_type = validator_for(schema)
        validator_type.check_schema(schema)
        # 只解析内嵌 Schema 引用，验证参数时不另行联网获取远程 $ref。
        validator = validator_type(schema, registry=Registry())
    except SchemaError:
        raise ValueError("服务返回了无效的工具参数 Schema。") from None

    def validate(arguments):
        try:
            error = next(validator.iter_errors(arguments), None)
        except Exception:
            raise ValueError("MCP 参数 Schema 无法解析，不支持外部 Schema 引用。") from None
        if error is not None:
            location = "/".join(str(part) for part in error.absolute_path) or "参数根节点"
            raise ValueError(f"MCP 工具参数不符合 Schema：{location}（{error.validator}）。")

    def run(**arguments):
        return tool_result(config, tool.name, connection.call(tool.name, arguments))

    def prepare(**arguments):
        preview = config.redact(json.dumps(arguments, ensure_ascii=False, indent=2))
        return DeferredTool("调用 MCP 工具", f"服务：{config.name}\n工具：{tool.name}", preview, "json",
                            action=lambda: run(**arguments))

    needs_approval = tool.name not in config.auto_approve
    description = config.redact(f"来自 MCP 服务 {config.name}。{config.description}\n{tool.description or ''}")
    return {"description": description, "parameters": schema, "validate_arguments": validate,
            "handler": prepare if needs_approval else run, "requires_approval": needs_approval,
            "mcp_server": config.name}


class MCPManager:
    """读取 zzcode 安装目录的配置，避免自动运行目标仓库自带的 MCP 配置。"""

    def __init__(self, config_path, workspace):
        self.config_path = Path(config_path)
        self.workspace = Path(workspace)
        self.connections = []
        self.notices = []
        self._servers = []
        self._config_error = ""
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._connecting = set()
        self._registry = {}
        self._revision = 0
        self._loading = False
        self._startup = None

    @property
    def revision(self):
        with self._lock:
            return self._revision

    def registry(self):
        return self.registry_state()[1]

    def registry_state(self):
        with self._lock:
            return self._revision, dict(self._registry)

    def start_background(self):
        with self._lock:
            if self._startup is not None or self._closed.is_set():
                return
            self._loading = True

            def load(_stop):
                try:
                    return self.start()
                except Exception as error:
                    with self._lock:
                        self._config_error = f"MCP 初始化失败（{type(error).__name__}）。"
                        self._loading = False
                    return {}

            self._startup = BackgroundJob(load, name="zzcode-mcp-startup")
            self._startup.start()

    def start(self):
        try:
            return self._start()
        finally:
            with self._lock:
                self._loading = False

    def _start(self):
        if self._closed.is_set():
            return {}
        if not self.config_path.is_file():
            return {}
        try:
            if self.config_path.stat().st_size > 256000:
                raise ValueError("配置文件过大。")
            data = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or not isinstance(data.get("servers"), dict):
                raise ValueError("配置应包含 servers 对象。")
        except (OSError, UnicodeError, ValueError):
            with self._lock:
                self._config_error = "MCP 配置无法读取，请检查 zzcode 安装目录的 mcp.json。"
                self.notices.append(self._config_error)
            return {}
        registry = {}
        try:
            for name, settings in data["servers"].items():
                connection = None
                state = {"name": name, "status": "连接中", "tools": 0, "detail": "", "connection": None}
                with self._lock:
                    if self._closed.is_set():
                        break
                    self._servers.append(state)
                try:
                    config = parse_server(name, settings, self.workspace)
                    if config is None:
                        with self._lock:
                            state.update(status="已禁用", detail="配置中 enabled=false。")
                        continue
                    connection = MCPConnection(config)
                    with self._lock:
                        if self._closed.is_set():
                            break
                        self._connecting.add(connection)
                    discovered = connection.start()
                    available = {tool.name for tool in discovered}
                    if len(available) != len(discovered):
                        raise ValueError("服务返回了重名工具。")
                    if config.tools is not None and config.tools - available:
                        raise ValueError("配置允许的部分工具未在服务中发现，请核对工具名。")
                    additions = {tool_alias(name, tool.name): make_tool(connection, tool) for tool in discovered
                                 if config.tools is None or tool.name in config.tools}
                    if len(additions) != sum(config.tools is None or tool.name in config.tools for tool in discovered):
                        raise ValueError("工具别名发生冲突。")
                    if registry.keys() & additions.keys():
                        raise ValueError("工具别名发生冲突。")
                    with self._lock:
                        if self._closed.is_set():
                            connection.close()
                            break
                        registry.update(additions)
                        self._registry.update(additions)
                        self._revision += 1
                        self.connections.append(connection)
                        state.update(status="已连接", tools=len(additions), connection=connection)
                        self.notices.append(f"MCP {name}：已连接，加载 {len(additions)} 个工具。")
                except (ValueError, MCPCallError) as error:
                    if connection is not None:
                        connection.close()
                    with self._lock:
                        state.update(status="未连接", detail=str(error))
                        self.notices.append(f"MCP {name} 未启用：{error}")
                finally:
                    with self._lock:
                        self._connecting.discard(connection)
        except BaseException:
            self.close()
            raise
        return registry

    def snapshot(self):
        """供本地命令和底栏共用；分别保留已加载工具总数和当前可用数。"""
        servers = []
        with self._lock:
            states = [dict(state) for state in self._servers]
            error, loading = self._config_error, self._loading
        for state in states:
            row = {key: value for key, value in state.items() if key != "connection"}
            if row["status"] == "已连接" and not state["connection"].connected:
                row.update(status="未连接", detail="连接已关闭，请重启后重试。")
            servers.append(row)
        connected = sum(row["status"] == "已连接" for row in servers)
        return {"connected": connected, "disconnected": len(servers) - connected,
                "tools": sum(row["tools"] for row in servers), "servers": servers,
                "available_tools": sum(row["tools"] for row in servers if row["status"] == "已连接"),
                "error": error, "loading": loading}

    def close(self):
        self._closed.set()
        failures = []
        with self._lock:
            connections = list(dict.fromkeys([*reversed(self.connections), *self._connecting]))
            self.connections.clear()
            self._registry.clear()
        for connection in connections:
            if not connection.close():
                failures.append(f"MCP {connection.config.name} 连接未及时关闭。")
        if self._startup is not None and not self._startup.close():
            failures.append("MCP 后台初始化未及时结束。")
        return failures
