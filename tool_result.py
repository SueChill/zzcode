"""工具结果的统一视图，避免调用链各处重复解析 JSON 字符串。"""

import json
from dataclasses import dataclass, field


FAILED_STATUSES = frozenset({
    "error", "timeout", "denied", "approval_required", "cancelled", "not_run",
})


@dataclass(frozen=True)
class ToolResult:
    """解析后的工具结果；原始字符串仍原样保存给模型。"""

    status: str = "ok"
    data: dict = field(default_factory=dict)
    message: str = ""
    error_code: str | None = None
    executed: bool = True

    @property
    def failed(self):
        return self.status in FAILED_STATUSES

    @classmethod
    def parse(cls, raw):
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = raw
        elif isinstance(raw, str):
            if raw.startswith("工具执行失败："):
                return cls(status="error", message=raw, error_code="tool_error")
            try:
                data = json.loads(raw)
            except (ValueError, TypeError):
                data = None
        else:
            data = None
        if not isinstance(data, dict):
            return cls(message="" if raw is None else str(raw))
        status = data.get("status")
        if not isinstance(status, str):
            status = "ok"
        error_code = data.get("error_code")
        message = data.get("message")
        return cls(
            status=status,
            data=data,
            message=message if isinstance(message, str) else "",
            error_code=error_code if isinstance(error_code, str) else None,
            executed=bool(data.get("executed", True)),
        )

    def to_json(self):
        payload = {"status": self.status}
        if self.error_code:
            payload["error_code"] = self.error_code
        if self.message:
            payload["message"] = self.message
        payload["executed"] = self.executed
        payload.update({key: value for key, value in self.data.items()
                        if key not in {"status", "error_code", "message", "executed"}})
        return json.dumps(payload, ensure_ascii=False)


def render_result(value):
    """工具返回值统一成字符串；字典仍按 JSON 交给模型。"""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)
