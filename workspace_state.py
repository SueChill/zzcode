from copy import deepcopy


class WorkspaceState:
    """合并尚未查询的关键变化，与聊天消息分开保存。"""

    NOTICE = "[工作区提示：状态已变化。如当前任务需要了解详情，请调用 query_workspace_state()。]"

    def __init__(self):
        self._data = {"revision": 0, "pending": False, "files": {}, "task": None}
        self._store = None
        self._session_id = None

    def attach_session(self, store, session_id):
        data = store.load_workspace_state(session_id)
        if not isinstance(data, dict):
            raise ValueError("会话的工作区状态格式无效。")
        if data:
            if (
                type(data.get("revision")) is not int
                or data["revision"] < 0
                or type(data.get("pending")) is not bool
                or not isinstance(data.get("files"), dict)
                or not all(isinstance(path, str) and action == "reverted" for path, action in data["files"].items())
                or not (data.get("task") is None or isinstance(data["task"], dict))
            ):
                raise ValueError("会话的工作区状态格式无效。")
            self._data = data
        else:
            self._data = {"revision": 0, "pending": False, "files": {}, "task": None}
        self._store = store
        self._session_id = session_id

    def _save(self):
        if self._store is not None:
            self._store.save_workspace_state(self._session_id, self._data)

    def _changed(self):
        # 已查询的旧批次由工具结果留档，新变化从空批次开始。
        if not self._data["pending"]:
            self._data["files"] = {}
            self._data["task"] = None
        self._data["revision"] += 1
        self._data["pending"] = True

    def record_undo(self, paths):
        if paths:
            self._changed()
            self._data["files"].update((path, "reverted") for path in paths)
            self._save()

    def record_task(self, status, message, tool_call_id=None):
        self._changed()
        self._data["task"] = {"status": status, "message": message}
        if tool_call_id is not None:
            self._data["task"]["tool_call_id"] = tool_call_id
        self._save()

    def notice(self):
        return self.NOTICE if self._data["pending"] else ""

    def query(self):
        """只返回已记录的变化，不扫描目录或读取文件正文。"""
        return {
            "revision": self._data["revision"],
            "files": [{"path": path, "action": action} for path, action in sorted(self._data["files"].items())],
            "task": deepcopy(self._data["task"]),
        }

    def acknowledge(self, revision):
        # 工具结果成功写进历史后才确认已查询，执行失败或中途取消不清提示。
        if type(revision) is int and revision == self._data["revision"] and self._data["pending"]:
            self._data["pending"] = False
            self._save()
