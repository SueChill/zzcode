import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class SessionStore:
    """用 SQLite 保存会话和有序的完整消息，不保存文件撤销快照。"""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=5)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    project_root TEXT NOT NULL,
                    model TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    workspace_state TEXT NOT NULL DEFAULT '{}',
                    context_state TEXT NOT NULL DEFAULT '{}',
                    plan_state TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    sequence INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, sequence)
                );
            """)
            columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(sessions)")}
            with self.connection:
                for column in ("workspace_state", "context_state", "plan_state"):
                    if column not in columns:
                        self.connection.execute(f"ALTER TABLE sessions ADD COLUMN {column} TEXT NOT NULL DEFAULT '{{}}'")
        except BaseException:
            self.connection.close()
            raise

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat(timespec="microseconds")

    @staticmethod
    def _project_key(root):
        return os.path.normcase(str(Path(root).resolve()))

    def create_session(self, project_root, model):
        session_id = uuid4().hex
        now = self._now()
        with self.connection:
            self.connection.execute(
                "INSERT INTO sessions (id, project_root, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, self._project_key(project_root), model, now, now),
            )
        return session_id

    def find_session(self, prefix, project_root):
        prefix = prefix.strip().lower()
        if not 4 <= len(prefix) <= 32 or any(character not in "0123456789abcdef" for character in prefix):
            raise ValueError("会话编号应为完整编号，或至少 4 位的唯一前缀。")
        rows = self.connection.execute(
            "SELECT * FROM sessions WHERE project_root = ? AND substr(id, 1, ?) = ? LIMIT 2",
            (self._project_key(project_root), len(prefix), prefix),
        ).fetchall()
        if not rows:
            raise ValueError("找不到该会话，可用 --sessions 查看当前项目的历史会话。")
        if len(rows) > 1:
            raise ValueError("该前缀匹配多个会话，请输入更长的编号。")
        return dict(rows[0])

    def list_sessions(self, project_root):
        return [dict(row) for row in self.connection.execute(
            "SELECT * FROM sessions WHERE project_root = ? ORDER BY updated_at DESC, id",
            (self._project_key(project_root),),
        )]

    def load_messages(self, session_id):
        messages = []
        for expected, row in enumerate(self.connection.execute(
            "SELECT sequence, data FROM messages WHERE session_id = ? ORDER BY sequence", (session_id,),
        )):
            message = json.loads(row["data"])
            if row["sequence"] != expected or not isinstance(message, dict):
                raise ValueError("会话消息顺序或格式无效，未恢复该会话。")
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError("会话包含未知消息类型，未恢复该会话。")
            messages.append(message)
        return messages

    def append_messages(self, session_id, messages, expected_count):
        if not messages:
            return
        now = self._now()
        rows = [
            (session_id, expected_count + index, message["role"], json.dumps(message, ensure_ascii=False), now)
            for index, message in enumerate(messages)
        ]
        title = next((
            " ".join(message["content"].split())[:60]
            for message in messages
            if message["role"] == "user" and isinstance(message.get("content"), str)
        ), "")
        # 检查并追加处于同一事务，防止两个进程同时续聊时互相覆盖记录。
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            count = self.connection.execute(
                "SELECT coalesce(max(sequence) + 1, 0) FROM messages WHERE session_id = ?", (session_id,),
            ).fetchone()[0]
            if count != expected_count:
                raise ValueError("该会话已被另一个进程更新，请重新启动并恢复会话。")
            self.connection.executemany(
                "INSERT INTO messages (session_id, sequence, role, data, created_at) VALUES (?, ?, ?, ?, ?)", rows,
            )
            self.connection.execute(
                "UPDATE sessions SET updated_at = ?, title = CASE WHEN title = '' THEN ? ELSE title END WHERE id = ?",
                (now, title, session_id),
            )

    def load_workspace_state(self, session_id):
        row = self.connection.execute(
            "SELECT workspace_state FROM sessions WHERE id = ?", (session_id,),
        ).fetchone()
        if row is None:
            raise ValueError("找不到该会话。")
        return json.loads(row["workspace_state"])

    def save_workspace_state(self, session_id, state):
        data = json.dumps(state, ensure_ascii=False)
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE sessions SET workspace_state = ? WHERE id = ?", (data, session_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("找不到该会话。")

    def close(self):
        self.connection.close()

    def load_plan_state(self, session_id):
        row = self.connection.execute("SELECT plan_state FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise ValueError("找不到该会话。")
        return json.loads(row["plan_state"])

    def save_plan_state(self, session_id, state, *, expected_count, previous_state):
        """计划独立于聊天历史保存，拒绝覆盖另一个进程的新消息或新计划。"""
        data = json.dumps(state, ensure_ascii=False)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            count = self.connection.execute(
                "SELECT count(*) FROM messages WHERE session_id = ?", (session_id,),
            ).fetchone()[0]
            if count != expected_count or self.load_plan_state(session_id) != previous_state:
                raise ValueError("该会话已被另一个进程更新，请重新启动并恢复会话。")
            self.connection.execute(
                "UPDATE sessions SET plan_state = ?, updated_at = ? WHERE id = ?",
                (data, self._now(), session_id),
            )

    def load_context_state(self, session_id):
        row = self.connection.execute("SELECT context_state FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise ValueError("找不到该会话。")
        return json.loads(row["context_state"])

    def save_context_state(self, session_id, state, *, expected_count, previous_state):
        """只更新请求上下文，不修改完整消息；拒绝覆盖其他进程的新状态。"""
        data = json.dumps(state, ensure_ascii=False)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            count = self.connection.execute(
                "SELECT count(*) FROM messages WHERE session_id = ?", (session_id,),
            ).fetchone()[0]
            if count != expected_count or self.load_context_state(session_id) != previous_state:
                raise ValueError("该会话已被另一个进程更新，请重新启动并恢复会话。")
            self.connection.execute(
                "UPDATE sessions SET context_state = ?, updated_at = ? WHERE id = ?",
                (data, self._now(), session_id),
            )
