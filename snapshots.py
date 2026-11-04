import errno
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


IGNORED_DIRECTORIES = frozenset({".venv", "venv", "__pycache__", ".previews", "node_modules", ".zzcode"})


@dataclass
class Snapshot:
    files: dict[str, bytes]
    directories: set[str]


class SnapshotManager:
    """在内存中保留最近一次有修改的请求，供用户手动撤销。"""

    def __init__(self, root, resolve_path):
        self.root = Path(root).resolve()
        self.resolve_path = resolve_path
        self.before = None
        self.last = None

    def _capture(self):
        files = {}
        directories = set()

        def allowed(path):
            # 复用工具的项目边界和配置检查，并跳过文件、目录链接。
            relative = path.relative_to(self.root).as_posix()
            try:
                return self.resolve_path(relative) == path
            except ValueError:
                return False

        def fail(error):
            raise error

        if not self.root.is_dir():
            raise ValueError("项目根目录不存在。")
        for current, directory_names, file_names in os.walk(self.root, onerror=fail):
            directory = Path(current)
            directory_names[:] = [
                name for name in directory_names
                if name.lower() not in IGNORED_DIRECTORIES and allowed(directory / name)
            ]
            directories.update(
                (directory / name).relative_to(self.root).as_posix()
                for name in directory_names
            )
            for name in file_names:
                path = directory / name
                if allowed(path):
                    files[path.relative_to(self.root).as_posix()] = path.read_bytes()
        return Snapshot(files, directories)

    def start(self):
        self.before = self._capture()

    def finish(self):
        if self.before is None:
            return
        before, self.before = self.before, None
        try:
            after = self._capture()
        except (OSError, ValueError):
            # 无法确认本次修改时，停止提供旧的撤销点，避免撤错请求。
            self.last = None
            raise
        files = sorted(
            name for name in before.files.keys() | after.files.keys()
            if before.files.get(name) != after.files.get(name)
        )
        directories = before.directories ^ after.directories
        if files or directories:
            self.last = (before, after, files, directories)

    def _restore_file(self, path, data):
        """先写临时文件再替换，写入失败时保留当前文件。"""
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".zzcode-undo-", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(data)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                if not temporary_path.resolve().is_relative_to(self.root):
                    raise ValueError("恢复临时文件超出项目范围。")
                temporary_path.unlink(missing_ok=True)

    def undo(self):
        if self.last is None:
            return []
        before, after, files, directories = self.last
        current = self._capture()

        # 先检查所有恢复路径；用户后来手动改过的内容不能直接覆盖。
        for name in set(files) | directories:
            path = self.root / name
            if self.resolve_path(name) != path:
                raise ValueError(f"恢复路径已变成链接，未撤销：{name}")
        for name in files:
            if current.files.get(name) not in (before.files.get(name), after.files.get(name)):
                raise ValueError(f"文件在请求结束后有变化，未撤销：{name}")
            if name in current.directories and name not in before.directories:
                raise ValueError(f"恢复路径已变成目录，未撤销：{name}")
        for name in directories:
            if name in current.files and name not in files:
                raise ValueError(f"恢复目录已变成文件，未撤销：{name}")

        # 先移除本次新建文件，再恢复原有目录与文件。
        for name in files:
            path = self.root / name
            if name not in before.files and not path.is_dir():
                path.unlink(missing_ok=True)
        for name in sorted(before.directories - after.directories, key=lambda value: value.count("/")):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        for name in files:
            if name in before.files:
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                self._restore_file(path, before.files[name])

        # 新建目录只在为空时移除，保留用户后来添加的文件。
        for name in sorted(after.directories - before.directories, key=lambda value: value.count("/"), reverse=True):
            path = self.root / name
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError as error:
                    if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                        raise
        self.last = None
        return sorted(set(files) | directories)
