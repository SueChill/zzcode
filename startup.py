"""后台初始化只发布结果，不在工作线程中改对话历史或工具列表。"""
import threading


class BackgroundJob:
    def __init__(self, load, *, name):
        self.stop = threading.Event()
        self.done = threading.Event()
        self.error = None
        self._result = None

        def run():
            try:
                self._result = load(self.stop)
            except Exception as error:
                # 不向终端输出网络异常正文，其中可能含服务地址或凭据。
                self.error = type(error).__name__
            finally:
                self.done.set()

        self._thread = threading.Thread(target=run, name=name, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def result(self):
        return self._result if self.done.is_set() and self.error is None else None

    def close(self, timeout=2):
        self.stop.set()
        if self._thread.ident is not None and threading.current_thread() is not self._thread:
            self._thread.join(timeout=timeout)
        return not self._thread.is_alive()
