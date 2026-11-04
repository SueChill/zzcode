"""从环境配置加载有序模型列表；不查询接口，不把密钥交给终端显示。"""
import re
import threading
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from model import ThinkingSettings


@dataclass(frozen=True)
class ModelProfile:
    alias: str
    model: str
    base_url: str
    api_key: str = field(repr=False)
    display_name: str = ""
    context_window: int = 0
    output_limit: int = 0
    thinking: str = "auto"
    effort: str = ""

    def client_options(self):
        return {"api_key": self.api_key, "base_url": self.base_url}


def load_profiles(getenv):
    names = (getenv("MODEL_PROFILES", "").strip() or "default").split(",")
    names = [name.strip().lower() for name in names]
    if (len(names) > 32 or len(set(names)) != len(names)
            or any(not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name) for name in names)):
        raise ValueError("MODEL_PROFILES 应为逗号分隔、互不重复的模型名称，只使用字母、数字和下划线。")
    profiles = []
    for alias in names:
        if alias == "default":
            keys = {"model": "DEEPSEEK_MODEL", "base_url": "DEEPSEEK_BASE_URL", "api_key": "DEEPSEEK_API_KEY"}
            defaults = {"model": "deepseek-flash", "base_url": "https://api.deepseek.com"}
            prefix = "MODEL_"
        else:
            prefix = "MODEL_" + alias.upper() + "_"
            keys = {"model": prefix + "ID", "base_url": prefix + "BASE_URL", "api_key": prefix + "API_KEY"}
            defaults = {}
        values = {name: getenv(key, defaults.get(name, "")).strip() for name, key in keys.items()}
        if not all(values.values()):
            continue
        try:
            endpoint = urlsplit(values["base_url"])
            if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
                raise ValueError()
            endpoint.port
        except ValueError:
            raise ValueError(f"{keys['base_url']} 应为有效的 HTTP 或 HTTPS 地址。") from None
        try:
            capacity = int(getenv(prefix + "CONTEXT_WINDOW", "").strip() or "0")
            output = int(getenv(prefix + "MAX_OUTPUT_TOKENS", "").strip() or "0")
            if capacity < 0 or output < 0:
                raise ValueError()
        except ValueError:
            raise ValueError(f"{prefix}CONTEXT_WINDOW、{prefix}MAX_OUTPUT_TOKENS 应填写正整数或留空。") from None
        profile = ModelProfile(alias=alias, **values,
                               display_name=getenv(prefix + "DISPLAY_NAME", "").strip(),
                               context_window=capacity, output_limit=output,
                               thinking=getenv(prefix + "THINKING", "auto"),
                               effort=getenv(prefix + "REASONING_EFFORT", ""))
        ThinkingSettings(profile.model, profile.base_url, profile.thinking, profile.effort)
        profiles.append(profile)
    return profiles


class ModelSelection:
    def __init__(self, profiles):
        self.profiles = tuple(profiles)
        self._index = 0
        self._lock = threading.RLock()
        self._thinking = {profile.alias: ThinkingSettings(profile.model, profile.base_url,
                                                         profile.thinking, profile.effort)
                          for profile in profiles}
        self._info = {}

    @property
    def current(self):
        with self._lock:
            return self.profiles[self._index]

    @property
    def thinking(self):
        return self.settings_for(self.current.alias)

    def settings_for(self, alias):
        return self._thinking[alias]

    def select(self, alias=None):
        with self._lock:
            if len(self.profiles) < 2 and alias is None:
                return False
            if alias is None:
                target = (self._index + 1) % len(self.profiles)
            else:
                target = next((i for i, profile in enumerate(self.profiles) if profile.alias == alias), None)
                if target is None:
                    return False
            changed = target != self._index
            self._index = target
            return changed

    def update_info(self, alias, info):
        with self._lock:
            self._info[alias] = dict(info)

    def view(self, alias=None):
        with self._lock:
            profile = (self.profiles[self._index] if alias is None else
                       next(profile for profile in self.profiles if profile.alias == alias))
            info = self._info.get(profile.alias, {})
            return {"alias": profile.alias, "model": profile.model,
                    "display_name": profile.display_name or info.get("name", ""),
                    "context_window": profile.context_window or info.get("context_window", 0)}
