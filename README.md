# zzcode

一个用 Python 编写的终端代码智能体，目前版本为 `0.1.0-dev`。

在终端里提出需求，模型可以读取项目、搜索代码、修改文件、运行命令，并根据工具结果继续处理。项目从简单的 API 对话逐步发展而来，用来实践智能体循环、工具注册、权限审批、会话保存和上下文管理。

## 快速开始

建议使用 Python 3.12 和支持颜色、鼠标事件的 Windows 终端。以下命令使用 **CMD** 语法。

在 zzcode 项目目录打开 CMD，首次安装依赖：

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
if not exist .env copy .env.example .env
```

编辑 zzcode 安装目录里的 `.env`，填写 API 密钥和账号可用的模型 ID：

```dotenv
DEEPSEEK_API_KEY=你的_API_Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=你的模型ID
```

启动：

```cmd
python main.py
```

输入普通文字后回车发送，输入 `exit` 退出。

### 在其他项目里使用

**启动命令执行时所在的目录，就是本次工作区。** 在目标项目目录打开 CMD，然后通过绝对路径运行 zzcode，无须切换到 zzcode 目录。

例如，本机已安装依赖的 Python 和 zzcode 位于以下位置时：

```cmd
"F:\Python\python.exe" "F:\1-todosomething\zzcode\main.py"
```

如果使用前面创建的虚拟环境，则改用虚拟环境的解释器：

```cmd
"F:\1-todosomething\zzcode\.venv\Scripts\python.exe" "F:\1-todosomething\zzcode\main.py"
```

程序仍从 **zzcode 安装目录**读取 `.env`、`SYSTEM.md` 和 `mcp.json`。工作区路径来自 `Path.cwd()`，启动后固定，不随模型调用变化。

## 终端操作

| 操作 | 效果 |
| --- | --- |
| 普通文字 + 回车 | 发送需求 |
| 输入 `/` | 显示本地命令候选，继续输入筛选 |
| 任务执行时按 `Ctrl+C` | 取消当前任务，回到输入处，保留已完成的修改 |
| 等待输入时连续按两次 `Ctrl+C`（2 秒内），或空输入时按 `Ctrl+D` | 退出程序；第一次 `Ctrl+C` 会清空输入并提示 |
| `Shift+Tab` | 循环只读 → 自动 → 完全访问；进入完全访问需确认 |
| 等待输入时按 `Ctrl+T` | 循环切换关闭思考和当前模型支持的力度，下一次请求生效 |
| 等待输入时按 `Alt+M` | 按配置顺序切换模型；少于两个完整配置时保持不变 |
| 审批时按 `y` / `n` | 允许或拒绝，也可用方向键、回车或鼠标选择 |
| 审批时按 `Tab` 或点击详情标题 | 展开、收起脚本、差异或完整参数 |
| 点击思考标题 | 在全屏终端界面中展开、收起该轮思考 |
| 点击长代码块标题 | 在全屏终端界面中展开、收起代码 |

思考内容使用灰色显示，正文开始后自动收起为可点击的「思考」标题；接口需要返回 `reasoning_content` 才有可展示的思考。非全屏模式仍提示用 `/reasoning` 查看最近一段。长代码默认收起，完整回复仍保存在历史中。

状态栏显示模型、上下文估算、思考力度、权限和任务状态，不再显示 MCP 与 Skill 数量；连接和技能详情仍可通过 `/mcp`、`/skills` 查询。

每次任务结束后会输出一行灰色摘要：状态、耗时、模型轮数、工具次数、改动文件和运行次数。摘要最多 140 字符，文件较多时只列前三个；纯聊天不输出摘要。

### 本地命令

本地查询和设置由程序直接处理。`/plan 需求`、`/execute` 和 `/continue` 会启动模型任务，并将用户输入的原始命令作为真实消息保存。

| 命令 | 用途 |
| --- | --- |
| `/help` | 在终端内查看按职责分组的命令和快捷键 |
| `/help 关键词` | 筛选命令，例如 `/help plan`、`/help 权限` |
| `/trace` | 查看最近任务摘要、最慢 5 轮及最多 5 条异常工具 |
| `/trace list` | 列出当前会话的任务和关键命令记录 |
| `/trace 编号前缀` | 查看指定追踪，前缀至少 4 位且必须唯一 |
| `/trace tree [编号前缀]` | 按需查看完整执行时间线 |
| `/trace report [编号前缀]` | 生成独立的离线 HTML 耗时报告 |
| `/undo` | 撤销最近一次产生文件变化的任务 |
| `/reasoning` | 查看最近一段完整思考 |
| `/thinking [off\|on\|auto\|力度]` | 查看或切换模型思考设置；auto 恢复接口默认 |
| `/model [next\|名称]` | 查看已配置模型、切换下一个或指定配置名称 |
| `/code` | 查看最近一段助手回复全文，包含完整代码 |
| `/usage` | 查看上下文估算、接口用量和输入输出预算 |
| `/compact` | 请求模型压缩较早历史，保留最近两轮原文 |
| `/permissions readOnly` | 切换到只读模式 |
| `/permissions auto` | 切换到自动模式 |
| `/permissions fullAccess` | 显式开启完全访问模式 |
| `/permissions` | 在只读和自动模式之间切换 |
| `/mcp` | 查看 MCP 连接、工具数量和失败原因 |
| `/skills` | 查看已发现的技能、安装位置和环境要求 |
| `/skills add "来源"` | 从本地目录或 GitHub 安装技能 |
| `/skills reload` | 重新扫描已安装技能 |
| `/project` | 查看项目配置 ZZCODE.md 及解析出的命令 |
| `/plan` | 进入规划模式，之后输入需求 |
| `/plan 需求` | 进入规划模式并直接开始分析需求 |
| `/plan show` | 查看已保存的目标、阶段、验收方式和进度 |
| `/plan off` | 回到普通对话，保留计划 |
| `/execute` | 确认执行已保存的计划，或继续未完成的计划 |
| `/continue` | 继续上次未完成的任务，例如达到轮数上限之后 |
