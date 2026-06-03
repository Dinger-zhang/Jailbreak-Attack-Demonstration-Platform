# Jailbreak-Attack-Demonstration-Platform

越狱攻击演示平台。当前平台已接入 `TAP`，提供前端控制台、后端任务调度、数据集管理、实时日志和结果导出。

## 功能

- `TAP` 子进程运行：后端调用 `TAP/main_TAP.py`，不重写 TAP 核心逻辑。
- Web 控制台：配置攻击者/目标/评估模型、TAP 参数、goal 和 target_str。
- 数据集管理：支持上传 `CSV`、`JSONL`、`JSON`、`TXT`，可选行填充测试目标。
- 任务管理：查看历史任务、状态、日志、parquet 结果摘要，并导出 CSV。
- 自部署模型：支持 TAP 已有的 `custom-api-model`，即 OpenAI Chat Completions 兼容接口。

## 目录结构

```text
backend/          FastAPI 后端、数据集解析、TAP 任务调度
frontend/         原生 HTML/CSS/JS 前端
datasets/         上传数据集运行时目录，默认不提交实际数据
platform_runs/    平台任务日志和结果目录，默认不提交
TAP/              已接入的 TAP 攻击方法
```

## 环境

本项目使用 conda 环境 `jailbreak`。

```powershell
conda activate jailbreak
python -m pip install -r requirements-platform.txt
```

如果你只缺单个包，也可以安装到同一环境：

```powershell
conda run -n jailbreak python -m pip install python-multipart
```

## 配置模型

如果使用实验室自部署 OpenAI 兼容服务，设置：

```powershell
$env:CUSTOM_API_URL="https://your-host/v1"
$env:CUSTOM_MODEL_NAME="your-model-name"
$env:CUSTOM_API_TOKEN="your-api-token"
```

也可以在前端页面填写这些字段。页面填写的 token 只传给当前 TAP 子进程，不写入任务元数据。

如果选择 OpenAI GPT 模型，设置：

```powershell
$env:OPENAI_API_KEY="sk-..."
```

后端默认对 TAP 子进程设置 `WANDB_MODE=offline`，避免必须先执行 `wandb login`。

## 启动

方式一：

```powershell
.\start_platform.ps1
```

方式二：

```powershell
conda run -n jailbreak python -m backend.main --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 数据集格式

推荐 CSV 或 JSONL 字段：

```text
goal,target_str,category
```

平台也会尝试识别以下别名：

- `goal`: `goal`, `prompt`, `question`, `behavior`, `instruction`, `task`
- `target_str`: `target_str`, `target`, `target_prefix`, `target_response`, `response_prefix`
- `category`: `category`, `label`, `type`, `class`

TXT 文件会把每个非空行作为 `goal`，此时需要在页面手动填写 `target_str`。

## API 摘要

- `GET /api/health`：检查后端、TAP 和依赖。
- `GET /api/methods`：查看已注册攻击方法和模型选项。
- `POST /api/datasets`：上传数据集。
- `GET /api/datasets/{id}/rows`：预览数据集行。
- `POST /api/jobs`：启动 TAP 任务。
- `GET /api/jobs/{id}/logs`：查看任务日志。
- `GET /api/jobs/{id}/results`：读取 TAP parquet 结果。
- `GET /api/jobs/{id}/results.csv`：导出 CSV。
