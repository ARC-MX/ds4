# DwarfStar (DS4) 用户手册

> **从模型下载到双机联动，手把手教你上手本地推理引擎 DwarfStar**

---

## 目录

1. [项目简介](#1-项目简介)
2. [硬件与环境要求](#2-硬件与环境要求)
3. [模型下载](#3-模型下载)
4. [编译构建](#4-编译构建)
5. [命令行交互模式](#5-命令行交互模式)
6. [服务器模式](#6-服务器模式)
7. [Agent 模式](#7-agent-模式)
8. [双机联动（分布式推理）](#8-双机联动分布式推理)
9. [SSD 流式推理（小内存跑大模型）](#9-ssd-流式推理小内存跑大模型)
10. [思考模式](#10-思考模式)
11. [性能调优](#11-性能调优)
12. [常见问题排查](#12-常见问题排查)

---

## 1. 项目简介

**DwarfStar** 是一个专为 DeepSeek V4 Flash / DeepSeek V4 PRO 优化的本地推理引擎。它不是一个通用的 GGUF 加载器，而是从模型加载、KV 缓存管理、HTTP API 到编码 Agent 全链路自包含的精简实现。

### 核心特点

- **极窄聚焦**：只跑 DeepSeek V4 系列模型，针对性优化
- **多后端支持**：Metal（macOS）、CUDA（NVIDIA/DGX Spark）、ROCm（Strix Halo）、CPU（仅调试）
- **分布式推理**：支持多机联动，将模型层拆分到多台机器协同推理
- **SSD 流式加载**：允许在内存小于模型大小的机器上运行
- **KV 缓存磁盘持久化**：支持会话保存/恢复，避免重复 prefill
- **完整 API 兼容**：支持 OpenAI Chat/Responses、Anthropic Messages 协议
- **原生 Agent**：内置终端编程助手，支持代码读写、搜索、Shell 执行

### 支持的模型

| 模型 | 总参数量 | 激活参数量 | 上下文长度 |
|---|---|---|---|
| DeepSeek-V4-Flash | 284B | 13B | 1M tokens |
| DeepSeek-V4-Pro | 1.6T | 49B | 1M tokens |

---

## 2. 硬件与环境要求

### macOS（Metal 后端，主要目标平台）

| 配置 | 推荐量化 | 最低内存 |
|---|---|---|
| MacBook Pro M3/M4/M5 Max | q2-imatrix (~81 GB) | 96 GB |
| MacBook Pro M3/M4/M5 Max | q2-q4-imatrix (~98 GB) | 128 GB |
| Mac Studio M3 Ultra | q4-imatrix (~153 GB) | 256 GB+ |
| Mac Studio M3 Ultra | pro-q2-imatrix (~430 GB) | 512 GB |

### Linux CUDA

| 配置 | 推荐量化 | 最低内存 |
|---|---|---|
| DGX Spark (GB10) | q2-imatrix | 128 GB |
| 通用 CUDA GPU | q2-imatrix | 128 GB+ |

### Linux ROCm（Strix Halo）

| 配置 | 推荐量化 | 最低内存 |
|---|---|---|
| Framework Desktop (Radeon 8060S) | q2-imatrix | 128 GB |

> **注意**：ROCm 环境需参考 [STRIXHALO.md](STRIXHALO.md) 进行额外配置（GTT aperture 扩容等）。

### 网络要求（分布式推理）

- **推荐**：Thunderbolt 5 直连（延迟 < 1ms）
- **可用**：千兆以太网
- **勉强**：WiFi（延迟 ~77ms，生成速度明显下降）
- **仅测试**：互联网/VPN（延迟 > 150ms，仅适合验证模型是否能跑）

---

## 3. 模型下载

### 3.1 配置 Hugging Face Token（可选）

公开下载通常不需要 token，但已登录用户可使用本地缓存的 token：

```sh
# 方式一：设置环境变量
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# 方式二：使用命令行参数
./download_model.sh q2-imatrix --token hf_xxxxxxxxxxxxxxxxxxxx
```

### 3.2 下载 Flash 模型

```sh
# 96/128 GB 内存机器首选（2-bit 量化，约 81 GB）
./download_model.sh q2-imatrix

# 128 GB 机器追求更高质量（混合量化，约 98 GB）
./download_model.sh q2-q4-imatrix

# 256 GB+ 内存机器（4-bit 量化，约 153 GB）
./download_model.sh q4-imatrix

# 可选：MTP 投机解码组件（约 3.5 GB）
./download_model.sh mtp
```

下载完成后，脚本会自动创建符号链接 `./ds4flash.gguf -> ./gguf/<模型文件名>`，后续命令无需指定 `-m` 参数。

### 3.3 下载 PRO 模型（高内存机器）

```sh
# 512 GB 内存机器（PRO 2-bit 单文件，约 430 GB）
./download_model.sh pro-q2-imatrix

# PRO 分布式部署：两台机器各自下载一半
# 协调者机器（layers 0-30）
./download_model.sh pro-q4-layers00-30

# 工作机（layers 31-output）
./download_model.sh pro-q4-layers31-output

# 或者在同一台机器下载全部（用于分发）
./download_model.sh pro-q4-split
```

> **注意**：PRO 文件因为体积巨大，使用 Hugging Face CLI（`hf`）下载而非 curl。需先安装：
> ```sh
> python3 -m pip install -U huggingface_hub hf_xet
> ```

### 3.4 自定义下载目录

```sh
# 默认下载到 ./gguf/，可通过环境变量修改
export DS4_GGUF_DIR=/path/to/models
./download_model.sh q2-imatrix
```

---

## 4. 编译构建

### 4.1 macOS（Metal）

```sh
# 普通构建
make -j$(sysctl -n hw.ncpu)

# 生成以下二进制文件：
#   ds4         - 命令行聊天工具
#   ds4-server  - HTTP API 服务器
#   ds4-agent   - 原生终端编程助手
#   ds4-bench   - 性能基准测试
#   ds4-eval    - 能力评估工具
```

### 4.2 Linux CUDA

```sh
# DGX Spark / GB10
make cuda-spark -j$(nproc)

# 通用 CUDA GPU
make cuda-generic -j$(nproc)

# 指定 CUDA 架构
make cuda CUDA_ARCH=sm_120 -j$(nproc)
```

### 4.3 Linux ROCm（Strix Halo）

```sh
# 需先按 STRIXHALO.md 配置 ROCm 环境
make strix-halo -j$(nproc)
# 或使用别名
make rocm -j$(nproc)
```

### 4.4 CPU 调试构建

```sh
make cpu -j$(nproc)
```

> **警告**：macOS 上运行 CPU 路径可能触发内核虚拟内存 bug 导致系统崩溃，仅用于诊断目的。

---

## 5. 命令行交互模式

### 5.1 快速启动

```sh
# 交互式对话（使用默认模型 ds4flash.gguf）
./ds4

# 指定模型
./ds4 -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf

# 指定上下文大小
./ds4 --ctx 100000

# 禁用思考模式，直接回答
./ds4 --nothink
```

### 5.2 单次提问

```sh
# 一句话提问
./ds4 -p "用一段话解释 Redis 流"

# 从文件读取长 prompt
./ds4 --prompt-file prompt.txt

# 带系统提示
./ds4 -sys "你是资深 C 语言专家" -p "解释函数指针"
```

### 5.3 交互命令

在 `ds4>` 提示符下可用：

| 命令 | 说明 |
|---|---|
| `/help` | 查看交互命令列表 |
| `/think` | 切换到思考模式 |
| `/think-max` | 切换到最大思考模式 |
| `/nothink` | 禁用思考 |
| `/ctx N` | 修改上下文大小并重建会话 |
| `/power N` | 设置 GPU 占空比 (1-100) |
| `/read FILE` | 读取文件内容并作为下一条用户消息 |
| `/quit` 或 `/exit` | 退出 |
| `Ctrl+C` | 中断当前生成，返回提示符 |

### 5.4 采样参数

```sh
# 确定性输出（贪心解码）
./ds4 -p "Hello" --temp 0

# 创造性输出
./ds4 -p "写一首诗" --temp 1.0 --top-p 0.9 --min-p 0.05

# 可复现的随机采样
./ds4 -p "Hello" --temp 0.8 --seed 42
```

---

## 6. 服务器模式

`ds4-server` 提供兼容 OpenAI 和 Anthropic 协议的 HTTP API，可用于各种编程 Agent 客户端。

### 6.1 启动服务器

```sh
# 基础启动
./ds4-server --ctx 100000

# 启用磁盘 KV 缓存（强烈推荐）
./ds4-server \
  --ctx 100000 \
  --kv-disk-dir ~/.ds4/server-kv \
  --kv-disk-space-mb 8192

# 允许局域网访问 + CORS 头
./ds4-server --ctx 100000 --host 0.0.0.0 --port 8000 --cors

# 降低功耗/发热（MacBook 推荐）
./ds4-server --ctx 100000 --power 60

# 启用 trace 调试
./ds4-server --ctx 100000 --trace /tmp/ds4-trace.txt
```

### 6.2 API 端点

| 端点 | 说明 |
|---|---|
| `GET /v1/models` | 获取模型列表 |
| `POST /v1/chat/completions` | OpenAI 聊天补全 |
| `POST /v1/responses` | OpenAI Responses（Codex CLI 推荐） |
| `POST /v1/completions` | 文本补全 |
| `POST /v1/messages` | Anthropic Messages（Claude Code 推荐） |

### 6.3 快速测试

```sh
# 测试模型列表
curl http://127.0.0.1:8000/v1/models

# 流式聊天
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash",
    "messages":[{"role":"user","content":"列出 Redis 的三个设计原则"}],
    "stream":true
  }'
```

### 6.4 配置外部 Agent 客户端

#### OpenCode（`~/.config/opencode/opencode.json`）

```json
{
  "provider": {
    "ds4": {
      "name": "ds4.c (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "dsv4-local"
      },
      "models": {
        "deepseek-v4-flash": {
          "name": "DeepSeek V4 Flash (ds4.c local)",
          "limit": { "context": 100000, "output": 384000 }
        }
      }
    }
  },
  "agent": {
    "ds4": {
      "description": "DeepSeek V4 Flash local",
      "model": "ds4/deepseek-v4-flash",
      "temperature": 0
    }
  }
}
```

#### Claude Code（Shell 包装脚本）

```sh
#!/bin/sh
export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_AUTH_TOKEN="dsv4-local"
export ANTHROPIC_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
exec "$HOME/.local/bin/claude" "$@"
```

#### Codex CLI（`codex.toml` 或项目配置）

```toml
[model_providers.ds4]
name = "DS4"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1000000
```

```sh
codex --model deepseek-v4-flash -c model_provider=ds4
```

---

## 7. Agent 模式

`ds4-agent` 是内建于推理引擎的原生终端编程助手，无需 API 边界，直接操控 KV 缓存。

### 7.1 启动 Agent

```sh
# 基础启动
./ds4-agent --ctx 100000

# 从其他目录启动时指定项目根目录
./ds4-agent --chdir /path/to/ds4 --ctx 100000

# 非交互模式（单次任务）
./ds4-agent --non-interactive -p "创建 /tmp/hello.c，输出 Hello World"

# 交互模式带初始 prompt
./ds4-agent -p "分析当前项目的 Makefile"
```

### 7.2 Agent 会话管理

```sh
# 保存当前会话
/save

# 列出已保存的会话
/list

# 切换到指定会话（秒级恢复，无需 prefill）
/switch <session-id>

# 删除会话
/del <session-id>

# 删除会话的 KV 数据（仅保留文本历史，之后可重建）
/strip <session-id>

# 查看当前会话历史
/history 20
```

### 7.3 会话文件位置

所有会话保存在 `~/.ds4/kvcache/`，以 SHA1 哈希命名。会话 ID 由首次用户输入和创建时间派生，跨保存操作保持稳定。

---

## 8. 双机联动（分布式推理）

分布式推理将模型的 Transformer 层拆分到多台机器，实现**单机放不下的模型在两台机器上联合推理**。

### 8.1 核心概念

1. **协调者（Coordinator）**：负责分词、采样、提示词管理，拥有部分层
2. **工作者（Worker）**：拥有其余层并负责计算
3. **层范围**：`--layers A:B` 表示从第 A 层到第 B 层（含两端）
4. **通信方式**：Worker 直连 Worker，无需协调者中转
5. **KV 缓存**：每个 Worker 管理自己层范围的 KV 缓存

### 8.2 预填充与生成的区别

| 阶段 | 分布式行为 | 性能 |
|---|---|---|
| **预填充 (Prefill)** | 流水线并行：协调者处理 chunk N+1 时，Worker 处理 chunk N | **可加速**（吞吐量更高） |
| **生成 (Decode)** | 严格自回归：token N+1 必须等 token N 完成全链路 | **比单机慢**（有通信延迟） |

### 8.3 两种使用场景

#### 场景 A：用两台机器跑更大的模型（例如 PRO Q4）

```
两台 Mac Studio M3 Ultra (各 512 GB RAM)：
  Coordinator:  layers 0-30       (~426 GB)
  Worker:       layers 31-output  (~412 GB)

能跑单台跑不动的完整 PRO Q4 模型
```

#### 场景 B：用两台机器加速预填充（例如 Flash Q4）

```
两台 MacBook M5 Max (各 128 GB RAM)：
  Coordinator:  layers 0-19
  Worker:       layers 20-output

预填充吞吐量可提升 1.4x ~ 1.8x，但生成速度略降
```

### 8.4 实战步骤：两台 MacBook 跑 Flash Q4 分布式

#### 准备工作

两台机器均需：
```sh
# 1. 编译 DS4
git clone <repo-url> && cd ds4
make -j$(sysctl -n hw.ncpu)

# 2. 下载模型（两台都下载，但各自只加载自己的层）
./download_model.sh q4-imatrix
```

#### 步骤一：启动工作机（Machine B）

```sh
./ds4 \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --role worker \
  --layers 20:output \
  --coordinator <Machine A 的 IP> 1234
```

> **注意**：`--layers 20:output` 包含从第 20 层到输出层的所有权重和输出头。最后的工作者应持有输出头以避免传回完整 hidden state。

#### 步骤二：启动协调者（Machine A）

```sh
./ds4 \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --role coordinator \
  --layers 0:19 \
  --listen 0.0.0.0 1234
```

启动后即进入正常的交互式对话界面，操作和单机模式完全一样。

#### 步骤三：验证连接

协调者启动时会在日志中打印路由信息。如果 Worker 成功连接，协调者日志会显示完整的层覆盖信息。

### 8.5 实战步骤：两台 Mac Studio 跑完整 PRO Q4

```sh
# 协调者机器（Machine A）
./download_model.sh pro-q4-layers00-30
./ds4 \
  -m gguf/DeepSeek-V4-Pro-Q4K-Layers00-30.gguf \
  --role coordinator \
  --layers 0:30 \
  --listen 169.254.43.68 1234

# 工作机（Machine B）
./download_model.sh pro-q4-layers31-output
./ds4 \
  -m gguf/DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf \
  --role worker \
  --layers 31:output \
  --coordinator 169.254.43.68 1234
```

### 8.6 分布式调优参数

```sh
# 降低激活值传输精度（减少网络流量）
--dist-activation-bits 16   # 从 32-bit 降到 16-bit，推荐以太网/WiFi

# 调整预填充并行度
--dist-prefill-chunk 4096   # 预填充分块大小（默认 4096）
--dist-prefill-window 4     # 允许同时在途的分块数（默认 workers+2，上限 8）

# 调试路由
--debug                     # 打印每跳耗时、字节数等遥测信息
--dist-replay-check         # 重置并重放 prompt，对比 logits
```

### 8.7 故障恢复

- **Worker 断连**：协调者从路由中移除该 Worker，当前请求可能失败。Worker 重连后协调者会重建 KV 状态。
- **KV 校验**：每次请求都携带 token 前缀的滚动哈希，Worker 会验证。不匹配时协调者重放转录历史。
- **Ctrl+C**：协调者等待当前分块/Token 完成后才释放控制权，避免 KV 分裂。
- **会话保存/恢复**：使用标准 DSV4 格式，保存时协调者从 Worker 拉取层张量，恢复时按当前路由分发。

### 8.8 链路性能对比

| 连接方式 | 延迟 | 预填充速度 | 生成速度 |
|---|---|---|---|
| Thunderbolt 5 | 0.45 ms | 582.99 t/s | 25.09 t/s |
| WiFi | 77.20 ms | 250.70 t/s | 10.70 t/s |
| 互联网/VPN | 152.10 ms | 114.88 t/s | 3.63 t/s |

> **建议**：能插网线就插网线。WiFi 和互联网模式仅适合临时验证。

---

## 9. SSD 流式推理（小内存跑大模型）

当模型大小超过可用内存时，可以将路由 MoE 专家放在 SSD 上按需加载。

### 9.1 启用 SSD 流式

```sh
# 自动缓存预算（推荐）
./ds4 -m ./ds4flash.gguf --ssd-streaming

# 手动设置专家缓存大小
./ds4 -m ./ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 32GB
```

### 9.2 实际案例

**64 GB MacBook 跑 Flash：**
```sh
./download_model.sh q2-imatrix
./ds4 \
  -m ./ds4flash.gguf \
  --ssd-streaming \
  --ssd-streaming-cache-experts 32GB \
  --ctx 32768 \
  --nothink
```

**128 GB MacBook 尝鲜 PRO：**
```sh
./download_model.sh pro-q2-imatrix
./ds4 \
  -m gguf/DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf \
  --ssd-streaming \
  --ctx 32768 \
  --nothink
```

> **注意**：SSD 流式模式下生成速度明显慢于全内存模式，尤其在有大量缓存未命中时。长预填充速度尚可，但逐 token 生成对缓存命中率很敏感。

---

## 10. 思考模式

DeepSeek V4 支持三种推理模式：

| 模式 | 行为 | 适用场景 |
|---|---|---|
| **Non-think** | 直接输出答案 | 简单问答、代码生成 |
| **Think (High)** | 先内部推理再输出 | 复杂任务（**服务器默认**） |
| **Think Max** | 最大推理预算 | 极难数学、逻辑问题（需 ≥ 384K 上下文） |

### 10.1 CLI 中使用

```sh
# 禁用思考
./ds4 --nothink -p "1+1=?"

# 普通思考（默认）
./ds4 --think -p "解释爱因斯坦相对论"

# 最大思考
./ds4 --think-max --ctx 393216 -p "解决这个数学难题..."
```

交互模式中：`/nothink`、`/think`、`/think-max` 随时切换。

### 10.2 服务器中的思考模式

- 默认使用 high-effort 思考
- `reasoning_effort=max` 触发 Think Max（需要 `--ctx >= 393216`）
- `thinking: {type: "disabled"}` 或 `think: false` 关闭思考

---

## 11. 性能调优

### 11.1 功耗控制

```sh
# 降低 GPU 占空比到 50%（减少发热和风扇噪音）
./ds4 --power 50

# 服务器模式降低功耗
./ds4-server --power 40 --ctx 100000

# Agent 模式
./ds4-agent --power 70
```

`--power` 的取值范围是 1-100，默认为 100（全速）。降低该值会在计算单元之间插入短暂休眠，不影响模型输出质量。

### 11.2 MTP 投机解码（实验性）

```sh
# 先下载 MTP 组件
./download_model.sh mtp

# 启用投机解码
./ds4 --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2

# 调整置信度门限（默认 3）
./ds4 --mtp gguf/... --mtp-draft 2 --mtp-margin 5
```

> MTP 投机解码目前仅小幅提升生成速度，属于实验性功能。仅在贪心解码模式下有效。

### 11.3 Context 大小选择

| 可用内存 | 推荐 Context |
|---|---|
| 96 GB + q2-imatrix | 100K ~ 200K |
| 128 GB + q2-imatrix | 100K ~ 300K |
| 128 GB + q2-q4-imatrix | 100K ~ 250K |
| 512 GB + q4-imatrix | 100K ~ 500K+ |
| 512 GB + pro-q2 | 100K ~ 300K |

> 1M context 需要约 26 GB 额外内存（仅压缩索引器就约 22 GB）。确保总内存（模型 + KV 缓存 + 运行时）不超出物理内存。

### 11.4 预填充分块大小

```sh
# 环境变量控制 Metal 预填充分块
export DS4_METAL_PREFILL_CHUNK=4096  # 默认值
export DS4_METAL_PREFILL_CHUNK=2048  # 与官方 vector 测试对齐
export DS4_METAL_PREFILL_CHUNK=0     # 整批预填充（需足够内存）
```

### 11.5 性能基准测试

```sh
# 测量不同 context 长度的预填充/生成吞吐量
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128

# 输出 CSV
./ds4-bench \
  --prompt-file long.txt \
  --ctx-max 32768 \
  --csv speed.csv

# 纯预填充测试（不生成）
./ds4-bench --prompt-file long.txt --gen-tokens 0
```

---

## 12. 常见问题排查

### 12.1 模型加载失败

```sh
# 先检查模型文件摘要
./ds4 --inspect -m <模型路径>

# 查看完整帮助
./ds4 --help
./ds4 --help runtime    # 运行时参数详解
./ds4 --help distributed  # 分布式参数详解
```

### 12.2 推理结果异常

```sh
# 检查分词是否正确
./ds4 --dump-tokens -p "你的提示文本"

# 导出 greedy 续写的 top logprobs
./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "你的提示"

# 导出完整 logits
./ds4 --dump-logits /tmp/logits.json --nothink --prompt-file prompt.txt

# 服务器 trace 调试
./ds4-server --trace /tmp/ds4-trace.txt ...
```

### 12.3 macOS 内存不足

```sh
# 1. 关闭内存占用大的进程
# 2. 尝试 SSD 流式模式
./ds4 --ssd-streaming --ssd-streaming-cache-experts 32GB

# 3. 模拟小内存诊断
./ds4 --simulate-used-memory 16GB --inspect
```

### 12.4 分布式连接问题

```sh
# 检查 Worker 是否能连上协调者
# 在 Worker 机器上测试端口连通性
nc -zv <协调者 IP> <端口>

# 在协调者上开启 debug 日志
./ds4 --role coordinator --layers 0:19 --listen 0.0.0.0 1234 --debug

# 常见问题：
#   - 防火墙阻止端口
#   - IP 地址不正确（Thunderbolt 通常使用 169.254.x.x 自分配地址）
#   - 协调者和 Worker 使用不同版本编译的 ds4
```

### 12.5 内核崩溃（macOS CPU 路径）

macOS 上使用 CPU 路径运行大模型可能触发虚拟内存 bug 导致内核崩溃。**避免在 macOS 上使用 `--cpu` 进行大模型推理**。CPU 模式仅限小型诊断使用。

### 12.6 磁盘 KV 缓存排查

```sh
# 缓存目录（可安全删除重建）
ls -la ~/.ds4/server-kv/

# 查看缓存文件中的 prompt 文本
hexdump -C ~/.ds4/server-kv/<sha1>.kv | head -100

# 清空缓存重新开始
rm -rf ~/.ds4/server-kv/
```

### 12.7 获取更多帮助

```sh
./ds4 --help all          # 查看所有参数
./ds4-server --help all   # 服务器所有参数
./ds4-agent --help all    # Agent 所有参数
./ds4-bench --help all    # 基准测试所有参数
./ds4-eval --help all     # 评估工具所有参数
```

---

## 附录：快速上手指南

### 第一次使用（macOS，128 GB MacBook）

```sh
# 1. 编译
make -j$(sysctl -n hw.ncpu)

# 2. 下载模型（约 81 GB，需几分钟到几十分钟）
./download_model.sh q2-imatrix

# 3. 测试推理
./ds4 -p "解释 Redis 的三种数据结构" --nothink

# 4. 启动交互模式
./ds4 --ctx 100000

# 5. 启动 API 服务器
./ds4-server --ctx 100000 --kv-disk-dir ~/.ds4/server-kv --kv-disk-space-mb 8192

# 6. 配置你的编程 Agent 客户端指向 http://127.0.0.1:8000
```

### 第一次使用（Linux DGX Spark）

```sh
# 1. 编译
make cuda-spark -j$(nproc)

# 2. 下载模型
./download_model.sh q2-imatrix

# 3. 测试推理
./ds4 -p "Hello" --nothink

# 4. 后续操作同 macOS
```

---

> **项目仓库**：[antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf)
>
> **更多文档**：[README.md](README.md) | [CONTRIBUTING.md](CONTRIBUTING.md) | [MODEL_CARD.md](MODEL_CARD.md) | [STRIXHALO.md](STRIXHALO.md)
>
> **许可**：MIT License
