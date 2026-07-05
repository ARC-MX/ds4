#!/usr/bin/env bash
#==============================================================================
# deploy_ds4.sh — DwarfStar (DS4) 自动化部署脚本 for DGX Spark / Linux CUDA
#
# 用法:
#   ./deploy_ds4.sh start       启动 ds4-server
#   ./deploy_ds4.sh stop        停止 ds4-server
#   ./deploy_ds4.sh restart     重启 ds4-server
#   ./deploy_ds4.sh status      查看服务器状态与系统内存
#   ./deploy_ds4.sh logs        查看实时日志
#   ./deploy_ds4.sh build       仅编译（不启动）
#   ./deploy_ds4.sh test        快速 API 验证（curl 测试）
#   ./deploy_ds4.sh bench <mode> [num] [max_tokens]   ShareGPT Benchmark
#     mode: cli  — CLI 模式（推荐，直调 ./ds4，不需 server）
#           api  — API 模式（需先 start，通过 HTTP streaming 测试）
#     num:  采样数（默认 50）
#     max_tokens: 最大输出 token 数（默认 256）
#     示例:
#       ./deploy_ds4.sh bench cli 50 256
#       ./deploy_ds4.sh bench api 100 512
#       DS4_CTX=8192 ./deploy_ds4.sh bench cli 100 256
#   ./deploy_ds4.sh client      打印客户端配置（Claude Code / OpenCode / Codex）
#   ./deploy_ds4.sh help        显示完整帮助
#
# 环境变量（均可选，有合理默认值）:
#   DS4_MODEL_PATH        模型文件路径（默认自动检测）
#   DS4_MODEL_SEARCH_DIR  模型搜索目录
#   DS4_HOST              监听地址（默认 0.0.0.0）
#   DS4_PORT              监听端口（默认 8000）
#   DS4_CTX               上下文长度（默认 65536）
#   DS4_KV_DISK_DIR       KV 缓存目录（默认 ~/.ds4/server-kv）
#   DS4_KV_DISK_SPACE     KV 缓存大小 MB（默认 4096）
#   DS4_POWER             GPU 占空比 1-100（默认 80）
#   DS4_CORS              启用 CORS（默认 1）
#   DS4_LOG_DIR           日志目录（默认 ./logs）
#   DS4_SKIP_BUILD        跳过编译（默认 0）
#   DS4_CLEAN_BUILD       清理后重新编译（默认 0）
#   DS4_BENCH_NUM         Benchmark 采样数（默认 50）
#   DS4_BENCH_MAX_TOKENS  Benchmark 最大输出 tokens（默认 256）
#   DS4_BENCH_DATASET     Benchmark 数据集路径
#==============================================================================

set -euo pipefail

# ─── 配置 ───────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

# 模型路径：优先环境变量，其次自动检测
if [ -n "${DS4_MODEL_PATH:-}" ]; then
    MODEL_PATH="${DS4_MODEL_PATH}"
else
    # 自动检测：环境变量指定目录 或 默认 models 目录
    MODEL_SEARCH_DIR="${DS4_MODEL_SEARCH_DIR:-/data/mxwang/Project/LLM/models/antirez-deepseek-v4-gguf}"
    MODEL_PATH=$(ls -1 "${MODEL_SEARCH_DIR}"/*.gguf 2>/dev/null | grep -iv mtp | head -1 || echo "")
    if [ -z "${MODEL_PATH}" ]; then
        # fallback: 在当前项目下找
        MODEL_PATH=$(ls -1 "${PROJECT_DIR}"/gguf/*.gguf 2>/dev/null | grep -iv mtp | head -1 || echo "")
    fi
fi

# 服务器配置
HOST="${DS4_HOST:-0.0.0.0}"
PORT="${DS4_PORT:-8000}"
CTX="${DS4_CTX:-65536}"
KV_DISK_DIR="${DS4_KV_DISK_DIR:-${HOME}/.ds4/server-kv}"
KV_DISK_SPACE="${DS4_KV_DISK_SPACE:-4096}"
POWER="${DS4_POWER:-80}"
CORS="${DS4_CORS:-1}"
LOG_DIR="${DS4_LOG_DIR:-${PROJECT_DIR}/logs}"
SKIP_BUILD="${DS4_SKIP_BUILD:-0}"

# 二进制路径
SERVER_BIN="${PROJECT_DIR}/ds4-server"
CLI_BIN="${PROJECT_DIR}/ds4"

# PID 文件
PID_FILE="${PROJECT_DIR}/.ds4-server.pid"

# 日志文件
LOG_FILE="${LOG_DIR}/ds4-server.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ─── 工具函数 ───────────────────────────────────────────────────────────────

log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date '+%Y-%m-%d %H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%Y-%m-%d %H:%M:%S') $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $*" >&2; }
log_step()  { echo -e "${BLUE}[STEP]${NC}  $*"; }

# ─── 环境检查 ───────────────────────────────────────────────────────────────

check_prerequisites() {
    log_step "检查运行环境..."

    # 1. 检查 NVIDIA GPU / CUDA
    if ! command -v nvidia-smi &>/dev/null; then
        log_error "未检测到 nvidia-smi，请确认 NVIDIA 驱动已安装"
        exit 1
    fi
    log_info "NVIDIA 驱动: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'unknown')"
    log_info "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

    # 2. 检查 CUDA
    if [ -z "${CUDA_HOME:-}" ]; then
        CUDA_HOME="/usr/local/cuda"
    fi
    if [ ! -f "${CUDA_HOME}/bin/nvcc" ]; then
        log_error "未找到 nvcc (${CUDA_HOME}/bin/nvcc)，请安装 CUDA Toolkit"
        exit 1
    fi
    CUDA_VER=$("${CUDA_HOME}/bin/nvcc" --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ',')
    log_info "CUDA 版本: ${CUDA_VER}"

    # 3. 检查模型文件
    if [ -z "${MODEL_PATH}" ] || [ ! -f "${MODEL_PATH}" ]; then
        log_error "未找到模型文件 (GGUF)"
        log_error "请设置环境变量 DS4_MODEL_PATH=/path/to/model.gguf"
        log_error "或确保模型文件在 ${MODEL_SEARCH_DIR}/ 下"
        exit 1
    fi
    MODEL_SIZE=$(du -h "${MODEL_PATH}" | cut -f1)
    log_info "模型: ${MODEL_PATH} (${MODEL_SIZE})"

    # 4. 检查可用内存
    AVAIL_MEM_MB=$(free -m | awk '/Mem:/ {print $7}')
    AVAIL_MEM_GB=$((AVAIL_MEM_MB / 1024))
    MODEL_SIZE_GB=$(du -b "${MODEL_PATH}" | cut -f1)
    MODEL_SIZE_GB=$((MODEL_SIZE_GB / 1024 / 1024 / 1024))
    log_info "可用内存: ${AVAIL_MEM_GB} GB, 模型大小: ${MODEL_SIZE_GB} GB"

    if [ "${AVAIL_MEM_GB}" -lt "$((MODEL_SIZE_GB + 4))" ]; then
        log_warn "可用内存 (${AVAIL_MEM_GB} GB) 小于模型大小 + 4 GB (${MODEL_SIZE_GB} + 4 GB)"
        log_warn "服务器可能因内存不足启动失败"
        log_warn "建议：降低上下文大小 (--ctx) 或关闭其他应用释放内存"
        log_warn "当前上下文大小配置: ${CTX}"
    fi

    # 5. 检查端口占用
    if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
        log_warn "端口 ${PORT} 已被占用:"
        ss -tlnp | grep ":${PORT} " || true
    fi

    log_info "环境检查通过 ✓"
}

# ─── 编译 ────────────────────────────────────────────────────────────────────

do_build() {
    log_step "开始编译 DS4 (CUDA/DGX Spark)..."

    cd "${PROJECT_DIR}"

    # 清理旧的构建产物（可选，避免 ABI 不兼容）
    if [ "${DS4_CLEAN_BUILD:-0}" = "1" ]; then
        log_info "清理旧的构建产物..."
        make clean
    fi

    local cpu_count
    cpu_count=$(nproc)
    log_info "使用 ${cpu_count} 个并行任务编译..."

    make cuda-spark -j"${cpu_count}"

    if [ ! -x "${SERVER_BIN}" ]; then
        log_error "编译失败: ${SERVER_BIN} 未生成"
        exit 1
    fi

    log_info "编译完成 ✓"
    log_info "已生成二进制文件:"
    for bin in ds4 ds4-server ds4-bench ds4-eval ds4-agent; do
        if [ -x "${PROJECT_DIR}/${bin}" ]; then
            echo "  - ${PROJECT_DIR}/${bin}"
        fi
    done
}

# ─── 启动服务器 ──────────────────────────────────────────────────────────────

do_start() {
    log_step "启动 DS4 Server..."

    # 检查是否已在运行
    if [ -f "${PID_FILE}" ]; then
        local old_pid
        old_pid=$(cat "${PID_FILE}")
        if kill -0 "${old_pid}" 2>/dev/null; then
            log_warn "ds4-server 已在运行 (PID: ${old_pid})"
            log_info "如需重启请执行: $0 restart"
            return 0
        else
            log_warn "清理过期的 PID 文件"
            rm -f "${PID_FILE}"
        fi
    fi

    # 创建日志目录
    mkdir -p "${LOG_DIR}"
    mkdir -p "${KV_DISK_DIR}"

    # 构建启动参数
    local args=(
        -m "${MODEL_PATH}"
        --host "${HOST}"
        --port "${PORT}"
        --ctx "${CTX}"
        --kv-disk-dir "${KV_DISK_DIR}"
        --kv-disk-space-mb "${KV_DISK_SPACE}"
        --power "${POWER}"
    )

    if [ "${CORS}" = "1" ]; then
        args+=(--cors)
    fi

    log_info "模型: ${MODEL_PATH}"
    log_info "监听: ${HOST}:${PORT}"
    log_info "上下文: ${CTX}"
    log_info "KV 缓存目录: ${KV_DISK_DIR} (${KV_DISK_SPACE} MB)"
    log_info "GPU 功率限制: ${POWER}%"
    log_info "日志文件: ${LOG_FILE}"

    # 启动服务器
    nohup "${SERVER_BIN}" "${args[@]}" >> "${LOG_FILE}" 2>&1 &
    local pid=$!
    echo "${pid}" > "${PID_FILE}"

    log_info "ds4-server 已启动 (PID: ${pid})"

    # 等待服务器就绪
    log_info "等待服务器就绪..."
    local max_wait=120
    local waited=0
    while [ "${waited}" -lt "${max_wait}" ]; do
        if curl -s "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
            log_info "服务器就绪 ✓ (耗时 ${waited}s)"
            do_status
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        if [ $((waited % 10)) -eq 0 ]; then
            log_info "等待中... (${waited}s/${max_wait}s)"
        fi
    done

    log_warn "服务器启动超时 (${max_wait}s)"
    log_warn "查看日志获取详情: tail -f ${LOG_FILE}"
    log_info "最近的日志输出:"
    tail -20 "${LOG_FILE}"
}

# ─── 停止服务器 ──────────────────────────────────────────────────────────────

do_stop() {
    log_step "停止 DS4 Server..."

    if [ ! -f "${PID_FILE}" ]; then
        log_warn "PID 文件不存在，尝试查找进程..."
        local pids
        pids=$(pgrep -f "ds4-server" 2>/dev/null || true)
        if [ -n "${pids}" ]; then
            log_info "找到 ds4-server 进程: ${pids}"
            kill ${pids} 2>/dev/null || true
            sleep 2
            # 强制杀死
            pids=$(pgrep -f "ds4-server" 2>/dev/null || true)
            if [ -n "${pids}" ]; then
                log_warn "强制终止: ${pids}"
                kill -9 ${pids} 2>/dev/null || true
            fi
        else
            log_info "未找到运行中的 ds4-server 进程"
        fi
        return 0
    fi

    local pid
    pid=$(cat "${PID_FILE}")
    if kill -0 "${pid}" 2>/dev/null; then
        log_info "发送 SIGTERM 到进程 ${pid}..."
        kill "${pid}"
        sleep 3

        if kill -0 "${pid}" 2>/dev/null; then
            log_warn "进程未响应 SIGTERM，发送 SIGKILL..."
            kill -9 "${pid}"
            sleep 1
        fi

        if kill -0 "${pid}" 2>/dev/null; then
            log_error "无法终止进程 ${pid}"
            return 1
        fi
        log_info "ds4-server 已停止 (PID: ${pid})"
    else
        log_warn "进程 ${pid} 不在运行"
    fi

    rm -f "${PID_FILE}"
}

# ─── 重启服务器 ──────────────────────────────────────────────────────────────

do_restart() {
    log_step "重启 DS4 Server..."
    do_stop
    sleep 1
    do_start
}

# ─── 查看状态 ────────────────────────────────────────────────────────────────

do_status() {
    log_step "DS4 Server 状态"

    # 检查进程
    if [ -f "${PID_FILE}" ]; then
        local pid
        pid=$(cat "${PID_FILE}")
        if kill -0 "${pid}" 2>/dev/null; then
            echo -e "  状态:     ${GREEN}运行中${NC}"
            echo -e "  PID:      ${pid}"
            local uptime_str
            uptime_str=$(ps -o etime= -p "${pid}" 2>/dev/null | tr -d ' ' || echo "unknown")
            echo -e "  运行时间: ${uptime_str}"
            local mem_usage
            mem_usage=$(ps -o rss= -p "${pid}" 2>/dev/null | awk '{printf "%.1f GB", $1/1024/1024}' || echo "unknown")
            echo -e "  内存使用: ${mem_usage}"
        else
            echo -e "  状态:     ${RED}已停止${NC} (PID 文件存在但进程不存活)"
            return 1
        fi
    else
        local pids
        pids=$(pgrep -f "ds4-server" 2>/dev/null || true)
        if [ -n "${pids}" ]; then
            echo -e "  状态:     ${YELLOW}运行中（无 PID 文件）${NC}"
            echo -e "  PID:      ${pids}"
        else
            echo -e "  状态:     ${RED}未运行${NC}"
            return 1
        fi
    fi

    # API 可用性检查
    echo ""
    if curl -s --max-time 5 "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
        echo -e "  API:      ${GREEN}可用${NC} (http://127.0.0.1:${PORT}/v1/models)"
        local models
        models=$(curl -s --max-time 5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null || echo "{}")
        echo "  模型列表: ${models}"
    else
        echo -e "  API:      ${RED}不可用${NC}"
    fi

    # 系统内存状态 (DGX Spark 使用统一内存架构，无独立显存)
    echo ""
    echo "  系统内存 (DDR 统一内存):"
    free -h | awk '/Mem:/ {printf "    总量: %s, 已用: %s, 可用: %s, 使用率: %.1f%%\n", $2, $3, $7, ($3/$2)*100}'

    # ds4-server 进程内存占用
    local ds4_pid
    ds4_pid=$(pgrep -f "ds4-server" 2>/dev/null | head -1)
    if [ -n "${ds4_pid}" ]; then
        local ds4_rss ds4_vsz
        ds4_rss=$(ps -o rss= -p "${ds4_pid}" 2>/dev/null | awk '{printf "%.1f GB", $1/1024/1024}')
        ds4_vsz=$(ps -o vsz= -p "${ds4_pid}" 2>/dev/null | awk '{printf "%.1f GB", $1/1024/1024}')
        echo "  ds4-server 进程: RSS=${ds4_rss}, VSZ=${ds4_vsz}"
    fi

    # GPU 温度和利用率
    if command -v nvidia-smi &>/dev/null; then
        echo "  GPU 状态:"
        nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null | \
            awk -F',' '{printf "    GPU 利用率: %s, 温度: %s\n", $1, $2}'
    fi
}

# ─── 查看日志 ────────────────────────────────────────────────────────────────

do_logs() {
    if [ ! -f "${LOG_FILE}" ]; then
        log_error "日志文件不存在: ${LOG_FILE}"
        exit 1
    fi
    log_info "实时日志 (Ctrl+C 退出): ${LOG_FILE}"
    echo "──────────────────────────────────────────────────────────────"
    tail -f "${LOG_FILE}"
}

# ─── 打印客户端配置 ──────────────────────────────────────────────────────────

do_client() {
    local base_url="http://127.0.0.1:${PORT}"

    echo "============================================================"
    echo "  客户端配置"
    echo "============================================================"
    echo ""

    echo "─── 快速测试 ───"
    echo ""
    echo "  # 模型列表"
    echo "  curl ${base_url}/v1/models"
    echo ""
    echo '  # 流式聊天'
    echo "  curl ${base_url}/v1/chat/completions \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo '    -d '\''{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"你好"}],"stream":true}'\'
    echo ""

    echo "─── Claude Code ─── (~/bin/claude-ds4.sh 或 shell 别名)"
    echo ""
    echo "  #!/bin/sh"
    echo "  export ANTHROPIC_BASE_URL=\"${base_url}\""
    echo "  export ANTHROPIC_AUTH_TOKEN=\"dsv4-local\""
    echo "  export ANTHROPIC_MODEL=\"deepseek-v4-flash\""
    echo "  export ANTHROPIC_DEFAULT_SONNET_MODEL=\"deepseek-v4-flash\""
    echo "  export ANTHROPIC_DEFAULT_HAIKU_MODEL=\"deepseek-v4-flash\""
    echo "  export ANTHROPIC_DEFAULT_OPUS_MODEL=\"deepseek-v4-flash\""
    echo "  export CLAUDE_CODE_SUBAGENT_MODEL=\"deepseek-v4-flash\""
    echo "  export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"
    echo "  export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1"
    echo "  exec \"\$HOME/.local/bin/claude\" \"\$@\""
    echo ""

    echo "─── OpenCode ─── (~/.config/opencode/opencode.json)"
    echo ""
    echo '  {'
    echo '    "provider": {'
    echo '      "ds4": {'
    echo '        "name": "DS4 (DGX Spark)",'
    echo '        "npm": "@ai-sdk/openai-compatible",'
    echo '        "options": {'
    echo "          \"baseURL\": \"${base_url}/v1\","
    echo '          "apiKey": "dsv4-local"'
    echo '        },'
    echo '        "models": {'
    echo '          "deepseek-v4-flash": {'
    echo '            "name": "DeepSeek V4 Flash (DS4 local)",'
    echo '            "limit": { "context": '$CTX', "output": 384000 }'
    echo '          }'
    echo '        }'
    echo '      }'
    echo '    },'
    echo '    "agent": {'
    echo '      "ds4": {'
    echo '        "description": "DeepSeek V4 Flash local",'
    echo '        "model": "ds4/deepseek-v4-flash",'
    echo '        "temperature": 0'
    echo '      }'
    echo '    }'
    echo '  }'
    echo ""

    echo "─── Codex CLI ─── (codex.toml)"
    echo ""
    echo '  [model_providers.ds4]'
    echo '  name = "DS4"'
    echo "  base_url = \"${base_url}/v1\""
    echo '  wire_api = "responses"'
    echo '  stream_idle_timeout_ms = 1000000'
    echo ""
}

# ─── 快速测试 ────────────────────────────────────────────────────────────────

do_test() {
    log_step "快速验证 API..."

    if ! curl -s --max-time 5 "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
        log_error "服务器未运行或不可达"
        log_info "请先启动服务器: $0 start"
        exit 1
    fi

    log_info "模型列表:"
    curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -m json.tool 2>/dev/null || \
    curl -s "http://127.0.0.1:${PORT}/v1/models"

    echo ""
    log_info "发送测试请求 (non-stream)..."
    curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{
        "model":"deepseek-v4-flash",
        "messages":[{"role":"user","content":"用一句话介绍 Redis"}],
        "max_tokens":100
      }' | python3 -m json.tool 2>/dev/null || true
}

# ─── Benchmark 性能测试 ──────────────────────────────────────────────────────

do_bench() {
    local bench_mode="${1:-api}"
    local bench_num="${2:-${DS4_BENCH_NUM:-50}}"
    local bench_tokens="${3:-${DS4_BENCH_MAX_TOKENS:-256}}"
    local bench_dataset="${4:-${DS4_BENCH_DATASET:-/data/mxwang/Project/LLM/models/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json}}"

    mkdir -p "${LOG_DIR}"

    if [ "${bench_mode}" = "cli" ]; then
        # CLI 模式：直接调用 ./ds4（绕过 server 内存开销）
        local bench_script="${PROJECT_DIR}/ds4_bench_cli.py"
        if [ ! -f "${bench_script}" ]; then
            log_error "CLI Benchmark 脚本不存在: ${bench_script}"
            exit 1
        fi
        log_step "DS4 CLI Benchmark — ShareGPT 性能测试 (绕过 server)"
        log_info "采样数: ${bench_num}, max_tokens: ${bench_tokens}, ctx: ${DS4_CTX:-4096}"
        log_info "数据集: ${bench_dataset}"
        python3 "${bench_script}" \
            --dataset "${bench_dataset}" \
            --num "${bench_num}" \
            --max-tokens "${bench_tokens}" \
            --ctx "${DS4_CTX:-4096}" \
            --output-csv "${LOG_DIR}/bench_cli_$(date +%Y%m%d_%H%M%S).csv" \
            --print-samples 3
    else
        # API 模式：通过 HTTP API 测试
        local bench_script="${PROJECT_DIR}/ds4_bench_api.py"
        if [ ! -f "${bench_script}" ]; then
            log_error "API Benchmark 脚本不存在: ${bench_script}"
            exit 1
        fi
        if ! curl -s --max-time 5 "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
            log_error "服务器未运行，请先执行: $0 start"
            exit 1
        fi
        log_step "DS4 API Benchmark — ShareGPT 性能测试"
        log_info "采样数: ${bench_num}, max_tokens: ${bench_tokens}"
        log_info "数据集: ${bench_dataset}"
        python3 "${bench_script}" \
            --base-url "http://127.0.0.1:${PORT}" \
            --dataset "${bench_dataset}" \
            --num "${bench_num}" \
            --max-tokens "${bench_tokens}" \
            --output-csv "${LOG_DIR}/bench_api_$(date +%Y%m%d_%H%M%S).csv" \
            --print-samples 3
    fi

    log_info "Benchmark 完成，CSV 报告保存在 ${LOG_DIR}/"
}

# ─── 帮助 ────────────────────────────────────────────────────────────────────

do_help() {
    echo "DwarfStar (DS4) 自动化部署脚本 — DGX Spark / Linux CUDA"
    echo ""
    echo "用法: $0 <command> [args...]"
    echo ""
    echo "命令:"
    echo "  start       编译（如需要）并启动 ds4-server"
    echo "  stop        停止 ds4-server"
    echo "  restart     重启 ds4-server"
    echo "  status      查看服务器状态"
    echo "  logs        查看实时日志"
    echo "  build       仅编译，不启动"
    echo "  test        快速 API 验证测试"
    echo "  bench       ShareGPT Benchmark 性能测试"
    echo "              $0 bench cli 50 256   (CLI 模式，推荐)"
    echo "              $0 bench api 50 256   (API 模式，需先 start)"
    echo "  client      打印客户端配置（Claude Code / OpenCode / Codex）"
    echo "  help        显示此帮助"
    echo ""
    echo "环境变量（均可选）:"
    echo "  DS4_MODEL_PATH       模型文件路径"
    echo "  DS4_MODEL_SEARCH_DIR  自动搜索模型的目录"
    echo "  DS4_HOST             监听地址（默认 0.0.0.0）"
    echo "  DS4_PORT             监听端口（默认 8000）"
    echo "  DS4_CTX              上下文长度（默认 65536）"
    echo "  DS4_KV_DISK_DIR      KV 缓存目录（默认 ~/.ds4/server-kv）"
    echo "  DS4_KV_DISK_SPACE    KV 缓存大小 MB（默认 4096）"
    echo "  DS4_POWER            GPU 占空比 1-100（默认 80）"
    echo "  DS4_CORS             启用 CORS（默认 1）"
    echo "  DS4_LOG_DIR          日志目录（默认 ./logs）"
    echo "  DS4_SKIP_BUILD       跳过编译（默认 0）"
    echo "  DS4_CLEAN_BUILD      清理后重新编译（默认 0）"
    echo "  DS4_BENCH_NUM        Benchmark 采样数（默认 50）"
    echo "  DS4_BENCH_MAX_TOKENS Benchmark 最大输出 tokens（默认 256）"
    echo ""
    echo "示例:"
    echo "  # 默认配置启动"
    echo "  $0 start"
    echo ""
    echo "  # 自定义上下文和端口"
    echo "  DS4_CTX=100000 DS4_PORT=8080 $0 start"
    echo ""
    echo "  # 指定模型路径"
    echo "  DS4_MODEL_PATH=/path/to/model.gguf $0 start"
    echo ""
    echo "  # 仅编译"
    echo "  DS4_CLEAN_BUILD=1 $0 build"
    echo ""
    echo "  # ── Benchmark ──"
    echo ""
    echo "  # CLI 模式（推荐）：直接调用 ./ds4，不需 server"
    echo "  $0 bench cli 50 256        # 50 条, 256 max_tokens"
    echo "  $0 bench cli 100 512       # 100 条, 512 max_tokens"
    echo "  DS4_CTX=8192 $0 bench cli 50 256  # 指定 context 大小"
    echo ""
    echo "  # API 模式：需先启动 server"
    echo "  $0 start                   # 先启动 server"
    echo "  $0 bench api 50 256        # 50 条, 256 max_tokens"
    echo ""
    echo "  # 使用自定义数据集"
    echo "  DS4_BENCH_DATASET=/path/to/data.json $0 bench cli 50 256"
}

# ─── 主入口 ──────────────────────────────────────────────────────────────────

main() {
    local cmd="${1:-help}"

    case "${cmd}" in
        start)
            check_prerequisites
            if [ "${SKIP_BUILD}" != "1" ] && [ ! -x "${SERVER_BIN}" ]; then
                do_build
            elif [ "${SKIP_BUILD}" != "1" ]; then
                log_info "二进制文件已存在，跳过编译（设置 DS4_SKIP_BUILD=1 可跳过）"
                log_info "如需重新编译，请运行: DS4_CLEAN_BUILD=1 $0 build"
            fi
            do_start
            ;;
        stop)
            do_stop
            ;;
        restart)
            check_prerequisites
            do_restart
            ;;
        status)
            do_status
            ;;
        logs)
            do_logs
            ;;
        build)
            check_prerequisites
            do_build
            ;;
        test)
            do_test
            ;;
        bench)
            do_bench "${2:-}" "${3:-}" "${4:-}"
            ;;
        client)
            do_client
            ;;
        help|--help|-h)
            do_help
            ;;
        *)
            log_error "未知命令: ${cmd}"
            echo ""
            do_help
            exit 1
            ;;
    esac
}

main "$@"
