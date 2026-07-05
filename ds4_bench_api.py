#!/usr/bin/env python3
"""
ds4_bench_api.py — DS4 Server API Benchmark Tool

使用 ShareGPT 数据集对 ds4-server 进行性能基准测试。
测量指标：TTFT、TPOT、吞吐量、延迟分布等。

用法:
  python3 ds4_bench_api.py                           # 默认参数运行
  python3 ds4_bench_api.py --num 50 --max-tokens 256   # 自定义参数
  python3 ds4_bench_api.py --concurrency 4              # 并发测试
  python3 ds4_bench_api.py --warmup 3                   # 先预热再测试
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict

# ─── 默认配置 ─────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = os.environ.get("DS4_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_DATASET = os.environ.get(
    "DS4_BENCH_DATASET",
    "/data/mxwang/Project/LLM/models/ShareGPT_Vicuna_unfiltered/"
    "ShareGPT_V3_unfiltered_cleaned_split.json",
)
DEFAULT_NUM_SAMPLES = 50
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT = 300
DEFAULT_SEED = 42


# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_sharegpt_prompts(path, num_samples, seed):
    """从 ShareGPT JSON 加载对话的第一轮 human prompt。"""
    print(f"📂 加载数据集: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"   总对话数: {len(data)}")

    prompts = []
    for item in data:
        conversations = item.get("conversations", [])
        if conversations and conversations[0].get("from") == "human":
            prompt = conversations[0]["value"].strip()
            if len(prompt) > 10:  # 过滤太短的 prompt
                prompts.append(prompt)

    print(f"   有效 prompts: {len(prompts)}")

    rng = random.Random(seed)
    if num_samples > len(prompts):
        num_samples = len(prompts)

    sampled = rng.sample(prompts, num_samples)
    # 按长度排序，先测短的（减少首条异常影响统计）
    sampled.sort(key=len)
    return sampled


# ─── API 请求 ────────────────────────────────────────────────────────────────

def send_chat_request(base_url, prompt, max_tokens, timeout, stream=True):
    """发送流式 chat completions 请求，返回 (metrics_dict, error_string)。"""
    url = f"{base_url}/v1/chat/completions"
    body = json.dumps({
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    metrics = {
        "ttft_ms": 0.0,
        "total_time_ms": 0.0,
        "output_tokens": 0,
        "prompt_chars": len(prompt),
        "first_token_text": "",
        "finish_reason": "",
        "error": None,
    }

    t_start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except Exception as e:
        metrics["error"] = str(e)
        return metrics

    first_token = True
    buffer = b""
    try:
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            buffer += chunk

            # 解析 SSE 事件
            while b"\n" in buffer:
                line_end = buffer.index(b"\n")
                line = buffer[:line_end].decode("utf-8", errors="replace").strip()
                buffer = buffer[line_end + 1:]

                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = event.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "") or ""
                finish = choices[0].get("finish_reason", "") or event.get("choices", [{}])[0].get("finish_reason", "")

                if first_token and content:
                    t_first = time.time()
                    metrics["ttft_ms"] = (t_first - t_start) * 1000
                    metrics["first_token_text"] = content[:50]
                    first_token = False

                if content:
                    metrics["output_tokens"] += 1

                if finish:
                    metrics["finish_reason"] = finish

    except Exception as e:
        if not metrics["error"]:
            metrics["error"] = str(e)

    t_end = time.time()
    metrics["total_time_ms"] = (t_end - t_start) * 1000

    if metrics["output_tokens"] > 0 and metrics["ttft_ms"] == 0:
        # 非流式或无 content delta，用总时间估算
        metrics["ttft_ms"] = metrics["total_time_ms"]

    return metrics


def send_chat_request_nonstream(base_url, prompt, max_tokens, timeout):
    """发送非流式请求，用于对比。"""
    url = f"{base_url}/v1/chat/completions"
    body = json.dumps({
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        raw = resp.read()
        t_end = time.time()
        result = json.loads(raw)

        usage = result.get("usage", {})
        choices = result.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""

        return {
            "ttft_ms": (t_end - t_start) * 1000,  # 非流式 TTFT ≈ 总时间
            "total_time_ms": (t_end - t_start) * 1000,
            "output_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "prompt_chars": len(prompt),
            "finish_reason": choices[0].get("finish_reason", "") if choices else "",
            "error": None,
        }
    except Exception as e:
        return {
            "ttft_ms": 0, "total_time_ms": 0,
            "output_tokens": 0, "prompt_chars": len(prompt),
            "finish_reason": "", "error": str(e),
        }


# ─── 统计计算 ─────────────────────────────────────────────────────────────────

def compute_stats(values):
    """计算基本统计量。"""
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0}
    s = sorted(values)
    n = len(s)
    return {
        "min": min(s), "max": max(s), "mean": sum(s) / n,
        "median": s[n // 2], "p50": s[n // 2],
        "p90": s[int(n * 0.90)], "p95": s[int(n * 0.95)], "p99": s[int(n * 0.99)],
        "count": n,
    }


def format_ms(ms):
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms/1000:.2f} s"


def format_tps(tps):
    if tps >= 100:
        return f"{tps:.1f}"
    return f"{tps:.2f}"


def print_report(metrics_list, args):
    """打印汇总报告。"""
    ok = [m for m in metrics_list if m["error"] is None]
    err = [m for m in metrics_list if m["error"] is not None]

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║            DS4 API Benchmark 测试报告                        ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  端点:     {args.base_url}                                  ")
    print(f"║  数据集:   ShareGPT (采样 {args.num} 条)")
    print(f"║  max_tokens: {args.max_tokens}")
    print(f"║  成功: {len(ok)}  失败: {len(err)}")
    print("╠══════════════════════════════════════════════════════════════╣")

    if not ok:
        print("║  ❌ 没有成功的请求                                          ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        return

    # 提取各项指标
    ttfts = [m["ttft_ms"] for m in ok if m["ttft_ms"] > 0]
    total_times = [m["total_time_ms"] for m in ok]
    output_tokens = [m["output_tokens"] for m in ok]
    tps_list = [m["output_tokens"] / (m["total_time_ms"] / 1000) if m["total_time_ms"] > 0 else 0 for m in ok]
    prompt_lens = [m["prompt_chars"] for m in ok]

    ttft_s = compute_stats(ttfts)
    total_s = compute_stats(total_times)
    tok_s = compute_stats(output_tokens)
    tps_s = compute_stats(tps_list)

    total_out_tokens = sum(output_tokens)
    total_time_s = sum(total_times) / 1000
    overall_tps = total_out_tokens / total_time_s if total_time_s > 0 else 0

    print("║")
    print("║  ── 延迟 (TTFT) ──")
    print(f"║    Mean:   {format_ms(ttft_s['mean']):>10}")
    print(f"║    Median: {format_ms(ttft_s['median']):>10}")
    print(f"║    P90:    {format_ms(ttft_s['p90']):>10}")
    print(f"║    P99:    {format_ms(ttft_s['p99']):>10}")
    print("║")
    print("║  ── 每请求总时间 ──")
    print(f"║    Mean:   {format_ms(total_s['mean']):>10}")
    print(f"║    Median: {format_ms(total_s['median']):>10}")
    print(f"║    P90:    {format_ms(total_s['p90']):>10}")
    print("║")
    print("║  ── Token 吞吐量 ──")
    print(f"║    每秒生成 (mean):      {format_tps(tps_s['mean']):>8} t/s")
    print(f"║    每秒生成 (median):    {format_tps(tps_s['median']):>8} t/s")
    print(f"║    每秒生成 (P90):       {format_tps(tps_s['p90']):>8} t/s")
    print(f"║    全局吞吐量:           {format_tps(overall_tps):>8} t/s")
    print(f"║    总输出 tokens:        {total_out_tokens:>8}")
    print(f"║    总耗时:               {format_ms(total_time_s*1000):>10}")
    print("║")
    print("║  ── 输出 Tokens 分布 ──")
    print(f"║    Min / Mean / Max:     {tok_s['min']} / {tok_s['mean']:.1f} / {tok_s['max']}")
    print("║")
    print("║  ── Prompt 长度分布 ──")
    print(f"║    Mean chars:           {sum(prompt_lens)/len(prompt_lens):.0f}")
    print(f"║    Min / Max chars:      {min(prompt_lens)} / {max(prompt_lens)}")
    print("║")
    print("║  ── 失败请求 ──")
    if err:
        for e in err:
            print(f"║    ❌ {e['error'][:60]}")
    else:
        print("║    ✅ 无失败请求")
    print("║")
    print("╚══════════════════════════════════════════════════════════════╝")


def save_csv(metrics_list, output_path):
    """保存详细结果到 CSV。"""
    with open(output_path, "w") as f:
        f.write("index,ttft_ms,total_time_ms,output_tokens,tps,prompt_chars,prompt_len_err,finish_reason,error\n")
        for i, m in enumerate(metrics_list):
            tps = m["output_tokens"] / (m["total_time_ms"] / 1000) if m["total_time_ms"] > 0 else 0
            err = (m["error"] or "").replace(",", ";")
            f.write(f"{i},{m['ttft_ms']:.1f},{m['total_time_ms']:.1f},"
                    f"{m['output_tokens']},{tps:.2f},"
                    f"{m['prompt_chars']},"
                    f"{m.get('prompt_tokens','')},{m['finish_reason']},{err}\n")
    print(f"\n📄 详细结果已保存: {output_path}")


# ─── 主流程 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DS4 API Benchmark — ShareGPT 数据集性能测试"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API 地址")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="ShareGPT JSON 路径")
    parser.add_argument("--num", type=int, default=DEFAULT_NUM_SAMPLES, help="采样数量")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="每个请求最大输出 tokens")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="单请求超时 (秒)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    parser.add_argument("--warmup", type=int, default=3, help="预热请求数")
    parser.add_argument("--output-csv", default="", help="输出 CSV 路径")
    parser.add_argument("--non-stream", action="store_true", help="使用非流式模式")
    parser.add_argument("--print-samples", type=int, default=0, help="打印前 N 个请求的详细结果")
    args = parser.parse_args()

    print("═" * 60)
    print("  DS4 API Benchmark")
    print("═" * 60)
    print(f"  端点:       {args.base_url}")
    print(f"  数据集:     {args.dataset}")
    print(f"  采样数:     {args.num}")
    print(f"  max_tokens: {args.max_tokens}")
    print(f"  超时:       {args.timeout}s")
    print(f"  流式:       {'否' if args.non_stream else '是'}")
    print(f"  预热:       {args.warmup} 条")
    print()

    # 1. 加载数据
    prompts = load_sharegpt_prompts(args.dataset, args.num + args.warmup, args.seed)

    warmup_prompts = prompts[:args.warmup]
    bench_prompts = prompts[args.warmup:args.warmup + args.num]

    send_fn = send_chat_request_nonstream if args.non_stream else send_chat_request

    # 2. 预热
    if warmup_prompts:
        print(f"\n🔥 预热 ({len(warmup_prompts)} 条)...")
        for i, p in enumerate(warmup_prompts):
            print(f"  预热 {i+1}/{len(warmup_prompts)}...", end=" ", flush=True)
            m = send_fn(args.base_url, p, min(args.max_tokens, 64), args.timeout)
            if m["error"]:
                print(f"❌ {m['error'][:50]}")
            else:
                print(f"✅ {m['output_tokens']} tokens, {m['ttft_ms']:.0f}ms TTFT")

    # 3. 正式测试
    print(f"\n🏃 正式测试 ({len(bench_prompts)} 条)...")
    results = []
    t_bench_start = time.time()

    for i, p in enumerate(bench_prompts):
        # 进度指示
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_bench_start
            done = len([r for r in results if r["error"] is None])
            tps_sofar = sum(r["output_tokens"] for r in results if r["error"] is None) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(bench_prompts)}] 已完成 {done}, "
                  f"累计 {elapsed:.0f}s, 整体 {tps_sofar:.1f} t/s",
                  flush=True)

        m = send_fn(args.base_url, p, args.max_tokens, args.timeout)
        results.append(m)

        if args.print_samples > 0 and i < args.print_samples:
            status = "❌" if m["error"] else "✅"
            print(f"    #{i} {status} TTFT={m['ttft_ms']:.0f}ms, "
                  f"tokens={m['output_tokens']}, "
                  f"total={m['total_time_ms']:.0f}ms, "
                  f"prompt_len={m['prompt_chars']}")

    t_bench_end = time.time()
    print(f"\n⏱️  总测试耗时: {t_bench_end - t_bench_start:.1f}s")

    # 4. 生成报告
    print_report(results, args)

    # 5. 保存 CSV
    csv_path = args.output_csv or f"logs/bench_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    save_csv(results, csv_path)


if __name__ == "__main__":
    main()
