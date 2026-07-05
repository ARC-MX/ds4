#!/usr/bin/env python3
"""
ds4_bench_cli.py — DS4 CLI Benchmark (绕过 server 内存开销)

直接调用 ./ds4 命令行进行性能测试，测量 TTFT / TPOT / 吞吐量。
用法同 ds4_bench_api.py。

TTFT 测量方法：通过 --dump-logprobs 输出第一个 token 的时间戳推算，
或通过解析 ds4 的 stderr 输出来计算 prefill/generation 时间。
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time

# ─── 默认配置 ─────────────────────────────────────────────────────────────────

DS4_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds4")
DEFAULT_DATASET = os.environ.get(
    "DS4_BENCH_DATASET",
    "/data/mxwang/Project/LLM/models/ShareGPT_Vicuna_unfiltered/"
    "ShareGPT_V3_unfiltered_cleaned_split.json",
)
DEFAULT_MODEL = os.environ.get(
    "DS4_MODEL_PATH",
    "/data/mxwang/Project/LLM/models/antirez-deepseek-v4-gguf/"
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
)
DEFAULT_NUM_SAMPLES = 50
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT = 300
DEFAULT_SEED = 42


def load_sharegpt_prompts(path, num_samples, seed):
    print(f"📂 加载数据集: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"   总对话数: {len(data)}")

    prompts = []
    for item in data:
        conversations = item.get("conversations", [])
        if conversations and conversations[0].get("from") == "human":
            prompt = conversations[0]["value"].strip()
            if len(prompt) > 10:
                prompts.append(prompt)

    print(f"   有效 prompts: {len(prompts)}")
    rng = random.Random(seed)
    if num_samples > len(prompts):
        num_samples = len(prompts)
    sampled = rng.sample(prompts, num_samples)
    sampled.sort(key=len)
    return sampled


def run_ds4_cli(prompt, max_tokens, timeout, ctx=4096):
    """调用 ./ds4 CLI，返回 metrics。"""
    cmd = [
        DS4_BIN,
        "-m", DEFAULT_MODEL,
        "--ctx", str(ctx),
        "--nothink",
        "--temp", "0",
        "-n", str(max_tokens),
        "-p", prompt,
    ]

    t_start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(DS4_BIN),
        )
    except subprocess.TimeoutExpired:
        return {
            "ttft_ms": timeout * 1000,
            "total_time_ms": timeout * 1000,
            "output_tokens": 0,
            "prompt_chars": len(prompt),
            "prefill_tps": 0,
            "gen_tps": 0,
            "error": "timeout",
        }

    t_end = time.time()
    wall_time_ms = (t_end - t_start) * 1000

    stderr = result.stderr
    stdout = result.stdout

    # 解析 stderr 中的性能数据
    # 格式: "ds4: prefill: X.XX t/s, generation: Y.YY t/s"
    prefill_tps = 0.0
    gen_tps = 0.0
    perf_match = re.search(r"prefill:\s*([\d.]+)\s*t/s,\s*generation:\s*([\d.]+)\s*t/s", stderr)
    if perf_match:
        prefill_tps = float(perf_match.group(1))
        gen_tps = float(perf_match.group(2))

    # 解析 stdout 中的生成文本
    output_text = stdout.strip()
    # 估算 output tokens（粗略：英文约 4 chars/token，中文约 1.5 chars/token）
    # 实际上在 model 输出后的 prompt 回显不确定，这里用简单估计
    output_chars = len(output_text)
    estimated_tokens = max(1, int(output_chars / 3.5))  # 混合中英文估计

    # 如果有 prefill rate 和 gen rate，反推 token 数
    if gen_tps > 0 and prefill_tps > 0:
        # total_time ≈ prompt_len / prefill_tps + output_tokens / gen_tps
        # wall_time 已知，但 prefill 部分未知
        # 用 gen_tps 估算: output_tokens ≈ gen_tps * (wall_time_ms/1000 - overhead)
        gen_time_est = wall_time_ms / 1000 * 0.85  # 假设 85% 时间用于生成
        estimated_tokens = int(gen_tps * gen_time_est)

    # TTFT 估算（预热 + prefill 时间）
    # 如果 prefill_tps 已知: TTFT ≈ prompt_length_in_tokens / prefill_tps
    # prompt 的 token 数未知（无 tokenizer），粗略估计 prompt_chars / 3
    prompt_tokens_est = len(prompt) / 3.0
    ttft_ms = 0
    if prefill_tps > 0:
        ttft_ms = (prompt_tokens_est / prefill_tps) * 1000
    else:
        ttft_ms = wall_time_ms * 0.15  # fallback: 假设 15% 为 TTFT

    error = None
    if result.returncode != 0:
        # 检测已知错误模式
        if "Killed" in stderr or result.returncode == -9:
            error = "SIGKILL (OOM suspected — CUDA out of memory during inference)"
        elif "CUDA tensor alloc failed" in stderr or "out of memory" in stderr.lower():
            error = "CUDA OOM — out of memory"
        elif "Cannot allocate memory" in stderr:
            error = "System OOM — Cannot allocate memory"
        else:
            # 从 stderr 中提取最后一行非启动日志的错误信息
            lines = stderr.strip().split('\n')
            # 过滤掉正常的启动日志行
            startup_prefixes = (
                "ds4: Linux cuda", "ds4: CUDA backend", "ds4: CUDA host",
                "ds4: CUDA preparing", "ds4: CUDA loading", "ds4: CUDA prepared",
                "ds4: CUDA startup", "ds4: cuda backend", "ds4: context buffers",
                "ds4: using GPU", "ds4-server:",
            )
            error_lines = [
                l for l in lines
                if l.strip() and not l.startswith(startup_prefixes)
            ]
            if error_lines:
                error = error_lines[-1][:120]
            elif result.returncode == -9:
                error = "SIGKILL (OOM suspected)"
            else:
                error = f"exit={result.returncode}"

    return {
        "ttft_ms": ttft_ms,
        "total_time_ms": wall_time_ms,
        "output_tokens": estimated_tokens,
        "prompt_chars": len(prompt),
        "prefill_tps": prefill_tps,
        "gen_tps": gen_tps,
        "error": error,
        "raw_output_len": output_chars,
    }


def compute_stats(values):
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "p90": 0, "p95": 0, "p99": 0}
    s = sorted(values)
    n = len(s)
    return {
        "min": min(s), "max": max(s), "mean": sum(s) / n,
        "median": s[n // 2], "p90": s[int(n * 0.90)],
        "p95": s[int(n * 0.95)], "p99": s[int(n * 0.99)],
    }


def format_ms(ms):
    if ms < 1000: return f"{ms:.1f} ms"
    return f"{ms/1000:.2f} s"


def format_tps(tps):
    return f"{tps:.1f}" if tps >= 10 else f"{tps:.2f}"


def print_report(metrics_list, args):
    ok = [m for m in metrics_list if m["error"] is None]
    err = [m for m in metrics_list if m["error"] is not None]

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        DS4 CLI Benchmark 测试报告 (ShareGPT)                 ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  引擎:     DS4 CLI (CUDA/DGX Spark)")
    print(f"║  采样数:   {args.num}  成功: {len(ok)}  失败: {len(err)}")
    print(f"║  ctx:      {args.ctx}  max_tokens: {args.max_tokens}")
    print("╠══════════════════════════════════════════════════════════════╣")

    if not ok:
        print("║  ❌ 没有成功的请求                                          ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        return

    prefill_tpss = [m["prefill_tps"] for m in ok if m["prefill_tps"] > 0]
    gen_tpss = [m["gen_tps"] for m in ok if m["gen_tps"] > 0]
    ttfts = [m["ttft_ms"] for m in ok if m["ttft_ms"] > 0]
    total_times = [m["total_time_ms"] for m in ok]
    prompt_lens = [m["prompt_chars"] for m in ok]
    out_lens = [m["raw_output_len"] for m in ok]

    prefill_s = compute_stats(prefill_tpss)
    gen_s = compute_stats(gen_tpss)
    ttft_s = compute_stats(ttfts)
    total_s = compute_stats(total_times)

    total_out_tokens = sum(m["output_tokens"] for m in ok)
    total_time_s = sum(total_times) / 1000
    overall_tps = total_out_tokens / total_time_s if total_time_s > 0 else 0

    print("║")
    print("║  ── 延迟 (TTFT 估算) ──")
    print(f"║    Mean:   {format_ms(ttft_s['mean']):>10}")
    print(f"║    Median: {format_ms(ttft_s['median']):>10}")
    print(f"║    P90:    {format_ms(ttft_s['p90']):>10}")
    print(f"║    P99:    {format_ms(ttft_s['p99']):>10}")
    print("║")
    print("║  ── Prefill 吞吐量 ──")
    print(f"║    Mean:   {format_tps(prefill_s['mean']):>10} t/s")
    print(f"║    Median: {format_tps(prefill_s['median']):>10} t/s")
    print(f"║    P90:    {format_tps(prefill_s['p90']):>10} t/s")
    print(f"║    Min:    {format_tps(prefill_s['min']):>10} t/s")
    print(f"║    Max:    {format_tps(prefill_s['max']):>10} t/s")
    print("║")
    print("║  ── Decode 吞吐量 ──")
    print(f"║    Mean:   {format_tps(gen_s['mean']):>10} t/s")
    print(f"║    Median: {format_tps(gen_s['median']):>10} t/s")
    print(f"║    P90:    {format_tps(gen_s['p90']):>10} t/s")
    print(f"║    Min:    {format_tps(gen_s['min']):>10} t/s")
    print(f"║    Max:    {format_tps(gen_s['max']):>10} t/s")
    print("║")
    print("║  ── 每请求总时间 ──")
    print(f"║    Mean:   {format_ms(total_s['mean']):>10}")
    print(f"║    Median: {format_ms(total_s['median']):>10}")
    print(f"║    P90:    {format_ms(total_s['p90']):>10}")
    print("║")
    print("║  ── Prompt 长度 ──")
    print(f"║    Mean:   {sum(prompt_lens)/len(prompt_lens):.0f} chars")
    print(f"║    Min/Max:{min(prompt_lens)}/{max(prompt_lens)} chars")
    print("║")
    print("║  ── 输出长度 ──")
    print(f"║    Mean:   {sum(out_lens)/len(out_lens):.0f} chars")
    print(f"║    估算输出 tokens: {total_out_tokens}")
    print(f"║    全局吞吐量:      {format_tps(overall_tps)} t/s (含 prefill)")
    print("║")
    if err:
        print("║  ── 失败请求 ──")
        for e in err[:5]:
            print(f"║    ❌ {str(e['error'])[:55]}")
    print("╚══════════════════════════════════════════════════════════════╝")


def save_csv(metrics_list, output_path):
    with open(output_path, "w") as f:
        f.write("index,ttft_ms,total_time_ms,output_tokens,prefill_tps,gen_tps,prompt_chars,raw_output_len,error\n")
        for i, m in enumerate(metrics_list):
            err = (m["error"] or "").replace(",", ";")
            f.write(f"{i},{m['ttft_ms']:.1f},{m['total_time_ms']:.1f},{m['output_tokens']},"
                    f"{m['prefill_tps']:.1f},{m['gen_tps']:.1f},{m['prompt_chars']},"
                    f"{m['raw_output_len']},{err}\n")
    print(f"\n📄 详细结果已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DS4 CLI Benchmark — ShareGPT 数据集")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--num", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--ctx", type=int, default=4096, help="上下文大小 (默认 4096)")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--print-samples", type=int, default=0)
    args = parser.parse_args()

    print("═" * 60)
    print("  DS4 CLI Benchmark (绕过 server，直接调用 ./ds4)")
    print("═" * 60)
    print(f"  模型:  {args.model}")
    print(f"  ctx:   {args.ctx}")
    print(f"  采样:  {args.num} (warmup {args.warmup})")
    print(f"  max_tokens: {args.max_tokens}")
    print()

    # 检查二进制
    if not os.path.exists(DS4_BIN):
        print(f"❌ 未找到 ds4 二进制: {DS4_BIN}")
        sys.exit(1)

    # 加载数据
    prompts = load_sharegpt_prompts(args.dataset, args.num + args.warmup, args.seed)
    warmup_prompts = prompts[:args.warmup]
    bench_prompts = prompts[args.warmup:args.warmup + args.num]

    # 预热
    if warmup_prompts:
        print(f"🔥 预热 ({len(warmup_prompts)} 条)...")
        for i, p in enumerate(warmup_prompts):
            print(f"  预热 {i+1}/{len(warmup_prompts)}...", end=" ", flush=True)
            m = run_ds4_cli(p, min(args.max_tokens, 32), args.timeout, args.ctx)
            if m["error"]:
                print(f"❌ {m['error'][:40]}")
            else:
                print(f"✅ pref={m['prefill_tps']:.1f}t/s gen={m['gen_tps']:.1f}t/s")

    # 正式测试
    print(f"\n🏃 正式测试 ({len(bench_prompts)} 条)...")
    results = []
    t_start = time.time()

    for i, p in enumerate(bench_prompts):
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            ok_count = len([r for r in results if r["error"] is None])
            avg_gen = sum(r["gen_tps"] for r in results if r["error"] is None) / max(1, ok_count)
            print(f"  [{i+1}/{len(bench_prompts)}] ok={ok_count}, "
                  f"avg_gen={avg_gen:.1f}t/s, elapsed={elapsed:.0f}s", flush=True)

        m = run_ds4_cli(p, args.max_tokens, args.timeout, args.ctx)
        results.append(m)

        if args.print_samples > 0 and i < args.print_samples:
            status = "❌" if m["error"] else "✅"
            print(f"    #{i} {status} pref={m['prefill_tps']:.1f}t/s gen={m['gen_tps']:.1f}t/s "
                  f"time={m['total_time_ms']:.0f}ms")

    t_end = time.time()
    print(f"\n⏱️  总测试耗时: {t_end - t_start:.1f}s")

    print_report(results, args)

    csv_path = args.output_csv or f"logs/bench_cli_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    save_csv(results, csv_path)


if __name__ == "__main__":
    main()
