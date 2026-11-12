
# task1_falsesharing.py

import os
import sys
import json
import time
import ctypes
import statistics
import multiprocessing as mp

from pathlib import Path
from multiprocessing import shared_memory

from harness import run_benchmark, generate_verify_hash


# ============================================================
# CONFIGURATION
# ============================================================

STUDENT_ID = "230103201"

TOTAL_OPERATIONS = 24_000_000
SHARED_MEMORY_SIZE = 4096

PHYSICAL_CORES = 8
LOGICAL_THREADS = 16

STRIDE_VALUES = [1, 2, 4, 8, 16, 32]

PROCESS_COUNTS = [1, 2, 4, 8, 16]

WARMUP_RUNS = 2
TIMED_RUNS = 7

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# WORKER FUNCTION
# Must be defined at module level for Windows spawn.
# ============================================================

def worker(shm_name, slot_idx, iterations, barrier, start_event):

    shm = shared_memory.SharedMemory(name=shm_name)

    # Cast shared memory into an array of uint64 values.
    array_type = ctypes.c_uint64 * (
        SHARED_MEMORY_SIZE // ctypes.sizeof(ctypes.c_uint64)
    )

    shared_array = array_type.from_buffer(shm.buf)

    try:
        # Wait until all workers are ready.
        barrier.wait()

        # Wait for the parent process to release workers.
        start_event.wait()

        # Deterministic read-modify-write accumulation.
        for _ in range(iterations):
            value = shared_array[slot_idx]
            shared_array[slot_idx] = value + 1

    finally:
        del shared_array
        shm.close()


# ============================================================
# SINGLE BENCHMARK TRIAL
# ============================================================

def run_trial(process_count, stride):

    if process_count > TOTAL_OPERATIONS:
        raise ValueError("Too many processes.")

    iterations = TOTAL_OPERATIONS // process_count

    if iterations * process_count != TOTAL_OPERATIONS:
        raise ValueError("Operations must divide evenly among workers.")

    max_index = (process_count - 1) * stride

    required_bytes = (max_index + 1) * 8

    if required_bytes > SHARED_MEMORY_SIZE:
        raise ValueError("Shared memory block is too small.")

    # Allocate shared memory.
    shm = shared_memory.SharedMemory(
        create=True,
        size=SHARED_MEMORY_SIZE
    )

    # Initialize shared memory.
    array_type = ctypes.c_uint64 * (
        SHARED_MEMORY_SIZE // ctypes.sizeof(ctypes.c_uint64)
    )

    shared_array = array_type.from_buffer(shm.buf)

    for i in range(len(shared_array)):
        shared_array[i] = 0

    # Windows multiprocessing context (spawn).
    ctx = mp.get_context("spawn")

    barrier = ctx.Barrier(process_count + 1)
    start_event = ctx.Event()

    processes = []

    try:

        # Create workers.
        for t in range(process_count):

            slot_idx = t * stride

            process = ctx.Process(
                target=worker,
                args=(
                    shm.name,
                    slot_idx,
                    iterations,
                    barrier,
                    start_event
                )
            )

            process.start()
            processes.append(process)

        # All workers are ready before starting the timer.
        barrier.wait()

        start = time.perf_counter_ns()

        # Release workers simultaneously.
        start_event.set()

        # Wait for all workers to finish.
        for process in processes:
            process.join()

        end = time.perf_counter_ns()

        # Check worker exit codes.
        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed with exit code {process.exitcode}"
                )

        elapsed_ms = (end - start) / 1_000_000

        # Verify deterministic accumulation.
        for t in range(process_count):

            slot_idx = t * stride

            actual = shared_array[slot_idx]

            if actual != iterations:
                raise RuntimeError(
                    f"Incorrect result at slot {slot_idx}: "
                    f"expected {iterations}, got {actual}"
                )

        return elapsed_ms

    finally:

        # Release exported buffer before closing shared memory.
        del shared_array

        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()

            process.close()

        shm.close()
        shm.unlink()


# ============================================================
# BENCHMARK EXECUTION
# ============================================================

def benchmark_configuration(process_count, stride, label):

    print("\n" + "=" * 65)
    print(f"EXPERIMENT: {label}")
    print(f"Processes: {process_count}")
    print(f"Stride: {stride}")
    print(f"Byte separation: {stride * 8} bytes")
    print(f"Operations: {TOTAL_OPERATIONS:,}")
    print("=" * 65)

    result = run_benchmark(
        run_trial,
        process_count,
        stride,
        warmup_runs=WARMUP_RUNS,
        timed_runs=TIMED_RUNS
    )

    median_ms = result["median_ms"]

    throughput = TOTAL_OPERATIONS / (median_ms * 1000)

    result["throughput_mops"] = throughput
    result["process_count"] = process_count
    result["stride"] = stride
    result["separation_bytes"] = stride * 8

    print(f"Median: {median_ms:.3f} ms")
    print(f"IQR: {result['iqr_ms']:.3f} ms")
    print(f"Throughput: {throughput:.3f} Mops/s")
    print(f"Min: {result['min_ms']:.3f} ms")
    print(f"Max: {result['max_ms']:.3f} ms")

    return result


# ============================================================
# TABLE 1A: STRIDE SENSITIVITY
# ============================================================

def run_stride_sweep():

    print("\n\n")
    print("#" * 65)
    print("TABLE 1A: STRIDE SENSITIVITY")
    print("#" * 65)

    results = {}

    for stride in STRIDE_VALUES:

        result = benchmark_configuration(
            PHYSICAL_CORES,
            stride,
            "STRIDE SWEEP"
        )

        results[stride] = result

    return results


# ============================================================
# TABLE 1B: PROCESS SCALING
# ============================================================

def run_scaling_sweep(stride):

    print("\n\n")
    print("#" * 65)
    print(f"PROCESS SCALING: STRIDE = {stride}")
    print("#" * 65)

    results = {}

    for process_count in PROCESS_COUNTS:

        result = benchmark_configuration(
            process_count,
            stride,
            "PROCESS SCALING"
        )

        results[process_count] = result

    return results


# ============================================================
# KARp-FLATT METRIC
# ============================================================

def calculate_scaling_metrics(results):

    baseline = results[1]["median_ms"]

    metrics = {}

    for p in PROCESS_COUNTS:

        tp = results[p]["median_ms"]

        speedup = baseline / tp

        efficiency = speedup / p

        if p >= 2:

            # Karp-Flatt experimental serial fraction.
            karp_flatt = (
                (1 / speedup) - (1 / p)
            ) / (
                1 - (1 / p)
            )

        else:
            karp_flatt = None

        metrics[p] = {
            "median_ms": tp,
            "speedup": speedup,
            "efficiency": efficiency,
            "karp_flatt": karp_flatt
        }

    return metrics


# ============================================================
# MARKDOWN REPORT GENERATION
# ============================================================

def generate_markdown(
    stride_results,
    scaling_stride1,
    scaling_stride16,
    metrics_stride1,
    metrics_stride16,
    hash_value
):

    lines = []

    lines.append("# Phase 1: False Sharing and Cache-Line Contention\n")

    lines.append("## Experimental Configuration\n")

    lines.append(f"- Student ID: {STUDENT_ID}")
    lines.append("- CPU: AMD Ryzen 7 7735HS")
    lines.append("- Physical cores: 8")
    lines.append("- Logical threads: 16")
    lines.append(f"- Total operations per run: {TOTAL_OPERATIONS:,}")
    lines.append(f"- Warm-up runs: {WARMUP_RUNS}")
    lines.append(f"- Timed runs: {TIMED_RUNS}")
    lines.append("- Shared memory size: 4096 bytes\n")

    # Table 1A
    lines.append("## Table 1A: Stride Sensitivity\n")

    lines.append(
        "| Stride | Separation (bytes) | Median (ms) "
        "| IQR (ms) | Throughput (Mops/s) |"
    )

    lines.append("|---:|---:|---:|---:|---:|")

    for stride in STRIDE_VALUES:

        r = stride_results[stride]

        lines.append(
            f"| {stride} "
            f"| {stride * 8} "
            f"| {r['median_ms']:.3f} "
            f"| {r['iqr_ms']:.3f} "
            f"| {r['throughput_mops']:.3f} |"
        )

    # Table 1B
    lines.append("\n## Table 1B: Karp-Flatt Diagnostic Series\n")

    lines.append(
        "| Processes | Stride | Median (ms) | Speedup "
        "| Efficiency | Karp-Flatt e(p) |"
    )

    lines.append("|---:|---:|---:|---:|---:|---:|")

    for stride, metrics in [
        (1, metrics_stride1),
        (16, metrics_stride16)
    ]:

        for p in PROCESS_COUNTS:

            m = metrics[p]

            e_value = (
                "N/A"
                if m["karp_flatt"] is None
                else f"{m['karp_flatt']:.5f}"
            )

            lines.append(
                f"| {p} "
                f"| {stride} "
                f"| {m['median_ms']:.3f} "
                f"| {m['speedup']:.3f} "
                f"| {m['efficiency']:.3f} "
                f"| {e_value} |"
            )

    # Forensic explanation
    lines.append("\n## Forensic Analysis\n")

    lines.append(
        "The experiment compares adjacent shared-memory writes "
        "with writes separated by larger memory offsets."
    )

    lines.append(
        "The CPU cache coherence protocol operates at cache-line "
        "granularity. Therefore, independent writes to different "
        "64-bit words can still cause cache-line invalidations "
        "when the words occupy the same cache line."
    )

    lines.append(
        "Stride 1 separates worker slots by 8 bytes, while stride "
        "16 separates them by 128 bytes. On this processor, the "
        "cache-line size is 64 bytes."
    )

    lines.append(
        "Under stride 1, multiple workers may repeatedly invalidate "
        "one another's cache lines. Under stride 16, each worker "
        "updates a separate cache-line region, reducing false sharing."
    )

    lines.append(
        "The Karp-Flatt metric helps identify the effective serial "
        "fraction and parallel overhead. A rising value indicates "
        "that additional parallelism is not producing proportional "
        "performance improvements."
    )

    lines.append(
        "\n**Note:** The measured Karp-Flatt value includes "
        "multiprocessing and synchronization overhead; it is an "
        "experimental effective serial fraction, not necessarily "
        "the algorithm's true serial fraction."
    )

    lines.append("\n## Verification\n")

    lines.append(f"```text\n{hash_value}\n```\n")

    lines.append("**Status:** Phase 1 benchmark completed.\n")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PHASE 1: FALSE SHARING FORENSIC EXPERIMENT")
    print("=" * 65)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical cores: {PHYSICAL_CORES}")
    print(f"Logical threads: {LOGICAL_THREADS}")
    print(f"Total operations: {TOTAL_OPERATIONS:,}")

    print("\nStarting experiments...")

    # 1. Stride sensitivity sweep at maximum physical cores.
    stride_results = run_stride_sweep()

    # 2. Full process scaling with severe false sharing.
    scaling_stride1 = run_scaling_sweep(stride=1)

    # 3. Full process scaling with cache-line isolation.
    scaling_stride16 = run_scaling_sweep(stride=16)

    # 4. Calculate Karp-Flatt metrics.
    metrics_stride1 = calculate_scaling_metrics(
        scaling_stride1
    )

    metrics_stride16 = calculate_scaling_metrics(
        scaling_stride16
    )

    # 5. Find largest relative performance drop.
    drops = []

    for previous, current in zip(
        STRIDE_VALUES[:-1],
        STRIDE_VALUES[1:]
    ):

        old_time = stride_results[previous]["median_ms"]
        new_time = stride_results[current]["median_ms"]

        drop_percent = (
            (old_time - new_time) / old_time
        ) * 100

        drops.append(
            (drop_percent, previous, current)
        )

    largest_drop = max(drops, key=lambda item: item[0])

    print("\n" + "=" * 65)
    print("STRIDE TRANSITION ANALYSIS")
    print("=" * 65)

    print(
        f"Largest measured relative time drop: "
        f"{largest_drop[0]:.2f}%"
    )

    print(
        f"Transition: stride {largest_drop[1]} "
        f"({largest_drop[1] * 8} bytes) -> "
        f"stride {largest_drop[2]} "
        f"({largest_drop[2] * 8} bytes)"
    )

    # 6. Generate verification hash.
    # Use stride sweep medians at Pphys = 8.
    hash_value = generate_verify_hash(
        "PHASE1",
        STUDENT_ID,
        round(stride_results[1]["median_ms"], 3),
        round(stride_results[16]["median_ms"], 3)
    )

    # 7. Save JSON data.
    all_results = {
        "student_id": STUDENT_ID,
        "total_operations": TOTAL_OPERATIONS,
        "physical_cores": PHYSICAL_CORES,
        "logical_threads": LOGICAL_THREADS,
        "stride_sweep": stride_results,
        "scaling_stride1": scaling_stride1,
        "scaling_stride16": scaling_stride16,
        "metrics_stride1": metrics_stride1,
        "metrics_stride16": metrics_stride16,
        "verification_hash": hash_value
    }

    json_path = RESULTS_DIR / "phase1_results.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=4)

    # 8. Generate Markdown report.
    report = generate_markdown(
        stride_results,
        scaling_stride1,
        scaling_stride16,
        metrics_stride1,
        metrics_stride16,
        hash_value
    )

    report_path = RESULTS_DIR / "phase1_report.md"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    # 9. Print Table 1A.
    print("\n\nTABLE 1A: STRIDE SENSITIVITY\n")
    print(
        f"{'Stride':<10}"
        f"{'Bytes':<12}"
        f"{'Median ms':<15}"
        f"{'IQR ms':<12}"
        f"{'Mops/s':<12}"
    )

    for stride in STRIDE_VALUES:

        r = stride_results[stride]

        print(
            f"{stride:<10}"
            f"{stride * 8:<12}"
            f"{r['median_ms']:<15.3f}"
            f"{r['iqr_ms']:<12.3f}"
            f"{r['throughput_mops']:<12.3f}"
        )

    # 10. Print Table 1B.
    print("\n\nTABLE 1B: KARp-FLATT DIAGNOSTICS\n")

    for stride, metrics in [
        (1, metrics_stride1),
        (16, metrics_stride16)
    ]:

        print(f"\nSTRIDE = {stride}")

        print(
            f"{'p':<8}"
            f"{'Median ms':<15}"
            f"{'Speedup':<12}"
            f"{'Efficiency':<13}"
            f"{'e(p)':<12}"
        )

        for p in PROCESS_COUNTS:

            m = metrics[p]

            e = (
                "N/A"
                if m["karp_flatt"] is None
                else f"{m['karp_flatt']:.5f}"
            )

            print(
                f"{p:<8}"
                f"{m['median_ms']:<15.3f}"
                f"{m['speedup']:<12.3f}"
                f"{m['efficiency']:<13.3f}"
                f"{e:<12}"
            )

    print("\n" + "=" * 65)
    print("PHASE 1 COMPLETED")
    print("=" * 65)

    print(f"\nJSON results: {json_path}")
    print(f"Markdown report: {report_path}")


if __name__ == "__main__":

    mp.freeze_support()
    main()
