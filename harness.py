# harness.py

import os
import sys
import time
import platform
import multiprocessing as mp
import hashlib
import statistics


def get_hardware_info():
    """
    Collect basic hardware and operating system information.
    Uses only Python standard library.
    """

    return {
        "cpu_model": platform.processor() or platform.machine(),
        "physical_cores": 8,  # AMD Ryzen 7 7735HS
        "logical_threads": mp.cpu_count(),
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "python": platform.python_version(),
    }


def timer_ns():
    """Return monotonic time in nanoseconds."""
    return time.perf_counter_ns()


def calculate_statistics(samples):
    """
    Calculate median and interquartile range.
    Samples are expected to be execution times in milliseconds.
    """

    ordered = sorted(samples)

    median = statistics.median(ordered)

    q1 = statistics.median(ordered[:len(ordered) // 2])
    q3 = statistics.median(
        ordered[(len(ordered) + 1) // 2:]
    )

    return {
        "median_ms": median,
        "iqr_ms": q3 - q1,
        "min_ms": min(ordered),
        "max_ms": max(ordered),
        "samples_ms": ordered,
    }


def run_benchmark(
    function,
    *args,
    warmup_runs=2,
    timed_runs=7,
    **kwargs
):
    """
    Execute warm-up runs followed by timed runs.

    Returns median, IQR, minimum, maximum and raw samples.
    """

    if warmup_runs < 0 or timed_runs < 1:
        raise ValueError("Invalid benchmark run count.")

    # Warm-up
    for _ in range(warmup_runs):
        function(*args, **kwargs)

    samples = []

    # Timed measurements
    for _ in range(timed_runs):
        start = timer_ns()

        function(*args, **kwargs)

        end = timer_ns()

        elapsed_ms = (end - start) / 1_000_000
        samples.append(elapsed_ms)

    return calculate_statistics(samples)


def generate_verify_hash(phase, student_seed, *values):
    """
    Generate a hardware-bound SHA-256 verification fingerprint.

    The hash depends on:
    - Student ID
    - Phase
    - Hardware fingerprint
    - Python version
    - Benchmark values
    """

    hardware = get_hardware_info()

    fingerprint_data = (
        f"STUDENT={student_seed}|"
        f"PHASE={phase}|"
        f"CPU={hardware['cpu_model']}|"
        f"CORES={hardware['physical_cores']}|"
        f"THREADS={hardware['logical_threads']}|"
        f"MACHINE={hardware['machine']}|"
        f"OS={hardware['system']}|"
        f"PYTHON={hardware['python']}|"
        f"VALUES={values}"
    )

    digest = hashlib.sha256(
        fingerprint_data.encode("utf-8")
    ).hexdigest()

    result = f"VERIFY_HASH[{phase}]: {digest}"

    print(result)

    return result


if __name__ == "__main__":
    print("Parallel Computing Benchmark Harness")
    print("-------------------------------------")

    for key, value in get_hardware_info().items():
        print(f"{key}: {value}")