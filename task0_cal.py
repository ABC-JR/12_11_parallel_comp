# task0_cal.py

import time
import multiprocessing as mp
import sys
import platform

from harness import (
    get_hardware_info,
    generate_verify_hash
)


STUDENT_ID = "230103201"

TOTAL_READS = 2_000_000


def measure_clock_latency():
    """
    Measure average perf_counter_ns() call latency.

    Performs exactly 2,000,000 clock reads.
    """

    print("\nStarting clock calibration...")

    # Warm-up
    for _ in range(10_000):
        time.perf_counter_ns()

    start = time.perf_counter_ns()

    for _ in range(TOTAL_READS):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_ns = end - start
    average_ns = total_ns / TOTAL_READS

    return total_ns, average_ns


def main():

    print("=" * 55)
    print("PHASE 0: HARDWARE TOPOLOGY & CALIBRATION")
    print("=" * 55)

    # 1. Hardware information
    hardware = get_hardware_info()

    print("\n[1] HARDWARE INFORMATION")

    for key, value in hardware.items():
        print(f"{key}: {value}")

    # 2. Multiprocessing CPU count
    cpu_count = mp.cpu_count()

    print("\n[2] MULTIPROCESSING INFORMATION")
    print(f"mp.cpu_count(): {cpu_count}")

    # 3. Python and OS information
    print("\n[3] SOFTWARE ENVIRONMENT")
    print(f"Python version: {sys.version}")
    print(f"Operating system: {platform.platform()}")
    print(f"Python executable: {sys.executable}")

    # 4. Clock calibration
    print("\n[4] CLOCK LATENCY MEASUREMENT")

    total_ns, average_ns = measure_clock_latency()

    print(f"Total clock reads: {TOTAL_READS:,}")
    print(f"Total elapsed time: {total_ns:,} ns")
    print(f"Average clock read latency: {average_ns:.3f} ns")

    # 5. Verification hash
    print("\n[5] VERIFICATION")

    generate_verify_hash(
        "PHASE0",
        STUDENT_ID,
        cpu_count,
        TOTAL_READS,
        round(average_ns, 3)
    )

    print("\n" + "=" * 55)
    print("PHASE 0 CALIBRATION COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    mp.freeze_support()
    main()