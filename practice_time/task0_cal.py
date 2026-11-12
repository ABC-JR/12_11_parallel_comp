
import multiprocessing as mp
import time

from harness import emit_verify_tag


STUDENT_ID = 230103201
CALLS = 1_000_000


def measure_timer_latency():
    start = time.perf_counter_ns()

    for _ in range(CALLS):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    average_latency = (end - start) / CALLS

    return average_latency


def main():
    cpu_count = mp.cpu_count()

    print("=" * 50)
    print("PHASE 0: HARDWARE CALIBRATION")
    print("=" * 50)

    print(f"Student ID: {STUDENT_ID}")
    print(f"CPU Count: {cpu_count}")
    print(f"Physical Cores: 8")
    print(f"Logical Threads: {cpu_count}")

    print(f"\nTimer calls: {CALLS:,}")

    latency = measure_timer_latency()

    print(f"Average perf_counter_ns latency: {latency:.2f} ns")

    verify = emit_verify_tag(
        phase="PHASE0",
        seed=STUDENT_ID,
        m1=latency,
        m2=0.0
    )

    print("\n" + verify)


if __name__ == "__main__":
    main()
