
import ctypes
import multiprocessing as mp
import time
import csv

from multiprocessing import shared_memory
from harness import run_benchmark, emit_verify_tag


# Hardware configuration
STUDENT_ID = 230103201
P_PHYS = 8
P_LOG = 16

# Experiment configuration
M_TOTAL = 16_000_000
SHM_SIZE = 4096
SLOT_COUNT = 512

STRIDES = [1, 2, 4, 8, 16, 32]
PROCESS_COUNTS = [1, 2, 4, 8, 16]

RUNS = 3


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)

    arr = (ctypes.c_uint64 * SLOT_COUNT).from_buffer(shm.buf)

    acc = arr[slot_idx]

    for i in range(iters):
        acc = (
            acc * 6364136223846793005 + (i | 1)
        ) & 0xFFFFFFFFFFFFFFFF

        arr[slot_idx] = acc

    del arr
    shm.close()


def run_case(p: int, stride: int) -> float:
    shm = shared_memory.SharedMemory(
        create=True,
        size=SHM_SIZE
    )

    iters = M_TOTAL // p

    processes = []

    for worker_id in range(p):
        slot_idx = worker_id * stride

        process = mp.Process(
            target=fs_worker,
            args=(shm.name, slot_idx, iters)
        )

        processes.append(process)

    try:
        t0 = time.perf_counter_ns()

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        elapsed_ms = (
            time.perf_counter_ns() - t0
        ) / 1e6

        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed: exit code {process.exitcode}"
                )

        return elapsed_ms

    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()

        shm.close()
        shm.unlink()


def benchmark_case(p: int, stride: int):
    stats = run_benchmark(
        lambda: run_case(p, stride),
        warmup=0,
        runs=RUNS
    )

    throughput = M_TOTAL / (stats.median_ms * 1000)

    return {
        "p": p,
        "stride": stride,
        "median_ms": stats.median_ms,
        "iqr_ms": stats.iqr_ms,
        "min_ms": stats.min_ms,
        "throughput": throughput
    }


def stride_sweep():
    print("\n" + "=" * 65)
    print("PHASE 1.1: FALSE SHARING STRIDE SWEEP")
    print("=" * 65)

    results = {}

    for stride in STRIDES:
        print(f"\nTesting stride={stride} ...", flush=True)

        result = benchmark_case(P_PHYS, stride)
        results[stride] = result

        print(
            f"Stride={stride:2d} | "
            f"Offset={stride * 8:3d} bytes | "
            f"Median={result['median_ms']:.2f} ms | "
            f"IQR={result['iqr_ms']:.2f} ms | "
            f"Throughput={result['throughput']:.3f} Mops/s",
            flush=True
        )

    return results


def karp_flatt(speedup: float, p: int) -> float:
    return (1 / speedup - 1 / p) / (1 - 1 / p)


def process_scaling():
    print("\n" + "=" * 65)
    print("PHASE 1.2: PROCESS SCALING")
    print("=" * 65)

    results = {}

    for stride in [1, 16]:

        print(f"\n--- STRIDE {stride} ---")

        results[stride] = {}

        for p in PROCESS_COUNTS:

            print(
                f"Testing p={p}, stride={stride} ...",
                flush=True
            )

            result = benchmark_case(p, stride)
            results[stride][p] = result

            print(
                f"p={p:2d} | "
                f"Median={result['median_ms']:.2f} ms | "
                f"IQR={result['iqr_ms']:.2f} ms | "
                f"Throughput={result['throughput']:.3f} Mops/s",
                flush=True
            )

    return results


def print_scaling_analysis(results):

    for stride in [1, 16]:

        print("\n" + "=" * 65)
        print(f"KARP-FLATT ANALYSIS: STRIDE {stride}")
        print("=" * 65)

        t1 = results[stride][1]["median_ms"]

        print(
            f"{'p':>5} {'Time(ms)':>14} "
            f"{'Speedup':>12} {'Efficiency':>14} "
            f"{'Karp-Flatt':>14}"
        )

        for p in PROCESS_COUNTS:

            tp = results[stride][p]["median_ms"]

            speedup = t1 / tp
            efficiency = speedup / p

            e = karp_flatt(speedup, p) if p >= 2 else None

            e_text = f"{e:.6f}" if e is not None else "N/A"

            print(
                f"{p:5d} {tp:14.2f} "
                f"{speedup:12.4f} "
                f"{efficiency:14.4f} "
                f"{e_text:>14}"
            )


def save_results(stride_results, scaling_results):

    with open(
        "phase1_results.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Experiment",
            "Stride",
            "Processes",
            "Median_ms",
            "IQR_ms",
            "Throughput_Mops"
        ])

        for stride, result in stride_results.items():
            writer.writerow([
                "Stride Sweep",
                stride,
                P_PHYS,
                result["median_ms"],
                result["iqr_ms"],
                result["throughput"]
            ])

        for stride, process_data in scaling_results.items():
            for p, result in process_data.items():

                writer.writerow([
                    "Process Scaling",
                    stride,
                    p,
                    result["median_ms"],
                    result["iqr_ms"],
                    result["throughput"]
                ])

    print("\nResults saved to phase1_results.csv")


def main():

    print("PHASE 1: FALSE SHARING & KAR​​P-FLATT")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical cores: {P_PHYS}")
    print(f"Logical threads: {P_LOG}")
    print(f"Total operations: {M_TOTAL:,}")
    print(f"Runs per configuration: {RUNS}")

    stride_results = stride_sweep()

    scaling_results = process_scaling()

    print_scaling_analysis(scaling_results)

    save_results(stride_results, scaling_results)

    median_stride_1 = stride_results[1]["median_ms"]
    median_stride_16 = stride_results[16]["median_ms"]

    print("\n" + "=" * 65)
    print("PHASE 1 VERIFICATION")
    print("=" * 65)

    print(
        emit_verify_tag(
            "PHASE1",
            STUDENT_ID,
            median_stride_1,
            median_stride_16
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
