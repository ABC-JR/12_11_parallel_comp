
import multiprocessing as mp
import math
import time
import csv

from harness import emit_verify_tag


STUDENT_ID = 230103201

P_LOG = mp.cpu_count()

STRONG_N = 1_200_000
WEAK_N0 = 200_000

RUNS = 3
WARMUP = 1


def heavy_calculation(x):
    result = 0.0

    for j in range(30):
        result += math.sin(x + j) * math.sqrt(x + j + 1)

    return result


def stage_a(n):
    result = 0.0

    count = max(1, int(n * 0.05))

    for i in range(count):
        x = i * 0.001 + 1.0

        for j in range(10):
            result += math.sin(x + j) * math.sqrt(x + j + 1)

    return result


def stage_b_worker(args):
    start, end = args

    result = 0.0

    for i in range(start, end):
        x = i * 0.001 + 1.0
        result += heavy_calculation(x)

    return result


def create_chunks(n, p):
    chunks = []

    chunk_size = (n + p - 1) // p

    for worker in range(p):
        start = worker * chunk_size
        end = min(start + chunk_size, n)

        if start < n:
            chunks.append((start, end))

    return chunks


def run_trial(pool, n, p):

    t0 = time.perf_counter()

    stage_a_result = stage_a(n)

    chunks = create_chunks(n, p)

    stage_b_results = pool.map(stage_b_worker, chunks)

    total_result = stage_a_result + sum(stage_b_results)

    elapsed = time.perf_counter() - t0

    return elapsed, total_result


def benchmark(pool, n, p):

    for _ in range(WARMUP):
        run_trial(pool, n, p)

    samples = []
    checksum = None

    for _ in range(RUNS):

        elapsed, result = run_trial(pool, n, p)

        samples.append(elapsed)

        checksum = result

    samples.sort()

    median = samples[len(samples) // 2]

    q1 = samples[len(samples) // 4]
    q3 = samples[(3 * len(samples)) // 4]

    iqr = q3 - q1

    return median, iqr, checksum


def strong_scaling():

    print("\n" + "=" * 65)
    print("PHASE 2.1: STRONG SCALING")
    print("=" * 65)

    results = {}

    for p in range(1, P_LOG + 1):

        print(f"Testing strong scaling p={p}...", flush=True)

        with mp.Pool(processes=p) as pool:

            median, iqr, checksum = benchmark(
                pool,
                STRONG_N,
                p
            )

        results[p] = {
            "median": median,
            "iqr": iqr,
            "checksum": checksum
        }

        print(
            f"p={p:2d} | "
            f"Median={median:.4f} s | "
            f"IQR={iqr:.4f} s"
        )

    t1 = results[1]["median"]
        # Measure serial Stage A separately
    t_stage_a = stage_a_timing(STRONG_N)

    serial_fraction = t_stage_a / t1

    amdahl_cap = 1 / serial_fraction

    print("\nAMDAHL'S LAW ANALYSIS")
    print(f"Stage A time: {t_stage_a:.6f} s")
    print(f"Serial fraction: {serial_fraction:.6f}")
    print(f"Amdahl's theoretical cap: {amdahl_cap:.4f}")

    print("\nSTRONG SCALING SUMMARY")

    print(
        f"{'p':>5} {'Time(s)':>12} "
        f"{'Speedup':>12} {'Efficiency':>14}"
    )

    for p in range(1, P_LOG + 1):

        tp = results[p]["median"]

        speedup = t1 / tp
        efficiency = speedup / p

        results[p]["speedup"] = speedup
        results[p]["efficiency"] = efficiency

        print(
            f"{p:5d} {tp:12.4f} "
            f"{speedup:12.4f} "
            f"{efficiency:14.4f}"
        )

    results["amdahl_cap"] = amdahl_cap
    results["stage_a_serial"] = t_stage_a

    return results


def weak_scaling():

    print("\n" + "=" * 65)
    print("PHASE 2.2: WEAK SCALING")
    print("=" * 65)

    results = {}

    for p in range(1, P_LOG + 1):

        n = p * WEAK_N0

        print(
            f"Testing weak scaling p={p}, N={n:,}...",
            flush=True
        )

        with mp.Pool(processes=p) as pool:

            median, iqr, checksum = benchmark(
                pool,
                n,
                p
            )

        stage_a_time = stage_a_timing(n)

        alpha = stage_a_time / median

        weak_efficiency = None
        gustafson_speedup = None

        results[p] = {
            "n": n,
            "median": median,
            "iqr": iqr,
            "stage_a_time": stage_a_time,
            "alpha": alpha
        }

        print(
            f"p={p:2d} | "
            f"N={n:,} | "
            f"Time={median:.4f} s | "
            f"Stage A={stage_a_time:.4f} s | "
            f"Alpha={alpha:.4f}"
        )

    t1 = results[1]["median"]

    print("\nWEAK SCALING SUMMARY")

    print(
        f"{'p':>5} {'Time(s)':>12} "
        f"{'Efficiency':>14} {'Gustafson':>14}"
    )

    for p in range(1, P_LOG + 1):

        tp = results[p]["median"]

        efficiency = t1 / tp

        alpha = results[p]["alpha"]

        gustafson = p - alpha * (p - 1)

        results[p]["weak_efficiency"] = efficiency
        results[p]["gustafson"] = gustafson

        print(
            f"{p:5d} {tp:12.4f} "
            f"{efficiency:14.4f} "
            f"{gustafson:14.4f}"
        )

    return results


def stage_a_timing(n):

    samples = []

    for _ in range(RUNS):

        t0 = time.perf_counter()

        stage_a(n)

        samples.append(time.perf_counter() - t0)

    samples.sort()

    return samples[len(samples) // 2]


def save_results(strong, weak):
    filename = "phase2_results.csv"

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Experiment",
            "Processes",
            "N",
            "Median_s",
            "IQR_s",
            "Speedup",
            "Efficiency",
            "Stage_A_s",
            "Alpha",
            "Gustafson"
        ])

        # Strong scaling
        for p, r in strong.items():
            if not isinstance(r, dict):
                continue

            writer.writerow([
                "Strong",
                p,
                STRONG_N,
                r["median"],
                r["iqr"],
                r["speedup"],
                r["efficiency"],
                "",
                "",
                ""
            ])

        # Weak scaling
        for p, r in weak.items():
            if not isinstance(r, dict):
                continue

            writer.writerow([
                "Weak",
                p,
                r["n"],
                r["median"],
                r["iqr"],
                "",
                r["efficiency"],
                r["stage_a"],
                r["alpha"],
                r["gustafson"]
            ])

    print(f"\nResults saved to {filename}")




def main():

    print("PHASE 2: STRONG VS WEAK SCALING")

    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical CPUs: {P_LOG}")

    print(f"Strong N: {STRONG_N:,}")
    print(f"Weak N0: {WEAK_N0:,}")

    strong = strong_scaling()
    # Architectural inflection: 8 -> 9 processes

    t7 = strong[7]["median"]
    t8 = strong[8]["median"]
    t9 = strong[9]["median"]

    gain_before = (t7 / t8) - 1
    gain_after = (t8 / t9) - 1

    if gain_before != 0:
        drop = (
            (gain_before - gain_after)
            / abs(gain_before)
        ) * 100
    else:
        drop = 0

    print("\nARCHITECTURAL INFLECTION: 8 -> 9")
    print(f"Marginal gain 7 -> 8: {gain_before * 100:.2f}%")
    print(f"Marginal gain 8 -> 9: {gain_after * 100:.2f}%")
    print(f"Marginal gain change: {drop:.2f}%")

    weak = weak_scaling()

    save_results(strong, weak)

    strong_speedup = strong[P_LOG]["speedup"]

    gustafson_speedup = weak[P_LOG]["gustafson"]

    print("\n" + "=" * 65)
    print("PHASE 2 VERIFICATION")
    print("=" * 65)

    print(
        emit_verify_tag(
            "PHASE2",
            STUDENT_ID,
            strong_speedup,
            gustafson_speedup
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
