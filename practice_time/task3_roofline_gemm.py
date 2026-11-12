
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import csv
import statistics
import math
from harness import emit_verify_tag


# ============================================================
# CONFIGURATION
# ============================================================

STUDENT_ID = 230103201

P_PHYS = 8
N = 256

MEMORY_SIZE = 64 * 1024 * 1024  # 64 MB per buffer
COMPUTE_FLOPS_PER_WORKER = 60_000_000

BLOCK_SIZES = [8, 16, 32, 64, 96, 128]

RUNS = 3


# ============================================================
# PART 3A.1: SUSTAINED DRAM BANDWIDTH
# ============================================================

def memory_copy_worker(args):
    name_a, name_b, start, end = args

    shm_a = shared_memory.SharedMemory(name=name_a)
    shm_b = shared_memory.SharedMemory(name=name_b)

    mv_a = memoryview(shm_a.buf)
    mv_b = memoryview(shm_b.buf)

    mv_a[start:end] = mv_b[start:end]

    del mv_a
    del mv_b

    shm_a.close()
    shm_b.close()


def measure_bandwidth():

    print("\n" + "=" * 65)
    print("PART 3A.1: SUSTAINED DRAM BANDWIDTH")
    print("=" * 65)

    shm_a = shared_memory.SharedMemory(create=True, size=MEMORY_SIZE)
    shm_b = shared_memory.SharedMemory(create=True, size=MEMORY_SIZE)

    chunk_size = MEMORY_SIZE // P_PHYS

    tasks = []

    for worker in range(P_PHYS):
        start = worker * chunk_size
        end = start + chunk_size

        tasks.append((
            shm_a.name,
            shm_b.name,
            start,
            end
        ))

    samples = []

    try:
        with mp.Pool(processes=P_PHYS) as pool:

            # Warm-up
            pool.map(memory_copy_worker, tasks)

            for run in range(RUNS):

                t0 = time.perf_counter()

                pool.map(memory_copy_worker, tasks)

                elapsed = time.perf_counter() - t0
                samples.append(elapsed)

                print(
                    f"Run {run + 1}: "
                    f"{elapsed:.6f} seconds"
                )

        median = statistics.median(samples)

        total_bytes = 2 * MEMORY_SIZE

        bandwidth = total_bytes / median / 1e9

        print(f"\nMedian time: {median:.6f} s")
        print(f"Sustained bandwidth: {bandwidth:.4f} GB/s")

        return median, bandwidth

    finally:
        shm_a.close()
        shm_b.close()

        shm_a.unlink()
        shm_b.unlink()


# ============================================================
# PART 3A.2: COMPUTE CEILING
# ============================================================

def compute_worker(iterations):

    # 16 independent accumulators
    x0 = x1 = x2 = x3 = 0.0
    x4 = x5 = x6 = x7 = 0.0
    x8 = x9 = x10 = x11 = 0.0
    x12 = x13 = x14 = x15 = 0.0

    a = 1.000001
    b = 1.000002

    for _ in range(iterations):

        x0 += a * b
        x1 += a * b
        x2 += a * b
        x3 += a * b

        x4 += a * b
        x5 += a * b
        x6 += a * b
        x7 += a * b

        x8 += a * b
        x9 += a * b
        x10 += a * b
        x11 += a * b

        x12 += a * b
        x13 += a * b
        x14 += a * b
        x15 += a * b

    return (
        x0 + x1 + x2 + x3 +
        x4 + x5 + x6 + x7 +
        x8 + x9 + x10 + x11 +
        x12 + x13 + x14 + x15
    )


def measure_compute_ceiling():

    print("\n" + "=" * 65)
    print("PART 3A.2: COMPUTE CEILING")
    print("=" * 65)

    # 16 accumulators * 2 FLOPs per accumulator per iteration
    iterations = COMPUTE_FLOPS_PER_WORKER // 32

    samples = []

    with mp.Pool(processes=P_PHYS) as pool:

        for run in range(RUNS):

            t0 = time.perf_counter()

            pool.map(
                compute_worker,
                [iterations] * P_PHYS
            )

            elapsed = time.perf_counter() - t0

            samples.append(elapsed)

            print(
                f"Run {run + 1}: "
                f"{elapsed:.6f} seconds"
            )

    median = statistics.median(samples)

    total_flops = COMPUTE_FLOPS_PER_WORKER * P_PHYS

    compute_gflops = total_flops / median / 1e9

    print(f"\nMedian time: {median:.6f} s")
    print(f"Compute ceiling: {compute_gflops:.4f} GFLOP/s")

    return median, compute_gflops


# ============================================================
# PART 3B: GEMM WORKER
# ============================================================

def gemm_worker(args):

    name_a, name_b, name_c, n, row_start, row_end, variant, bs = args

    shm_a = shared_memory.SharedMemory(name=name_a)
    shm_b = shared_memory.SharedMemory(name=name_b)
    shm_c = shared_memory.SharedMemory(name=name_c)

    A = memoryview(shm_a.buf).cast("d")
    B = memoryview(shm_b.buf).cast("d")
    C = memoryview(shm_c.buf).cast("d")

    if variant == 1:

        # Variant 1: Naive i-j-k
        for i in range(row_start, row_end):
            for j in range(n):

                total = 0.0

                for k in range(n):
                    total += A[i * n + k] * B[k * n + j]

                C[i * n + j] = total

    elif variant == 2:

        # Variant 2: Reordered i-k-j
        for i in range(row_start, row_end):

            for k in range(n):

                a_ik = A[i * n + k]

                for j in range(n):
                    C[i * n + j] += a_ik * B[k * n + j]

    elif variant == 3:

        # Variant 3: Six-loop cache-blocked GEMM
        for ii in range(row_start, row_end, bs):

            i_end = min(ii + bs, row_end)

            for kk in range(0, n, bs):

                k_end = min(kk + bs, n)

                for jj in range(0, n, bs):

                    j_end = min(jj + bs, n)

                    # i-k-j micro-kernel
                    for i in range(ii, i_end):

                        for k in range(kk, k_end):

                            a_ik = A[i * n + k]

                            for j in range(jj, j_end):

                                C[i * n + j] += (
                                    a_ik * B[k * n + j]
                                )

    del A
    del B
    del C

    shm_a.close()
    shm_b.close()
    shm_c.close()


# ============================================================
# GEMM HELPERS
# ============================================================

def create_tasks(names, n, variant, bs):

    name_a, name_b, name_c = names

    tasks = []

    chunk = (n + P_PHYS - 1) // P_PHYS

    for worker in range(P_PHYS):

        start = worker * chunk
        end = min(start + chunk, n)

        tasks.append((
            name_a,
            name_b,
            name_c,
            n,
            start,
            end,
            variant,
            bs
        ))

    return tasks


def clear_matrix(shm_c):

    C = memoryview(shm_c.buf).cast("d")

    for i in range(N * N):
        C[i] = 0.0

    del C


def benchmark_gemm(pool, names, shm_c, variant, bs=0):

    tasks = create_tasks(names, N, variant, bs)

    samples = []

    for _ in range(RUNS):

        # Reset C outside the timed region
        clear_matrix(shm_c)

        t0 = time.perf_counter()

        pool.map(gemm_worker, tasks)

        elapsed = time.perf_counter() - t0

        samples.append(elapsed)

    return statistics.median(samples)


def get_matrix_values(shm_c):

    C = memoryview(shm_c.buf).cast("d")

    values = list(C)

    del C

    return values


def max_difference(shm_c, reference):

    C = memoryview(shm_c.buf).cast("d")

    difference = max(
        abs(C[i] - reference[i])
        for i in range(N * N)
    )

    del C

    return difference


# ============================================================
# PART 3B: PROGRESSIVE GEMM AND TILE SWEEP
# ============================================================

def run_gemm():

    print("\n" + "=" * 65)
    print("PART 3B: PROGRESSIVE GEMM")
    print("=" * 65)

    size = N * N * 8

    shm_a = shared_memory.SharedMemory(create=True, size=size)
    shm_b = shared_memory.SharedMemory(create=True, size=size)
    shm_c = shared_memory.SharedMemory(create=True, size=size)

    A = memoryview(shm_a.buf).cast("d")
    B = memoryview(shm_b.buf).cast("d")

    # Deterministic input matrices
    for i in range(N):
        for j in range(N):

            A[i * N + j] = (
                ((i * 7 + j * 3) % 100) / 100.0
            )

            B[i * N + j] = (
                ((i * 5 + j * 11) % 100) / 100.0
            )

    del A
    del B

    names = (shm_a.name, shm_b.name, shm_c.name)

    results = []

    try:

        with mp.Pool(processes=P_PHYS) as pool:

            # ------------------------------------------------
            # Variant 1
            # ------------------------------------------------

            print("\nVariant 1: Naive i-j-k")

            t1 = benchmark_gemm(
                pool, names, shm_c, variant=1
            )

            gflops1 = (2 * N**3) / t1 / 1e9

            print(f"Median: {t1:.6f} s")
            print(f"Performance: {gflops1:.6f} GFLOP/s")

            results.append({
                "variant": "Naive",
                "BS": "",
                "median": t1,
                "gflops": gflops1,
                "tile_bytes": "",
                "intensity": "",
                "max_error": ""
            })

            # ------------------------------------------------
            # Variant 2
            # ------------------------------------------------

            print("\nVariant 2: Reordered i-k-j")

            t2 = benchmark_gemm(
                pool, names, shm_c, variant=2
            )

            gflops2 = (2 * N**3) / t2 / 1e9

            reference = get_matrix_values(shm_c)

            print(f"Median: {t2:.6f} s")
            print(f"Performance: {gflops2:.6f} GFLOP/s")

            results.append({
                "variant": "Reordered",
                "BS": "",
                "median": t2,
                "gflops": gflops2,
                "tile_bytes": "",
                "intensity": "",
                "max_error": ""
            })

            # ------------------------------------------------
            # Variant 3: Tile sweep
            # ------------------------------------------------

            print("\nVariant 3: Cache-blocked GEMM")

            blocked_results = []

            for bs in BLOCK_SIZES:

                t3 = benchmark_gemm(
                    pool,
                    names,
                    shm_c,
                    variant=3,
                    bs=bs
                )

                gflops3 = (2 * N**3) / t3 / 1e9

                tile_bytes = 3 * bs**2 * 8

                intensity = bs / 12.0

                error = max_difference(shm_c, reference)

                print(
                    f"BS={bs:3d} | "
                    f"Time={t3:.6f} s | "
                    f"GFLOP/s={gflops3:.6f} | "
                    f"Tile={tile_bytes / 1024:.2f} KB | "
                    f"I={intensity:.4f} | "
                    f"Error={error:.3e}"
                )

                result = {
                    "variant": "Blocked",
                    "BS": bs,
                    "median": t3,
                    "gflops": gflops3,
                    "tile_bytes": tile_bytes,
                    "intensity": intensity,
                    "max_error": error
                }

                results.append(result)
                blocked_results.append(result)

            # ------------------------------------------------
            # Select optimal tile
            # ------------------------------------------------

            best = max(
                blocked_results,
                key=lambda item: item["gflops"]
            )

            print("\nOPTIMAL TILE")
            print(f"BS*: {best['BS']}")
            print(f"GFLOP/s: {best['gflops']:.6f}")
            print(f"Tile footprint: {best['tile_bytes']} bytes")
            print(f"Operational intensity: {best['intensity']:.4f}")

            print("\nNUMERICAL VERIFICATION")

            for item in blocked_results:

                if item["max_error"] >= 1e-9:
                    raise RuntimeError(
                        f"Verification failed for BS={item['BS']}: "
                        f"error={item['max_error']}"
                    )

            print("All blocked variants passed verification.")

        return results, best, gflops1, gflops2

    finally:

        shm_a.close()
        shm_b.close()
        shm_c.close()

        shm_a.unlink()
        shm_b.unlink()
        shm_c.unlink()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    filename = "phase3_results.csv"

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Variant",
            "BS",
            "Median_s",
            "GFLOPS",
            "Tile_Bytes",
            "Operational_Intensity",
            "Max_Error"
        ])

        for item in results:

            writer.writerow([
                item["variant"],
                item["BS"],
                item["median"],
                item["gflops"],
                item["tile_bytes"],
                item["intensity"],
                item["max_error"]
            ])

    print(f"\nResults saved to {filename}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("PHASE 3: EMPIRICAL ROOFLINE & CACHE-BLOCKED GEMM")
    print("=" * 65)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical workers: {P_PHYS}")
    print(f"Logical CPUs: {mp.cpu_count()}")
    print(f"Matrix size: {N} x {N}")

    # Part 3A
    _, bandwidth = measure_bandwidth()

    _, compute = measure_compute_ceiling()

    ridge_point = compute / bandwidth

    print("\nROOFLINE CEILINGS")
    print(f"Bandwidth: {bandwidth:.4f} GB/s")
    print(f"Compute ceiling: {compute:.4f} GFLOP/s")
    print(f"Ridge point: {ridge_point:.4f} FLOPs/byte")

    # Part 3B
    results, best, gflops1, gflops2 = run_gemm()

    save_results(results)

    print("\n" + "=" * 65)
    print("PHASE 3 SUMMARY")
    print("=" * 65)

    print(f"Bandwidth: {bandwidth:.4f} GB/s")
    print(f"Compute ceiling: {compute:.4f} GFLOP/s")
    print(f"Ridge point: {ridge_point:.4f} FLOPs/byte")
    print(f"Variant 1: {gflops1:.6f} GFLOP/s")
    print(f"Variant 2: {gflops2:.6f} GFLOP/s")
    print(f"Variant 3 best: {best['gflops']:.6f} GFLOP/s")
    print(f"Optimal BS: {best['BS']}")

    print(
        emit_verify_tag(
            "PHASE3",
            STUDENT_ID,
            gflops1,
            best["gflops"]
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
