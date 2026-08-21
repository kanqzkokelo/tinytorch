# CPU matmul benchmark results

- Date: 2026-08-21
- CPU: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz (12 logical cores)
- GPU clocks (context): 405 MHz, 405 MHz
- Protocol: warmup 5, median of 20 runs, fp32
- NumPy backend: OpenBLAS (pinned to 1 thread for np-1t row)
- OMP kernel threads: 6

| N | naive | AVX2 1T | AVX2 OMP | NumPy 1T | NumPy MT | AVX2/naive | AVX2/NumPy1T |
|---|-------|---------|----------|----------|----------|------------|--------------|
| 256 | 7.5 | 70.8 | 134.6 | 99.0 | nan | 9.4x | 0.71x |
| 512 | 7.2 | 71.6 | 189.4 | 82.7 | nan | 10.0x | 0.86x |
| 1024 | 7.0 | 70.5 | 174.0 | 90.5 | nan | 10.1x | 0.78x |

GFLOPS. Gate (AVX2 1T >= NumPy 1T @ 1024^3): FAIL

## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 512 | 369.1 | 1502.1 | 2043.7 | 2819.2 | 4.1x | 53.3% | 72.5% |
| 1024 | 376.3 | 2803.2 | 1418.1 | 4580.5 | 7.5x | 61.2% | 31.0% |
| 2048 | 426.7 | 3506.1 | 2850.7 | 4226.5 | 8.2x | 83.0% | 67.4% |


## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 512 | 370.1 | 1352.1 | 2043.4 | 2833.1 | 3.7x | 47.7% | 72.1% |
| 1024 | 375.9 | 1866.8 | 1544.2 | 4558.0 | 5.0x | 41.0% | 33.9% |
| 2048 | 426.8 | 2419.1 | 2872.5 | 4234.1 | 5.7x | 57.1% | 67.8% |


## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 512 | 88.6 | 1358.3 | 2050.1 | 3233.2 | 15.3x | 42.0% | 63.4% |
| 1024 | 108.9 | 2207.2 | 1639.4 | 4558.3 | 20.3x | 48.4% | 36.0% |
| 2048 | 111.2 | 2410.9 | 2859.2 | 4208.7 | 21.7x | 57.3% | 67.9% |

