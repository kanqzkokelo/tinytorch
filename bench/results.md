# CPU matmul benchmark results

- Date: 2026-08-21
- CPU: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz (12 logical cores)
- GPU clocks (context): 1500 MHz, 6001 MHz
- Protocol: warmup 5, median of 20 runs, fp32
- NumPy backend: OpenBLAS (pinned to 1 thread for np-1t row)
- OMP kernel threads: 6

| N | naive | AVX2 1T | AVX2 OMP | NumPy 1T | NumPy MT | AVX2/naive | AVX2/NumPy1T |
|---|-------|---------|----------|----------|----------|------------|--------------|
| 256 | 7.8 | 54.7 | 81.9 | 96.1 | nan | 7.0x | 0.57x |
| 512 | 7.2 | 64.0 | 115.2 | 80.4 | nan | 8.9x | 0.80x |
| 1024 | 7.0 | 64.4 | 183.3 | 101.5 | nan | 9.2x | 0.63x |

GFLOPS. Gate (AVX2 1T >= NumPy 1T @ 1024^3): FAIL
