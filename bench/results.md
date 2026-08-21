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

## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 512 | 92579.9 | 93466.4 | 421.8 | 28154.1 | 1.0x | 332.0% | 1.5% |
| 1024 | 289087.1 | 289087.2 | 1413.3 | 579148.9 | 1.0x | 49.9% | 0.2% |
| 2048 | 5965231.4 | 6039679.5 | 2803.8 | 4137235.2 | 1.0x | 146.0% | 0.1% |


## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 512 | 369.6 | 1705.0 | 2038.9 | 2824.5 | 4.6x | 60.4% | 72.2% |


## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 1024 | 375.3 | 1847.6 | 1325.3 | 4528.5 | 4.9x | 40.8% | 29.3% |


## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 1024 | 376.0 | 7559.6 | 1443.4 | 4529.2 | 20.1x | 166.9% | 31.9% |


## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 512 | 369.6 | 1679.6 | 2044.7 | 2826.7 | 4.5x | 59.4% | 72.3% |
| 1024 | 376.6 | 2075.8 | 1523.3 | 4545.0 | 5.5x | 45.7% | 33.5% |
| 2048 | nan | nan | nan | nan | nanx | nan% | nan% |


## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 1024 | 374.2 | 2934.5 | 1314.9 | 4551.5 | 7.8x | 64.5% | 28.9% |


## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 512 | 369.9 | 1599.0 | 2040.4 | 2824.1 | 4.3x | 56.6% | 72.2% |
| 1024 | 375.9 | 2179.7 | 1559.1 | 4557.4 | 5.8x | 47.8% | 34.2% |
| 2048 | nan | nan | nan | nan | nanx | nan% | nan% |


## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 1024 | 376.7 | 1095.5 | 1402.0 | 4583.4 | 2.9x | 23.9% | 30.6% |


## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 1024 | 376.7 | 1321.9 | 1395.9 | 4570.1 | 3.5x | 28.9% | 30.5% |


## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 512 | 370.4 | 1232.4 | 2054.8 | 2838.0 | 3.3x | 43.4% | 72.4% |
| 1024 | 376.5 | 1316.5 | 1499.7 | 4564.8 | 3.5x | 28.8% | 32.9% |
| 2048 | nan | nan | nan | nan | nanx | nan% | nan% |


## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 512 | 370.3 | 1680.8 | 2050.6 | 2832.6 | 4.5x | 59.3% | 72.4% |
| 1024 | 376.1 | 2090.3 | 1551.4 | 4583.6 | 5.6x | 45.6% | 33.8% |
| 2048 | nan | nan | nan | nan | nanx | nan% | nan% |


## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 512 | 370.2 | 1680.2 | 2055.9 | 2842.8 | 4.5x | 59.1% | 72.3% |
| 1024 | 374.2 | 1780.9 | 1468.7 | 4556.2 | 4.8x | 39.1% | 32.2% |
| 2048 | 449.7 | 2097.3 | 2837.9 | 4193.4 | 4.7x | 50.0% | 67.7% |


## CUDA matmul ladder (RTX 3050 laptop, sm_86)

| N | naive | tiled | WMMA fp16 | cuBLAS | tiled/naive | tiled %cuBLAS | wmma %cuBLAS |
|---|-------|-------|-----------|--------|-------------|--------------|---------------|
| 512 | 369.5 | 1671.6 | 2027.5 | 2810.6 | 4.5x | 59.5% | 72.1% |
| 1024 | 385.9 | 1837.5 | 1421.8 | 3838.5 | 4.8x | 47.9% | 37.0% |
| 2048 | 450.3 | 2112.1 | 2872.8 | 4168.7 | 4.7x | 50.7% | 68.9% |

