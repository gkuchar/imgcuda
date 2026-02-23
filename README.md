# imgcuda — CUDA GPU Image Processing Library
**Griffin Kuchar** | C++ · CUDA · Parallel Computing · Computer Vision

📧 [griffin.kuchar@gmail.com](mailto:griffin.kuchar@gmail.com) · 💼 [linkedin.com/in/griffin-kuchar-95081124b](https://www.linkedin.com/in/griffin-kuchar-95081124b/)

A high-performance image processing library built in C++ and CUDA, demonstrating GPU parallel programming, memory management, and software engineering best practices. Implements 8 image filters that run on NVIDIA GPUs with CPU reference implementations for correctness validation.

---

## Highlights

- **8 GPU-accelerated image filters** including convolution-based edge detection, blurring, and color transforms
- **CUDA kernel design** with 2D thread/block grid mapping for efficient pixel-parallel execution
- **Performance benchmarking** using CUDA events to measure and report H2D transfer, kernel execution, and D2H transfer times independently
- **CPU/GPU correctness verification** with pixel-level comparison and floating-point tolerance
- **Clean library API** with separate namespaces for GPU (`imgcuda`) and CPU reference (`imgcpu`) implementations
- **Robust CLI** with full error handling, input validation, and timing reports

---

## Tech Stack

| Area | Technology |
|------|-----------|
| Language | C++17 |
| GPU Computing | CUDA |
| Build System | GNU Make |
| Platform | Linux (NCSA Delta HPC Cluster) |
| GPU Architecture | NVIDIA A100 (sm_80) |

---

## Project Structure

```
imgcuda/
├── src/
│   ├── main.cpp            # CLI driver — argument parsing, timing, verification
│   ├── imgcuda.h           # Public GPU API declarations
│   ├── imgcuda.cu          # CUDA kernel implementations
│   ├── imgreference.h      # CPU reference API declarations
│   ├── imgreference.cpp    # CPU reference implementations
│   ├── ppm.h / ppm.cpp     # PPM image I/O module
│   └── ppmtest.cpp         # PPM I/O smoke test
├── Makefile
├── README.md
└── .gitignore
```

---

## Implemented Filters

| Filter | Type | Description |
|--------|------|-------------|
| `drop_red/green/blue` | Pixel-wise | Zero out a single RGB channel |
| `grayscale` | Pixel-wise | Weighted luminance conversion (0.21R + 0.71G + 0.07B) |
| `blur` | Convolution | Box blur with configurable radius (1–5) |
| `sharpen` | Convolution | 3×3 sharpening kernel |
| `sobel_x` | Convolution | Vertical edge detection |
| `sobel_y` | Convolution | Horizontal edge detection |

---

## Engineering Design Decisions

**Dual API pattern** — Each function has a standard overload matching the required public interface and a timed overload that accumulates `H2D / kernel / D2H` metrics, keeping the public API clean while enabling precise performance measurement:
```cpp
ppm::Image drop_red(const ppm::Image& in);              // standard
ppm::Image drop_red(const ppm::Image& in, Timing& t);   // timed
```

**Separate input/output buffers** — Convolution filters (blur, sharpen, Sobel) use separate `d_in` and `d_out` device buffers to prevent threads from reading partially-modified data, ensuring correctness.

**Zero-padding boundary strategy** — Out-of-bounds reads during convolution are treated as 0. Only valid in-bounds pixels contribute to averages and kernel sums.

**AoS memory layout** — Interleaved `RGBRGB...` layout matches the PPM format directly, avoiding reformatting overhead.

---

## Usage

```bash
make
./bin/imgtool input.ppm output.ppm --filter <name> [--radius <r>] [--runs <n>] [--verify]
```

```bash
# Blur with radius 3
./bin/imgtool tests/lenna.ppm tests/out.ppm --filter blur --radius 3

# Grayscale with CPU vs GPU correctness check
./bin/imgtool tests/lenna.ppm tests/out.ppm --filter grayscale --verify

# Sobel edge detection, 15 timed runs
./bin/imgtool tests/lenna.ppm tests/out.ppm --filter sobel_x --runs 15
```

### Example Timing Output
```
Filter: blur radius=3
Runs: 10
Average H2D: 0.123 ms
Average Kernel: 0.456 ms
Average D2H: 0.118 ms

!VERIFY REPORT!
There were 0 mismatched RGB bytes between CPU and GPU output.
```

---

## Performance Notes

For simple pixel-wise filters on large images, memory transfer time (H2D + D2H) typically dominates over kernel execution time; a characteristic behavior of GPU workloads with low arithmetic intensity. Convolution filters show higher relative kernel time due to neighborhood memory access patterns.

---

## Building

Requires CUDA toolkit and `g++` with C++17. Targets CUDA architecture `sm_80` by default (NVIDIA A100).

```bash
make                    # build imgtool
make clean              # remove build artifacts
```