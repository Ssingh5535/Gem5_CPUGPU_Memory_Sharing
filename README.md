# Efficient Memory Sharing Between CPU and GPU in Gem5

A comprehensive gem5-based project exploring heterogeneous memory sharing and cache coherence between CPUs and GPUs.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Background](#background)  
3. [Objectives](#objectives)  
4. [Team Roles](#team-roles)  
5. [System Architecture](#system-architecture)  
   1. [GPGPU Model](#gpgpu-model)  
   2. [Full-System AMD GPU Model](#full-system-amd-gpu-model)  
   3. [Gem5-GPU Model](#gem5-gpu-model)  
   4. [GCN3 GPU Model](#gcn3-gpu-model)  
6. [Setup & Prerequisites](#setup--prerequisites)  
7. [Simulation Workflow](#simulation-workflow)  
   1. [Building Gem5 with GPU Support](#building-gem5-with-gpu-support)  
   2. [Configuring Heterogeneous Memory](#configuring-heterogeneous-memory)  
   3. [Running Standard APU Workload](#running-standard-apu-workload)  
   4. [Running Custom APU Workload](#running-custom-apu-workload)  
8. [Performance Metrics & Results](#performance-metrics--results)  
   1. [Standard APU Workload Results](#standard-apu-workload-results)  
   2. [Custom APU Workload Results](#custom-apu-workload-results)  
9. [Analysis & Comparison](#analysis--comparison)  
10. [Limitations](#limitations)  
11. [Future Work](#future-work)  
12. [License](#license)  
13. [References](#references)  

---

## Project Overview

This project investigates mechanisms for **efficient memory sharing** and **cache coherence** in heterogeneous CPU–GPU systems using the gem5 simulator. We model unified memory access, shared TLB management, and evaluate performance under varied APU workloads to identify optimal cache configurations and coherence protocols.

## Background

Modern heterogeneous platforms (e.g., integrated APUs and discrete GPUs) often employ **unified memory architectures** (e.g., hUMA) to allow seamless data sharing between CPU and GPU domains. Accurately simulating such systems requires extending gem5’s CPU and GPU models, integrating cache coherence across different memory hierarchies, and instrumenting performance metrics such as demand hits/misses and array access counts.

## Objectives

- **Develop a shared-memory model** that leverages both CPU and GPU caches for unified data access.  
- **Design and implement cache-coherence mechanisms** suitable for heterogeneous workloads within gem5.  
- **Benchmark performance** under both standard and custom APU workloads to quantify the benefits of our approach.

## Team Roles

| Member             | Responsibility                                 |
|--------------------|------------------------------------------------|
| Stephen Singh      | Gem5 heterogeneous setup & memory optimization |
| Austin Kee         | GPU model integration & ROCm/CUDA support      |
| Thomas Tymecki     | Performance analysis & metrics instrumentation |
| Andrew Femiano     | Debugging & workflow automation                |

## System Architecture

Our gem5-based platform comprises four complementary models:

### GPGPU Model

- **Detailed heterogeneous system architecture** supporting CUDA-based workloads.  
- **CUDA Toolkit (≤8.0)** integration on Ubuntu 16.04 for NVIDIA GPU simulation.  
- **Built-in gem5 samples** for gathering cache-coherence statistics (e.g., demand hits/misses).

### Full-System AMD GPU Model

- **x86 CPU + AMD GPU** heterogeneous simulation using KVM for full-system emulation.  
- **ROCm integration** enabling execution of OpenCL/HIP kernels.  
- **Active community support** and up-to-date AMD GPU models within gem5.

### Gem5-GPU Model

- **Custom gem5 builds** with x86 support extended to GPU modeling.  
- **OpenCL support** for parallel kernel execution.  
- **Specialized configuration files** for different cache-coherence protocols.

### GCN3 GPU Model

1. **Compute Units (CUs)**
   - Scalar Units for lightweight operations  
   - SIMD Units for vectorized parallelism  
   - LDS for intra-CU shared memory  
2. **Memory Hierarchy**
   - Private, Shared, and Global memory regions with simulated latency and contention  
3. **Heterogeneous Unified Memory Access (hUMA)** support  
4. **ROCm & HIP** workload compatibility  
5. **AMD GCN3 ISA implementation** for realistic instruction-level behavior  
6. **Full-system simulation** including I/O and OS in gem5  

## Setup & Prerequisites

- **Operating System**: Ubuntu 16.04 LTS (recommended for CUDA 8.0 compatibility)  
- **Gem5 Version**: v21.2 or later, built with `X86_MESI_Two_Level` and Ruby  
- **Toolchain**:  
  - SCons (`sudo apt install scons python3-dev`)  
  - Python 3.6+ and pip packages (`numpy`, `pandas`)  
  - CUDA 8.0 for NVIDIA models or ROCm for AMD models  
- **Hardware**: Multi-core x86 host with ≥16 GB RAM and GPU driver support (VM access for discrete GPU models)

## Simulation Workflow

### Building Gem5 with GPU Support

```bash
git clone https://github.com/yourusername/gem5-hetero-memory.git
cd gem5-hetero-memory
scons build/X86_MESI_Two_Level/gem5.opt -j8
```

### Configuring Heterogeneous Memory

1. Edit `configs/hetero_apu.py` to enable:
   - `use_ruby = True`  
   - `dgpu_mem_size = "16GB"`  
   - `hsa_unified = True`  
2. Select CPU type (`AtomicSimpleCPU` or `X86KvmCPU`) and GPU model flags.

### Running Standard APU Workload

```bash
build/X86_MESI_Two_Level/gem5.opt configs/hetero_apu.py     --workload=standard --output-dir=m5out_standard
```

### Running Custom APU Workload

```bash
build/X86_MESI_Two_Level/gem5.opt configs/hetero_apu.py     --workload=custom --output-dir=m5out_custom
```

## Performance Metrics & Results

### Standard APU Workload Results

| Metric               | L1DCache         | L1ICache         |
|----------------------|------------------|------------------|
| Data Array Reads     | ~7.37 M          | ~50.99 M         |
| Data Array Writes    | ~3.22 M          | ~0.483 M         |
| Tag Array Reads      | ~11.46 M         | ~51.55 M        |
| Tag Array Writes     | ~4.74 M          | ~0.0235 M       |
| Demand Hits          | ~10.31 M         | ~51.45 M        |
| Demand Misses        | ~0.454 M         | ~0.0478 M       |

### Custom APU Workload Results

| Metric               | L1DCache | L1ICache  |
|----------------------|----------|-----------|
| Data Array Reads     | 16,571   | 93,023    |
| Data Array Writes    | 9,150    | 40        |
| Tag Array Reads      | 25,214   | 93,858    |
| Tag Array Writes     | 1,288    | 40        |
| Demand Hits          | 24,481   | 93,023    |
| Demand Misses        | 655      | 626       |

## Analysis & Comparison

- **Workload Intensity**  
  - Standard workload exhibits **higher cache activity**, suitable for stress-testing scenarios.  
  - Custom workload shows **lower overall accesses** with marginally higher hit rates (L1D: 97.4 % vs. 95.8 %).
- **Cache Coherence**  
  - Custom configuration demonstrates **better coherence** for lighter loads, reducing miss penalties.
- **Protocol Selection**  
  - Use standard setup for benchmarking throughput under heavy loads.  
  - Adopt custom setup when aiming for **low-latency** and **efficient** memory sharing.

## Limitations

- gem5 currently only supports **AtomicSimpleCPU** and **X86KvmCPU** for GPU-coherent simulations; timing CPUs are not compatible with the Ruby network required for discrete GPUs.  
- **CPU timing models are limited**, so results assume idealized access patterns without fine-grained race conditions.  
- Certain parameters (e.g., `dgpu_mem_size`) do not change simulated device properties due to hardcoded defaults in gem5’s C++ code.

## Future Work

- **Integrate timing CPU support** alongside GPU models to capture cache contention dynamics.  
- **Extend Ruby/Garnet 2.0** network models to include simplified GPU traffic flows.  
- **Enable full parameter customization** (e.g., memory sizes, bandwidth limits) on simulated devices.  
- **Simulate emerging unified-memory architectures** (e.g., Apple M-series, ARM Mali) with zero-copy features in gem5.

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

## References

- CATeam13 Proposal: *Efficient Memory Sharing Between CPU and GPU in Gem5*
