# N-body-Simulation-O-nlog-n-
This project implements a GPU-accelerated 2D Barnes-Hut algorithm for large-scale N-body simulations using NVIDIA's CUDA framework. The Barnes-Hut method reduces computational complexity from O(N²) to O(N log N) by approximating distant particle interactions via hierarchical quadtree structures.

Overview
Naive vs. Barnes-Hut: Naive method becomes infeasible beyond 100K particles. Our Barnes-Hut implementation achieves real-time performance up to 1 million particles.

GPU Acceleration: Optimized CUDA kernels for force computation, shared memory caching, warp-level operations, and asynchronous transfers.

Quadtree Bottleneck: Spatial hierarchy is built on the CPU, which becomes the primary bottleneck at large scales.

Features
CUDA 12.2 implementation

Morton code generation and radix sort for spatial locality

Real-time visualization via OpenGL + CUDA interop

Asynchronous memory transfers

Performance profiling on RTX 4090

Challenges
Recursive nature of quadtree construction limits parallelism

CPU-side spatial indexing causes memory transfer bottlenecks

Control-flow divergence and memory fragmentation on GPU

Future Work
We aim to implement fully parallel GPU-native quadtree construction using:

Bottom-up prefix scan-based methods

Top-down bitmask-based approaches

Structure-of-Arrays memory layout

Warp-cooperative primitives

Contributors
Developed collaboratively by a team of students as part of a high-performance computing project.

References
Zhang & Gruenwald (2023) - Efficient Quadtree Construction on GPUs

NVIDIA CUDA C Programming Guide v12.2

Thrust Parallel Algorithms Library Documentation

