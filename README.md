# N-body-Simulation-O-nlog-n-
This project implements a GPU-accelerated 2D N-body simulation using the Barnes-Hut algorithm and NVIDIA's CUDA framework. The Barnes-Hut algorithm reduces the computational complexity of particle interaction calculations from O(N²) to O(N log N) by approximating distant forces using a hierarchical quadtree structure. Our implementation leverages CUDA 12.2 and demonstrates real-time performance with up to 1 million particles on an RTX 4090 GPU.

To optimize GPU performance, we use Morton code (Z-order) transformations for spatial indexing, enabling efficient particle sorting and proximity-based grouping. The force computation kernel uses shared memory, warp-level operations, and asynchronous memory transfers to maximize throughput. Particle data is also visualized in real-time using CUDA-OpenGL interoperation.

Despite significant speedups over the naive approach, the current implementation relies on CPU-based quadtree construction, which introduces a performance bottleneck at higher particle counts. This limitation becomes especially apparent beyond 800k particles, where tree construction dominates total runtime. We analyze this bottleneck and discuss the challenges of implementing recursive data structures like quadtrees on massively parallel GPU architectures.

As future work, we plan to explore GPU-native quadtree construction techniques using both bottom-up and top-down strategies. These approaches utilize parallel primitives such as prefix scans, warp-cooperative sorting, and ballot/shuffle intrinsics to avoid dynamic memory allocation and improve scalability. By adopting these methods, we aim to push the simulation toward real-time performance for tens or even hundreds of millions of particles.

This project was developed collaboratively by a team of students as part of a high-performance computing research effort. Our work builds on recent literature in GPU-accelerated spatial indexing and contributes to the growing body of research in real-time, large-scale physical simulations.

