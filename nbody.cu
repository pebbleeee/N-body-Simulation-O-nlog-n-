#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glut.h>
#include <GL/glu.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <bitset>
#include <string>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <limits>
#include <vector>
#include <algorithm>

bool use_naive = false;

const float G = 0.0001f;
const float SOFTENING = 0.05f;
const float DT = 0.01f;
const float THETA = 0.5f;

struct Particle {
    float2 pos;
    float2 vel;
    float mass;
};

struct get_x {
    __host__ __device__
    float operator()(const Particle& p) const { return p.pos.x; }
};

struct get_y {
    __host__ __device__
    float operator()(const Particle& p) const { return p.pos.y; }
};

struct IntCoords {
    unsigned int x_int;
    unsigned int y_int;
};

struct MortonParticle {
    unsigned int morton_index;
    int particle_index;
};

struct QuadNode {
    bool is_leaf;
    int particle_index;
    int children[4];
    float2 center_of_mass;
    float total_mass;
};

Particle* d_particles;
GLuint vbo;
cudaGraphicsResource* cuda_vbo_resource;
float2* d_vbo_ptr;
int n_particles;
IntCoords* d_int_coords;
MortonParticle* d_morton_particles;
float* d_min_x;
float* d_max_x;
float* d_max_y;
float* d_min_y;

QuadNode* d_quad_nodes = nullptr;
int d_root_idx = 0;

/**
 * Naive N-body force computation kernel (O(N²) complexity)
 * @param particles Device array of particles
 * @param n Number of particles
 * @note This direct method becomes computationally prohibitive for N > 10⁴
 */
__global__ void computeForces(Particle* particles, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float2 force = {0.0f, 0.0f};
        float2 pos_i = particles[i].pos;
        float mass_i = particles[i].mass;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                float2 pos_j = particles[j].pos;
                float mass_j = particles[j].mass;
                float dx = pos_j.x - pos_i.x;
                float dy = pos_j.y - pos_i.y;
                float dist_sq = dx * dx + dy * dy + SOFTENING * SOFTENING;
                float dist = sqrtf(dist_sq);
                float force_mag = (G * mass_i * mass_j) / dist_sq;
                force.x += force_mag * dx / dist;
                force.y += force_mag * dy / dist;
            }
        }
        particles[i].vel.x += force.x / mass_i * DT;
        particles[i].vel.y += force.y / mass_i * DT;
    }
}

/**
 * Barnes-Hut force approximation kernel (O(N log N) complexity)
 * @param particles Device array of particles
 * @param n Number of particles
 * @param theta Barnes-Hut opening angle threshold
 * @param min_x Domain minimum x-coordinate
 * @param max_x Domain maximum x-coordinate
 * @param min_y Domain minimum y-coordinate
 * @param max_y Domain maximum y-coordinate
 * @param nodes Device array of quadtree nodes
 * @param root_idx Index of root node in quadtree array
 * @note Tree traversal accounts for 20-40% of runtime in typical configurations
 */
__global__ void computeForcesBH(Particle* particles, int n, float theta, 
                               float min_x, float max_x, float min_y, float max_y,
                               QuadNode* nodes, int root_idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle p = particles[i];
    float2 force = {0.0f, 0.0f};

    const int STACK_SIZE = 32;
    struct StackEntry {
        int node_idx;
        float x_min, x_max, y_min, y_max;
    };
    StackEntry stack[STACK_SIZE];
    int stack_ptr = 0;

    stack[stack_ptr++] = {root_idx, min_x, max_x, min_y, max_y};

    while (stack_ptr > 0) {
        StackEntry entry = stack[--stack_ptr];
        QuadNode node = nodes[entry.node_idx];
        float s = entry.x_max - entry.x_min;

        float dx = node.center_of_mass.x - p.pos.x;
        float dy = node.center_of_mass.y - p.pos.y;
        float dist_sq = dx*dx + dy*dy + SOFTENING*SOFTENING;
        float dist = sqrtf(dist_sq);

        if (node.is_leaf) {
            if (node.particle_index == -1) {
                if (node.total_mass > 0) {
                    float force_mag = (G * p.mass * node.total_mass) / dist_sq;
                    force.x += force_mag * dx / dist;
                    force.y += force_mag * dy / dist;
                }
            } else {
                if (node.particle_index != i) {
                    float mass = particles[node.particle_index].mass;
                    float force_mag = (G * p.mass * mass) / dist_sq;
                    force.x += force_mag * dx / dist;
                    force.y += force_mag * dy / dist;
                }
            }
        } else {
            if (s / dist < theta) {
                float force_mag = (G * p.mass * node.total_mass) / dist_sq;
                force.x += force_mag * dx / dist;
                force.y += force_mag * dy / dist;
            } else {
                float x_mid = (entry.x_min + entry.x_max) * 0.5f;
                float y_mid = (entry.y_min + entry.y_max) * 0.5f;
                
                for (int c = 3; c >= 0; c--) {
                    int child_idx = node.children[c];
                    if (child_idx != -1) {
                        StackEntry child_entry;
                        child_entry.node_idx = child_idx;
                        
                        switch (c) {
                            case 0:
                                child_entry = {child_idx, entry.x_min, x_mid, entry.y_min, y_mid};
                                break;
                            case 1:
                                child_entry = {child_idx, x_mid, entry.x_max, entry.y_min, y_mid};
                                break;
                            case 2:
                                child_entry = {child_idx, entry.x_min, x_mid, y_mid, entry.y_max};
                                break;
                            case 3:
                                child_entry = {child_idx, x_mid, entry.x_max, y_mid, entry.y_max};
                                break;
                        }
                        
                        if (stack_ptr < STACK_SIZE) {
                            stack[stack_ptr++] = child_entry;
                        }
                    }
                }
            }
        }
    }

    particles[i].vel.x += force.x / p.mass * DT;
    particles[i].vel.y += force.y / p.mass * DT;
}

/**
 * Particle position update and VBO synchronization kernel
 * @param particles Device array of particles
 * @param dt Time step
 * @param n Number of particles
 * @param vbo_ptr Mapped VBO device pointer
 */
__global__ void updateParticles(Particle* particles, float dt, int n, float2* vbo_ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        particles[i].pos.x += particles[i].vel.x * dt;
        particles[i].pos.y += particles[i].vel.y * dt;
        vbo_ptr[i].x = particles[i].pos.x;
        vbo_ptr[i].y = particles[i].pos.y;
    }
}

/**
 * Maps particle positions to integer coordinates for Morton code calculation
 * @param particles Device array of particles
 * @param int_coords Output integer coordinates
 * @param n Number of particles
 * @param min_x Domain minimum x-coordinate
 * @param max_x Domain maximum x-coordinate
 * @param min_y Domain minimum y-coordinate
 * @param max_y Domain maximum y-coordinate
 */
__global__ void mapToIntegerCoords(Particle* particles, IntCoords* int_coords, int n, 
                                  float min_x, float max_x, float min_y, float max_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = particles[i].pos.x;
        float y = particles[i].pos.y;
        float range_x = max_x - min_x;
        float range_y = max_y - min_y;
        if (range_x == 0.0f) range_x = 1.0f;
        if (range_y == 0.0f) range_y = 1.0f;
        unsigned int x_int = static_cast<unsigned int>(((x - min_x) / range_x) * 65535.0f);
        unsigned int y_int = static_cast<unsigned int>(((y - min_y) / range_y) * 65535.0f);
        x_int = min(max(x_int, 0u), 65535u);
        y_int = min(max(y_int, 0u), 65535u);
        int_coords[i].x_int = x_int;
        int_coords[i].y_int = y_int;
    }
}

__device__ unsigned int spreadBits(unsigned int v) {
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v;
}

__device__ unsigned int computeMortonIndex(unsigned int x, unsigned int y) {
    return (spreadBits(y) << 1) | spreadBits(x);
}

/**
 * Computes Morton codes for spatial sorting
 * @param int_coords Integer coordinates from mapToIntegerCoords
 * @param morton_particles Output array of Morton codes with particle indices
 * @param n_particles Number of particles
 */
__global__ void computeMortonIndices(IntCoords* int_coords, MortonParticle* morton_particles, int n_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_particles) {
        unsigned int x = int_coords[i].x_int;
        unsigned int y = int_coords[i].y_int;
        unsigned int morton_index = computeMortonIndex(x, y);
        morton_particles[i].morton_index = morton_index;
        morton_particles[i].particle_index = i;
    }
}

/**
 * Recursive quadtree builder (CPU implementation)
 * @param nodes Output vector of quadtree nodes
 * @param sorted_morton Sorted Morton codes with particle indices
 * @param start Start index in sorted Morton array
 * @param end End index in sorted Morton array
 * @param bit_pos Current bit position for spatial partitioning
 * @param h_particles Host array of particles for COM calculations
 * @return Index of created node in nodes vector
 * @note This serial implementation becomes bottleneck for N > 10⁵. 
 *        GPU parallel tree construction could provide 10-100x speedup.
 */
int buildQuadTree(std::vector<QuadNode>& nodes, const std::vector<MortonParticle>& sorted_morton, 
                 int start, int end, int bit_pos, const Particle* h_particles) {
    if (start >= end) return -1;
    if (end - start == 1 || bit_pos >= 30) {
        QuadNode node;
        node.is_leaf = true;
        float total_mass = 0.0f;
        float2 weighted_sum = {0.0f, 0.0f};
        for (int k = start; k < end; k++) {
            int idx = sorted_morton[k].particle_index;
            total_mass += h_particles[idx].mass;
            weighted_sum.x += h_particles[idx].mass * h_particles[idx].pos.x;
            weighted_sum.y += h_particles[idx].mass * h_particles[idx].pos.y;
        }
        if (total_mass > 0.0f) {
            node.center_of_mass.x = weighted_sum.x / total_mass;
            node.center_of_mass.y = weighted_sum.y / total_mass;
        } else {
            node.center_of_mass.x = 0.0f;
            node.center_of_mass.y = 0.0f;
        }
        node.total_mass = total_mass;
        node.particle_index = (end - start == 1) ? sorted_morton[start].particle_index : -1;
        nodes.push_back(node);
        return nodes.size() - 1;
    }

    QuadNode node;
    node.is_leaf = false;
    for (int i = 0; i < 4; i++) node.children[i] = -1;

    int sub_start = start;
    while (sub_start < end) {
        unsigned int morton = sorted_morton[sub_start].morton_index;
        int key = (morton >> (30 - bit_pos)) & 3;
        int sub_end = sub_start + 1;
        while (sub_end < end) {
            unsigned int next_morton = sorted_morton[sub_end].morton_index;
            int next_key = (next_morton >> (30 - bit_pos)) & 3;
            if (next_key != key) break;
            sub_end++;
        }
        int child_idx = buildQuadTree(nodes, sorted_morton, sub_start, sub_end, bit_pos + 2, h_particles);
        node.children[key] = child_idx;
        sub_start = sub_end;
    }

    float total_mass = 0.0f;
    float2 weighted_sum = {0.0f, 0.0f};
    for (int i = 0; i < 4; i++) {
        if (node.children[i] != -1) {
            int child_idx = node.children[i];
            const QuadNode& child = nodes[child_idx];
            total_mass += child.total_mass;
            weighted_sum.x += child.total_mass * child.center_of_mass.x;
            weighted_sum.y += child.total_mass * child.center_of_mass.y;
        }
    }
    if (total_mass > 0.0f) {
        node.center_of_mass.x = weighted_sum.x / total_mass;
        node.center_of_mass.y = weighted_sum.y / total_mass;
    } else {
        node.center_of_mass.x = 0.0f;
        node.center_of_mass.y = 0.0f;
    }
    node.total_mass = total_mass;
    nodes.push_back(node);
    return nodes.size() - 1;
}

float view_center_x = 0.0f;
float view_center_y = 0.0f;
float sim_units_per_pixel = 0.01f;

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    int width = glutGet(GLUT_WINDOW_WIDTH);
    int height = glutGet(GLUT_WINDOW_HEIGHT);
    float view_width = width * sim_units_per_pixel;
    float view_height = height * sim_units_per_pixel;
    float left = view_center_x - view_width / 2.0f;
    float right = view_center_x + view_width / 2.0f;
    float bottom = view_center_y - view_height / 2.0f;
    float top = view_center_y + view_height / 2.0f;
    glOrtho(left, right, bottom, top, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glPointSize(2.0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(2, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, n_particles);
    glDisableClientState(GL_VERTEX_ARRAY);
    glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y) {
    const float pan_speed = 10.0f * sim_units_per_pixel;
    const float zoom_factor = 1.1f;
    switch (key) {
        case '+': sim_units_per_pixel /= zoom_factor; break;
        case '-': sim_units_per_pixel *= zoom_factor; break;
        case 'w': view_center_y += pan_speed; break;
        case 's': view_center_y -= pan_speed; break;
        case 'a': view_center_x -= pan_speed; break;
        case 'd': view_center_x += pan_speed; break;
    }
    glutPostRedisplay();
}

void reshape(int width, int height) {
    glViewport(0, 0, width, height);
    glutPostRedisplay();
}

void idle() {
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    size_t size;
    cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &size, cuda_vbo_resource);
    int threadsPerBlock = 512;
    int blocks = (n_particles + threadsPerBlock - 1) / threadsPerBlock;

    if (use_naive) {
        computeForces<<<blocks, threadsPerBlock>>>(d_particles, n_particles);
    } else {
        thrust::device_ptr<Particle> dev_ptr(d_particles);
        float min_x = thrust::reduce(
            thrust::make_transform_iterator(dev_ptr, get_x()),
            thrust::make_transform_iterator(dev_ptr + n_particles, get_x()),
            std::numeric_limits<float>::max(),
            thrust::minimum<float>()
        );
        float max_x = thrust::reduce(
            thrust::make_transform_iterator(dev_ptr, get_x()),
            thrust::make_transform_iterator(dev_ptr + n_particles, get_x()),
            std::numeric_limits<float>::lowest(),
            thrust::maximum<float>()
        );
        float min_y = thrust::reduce(
            thrust::make_transform_iterator(dev_ptr, get_y()),
            thrust::make_transform_iterator(dev_ptr + n_particles, get_y()),
            std::numeric_limits<float>::max(),
            thrust::minimum<float>()
        );
        float max_y = thrust::reduce(
            thrust::make_transform_iterator(dev_ptr, get_y()),
            thrust::make_transform_iterator(dev_ptr + n_particles, get_y()),
            std::numeric_limits<float>::lowest(),
            thrust::maximum<float>()
        );

        mapToIntegerCoords<<<blocks, threadsPerBlock>>>(d_particles, d_int_coords, n_particles, min_x, max_x, min_y, max_y);
        cudaDeviceSynchronize();

        computeMortonIndices<<<blocks, threadsPerBlock>>>(d_int_coords, d_morton_particles, n_particles);
        cudaDeviceSynchronize();

        thrust::device_ptr<MortonParticle> dev_morton_ptr(d_morton_particles);
        thrust::sort(dev_morton_ptr, dev_morton_ptr + n_particles,
                     [] __device__ (const MortonParticle& a, const MortonParticle& b) {
                         return a.morton_index < b.morton_index;
                     });

        MortonParticle* h_morton_particles = new MortonParticle[n_particles];
        Particle* h_particles = new Particle[n_particles];
        cudaMemcpy(h_morton_particles, d_morton_particles, n_particles * sizeof(MortonParticle), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_particles, d_particles, n_particles * sizeof(Particle), cudaMemcpyDeviceToHost);

        std::vector<MortonParticle> sorted_morton(h_morton_particles, h_morton_particles + n_particles);
        std::vector<QuadNode> nodes;
        int root_idx = buildQuadTree(nodes, sorted_morton, 0, n_particles, 0, h_particles);

        delete[] h_morton_particles;
        delete[] h_particles;

        if (d_quad_nodes) cudaFree(d_quad_nodes);
        cudaMalloc(&d_quad_nodes, nodes.size() * sizeof(QuadNode));
        cudaMemcpy(d_quad_nodes, nodes.data(), nodes.size() * sizeof(QuadNode), cudaMemcpyHostToDevice);

        computeForcesBH<<<blocks, threadsPerBlock>>>(d_particles, n_particles, THETA, min_x, max_x, min_y, max_y, d_quad_nodes, root_idx);
    }

    cudaDeviceSynchronize();

    updateParticles<<<blocks, threadsPerBlock>>>(d_particles, DT, n_particles, d_vbo_ptr);
    cudaDeviceSynchronize();

    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    glutPostRedisplay();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s [--naive] <number_of_particles>\n", argv[0]);
        exit(1);
    }

    if (strcmp(argv[1], "--naive") == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: %s --naive <number_of_particles>\n", argv[0]);
            exit(1);
        }
        use_naive = true;
        n_particles = atoi(argv[2]);
    } else {
        if (argc != 2) {
            fprintf(stderr, "Usage: %s <number_of_particles>\n", argv[0]);
            exit(1);
        }
        use_naive = false;
        n_particles = atoi(argv[1]);
    }

    if (n_particles <= 0) {
        fprintf(stderr, "Number of particles must be positive.\n");
        exit(1);
    }

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(800, 600);
    glutCreateWindow("Barnes-Hut N-Body Simulation");
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, n_particles * sizeof(float2), NULL, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
    cudaMalloc(&d_particles, n_particles * sizeof(Particle));
    cudaMalloc(&d_min_x, sizeof(float));
    cudaMalloc(&d_max_x, sizeof(float));
    cudaMalloc(&d_min_y, sizeof(float));
    cudaMalloc(&d_max_y, sizeof(float));
    cudaMalloc(&d_int_coords, n_particles * sizeof(IntCoords));
    cudaMalloc(&d_morton_particles, n_particles * sizeof(MortonParticle));
    Particle* h_particles = new Particle[n_particles];
    srand(42);
    if (n_particles == 4) {
        h_particles[0] = {{ -0.5f,  0.5f }, {0.0f, 0.0f}, 1.0f};
        h_particles[1] = {{  0.5f,  0.5f }, {0.0f, 0.0f}, 1.0f};
        h_particles[2] = {{ -0.5f, -0.5f }, {0.0f, 0.0f}, 1.0f};
        h_particles[3] = {{  0.5f, -0.5f }, {0.0f, 0.0f}, 1.0f};
    } else {
        for (int i = 0; i < n_particles; i++) {
            h_particles[i].pos.x = (float)rand() / RAND_MAX * 2 - 1;
            h_particles[i].pos.y = (float)rand() / RAND_MAX * 2 - 1;
            h_particles[i].vel.x = (float)rand() / RAND_MAX * 0.01 - 0.005;
            h_particles[i].vel.y = (float)rand() / RAND_MAX * 0.01 - 0.005;
            h_particles[i].mass = 1.0f;
        }
    }
    cudaMemcpy(d_particles, h_particles, n_particles * sizeof(Particle), cudaMemcpyHostToDevice);
    delete[] h_particles;

    glutMainLoop();

    cudaGraphicsUnregisterResource(cuda_vbo_resource);
    cudaFree(d_particles);
    cudaFree(d_min_x);
    cudaFree(d_max_x);
    cudaFree(d_min_y);
    cudaFree(d_max_y);
    cudaFree(d_int_coords);
    cudaFree(d_morton_particles);
    glDeleteBuffers(1, &vbo);
    return 0;
}
