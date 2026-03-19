#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr int kMaxNeighbors = 14;
constexpr int kNumKeys = 27;
constexpr int kLutSize = 6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + 6 * (1 << 10) + (1 << 14);

constexpr std::array<int, kNumKeys> kKeys = {
    277, 153, 1050, 2085, 102, 554,
    413, 1179, 2357, 1594, 2151, 622,
    5462, 10921,
    5463, 5470, 5494, 10937, 10925, 10923,
    5599, 7543, 11197, 6014, 11963, 10991,
    16383
};

constexpr std::array<int, kNumKeys> kLutOffsets = {
    0, (1 << 4), (2 << 4), (3 << 4), (4 << 4), (5 << 4),
    6 * (1 << 4), 6 * (1 << 4) + (1 << 6), 6 * (1 << 4) + 2 * (1 << 6),
    6 * (1 << 4) + 3 * (1 << 6), 6 * (1 << 4) + 4 * (1 << 6), 6 * (1 << 4) + 5 * (1 << 6),
    6 * (1 << 4) + 6 * (1 << 6), 6 * (1 << 4) + 6 * (1 << 6) + (1 << 7),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + (1 << 8),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 2 * (1 << 8),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 3 * (1 << 8),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 4 * (1 << 8),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 5 * (1 << 8),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + (1 << 10),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + 2 * (1 << 10),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + 3 * (1 << 10),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + 4 * (1 << 10),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + 5 * (1 << 10),
    6 * (1 << 4) + 6 * (1 << 6) + 2 * (1 << 7) + 6 * (1 << 8) + 6 * (1 << 10)
};

constexpr std::array<std::array<int, 3>, kMaxNeighbors> kDirections = {{
    {{1, 0, 0}}, {{-1, 0, 0}}, {{0, 1, 0}}, {{0, -1, 0}},
    {{0, 0, 1}}, {{0, 0, -1}}, {{-1, 1, 0}}, {{1, -1, 0}},
    {{0, 1, 1}}, {{0, -1, -1}}, {{-1, 0, 1}}, {{1, 0, -1}},
    {{-1, 1, 1}}, {{1, -1, -1}}
}};

__constant__ int kDirectionsDev[42] = {
    1, 0, 0, -1, 0, 0,
    0, 1, 0, 0, -1, 0,
    0, 0, 1, 0, 0, -1,
    -1, 1, 0, 1, -1, 0,
    0, 1, 1, 0, -1, -1,
    -1, 0, 1, 1, 0, -1,
    -1, 1, 1, 1, -1, -1
};

struct NodeRecord {
    int id;
    int grid_id;
    int type;
    int x;
    int y;
    int z;
    double value;
};

struct EdgeRecord {
    int u;
    int v;
    int edge_type;
    double weight;
};

struct EdgeKey {
    int u;
    int v;
    int edge_type;

    bool operator==(const EdgeKey &other) const {
        return u == other.u && v == other.v && edge_type == other.edge_type;
    }
};

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey &k) const {
        const std::size_t h1 = std::hash<int>{}(k.u);
        const std::size_t h2 = std::hash<int>{}(k.v);
        const std::size_t h3 = std::hash<int>{}(k.edge_type);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

inline bool is_larger_shared(int v, int u, double value_v, double value_u) {
    return (value_v > value_u) || ((value_v == value_u) && (v > u));
}

inline bool is_less_shared(int v, int u, double value_v, double value_u) {
    return (value_v < value_u) || ((value_v == value_u) && (v < u));
}

inline int linear_index(int x, int y, int z, int nx, int ny, int nz) {
    (void)nz;
    return z * nx * ny + y * nx + x;
}

inline std::array<int, 3> idx_to_xyz(int idx, int nx, int ny) {
    std::array<int, 3> xyz{};
    xyz[0] = idx % nx;
    xyz[1] = (idx / nx) % ny;
    xyz[2] = idx / (nx * ny);
    return xyz;
}

std::vector<int> gather_neighbors(int idx, int nx, int ny, int nz) {
    std::vector<int> neighbors;
    neighbors.reserve(kMaxNeighbors);

    const auto xyz = idx_to_xyz(idx, nx, ny);
    const int x = xyz[0];
    const int y = xyz[1];
    const int z = xyz[2];

    for (const auto &d : kDirections) {
        const int new_x = x + d[0];
        const int new_y = y + d[1];
        const int new_z = z + d[2];
        if (new_x < 0 || new_x >= nx || new_y < 0 || new_y >= ny || new_z < 0 || new_z >= nz) {
            continue;
        }
        neighbors.push_back(linear_index(new_x, new_y, new_z, nx, ny, nz));
    }

    return neighbors;
}

bool read_double_bin(const std::string &path, std::size_t count, std::vector<double> &out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open scalar file: " << path << '\n';
        return false;
    }
    out.resize(count);
    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(count * sizeof(double)));
    if (!in) {
        std::cerr << "Failed to read scalar data from: " << path << '\n';
        return false;
    }
    return true;
}

bool read_int_bin(const std::string &path, std::size_t count, std::vector<int> &out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open int file: " << path << '\n';
        return false;
    }
    out.resize(count);
    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(count * sizeof(int)));
    if (!in) {
        std::cerr << "Failed to read int data from: " << path << '\n';
        return false;
    }
    return true;
}

int binary_search_lut(const std::array<int, kNumKeys> &keys, int target) {
    for (int i = 0; i < kNumKeys; ++i) {
        if (keys[i] == target) {
            return i;
        }
    }
    return -1;
}

bool classify_vertices_from_scalar(
    const std::vector<double> &scalar,
    int nx,
    int ny,
    int nz,
    const std::string &lut_path,
    std::vector<int> &vertex_triplets_out,
    std::vector<int> &lut_result_out) {

    std::vector<int> lut_table;
    if (!read_int_bin(lut_path, kLutSize, lut_table)) {
        std::cerr << "Failed to read LUT table from " << lut_path << '\n';
        return false;
    }

    const std::size_t n = scalar.size();
    vertex_triplets_out.assign(n * 3, -1);
    lut_result_out.assign(n, -1);

    for (std::size_t g_idx = 0; g_idx < n; ++g_idx) {
        const double data_value = scalar[g_idx];
        const auto xyz = idx_to_xyz(static_cast<int>(g_idx), nx, ny);
        const int gx = xyz[0];
        const int gy = xyz[1];
        const int gz = xyz[2];

        int lower_count = 0;
        int upper_count = 0;
        int largest_index = static_cast<int>(g_idx);
        double largest_value = data_value;
        int smallest_index = static_cast<int>(g_idx);
        double smallest_value = data_value;
        int neighbor_size = 0;
        std::array<int, kMaxNeighbors> binary{};
        std::array<int, kMaxNeighbors> vertex_binary{};

        for (int d = 0; d < kMaxNeighbors; ++d) {
            const int new_x = gx + kDirections[d][0];
            const int new_y = gy + kDirections[d][1];
            const int new_z = gz + kDirections[d][2];
            if (new_x < 0 || new_x >= nx || new_y < 0 || new_y >= ny || new_z < 0 || new_z >= nz) {
                continue;
            }

            const int r = linear_index(new_x, new_y, new_z, nx, ny, nz);
            const double neighbor_value = scalar[r];

            if (neighbor_value < data_value || ((neighbor_value == data_value) && (r < static_cast<int>(g_idx)))) {
                binary[kMaxNeighbors - 1 - neighbor_size] = 0;
                ++lower_count;
            } else if (neighbor_value > data_value || ((neighbor_value == data_value) && (r > static_cast<int>(g_idx)))) {
                binary[kMaxNeighbors - 1 - neighbor_size] = 1;
                ++upper_count;
            }

            vertex_binary[kMaxNeighbors - 1 - d] = 1;
            ++neighbor_size;

            if (is_larger_shared(r, largest_index, neighbor_value, largest_value)) {
                largest_index = r;
                largest_value = neighbor_value;
            }
            if (is_less_shared(r, smallest_index, neighbor_value, smallest_value)) {
                smallest_index = r;
                smallest_value = neighbor_value;
            }
        }

        int decimal_value = 0;
        for (int i = 0; i < kMaxNeighbors; ++i) {
            decimal_value = (decimal_value << 1) | binary[i];
        }

        int vertex_types = 0;
        for (int i = 0; i < kMaxNeighbors; ++i) {
            vertex_types = (vertex_types << 1) | vertex_binary[i];
        }

        const int key_index = binary_search_lut(kKeys, vertex_types);
        int lut_result = 5;
        if (upper_count == 0) {
            lut_result = 4;
        } else if (lower_count == 0) {
            lut_result = 0;
        } else if (key_index != -1) {
            lut_result = lut_table[kLutOffsets[key_index] + decimal_value];
        } else {
            lut_result = -1;
        }

        vertex_triplets_out[3 * g_idx] = lut_result;
        vertex_triplets_out[3 * g_idx + 1] = smallest_index;
        vertex_triplets_out[3 * g_idx + 2] = largest_index;
        lut_result_out[g_idx] = lut_result;
    }

    return true;
}

template <typename T>
void write_one(std::ofstream &out, const T &value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

bool write_nodes_bin(const std::string &path, const std::vector<NodeRecord> &nodes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open nodes output: " << path << '\n';
        return false;
    }
    for (const auto &n : nodes) {
        write_one(out, n.id);
        write_one(out, n.grid_id);
        write_one(out, n.type);
        write_one(out, n.x);
        write_one(out, n.y);
        write_one(out, n.z);
        write_one(out, n.value);
    }
    return static_cast<bool>(out);
}

bool write_edges_bin(const std::string &path, const std::vector<EdgeRecord> &edges) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open edges output: " << path << '\n';
        return false;
    }
    for (const auto &e : edges) {
        write_one(out, e.u);
        write_one(out, e.v);
        write_one(out, e.edge_type);
        write_one(out, e.weight);
    }
    return static_cast<bool>(out);
}

inline int map_node_type(int lut_result) {
    if (lut_result == 0) {
        return 0;
    }
    if (lut_result == 4) {
        return 1;
    }
    if (lut_result == 1) {
        return 2;
    }
    if (lut_result == 2) {
        return 3;
    }
    return -1;
}

bool is_integer_string(const std::string &s) {
    if (s.empty()) {
        return false;
    }
    std::size_t i = 0;
    if (s[0] == '+' || s[0] == '-') {
        if (s.size() == 1) {
            return false;
        }
        i = 1;
    }
    for (; i < s.size(); ++i) {
        if (s[i] < '0' || s[i] > '9') {
            return false;
        }
    }
    return true;
}

#define CUDA_CHECK(expr)                                                            \
    do {                                                                             \
        cudaError_t _err = (expr);                                                   \
        if (_err != cudaSuccess) {                                                   \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__          \
                      << " -> " << cudaGetErrorString(_err) << "\n";               \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

__device__ inline bool is_larger_shared_dev(int v, int u, double value_v, double value_u) {
    return (value_v > value_u) || ((value_v == value_u) && (v > u));
}

__device__ inline bool is_less_shared_dev(int v, int u, double value_v, double value_u) {
    return (value_v < value_u) || ((value_v == value_u) && (v < u));
}

__global__ void trace_seeds_kernel(
    const double *scalar,
    const int *lut,
    int nx,
    int ny,
    int nz,
    int n,
    const int *seed_start,
    const int *seed_target_lut,
    const int *seed_mode,
    int seed_count,
    int *out_reached,
    unsigned char *out_valid) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= seed_count) {
        return;
    }

    int current = seed_start[tid];
    int target = seed_target_lut[tid];
    int mode = seed_mode[tid];

    for (int iter = 0; iter < n; ++iter) {
        if (current < 0 || current >= n) {
            out_reached[tid] = -1;
            out_valid[tid] = 0;
            return;
        }

        if (lut[current] == target) {
            out_reached[tid] = current;
            out_valid[tid] = 1;
            return;
        }

        int x = current % nx;
        int y = (current / nx) % ny;
        int z = current / (nx * ny);

        int best_next = -1;
        double best_value = 0.0;

        for (int d = 0; d < kMaxNeighbors; ++d) {
            int dir_x = kDirectionsDev[d * 3];
            int dir_y = kDirectionsDev[d * 3 + 1];
            int dir_z = kDirectionsDev[d * 3 + 2];

            int new_x = x + dir_x;
            int new_y = y + dir_y;
            int new_z = z + dir_z;
            if (new_x < 0 || new_x >= nx || new_y < 0 || new_y >= ny || new_z < 0 || new_z >= nz) {
                continue;
            }

            int nb = new_z * nx * ny + new_y * nx + new_x;
            double nb_value = scalar[nb];
            double cur_value = scalar[current];

            if (mode == 1) {
                if (!is_larger_shared_dev(nb, current, nb_value, cur_value)) {
                    continue;
                }
                if (best_next == -1 || is_larger_shared_dev(nb, best_next, nb_value, best_value)) {
                    best_next = nb;
                    best_value = nb_value;
                }
            } else {
                if (!is_less_shared_dev(nb, current, nb_value, cur_value)) {
                    continue;
                }
                if (best_next == -1 || is_less_shared_dev(nb, best_next, nb_value, best_value)) {
                    best_next = nb;
                    best_value = nb_value;
                }
            }
        }

        if (best_next == -1) {
            out_reached[tid] = -1;
            out_valid[tid] = 0;
            return;
        }

        current = best_next;
    }

    out_reached[tid] = -1;
    out_valid[tid] = 0;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7 && argc != 8) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0]
                  << " <scalar.bin(double)> <nx> <ny> <nz> <out_dir>\n"
                  << "  " << argv[0]
                  << " <scalar.bin(double)> <nx> <ny> <nz> <out_dir> <LUT.bin>\n"
                  << "  " << argv[0]
                  << " <scalar.bin(double)> <vertex_type.bin(int,3*N)> <nx> <ny> <nz> <out_dir>\n";
        return 1;
    }

    bool has_vertex_type_input = false;
    if (argc == 7) {
        has_vertex_type_input = !is_integer_string(argv[2]);
    }
    if (argc == 8) {
        has_vertex_type_input = false;
    }

    const std::string scalar_path = argv[1];
    const int nx = std::stoi(argv[has_vertex_type_input ? 3 : 2]);
    const int ny = std::stoi(argv[has_vertex_type_input ? 4 : 3]);
    const int nz = std::stoi(argv[has_vertex_type_input ? 5 : 4]);

    std::string out_dir;
    std::string lut_path = "LUT.bin";

    if (has_vertex_type_input) {
        out_dir = argv[6];
    } else {
        if (argc == 6) {
            const std::string arg5 = argv[5];
            if (arg5.size() >= 4 && arg5.substr(arg5.size() - 4) == ".bin") {
                out_dir = ".";
                lut_path = arg5;
                std::cout << "[info] Detected scalar+dims+LUT form. Using current directory as out_dir.\n";
            } else {
                out_dir = arg5;
            }
        } else if (argc == 7) {
            out_dir = argv[5];
            lut_path = argv[6];
        } else {
            std::cerr << "Invalid arguments.\n";
            return 1;
        }
    }

    if (nx <= 0 || ny <= 0 || nz <= 0) {
        std::cerr << "Invalid dimensions.\n";
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);

    std::vector<double> scalar;
    if (!read_double_bin(scalar_path, n, scalar)) {
        return 1;
    }

    std::vector<int> lut(n, -1);
    std::vector<int> vertex_triplets;

    if (has_vertex_type_input) {
        const std::string vertex_type_path = argv[2];
        if (!read_int_bin(vertex_type_path, n * 3, vertex_triplets)) {
            return 1;
        }
        for (std::size_t i = 0; i < n; ++i) {
            lut[i] = vertex_triplets[3 * i];
        }
    } else {
        if (!classify_vertices_from_scalar(scalar, nx, ny, nz, lut_path, vertex_triplets, lut)) {
            return 1;
        }
    }

    {
        std::error_code ec;
        if (std::filesystem::exists(out_dir, ec) && std::filesystem::is_regular_file(out_dir, ec)) {
            std::cerr << "Output path is a file, not a directory: " << out_dir << "\n";
            std::cerr << "Please provide a directory path for out_dir.\n";
            return 1;
        }
        if (!std::filesystem::create_directories(out_dir, ec) && ec) {
            std::cerr << "Failed to create output directory: " << out_dir << "\n";
            std::cerr << "Reason: " << ec.message() << "\n";
            return 1;
        }
    }

    std::vector<NodeRecord> nodes;
    nodes.reserve(n / 20);
    std::unordered_map<int, int> grid_to_node;
    grid_to_node.reserve(n / 20);
    std::vector<int> node_id_of_grid(n, -1);

    std::vector<int> saddle1_grid_ids;
    std::vector<int> saddle2_grid_ids;

    for (std::size_t grid_id = 0; grid_id < n; ++grid_id) {
        const int node_type = map_node_type(lut[grid_id]);
        if (node_type < 0) {
            continue;
        }

        const auto xyz = idx_to_xyz(static_cast<int>(grid_id), nx, ny);
        NodeRecord rec{};
        rec.id = static_cast<int>(nodes.size());
        rec.grid_id = static_cast<int>(grid_id);
        rec.type = node_type;
        rec.x = xyz[0];
        rec.y = xyz[1];
        rec.z = xyz[2];
        rec.value = scalar[grid_id];
        nodes.push_back(rec);

        grid_to_node[rec.grid_id] = rec.id;
        node_id_of_grid[rec.grid_id] = rec.id;
        if (lut[grid_id] == 1) {
            saddle1_grid_ids.push_back(rec.grid_id);
        } else if (lut[grid_id] == 2) {
            saddle2_grid_ids.push_back(rec.grid_id);
        }
    }

    std::vector<int> seed_start;
    std::vector<int> seed_saddle_node;
    std::vector<int> seed_edge_type;
    std::vector<int> seed_target_lut;
    std::vector<int> seed_mode;

    seed_start.reserve((saddle1_grid_ids.size() + saddle2_grid_ids.size()) * 4);
    seed_saddle_node.reserve((saddle1_grid_ids.size() + saddle2_grid_ids.size()) * 4);
    seed_edge_type.reserve((saddle1_grid_ids.size() + saddle2_grid_ids.size()) * 4);
    seed_target_lut.reserve((saddle1_grid_ids.size() + saddle2_grid_ids.size()) * 4);
    seed_mode.reserve((saddle1_grid_ids.size() + saddle2_grid_ids.size()) * 4);

    for (int saddle_grid_id : saddle1_grid_ids) {
        int saddle_node_id = node_id_of_grid[saddle_grid_id];
        const auto neighbors = gather_neighbors(saddle_grid_id, nx, ny, nz);
        for (int nb : neighbors) {
            if (!is_less_shared(nb, saddle_grid_id, scalar[nb], scalar[saddle_grid_id])) {
                continue;
            }
            seed_start.push_back(nb);
            seed_saddle_node.push_back(saddle_node_id);
            seed_edge_type.push_back(0);
            seed_target_lut.push_back(4);
            seed_mode.push_back(1); // ascending
        }
    }

    for (int saddle_grid_id : saddle2_grid_ids) {
        int saddle_node_id = node_id_of_grid[saddle_grid_id];
        const auto neighbors = gather_neighbors(saddle_grid_id, nx, ny, nz);
        for (int nb : neighbors) {
            if (!is_larger_shared(nb, saddle_grid_id, scalar[nb], scalar[saddle_grid_id])) {
                continue;
            }
            seed_start.push_back(nb);
            seed_saddle_node.push_back(saddle_node_id);
            seed_edge_type.push_back(1);
            seed_target_lut.push_back(0);
            seed_mode.push_back(-1); // descending
        }
    }

    const int seed_count = static_cast<int>(seed_start.size());
    std::vector<int> reached(seed_count, -1);
    std::vector<unsigned char> valid(seed_count, 0);

    if (seed_count > 0) {
        double *d_scalar = nullptr;
        int *d_lut = nullptr;
        int *d_seed_start = nullptr;
        int *d_seed_target_lut = nullptr;
        int *d_seed_mode = nullptr;
        int *d_reached = nullptr;
        unsigned char *d_valid = nullptr;

        CUDA_CHECK(cudaMalloc(&d_scalar, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_lut, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_seed_start, seed_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_seed_target_lut, seed_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_seed_mode, seed_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_reached, seed_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_valid, seed_count * sizeof(unsigned char)));

        CUDA_CHECK(cudaMemcpy(d_scalar, scalar.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lut, lut.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seed_start, seed_start.data(), seed_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seed_target_lut, seed_target_lut.data(), seed_count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seed_mode, seed_mode.data(), seed_count * sizeof(int), cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (seed_count + threads - 1) / threads;
        trace_seeds_kernel<<<blocks, threads>>>(
            d_scalar,
            d_lut,
            nx,
            ny,
            nz,
            static_cast<int>(n),
            d_seed_start,
            d_seed_target_lut,
            d_seed_mode,
            seed_count,
            d_reached,
            d_valid);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(reached.data(), d_reached, seed_count * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(valid.data(), d_valid, seed_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_scalar));
        CUDA_CHECK(cudaFree(d_lut));
        CUDA_CHECK(cudaFree(d_seed_start));
        CUDA_CHECK(cudaFree(d_seed_target_lut));
        CUDA_CHECK(cudaFree(d_seed_mode));
        CUDA_CHECK(cudaFree(d_reached));
        CUDA_CHECK(cudaFree(d_valid));
    }

    std::unordered_map<EdgeKey, double, EdgeKeyHash> best_edge_weight;
    best_edge_weight.reserve(seed_count);

    for (int i = 0; i < seed_count; ++i) {
        if (!valid[i]) {
            continue;
        }
        int reached_grid_id = reached[i];
        if (reached_grid_id < 0 || reached_grid_id >= static_cast<int>(n)) {
            continue;
        }
        int target_node_id = node_id_of_grid[reached_grid_id];
        if (target_node_id < 0) {
            continue;
        }

        EdgeKey key{seed_saddle_node[i], target_node_id, seed_edge_type[i]};
        best_edge_weight[key] = 1.0;
    }

    std::vector<EdgeRecord> edges;
    edges.reserve(best_edge_weight.size());
    for (const auto &kv : best_edge_weight) {
        edges.push_back({kv.first.u, kv.first.v, kv.first.edge_type, kv.second});
    }

    std::sort(edges.begin(), edges.end(), [](const EdgeRecord &a, const EdgeRecord &b) {
        if (a.u != b.u) return a.u < b.u;
        if (a.v != b.v) return a.v < b.v;
        return a.edge_type < b.edge_type;
    });

    const std::string nodes_path = (std::filesystem::path(out_dir) / "nodes.bin").string();
    const std::string edges_path = (std::filesystem::path(out_dir) / "edges.bin").string();

    if (!write_nodes_bin(nodes_path, nodes)) {
        return 1;
    }
    if (!write_edges_bin(edges_path, edges)) {
        return 1;
    }

    std::cout << "Done (CUDA tracing).\n"
              << "nodes: " << nodes.size() << " -> " << nodes_path << "\n"
              << "edges: " << edges.size() << " -> " << edges_path << "\n";

    return 0;
}
