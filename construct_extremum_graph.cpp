#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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

struct TraceResult {
    int reached_grid_id;
    double length;
    bool ok;
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

inline double step_distance(int a, int b, int nx, int ny) {
    const auto xyz_a = idx_to_xyz(a, nx, ny);
    const auto xyz_b = idx_to_xyz(b, nx, ny);
    const int dx = xyz_b[0] - xyz_a[0];
    const int dy = xyz_b[1] - xyz_a[1];
    const int dz = xyz_b[2] - xyz_a[2];
    return std::sqrt(static_cast<double>(dx * dx + dy * dy + dz * dz));
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

TraceResult trace_ascending_to_max(
    int start,
    const std::vector<double> &scalar,
    const std::vector<int> &lut,
    int nx,
    int ny,
    int nz) {

    const int n = static_cast<int>(scalar.size());
    int current = start;
    double length = 0.0;
    std::unordered_set<int> visited;
    visited.reserve(256);

    for (int iter = 0; iter < n; ++iter) {
        if (lut[current] == 4) {
            return {current, length, true};
        }

        if (visited.find(current) != visited.end()) {
            return {-1, length, false};
        }
        visited.insert(current);

        const auto neighbors = gather_neighbors(current, nx, ny, nz);
        int best_next = -1;
        double best_value = 0.0;

        for (int nb : neighbors) {
            const double nb_value = scalar[nb];
            const double cur_value = scalar[current];
            if (!is_larger_shared(nb, current, nb_value, cur_value)) {
                continue;
            }

            if (best_next == -1 || is_larger_shared(nb, best_next, nb_value, best_value)) {
                best_next = nb;
                best_value = nb_value;
            }
        }

        if (best_next == -1) {
            if (lut[current] == 4) {
                return {current, length, true};
            }
            return {-1, length, false};
        }

        length += step_distance(current, best_next, nx, ny);
        current = best_next;
    }

    return {-1, length, false};
}

TraceResult trace_descending_to_min(
    int start,
    const std::vector<double> &scalar,
    const std::vector<int> &lut,
    int nx,
    int ny,
    int nz) {

    const int n = static_cast<int>(scalar.size());
    int current = start;
    double length = 0.0;
    std::unordered_set<int> visited;
    visited.reserve(256);

    for (int iter = 0; iter < n; ++iter) {
        if (lut[current] == 0) {
            return {current, length, true};
        }

        if (visited.find(current) != visited.end()) {
            return {-1, length, false};
        }
        visited.insert(current);

        const auto neighbors = gather_neighbors(current, nx, ny, nz);
        int best_next = -1;
        double best_value = 0.0;

        for (int nb : neighbors) {
            const double nb_value = scalar[nb];
            const double cur_value = scalar[current];
            if (!is_less_shared(nb, current, nb_value, cur_value)) {
                continue;
            }

            if (best_next == -1 || is_less_shared(nb, best_next, nb_value, best_value)) {
                best_next = nb;
                best_value = nb_value;
            }
        }

        if (best_next == -1) {
            if (lut[current] == 0) {
                return {current, length, true};
            }
            return {-1, length, false};
        }

        length += step_distance(current, best_next, nx, ny);
        current = best_next;
    }

    return {-1, length, false};
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

}  // namespace

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
        if (lut[grid_id] == 1) {
            saddle1_grid_ids.push_back(rec.grid_id);
        } else if (lut[grid_id] == 2) {
            saddle2_grid_ids.push_back(rec.grid_id);
        }
    }

    std::unordered_map<EdgeKey, double, EdgeKeyHash> best_edge_weight;
    best_edge_weight.reserve((saddle1_grid_ids.size() + saddle2_grid_ids.size()) * 4);

    for (int saddle_grid_id : saddle1_grid_ids) {
        const int saddle_node_id = grid_to_node[saddle_grid_id];
        const auto neighbors = gather_neighbors(saddle_grid_id, nx, ny, nz);

        for (int nb : neighbors) {
            if (!is_less_shared(nb, saddle_grid_id, scalar[nb], scalar[saddle_grid_id])) {
                continue;
            }
            const TraceResult tr = trace_ascending_to_max(nb, scalar, lut, nx, ny, nz);
            if (!tr.ok) {
                continue;
            }
            auto it = grid_to_node.find(tr.reached_grid_id);
            if (it == grid_to_node.end()) {
                continue;
            }
            const int max_node_id = it->second;
            const int edge_type = 0;
            const EdgeKey key{saddle_node_id, max_node_id, edge_type};
            auto old = best_edge_weight.find(key);
            if (old == best_edge_weight.end() || tr.length < old->second) {
                best_edge_weight[key] = tr.length;
            }
        }
    }

    for (int saddle_grid_id : saddle2_grid_ids) {
        const int saddle_node_id = grid_to_node[saddle_grid_id];
        const auto neighbors = gather_neighbors(saddle_grid_id, nx, ny, nz);

        for (int nb : neighbors) {
            if (!is_larger_shared(nb, saddle_grid_id, scalar[nb], scalar[saddle_grid_id])) {
                continue;
            }
            const TraceResult tr = trace_descending_to_min(nb, scalar, lut, nx, ny, nz);
            if (!tr.ok) {
                continue;
            }
            auto it = grid_to_node.find(tr.reached_grid_id);
            if (it == grid_to_node.end()) {
                continue;
            }
            const int min_node_id = it->second;
            const int edge_type = 1;
            const EdgeKey key{saddle_node_id, min_node_id, edge_type};
            auto old = best_edge_weight.find(key);
            if (old == best_edge_weight.end() || tr.length < old->second) {
                best_edge_weight[key] = tr.length;
            }
        }
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

    std::cout << "Done.\n"
              << "nodes: " << nodes.size() << " -> " << nodes_path << "\n"
              << "edges: " << edges.size() << " -> " << edges_path << "\n";

    return 0;
}
