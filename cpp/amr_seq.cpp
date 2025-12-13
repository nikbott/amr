#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <memory>
#include <iomanip>
#include <string>
#include <chrono>
#include <fstream>

// =========================================================
// 1. CONFIGURATION (config.py)
// =========================================================

struct AMRConfig {
    int max_level = 21;
    std::vector<double> center = {0.5, 0.5, 0.5};
    double radius = 0.25;
    double bandwidth = 0.05;
    int coarse_level = 4;
    int fine_level = 10;

    uint64_t domain_width() const {
        return 1ULL << max_level;
    }

    std::vector<uint64_t> get_int_center(int ndim) const {
        uint64_t w = domain_width();
        std::vector<uint64_t> c(ndim);
        for(int i=0; i<ndim; ++i) {
            c[i] = static_cast<uint64_t>(center[i] * w);
        }
        return c;
    }

    double get_int_radius() const { return radius * domain_width(); }
    double get_int_bandwidth() const { return bandwidth * domain_width(); }
};

// =========================================================
// 2. MORTON CODES (morton.py)
// =========================================================

const uint64_t INVALID_CODE = UINT64_MAX;

class Morton2D {
    static const uint64_t MASK_1 = 0x00000000FFFFFFFF;
    static const uint64_t MASK_2 = 0x0000FFFF0000FFFF;
    static const uint64_t MASK_3 = 0x00FF00FF00FF00FF;
    static const uint64_t MASK_4 = 0x0F0F0F0F0F0F0F0F;
    static const uint64_t MASK_5 = 0x3333333333333333;
    static const uint64_t MASK_6 = 0x5555555555555555;

    static uint64_t spread(uint64_t n) {
        n &= MASK_1;
        n = (n | (n << 16)) & MASK_2;
        n = (n | (n << 8))  & MASK_3;
        n = (n | (n << 4))  & MASK_4;
        n = (n | (n << 2))  & MASK_5;
        n = (n | (n << 1))  & MASK_6;
        return n;
    }

    static uint64_t compact(uint64_t n) {
        n &= MASK_6;
        n = (n | (n >> 1)) & MASK_5;
        n = (n | (n >> 2)) & MASK_4;
        n = (n | (n >> 4)) & MASK_3;
        n = (n | (n >> 8)) & MASK_2;
        n = (n | (n >> 16)) & MASK_1;
        return n;
    }

public:
    static uint64_t encode(uint64_t x, uint64_t y) {
        return (spread(y) << 1) | spread(x);
    }

    static std::pair<uint64_t, uint64_t> decode(uint64_t code) {
        return {compact(code), compact(code >> 1)};
    }

    static uint64_t get_neighbor(uint64_t code, int level, int dx, int dy, int max_level) {
        auto [x, y] = decode(code);
        int64_t size = 1ULL << (max_level - level);
        
        int64_t nx = static_cast<int64_t>(x) + dx * size;
        int64_t ny = static_cast<int64_t>(y) + dy * size;
        int64_t limit = 1ULL << max_level;

        if (nx < 0 || nx >= limit || ny < 0 || ny >= limit) return INVALID_CODE;
        return encode(static_cast<uint64_t>(nx), static_cast<uint64_t>(ny));
    }
};

class Morton3D {
    static const uint64_t MASK_1 = 0x1FFFFF;
    static const uint64_t MASK_2 = 0x1F00000000FFFF;
    static const uint64_t MASK_3 = 0x1F0000FF0000FF;
    static const uint64_t MASK_4 = 0x100F00F00F00F00F;
    static const uint64_t MASK_5 = 0x10C30C30C30C30C3;
    static const uint64_t MASK_6 = 0x1249249249249249;

    static uint64_t spread(uint64_t n) {
        n &= MASK_1;
        n = (n | (n << 32)) & MASK_2;
        n = (n | (n << 16)) & MASK_3;
        n = (n | (n << 8))  & MASK_4;
        n = (n | (n << 4))  & MASK_5;
        n = (n | (n << 2))  & MASK_6;
        return n;
    }

    static uint64_t compact(uint64_t n) {
        n &= MASK_6;
        n = (n | (n >> 2))  & MASK_5;
        n = (n | (n >> 4))  & MASK_4;
        n = (n | (n >> 8))  & MASK_3;
        n = (n | (n >> 16)) & MASK_2;
        n = (n | (n >> 32)) & MASK_1;
        return n;
    }

public:
    static uint64_t encode(uint64_t x, uint64_t y, uint64_t z) {
        return (spread(z) << 2) | (spread(y) << 1) | spread(x);
    }

    static std::tuple<uint64_t, uint64_t, uint64_t> decode(uint64_t code) {
        return {compact(code), compact(code >> 1), compact(code >> 2)};
    }

    static uint64_t get_neighbor(uint64_t code, int level, int dx, int dy, int dz, int max_level) {
        auto [x, y, z] = decode(code);
        int64_t size = 1ULL << (max_level - level);
        
        int64_t nx = static_cast<int64_t>(x) + dx * size;
        int64_t ny = static_cast<int64_t>(y) + dy * size;
        int64_t nz = static_cast<int64_t>(z) + dz * size;
        int64_t limit = 1ULL << max_level;

        if (nx < 0 || nx >= limit || ny < 0 || ny >= limit || nz < 0 || nz >= limit) return INVALID_CODE;
        return encode(static_cast<uint64_t>(nx), static_cast<uint64_t>(ny), static_cast<uint64_t>(nz));
    }
};

// =========================================================
// 3. TREE STRUCTURE (tree.py)
// =========================================================

struct Node {
    uint64_t code;
    int level;

    // Necessário para std::sort e std::lower_bound
    bool operator<(const Node& other) const {
        return code < other.code;
    }
};

// Interface abstrata
class IOracle {
public:
    virtual bool evaluate(const Node& node, int max_level_grid) const = 0;
    virtual ~IOracle() = default;
};

class VectorOracle : public IOracle {
    const std::vector<uint64_t>& targets; // Referência para vetor ORDENADO
public:
    VectorOracle(const std::vector<uint64_t>& t) : targets(t) {}

    bool evaluate(const Node& n, int) const override {
        // Pesquisa binária rápida (O(log K))
        return std::binary_search(targets.begin(), targets.end(), n.code);
    }
};

class AbstractLinearTree {
protected:
    int max_level;
    
public:
    std::vector<Node> leaves;

    AbstractLinearTree(int max_lvl) : max_level(max_lvl) {
        leaves.push_back({0, 0}); // Root node
    }

    virtual ~AbstractLinearTree() = default;
    virtual int get_ndim() const = 0;
    virtual std::vector<uint64_t> decode_impl(uint64_t code) const = 0;
    virtual uint64_t encode_impl(const std::vector<uint64_t>& coords) const = 0;
    virtual uint64_t get_neighbor_code(uint64_t code, int level, const std::vector<int>& offset) const = 0;

    uint64_t domain_width() const {
        return 1ULL << max_level;
    }

    std::pair<std::vector<uint64_t>, uint64_t> get_geometry(const Node& node) const {
        auto coords = decode_impl(node.code);
        uint64_t size = 1ULL << (max_level - node.level);
        return {coords, size};
    }

    bool refine(const IOracle& oracle) {
        std::vector<Node> new_leaves;
        bool has_changed = false;
        new_leaves.reserve(leaves.size() * 1.5); 

        int ndim = get_ndim();
        std::vector<std::vector<int>> offsets;
        if (ndim == 2) {
            offsets = {{0,0}, {1,0}, {0,1}, {1,1}};
        } else {
            offsets = {{0,0,0}, {1,0,0}, {0,1,0}, {1,1,0},
                       {0,0,1}, {1,0,1}, {0,1,1}, {1,1,1}};
        }

        for (const auto& node : leaves) {
            if (oracle.evaluate(node, max_level)) {
                has_changed = true;
                auto current_coords = decode_impl(node.code);
                int new_lvl = node.level + 1;
                uint64_t step = 1ULL << (max_level - new_lvl);

                for (const auto& off : offsets) {
                    std::vector<uint64_t> child_coords(ndim);
                    for (int i = 0; i < ndim; ++i) {
                        child_coords[i] = current_coords[i] + off[i] * step;
                    }
                    new_leaves.push_back({encode_impl(child_coords), new_lvl});
                }
            } else {
                new_leaves.push_back(node);
            }
        }

        if (has_changed) {
            std::sort(new_leaves.begin(), new_leaves.end());
            leaves = std::move(new_leaves);
        }
        return has_changed;
    }

    // --- Versão Otimizada do Balance (Vector-based) ---
    void balance() {
        int max_iter = 20;
        int ndim = get_ndim();

        std::vector<std::vector<int>> directions;
        if (ndim == 2) {
            for(int dx=-1; dx<=1; ++dx)
                for(int dy=-1; dy<=1; ++dy)
                    if(!(dx==0 && dy==0)) directions.push_back({dx, dy});
        } else {
            for(int dx=-1; dx<=1; ++dx)
                for(int dy=-1; dy<=1; ++dy)
                    for(int dz=-1; dz<=1; ++dz)
                        if(!(dx==0 && dy==0 && dz==0)) directions.push_back({dx, dy, dz});
        }

        for (int iter = 0; iter < max_iter; ++iter) {
            // Usa vector em vez de set para performance
            std::vector<uint64_t> to_refine_codes;

            for (const auto& node : leaves) {
                int min_valid_level = node.level - 1;
                if (min_valid_level < 1) continue;

                for (const auto& dir : directions) {
                    uint64_t base_n_code = get_neighbor_code(node.code, node.level, dir);
                    if (base_n_code == INVALID_CODE) continue;

                    int curr_search_level = node.level - 2;
                    while (curr_search_level >= 0) {
                        int shift = (max_level - curr_search_level) * ndim;
                        uint64_t mask = ~((1ULL << shift) - 1ULL);
                        uint64_t coarse_n_code = base_n_code & mask;

                        // OTIMIZAÇÃO: std::lower_bound em vez de Map lookup
                        Node target_search = {coarse_n_code, 0};
                        auto it = std::lower_bound(leaves.begin(), leaves.end(), target_search);

                        if (it != leaves.end() && it->code == coarse_n_code) {
                            if (it->level == curr_search_level) {
                                to_refine_codes.push_back(it->code);
                                break; 
                            }
                        }
                        curr_search_level--;
                    }
                }
            }

            if (to_refine_codes.empty()) break;

            // Ordena e remove duplicados para preparar para o VectorOracle
            std::sort(to_refine_codes.begin(), to_refine_codes.end());
            to_refine_codes.erase(std::unique(to_refine_codes.begin(), to_refine_codes.end()), to_refine_codes.end());

            VectorOracle vo(to_refine_codes);
            refine(vo);
        }
    }
};

class Quadtree : public AbstractLinearTree {
public:
    Quadtree(int max_level) : AbstractLinearTree(max_level) {}
    int get_ndim() const override { return 2; }
    
    std::vector<uint64_t> decode_impl(uint64_t code) const override {
        auto p = Morton2D::decode(code);
        return {p.first, p.second};
    }
    uint64_t encode_impl(const std::vector<uint64_t>& c) const override {
        return Morton2D::encode(c[0], c[1]);
    }
    uint64_t get_neighbor_code(uint64_t code, int level, const std::vector<int>& off) const override {
        return Morton2D::get_neighbor(code, level, off[0], off[1], max_level);
    }
};

class Octree : public AbstractLinearTree {
public:
    Octree(int max_level) : AbstractLinearTree(max_level) {}
    int get_ndim() const override { return 3; }
    
    std::vector<uint64_t> decode_impl(uint64_t code) const override {
        auto t = Morton3D::decode(code);
        return {std::get<0>(t), std::get<1>(t), std::get<2>(t)};
    }
    uint64_t encode_impl(const std::vector<uint64_t>& c) const override {
        return Morton3D::encode(c[0], c[1], c[2]);
    }
    uint64_t get_neighbor_code(uint64_t code, int level, const std::vector<int>& off) const override {
        return Morton3D::get_neighbor(code, level, off[0], off[1], off[2], max_level);
    }
};

// =========================================================
// 4. PHYSICS (physics.py)
// =========================================================

class CircleOracle2D : public IOracle {
    AMRConfig config;
    uint64_t cx, cy;
    double radius, bandwidth;
    int max_depth, min_depth;

public:
    CircleOracle2D(AMRConfig cfg) : config(cfg) {
        auto center_ints = config.get_int_center(2);
        cx = center_ints[0]; cy = center_ints[1];
        radius = config.get_int_radius();
        bandwidth = config.get_int_bandwidth();
        max_depth = config.fine_level;
        min_depth = config.coarse_level;
    }

    bool evaluate(const Node& node, int max_level_grid) const override {
        auto [x, y] = Morton2D::decode(node.code);
        uint64_t size = 1ULL << (max_level_grid - node.level);

        double node_cx = x + size * 0.5;
        double node_cy = y + size * 0.5;

        double dist_sq = std::pow(node_cx - cx, 2) + std::pow(node_cy - cy, 2);

        double extent = size * 0.70710678; 
        double threshold = bandwidth + extent;

        double upper_bound = radius + threshold;
        double upper_sq = upper_bound * upper_bound;

        double lower_bound = radius - threshold;
        double lower_sq = (lower_bound < 0) ? 0.0 : lower_bound * lower_bound;

        bool is_refining = (dist_sq < upper_sq) && (dist_sq > lower_sq);

        return (node.level < min_depth) || (is_refining && node.level < max_depth);
    }
};

// =========================================================
// 5. VISUALIZATION
// =========================================================

class MeshVisualizer {
public:
    // Exports the Quadtree to an SVG file viewable in any browser
    static void save_svg(const Quadtree& tree, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        uint64_t limit = tree.domain_width();
        double scale_factor = 1000.0 / limit; // Map domain to 1000x1000 pixels

        // SVG Header
        file << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" "
             << "width=\"1000\" height=\"1000\" viewBox=\"0 0 1000 1000\">\n";
        
        // Background
        file << "<rect width=\"1000\" height=\"1000\" fill=\"white\"/>\n";

        // Draw Leaves
        for (const auto& node : tree.leaves) {
            auto [coords, size] = tree.get_geometry(node);
            
            // SVG coordinate system has (0,0) at top-left. 
            // We need to flip Y to match standard cartesian (bottom-left).
            double x = coords[0] * scale_factor;
            double raw_y = coords[1] * scale_factor;
            double h = size * scale_factor;
            double y = 1000.0 - raw_y - h; // Flip Y

            file << "<rect x=\"" << x << "\" y=\"" << y 
                 << "\" width=\"" << h << "\" height=\"" << h 
                 << "\" style=\"fill:none;stroke:red;stroke-width:0.5\" />\n";
        }

        // Add Title/Info text
        file << "<text x=\"10\" y=\"25\" font-family=\"Arial\" font-size=\"20\" fill=\"black\">"
             << "Elements: " << tree.leaves.size() << "</text>\n";

        file << "</svg>";
        file.close();
        std::cout << "[Viz] Saved 2D mesh to " << filename << std::endl;
    }
};

// =========================================================
// 6. MAIN
// =========================================================

int main(int argc, char** argv) {
    // Parse command-line arguments (naive check for help)
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --max_level <int>    Set maximum refinement level (default: 15)" << std::endl;
            std::cout << "  --fine_level <int>   Set fine refinement level (default: 9)" << std::endl;
            std::cout << "  --help, -h           Show this help message" << std::endl;
            return 0;
        }
    }

    std::cout << "--- C++ Sequential AMR Benchmark ---" << std::endl;
    
    AMRConfig cfg;
    cfg.max_level = 20; 
    cfg.fine_level = 12;

    // Parse configuration arguments
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--max_level" && i + 1 < argc) {
            cfg.max_level = std::stoi(argv[++i]);
        } else if (std::string(argv[i]) == "--fine_level" && i + 1 < argc) {
            cfg.fine_level = std::stoi(argv[++i]);
        }
    }
    cfg.coarse_level = 4;
    cfg.center = {0.5, 0.5};
    cfg.radius = 0.25;

    std::cout << "Config: 2D Quadtree, MaxLvl=" << cfg.max_level << std::endl;

    Quadtree tree(cfg.max_level);
    CircleOracle2D oracle(cfg);

    int max_steps = 14;
    
    std::cout << std::left << std::setw(10) << "Step" 
              << "| " << std::setw(15) << "Elements" 
              << "| " << std::setw(12) << "Time (ms)" << std::endl;
    std::cout << std::string(45, '-') << std::endl;

    auto total_start = std::chrono::high_resolution_clock::now();

    for(int step = 0; step < max_steps; ++step) {
        auto step_start = std::chrono::high_resolution_clock::now();
        
        bool changed = tree.refine(oracle);
        
        auto step_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start);
        
        std::cout << std::left << std::setw(10) << step 
                  << "| " << std::setw(15) << tree.leaves.size() 
                  << "| " << std::setw(12) << duration.count() << std::endl;

        if (!changed) {
            std::cout << "Converged early." << std::endl;
            break;
        }
    }

    std::cout << "\nRunning Balance Constraint..." << std::endl;
    size_t before = tree.leaves.size();
    
    auto balance_start = std::chrono::high_resolution_clock::now();
    tree.balance();
    auto balance_end = std::chrono::high_resolution_clock::now();
    
    auto balance_duration = std::chrono::duration_cast<std::chrono::milliseconds>(balance_end - balance_start);
    
    size_t after = tree.leaves.size();
    std::cout << "Balance complete. Elements: " << before << " -> " << after << std::endl;
    std::cout << "Balance time: " << balance_duration.count() << " ms" << std::endl;

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    
    std::cout << "\nTotal execution time: " << total_duration.count() << " ms" << std::endl;

    // Save SVG
    std::string filename = "mesh_2d_seq.svg";
    MeshVisualizer::save_svg(tree, filename);

    return 0;
}
