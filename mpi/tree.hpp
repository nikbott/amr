/**
 * @file tree.hpp
 * @brief Distributed Linear Tree Implementation (MPI + OpenMP).
 */
#pragma once
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <vector>

#include "core.hpp"

namespace amr {

struct Node {
    MortonCode code;
    int level;
    bool operator<(const Node& other) const {
        if (code.value != other.code.value)
            return code.value < other.code.value;
        return level < other.level;
    }
    bool operator==(const Node& other) const {
        return code.value == other.code.value && level == other.level;
    }
};

template <typename T>
concept RefinementOracle = requires(T t, const Node& n, int max_lvl) {
    { t(n, max_lvl) } -> std::convertible_to<bool>;
};

template <int DIM>
class DistributedTree {
    static_assert(DIM == 2 || DIM == 3, "Only 2D or 3D trees supported.");

public:
    using Point = std::array<Coordinate, DIM>;
    const int max_level;

private:
    std::vector<uint64_t> leaf_codes;
    std::vector<uint8_t> leaf_levels;
    std::vector<Node> ghost_nodes;
    std::vector<uint64_t> partition_starts;

    int mpi_rank, mpi_size;

    std::vector<uint64_t> wksp_counts;
    std::vector<uint64_t> wksp_offsets;
    std::vector<uint64_t> wksp_scan_buffer;
    std::vector<uint8_t> wksp_flags;

public:
    explicit DistributedTree(int max_lvl) : max_level(max_lvl) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        if (mpi_rank == 0) {
            leaf_codes.push_back(0);
            leaf_levels.push_back(0);
        }
        update_partition_map();
    }

    [[nodiscard]] size_t local_size() const { return leaf_codes.size(); }
    [[nodiscard]] size_t global_size() const {
        return mpi::all_reduce_sum<uint64_t>(leaf_codes.size());
    }
    [[nodiscard]] const std::vector<Node>& get_ghosts() const { return ghost_nodes; }
    [[nodiscard]] uint64_t domain_width() const { return 1ULL << max_level; }

    Node operator[](size_t i) const {
        return {MortonCode{leaf_codes[i]}, static_cast<int>(leaf_levels[i])};
    }

    class ConstIterator {
        const DistributedTree* tree;
        size_t index;

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = Node;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = Node;
        ConstIterator(const DistributedTree* t, size_t i) : tree(t), index(i) {}
        reference operator*() const { return (*tree)[index]; }
        struct ArrowProxy {
            Node value;
            const Node* operator->() const { return &value; }
        };
        ArrowProxy operator->() const { return ArrowProxy{**this}; }
        ConstIterator& operator++() {
            ++index;
            return *this;
        }
        ConstIterator operator++(int) {
            ConstIterator tmp = *this;
            ++index;
            return tmp;
        }
        ConstIterator& operator--() {
            --index;
            return *this;
        }
        ConstIterator operator--(int) {
            ConstIterator tmp = *this;
            --index;
            return tmp;
        }
        ConstIterator& operator+=(difference_type n) {
            index += n;
            return *this;
        }
        ConstIterator& operator-=(difference_type n) {
            index -= n;
            return *this;
        }
        friend ConstIterator operator+(ConstIterator it, difference_type n) {
            return {it.tree, it.index + n};
        }
        friend ConstIterator operator+(difference_type n, ConstIterator it) {
            return {it.tree, it.index + n};
        }
        friend ConstIterator operator-(ConstIterator it, difference_type n) {
            return {it.tree, it.index - n};
        }
        friend difference_type operator-(const ConstIterator& a, const ConstIterator& b) {
            return static_cast<difference_type>(a.index) - static_cast<difference_type>(b.index);
        }
        auto operator<=>(const ConstIterator& other) const = default;
    };
    ConstIterator begin() const { return ConstIterator(this, 0); }
    ConstIterator end() const { return ConstIterator(this, leaf_codes.size()); }

    [[nodiscard]] Point decode(MortonCode code) const {
        if constexpr (DIM == 2)
            return morton::decode_2d(code);
        else
            return morton::decode_3d(code);
    }

    MortonCode get_neighbor_code(MortonCode code,
                                 int level,
                                 const std::array<int, DIM>& dir) const {
        uint64_t c = code.value;
        uint64_t mask_x = (DIM == 3) ? morton::MASK3_X : morton::MASK2_X;
        uint64_t mask_y = (DIM == 3) ? morton::MASK3_Y : 0xAAAAAAAAAAAAAAAA;
        uint64_t mask_z = (DIM == 3) ? morton::MASK3_Z : 0;
        auto add_dim = [&](uint64_t current, uint64_t dim_mask, int d) -> uint64_t {
            if (current == std::numeric_limits<uint64_t>::max())
                return current;
            if (d == 0)
                return current;
            // A level-0 cell spans the whole domain: any neighbour is out of
            // bounds, and returning here avoids the 1<<64 shift at the max level.
            if (level == 0)
                return std::numeric_limits<uint64_t>::max();
            uint64_t shift_coord = max_level - level;
            uint64_t one_dilated =
                (DIM == 3) ? (1ULL << (shift_coord * 3)) : (1ULL << (shift_coord * 2));
            if (dim_mask == mask_y)
                one_dilated <<= 1;
            if (dim_mask == mask_z)
                one_dilated <<= 2;
            if (d > 0) {
                // Out of the domain iff this axis already sits at its maximum
                // aligned coordinate for this level. (The old all-ones test only
                // fired at full resolution, so a coarser boundary cell's +neighbour
                // wrapped instead of returning the sentinel.)
                const uint64_t domain_mask =
                    (DIM * max_level >= 64) ? ~0ULL : ((1ULL << (DIM * max_level)) - 1);
                const uint64_t axis_max = (dim_mask & domain_mask) & ~(one_dilated - 1);
                if ((current & dim_mask) == axis_max)
                    return std::numeric_limits<uint64_t>::max();
                uint64_t sum = (current | ~dim_mask) + one_dilated;
                return (sum & dim_mask) | (current & ~dim_mask);
            } else {
                if ((current & dim_mask) < one_dilated)
                    return std::numeric_limits<uint64_t>::max();
                uint64_t diff = (current & dim_mask) - one_dilated;
                return (diff & dim_mask) | (current & ~dim_mask);
            }
        };
        uint64_t next = add_dim(c, mask_x, dir[0]);
        next = add_dim(next, mask_y, dir[1]);
        if constexpr (DIM == 3)
            next = add_dim(next, mask_z, dir[2]);
        return MortonCode{next};
    }

    template <typename Oracle>
    bool refine(const Oracle& oracle) {
        bool changed = refine_local_pass(oracle);
        if (!mpi::all_reduce_or(changed))
            return false;
        update_partition_map();
        return true;
    }

    template <typename Oracle>
    bool coarsen(const Oracle& oracle) {
        size_t n = leaf_codes.size();
        constexpr int siblings = 1 << DIM;
        bool changed = false;

        // Local marking pass. A rank whose partition is empty (n == 0) does no
        // work here, but it MUST still reach the collectives below
        // (all_reduce_or and update_partition_map). Returning early on n == 0
        // would skip those MPI_Allreduce/Allgather calls and deadlock every
        // rank that still has leaves to coarsen.
        if (n > 0) {
            wksp_flags.assign(n, 0);
#pragma omp parallel for schedule(static) reduction(| : changed)
            for (size_t i = 0; i < n; ++i) {
                if (i + siblings > n)
                    continue;
                uint64_t raw_code = leaf_codes[i];
                int lvl = leaf_levels[i];
                if (lvl == 0)
                    continue;
                uint64_t shift = static_cast<uint64_t>(DIM) * (max_level - lvl);
                uint64_t mask_siblings = (1ULL << DIM) - 1;
                if (((raw_code >> shift) & mask_siblings) != 0)
                    continue;
                bool valid_family = true;
                for (int k = 1; k < siblings; ++k) {
                    if (leaf_levels[i + k] != lvl) {
                        valid_family = false;
                        break;
                    }
                }
                if (valid_family && !oracle({MortonCode{raw_code}, lvl - 1}, max_level)) {
                    wksp_flags[i] = 1;
                    for (int k = 1; k < siblings; ++k)
                        wksp_flags[i + k] = 2;
                    changed = true;
                }
            }
        }

        // Collective: every rank participates regardless of local emptiness.
        if (!mpi::all_reduce_or(changed))
            return false;

        if (n > 0) {
            std::vector<uint64_t> keep_mask(n);
#pragma omp parallel for
            for (size_t i = 0; i < n; ++i)
                keep_mask[i] = (wksp_flags[i] != 2) ? 1 : 0;
            wksp_offsets.resize(n);
            parallel::exclusive_scan(keep_mask, wksp_offsets, wksp_scan_buffer);
            size_t new_size = wksp_offsets.back() + keep_mask.back();
            std::vector<uint64_t> nc(new_size);
            std::vector<uint8_t> nl(new_size);
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                if (wksp_flags[i] == 2)
                    continue;
                size_t pos = wksp_offsets[i];
                nc[pos] = leaf_codes[i];
                nl[pos] = (wksp_flags[i] == 1) ? (uint8_t)(leaf_levels[i] - 1) : leaf_levels[i];
            }
            leaf_codes = std::move(nc);
            leaf_levels = std::move(nl);
        }

        // Collective: rebuilds the global partition map across all ranks.
        update_partition_map();
        return true;
    }

    void balance() {
        // Full 2:1 balance -- 6 face + 12 edge directions in 3D so every shared
        // edge is within one level, matching omp/tree.hpp. See there for why the
        // DIC bridge needs edge completeness; 2D is already edge-complete.
        std::vector<std::array<int, DIM>> dirs;
        if constexpr (DIM == 2)
            dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        else
            dirs = {{1, 0, 0},
                    {-1, 0, 0},
                    {0, 1, 0},
                    {0, -1, 0},
                    {0, 0, 1},
                    {0, 0, -1},
                    {1, 1, 0},
                    {1, -1, 0},
                    {-1, 1, 0},
                    {-1, -1, 0},
                    {1, 0, 1},
                    {1, 0, -1},
                    {-1, 0, 1},
                    {-1, 0, -1},
                    {0, 1, 1},
                    {0, 1, -1},
                    {0, -1, 1},
                    {0, -1, -1}};
        while (true) {
            update_partition_map();
            fetch_ghosts();
            bool changed = false;
            size_t n = leaf_codes.size();
            wksp_flags.assign(n, 0);
#pragma omp parallel for reduction(| : changed)
            for (size_t i = 0; i < n; ++i) {
                MortonCode code{leaf_codes[i]};
                int lvl = leaf_levels[i];
                uint64_t my_size = 1ULL << (DIM * (max_level - lvl));
                for (const auto& dir : dirs) {
                    MortonCode n_code = get_neighbor_code(code, lvl, dir);
                    if (n_code.value == std::numeric_limits<uint64_t>::max())
                        continue;
                    uint64_t search_start = n_code.value;
                    uint64_t search_end = n_code.value + my_size;
                    int max_n_level = -1;
                    auto it_local =
                        std::lower_bound(leaf_codes.begin(), leaf_codes.end(), search_start);
                    if (it_local != leaf_codes.begin()) {
                        size_t prev_idx = std::distance(leaf_codes.begin(), it_local - 1);
                        uint64_t prev_c = leaf_codes[prev_idx];
                        uint64_t prev_sz = 1ULL << (DIM * (max_level - leaf_levels[prev_idx]));
                        if (prev_c + prev_sz > search_start)
                            max_n_level = std::max(max_n_level, (int)leaf_levels[prev_idx]);
                    }
                    while (it_local != leaf_codes.end() && *it_local < search_end) {
                        max_n_level =
                            std::max(max_n_level,
                                     (int)leaf_levels[std::distance(leaf_codes.begin(), it_local)]);
                        ++it_local;
                    }
                    Node key{MortonCode{search_start}, 0};
                    auto it_ghost = std::lower_bound(ghost_nodes.begin(), ghost_nodes.end(), key);
                    if (it_ghost != ghost_nodes.begin()) {
                        auto prev = it_ghost - 1;
                        uint64_t prev_sz = 1ULL << (DIM * (max_level - prev->level));
                        if (prev->code.value + prev_sz > search_start)
                            max_n_level = std::max(max_n_level, prev->level);
                    }
                    while (it_ghost != ghost_nodes.end() && it_ghost->code.value < search_end) {
                        max_n_level = std::max(max_n_level, it_ghost->level);
                        ++it_ghost;
                    }
                    if (max_n_level != -1 && max_n_level > lvl + 1) {
                        wksp_flags[i] = 1;
                        changed = true;
                    }
                }
            }
            if (!mpi::all_reduce_or(changed))
                break;
            apply_refinement_from_flags();
        }
    }

    void verify_global() const {
        double local_vol = 0.0;
        for (size_t i = 0; i < leaf_codes.size(); ++i)
            local_vol += std::pow(1.0 / (1ULL << leaf_levels[i]), DIM);
        double total_vol = mpi::all_reduce_sum(local_vol);
        if (std::abs(total_vol - 1.0) > 1e-9)
            throw std::runtime_error("Partition of Unity Violation");
    }

    /**
     * @brief Global Load Balancing (Z-Curve Partitioning).
     * @details Redistributes leaves so each rank owns ~N/P elements.
     */
    void repartition() {
        // 1. Calculate Global Offsets
        size_t n_local = leaf_codes.size();
        size_t n_global = global_size();
        size_t offset_global = 0;
        MPI_Exscan(&n_local, &offset_global, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

        // 2. Determine Target Ranges
        // Rank i owns global indices [start, end)
        auto get_rank_range = [&](int r) -> std::pair<size_t, size_t> {
            size_t size_u = static_cast<size_t>(mpi_size);
            size_t r_u = static_cast<size_t>(r);

            size_t avg = n_global / size_u;
            size_t rem = n_global % size_u;

            // Distribute the remainder 'rem' among the first 'rem' ranks
            size_t start = r_u * avg + std::min(r_u, rem);
            size_t end =
                start + avg + (r_u < rem ? 1 : 0);  // Fixed: Comparison is now size_t vs size_t

            return {start, end};
        };

        // 3. Identify Nodes to Send
        std::vector<std::vector<uint64_t>> send_c(mpi_size);
        std::vector<std::vector<uint8_t>> send_l(mpi_size);

        for (int r = 0; r < mpi_size; ++r) {
            auto [r_start, r_end] = get_rank_range(r);

            // Intersection of "My Current Data" [offset, offset+n) and "Rank R's Target"
            size_t i_start = std::max(offset_global, r_start);
            size_t i_end = std::min(offset_global + n_local, r_end);

            if (i_start < i_end) {
                size_t local_idx_start = i_start - offset_global;
                size_t count = i_end - i_start;

                send_c[r].insert(send_c[r].end(),
                                 leaf_codes.begin() + local_idx_start,
                                 leaf_codes.begin() + local_idx_start + count);
                send_l[r].insert(send_l[r].end(),
                                 leaf_levels.begin() + local_idx_start,
                                 leaf_levels.begin() + local_idx_start + count);
            }
        }

        // 4. Exchange Data (Reuse existing pattern)
        auto exchange_vec = [&](auto& sends, auto& recvs) {
            std::vector<int> sc(mpi_size), rc(mpi_size), sdisp(mpi_size + 1), rdisp(mpi_size + 1);

            // Safe downcast to int for MPI counts (assuming < 2GB per message)
            for (int i = 0; i < mpi_size; ++i)
                sc[i] = static_cast<int>(
                    sends[i].size() *
                    sizeof(typename std::remove_reference_t<decltype(sends[0])>::value_type));

            MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, MPI_COMM_WORLD);
            for (int i = 0; i < mpi_size; ++i) {
                sdisp[i + 1] = sdisp[i] + sc[i];
                rdisp[i + 1] = rdisp[i] + rc[i];
            }

            std::vector<uint8_t> sflat(sdisp.back()), rflat(rdisp.back());
            for (int i = 0; i < mpi_size; ++i)
                memcpy(sflat.data() + sdisp[i], sends[i].data(), sc[i]);

            MPI_Alltoallv(sflat.data(),
                          sc.data(),
                          sdisp.data(),
                          MPI_BYTE,
                          rflat.data(),
                          rc.data(),
                          rdisp.data(),
                          MPI_BYTE,
                          MPI_COMM_WORLD);

            for (int i = 0; i < mpi_size; ++i) {
                recvs[i].resize(
                    rc[i] /
                    sizeof(typename std::remove_reference_t<decltype(recvs[0])>::value_type));
                memcpy(recvs[i].data(), rflat.data() + rdisp[i], rc[i]);
            }
        };

        std::vector<std::vector<uint64_t>> recv_c(mpi_size);
        std::vector<std::vector<uint8_t>> recv_l(mpi_size);

        exchange_vec(send_c, recv_c);
        exchange_vec(send_l, recv_l);

        // 5. Rebuild Local Tree
        leaf_codes.clear();
        leaf_levels.clear();
        for (int r = 0; r < mpi_size; ++r) {
            leaf_codes.insert(leaf_codes.end(), recv_c[r].begin(), recv_c[r].end());
            leaf_levels.insert(leaf_levels.end(), recv_l[r].begin(), recv_l[r].end());
        }

        update_partition_map();
    }

private:
    void update_partition_map() {
        uint64_t first =
            leaf_codes.empty() ? std::numeric_limits<uint64_t>::max() : leaf_codes.front();
        partition_starts = mpi::build_partition_map(first);
    }

    int find_owner_of_point(uint64_t code) const {
        auto it = std::upper_bound(partition_starts.begin(), partition_starts.end(), code);
        int rank = std::distance(partition_starts.begin(), it) - 1;
        if (rank < 0)
            return 0;
        if (rank >= mpi_size)
            return mpi_size - 1;
        return rank;
    }

    void fetch_ghosts() {
        ghost_nodes.clear();
        std::vector<Node> needed;
        // Must match balance()'s stencil: edge-diagonal neighbours can live on
        // another rank, so we fetch ghosts along all 12 edge directions too --
        // otherwise balance() cannot see a cross-rank edge neighbour to correct.
        std::vector<std::array<int, DIM>> dirs;
        if constexpr (DIM == 2)
            dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        else
            dirs = {{1, 0, 0},
                    {-1, 0, 0},
                    {0, 1, 0},
                    {0, -1, 0},
                    {0, 0, 1},
                    {0, 0, -1},
                    {1, 1, 0},
                    {1, -1, 0},
                    {-1, 1, 0},
                    {-1, -1, 0},
                    {1, 0, 1},
                    {1, 0, -1},
                    {-1, 0, 1},
                    {-1, 0, -1},
                    {0, 1, 1},
                    {0, 1, -1},
                    {0, -1, 1},
                    {0, -1, -1}};
        uint64_t my_start = leaf_codes.empty() ? 0 : leaf_codes.front();
        uint64_t my_end =
            leaf_codes.empty()
                ? 0
                : leaf_codes.back() + (1ULL << (DIM * (max_level - leaf_levels.back())));
#pragma omp parallel
        {
            std::vector<Node> local_n;
#pragma omp for
            for (size_t i = 0; i < leaf_codes.size(); ++i) {
                MortonCode code{leaf_codes[i]};
                int lvl = leaf_levels[i];
                for (auto& d : dirs) {
                    MortonCode n = get_neighbor_code(code, lvl, d);
                    if (n.value == std::numeric_limits<uint64_t>::max())
                        continue;
                    uint64_t n_end = n.value + (1ULL << (DIM * (max_level - lvl)));
                    if (n.value < my_start || n_end > my_end)
                        local_n.push_back({n, lvl});
                }
            }
#pragma omp critical
            needed.insert(needed.end(), local_n.begin(), local_n.end());
        }
        std::sort(needed.begin(), needed.end());
        needed.erase(std::unique(needed.begin(), needed.end()), needed.end());
        std::vector<std::vector<Node>> send_bufs(mpi_size), recv_bufs(mpi_size);
        // Locate the owner of req_start by binary search on the sorted partition
        // map, then walk only the contiguous run of ranks that overlaps
        // [req_start, req_end). Replaces the old O(needed * P) scan over every
        // rank with O(needed * (log P + k)); the resulting send_bufs are
        // identical (same nodes routed to the same ranks).
        for (const auto& node : needed) {
            uint64_t req_start = node.code.value;
            uint64_t req_end = node.code.value + (1ULL << (DIM * (max_level - node.level)));
            for (int r = find_owner_of_point(req_start); r < mpi_size; ++r) {
                uint64_t r_start = partition_starts[r];
                if (r_start >= req_end)
                    break;  // map is non-decreasing: no later rank can overlap
                uint64_t r_end = (r == mpi_size - 1) ? std::numeric_limits<uint64_t>::max()
                                                     : partition_starts[r + 1];
                if (r == mpi_rank)
                    continue;
                if (std::max(req_start, r_start) < std::min(req_end, r_end))
                    send_bufs[r].push_back(node);
            }
        }
        exchange_data(send_bufs, recv_bufs);
        std::vector<std::vector<Node>> reply_send(mpi_size), reply_recv(mpi_size);
        for (int r = 0; r < mpi_size; ++r) {
            for (const auto& req : recv_bufs[r]) {
                uint64_t req_s = req.code.value;
                uint64_t req_e = req_s + (1ULL << (DIM * (max_level - req.level)));
                auto it = std::lower_bound(leaf_codes.begin(), leaf_codes.end(), req_s);
                if (it != leaf_codes.begin()) {
                    size_t p_idx = std::distance(leaf_codes.begin(), it - 1);
                    uint64_t p_c = leaf_codes[p_idx];
                    uint64_t p_sz = 1ULL << (DIM * (max_level - leaf_levels[p_idx]));
                    if (p_c + p_sz > req_s)
                        reply_send[r].push_back({MortonCode{p_c}, (int)leaf_levels[p_idx]});
                }
                while (it != leaf_codes.end() && *it < req_e) {
                    size_t idx = std::distance(leaf_codes.begin(), it);
                    reply_send[r].push_back({MortonCode{*it}, (int)leaf_levels[idx]});
                    ++it;
                }
            }
        }
        exchange_data(reply_send, reply_recv);
        for (const auto& vec : reply_recv)
            ghost_nodes.insert(ghost_nodes.end(), vec.begin(), vec.end());
        std::sort(ghost_nodes.begin(), ghost_nodes.end());
        ghost_nodes.erase(std::unique(ghost_nodes.begin(), ghost_nodes.end()), ghost_nodes.end());
    }

    // Distinct-neighbour count at or below which point-to-point (Isend/Irecv)
    // is used instead of a neighbourhood collective. For a handful of partners
    // the zero setup cost of non-blocking sends beats building and freeing a
    // distributed-graph communicator; denser neighbour sets amortise the
    // communicator and profit from the topology-aware collective ([IBWG2015]).
    static constexpr int SPARSE_NEIGHBOR_LIMIT = 4;

    // Sparse neighbourhood ghost exchange. sends[r] holds the payload destined
    // for rank r; on return recvs[r] holds what rank r sent us. This moves
    // byte-for-byte the same data a dense MPI_Alltoallv would -- only the
    // routing differs, so the resulting ghost set is unchanged.
    template <typename T>
    void exchange_data(const std::vector<std::vector<T>>& sends,
                       std::vector<std::vector<T>>& recvs) {
        // 1. Per-rank send counts (bytes) and the reciprocal recv counts. This
        //    counts exchange is one int per rank (cheap) and simultaneously
        //    reveals the source neighbour set -- exactly as before.
        std::vector<int> sc(mpi_size), rc(mpi_size);
        for (int i = 0; i < mpi_size; ++i)
            sc[i] = static_cast<int>(sends[i].size() * sizeof(T));
        MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // 2. Neighbour sets: destinations (I send to) and sources (I recv from).
        std::vector<int> dests, srcs;
        for (int r = 0; r < mpi_size; ++r) {
            if (sc[r] > 0)
                dests.push_back(r);
            if (rc[r] > 0)
                srcs.push_back(r);
        }
        for (int r = 0; r < mpi_size; ++r)
            recvs[r].clear();
        for (int r : srcs)
            recvs[r].resize(rc[r] / sizeof(T));

        // The sparse-vs-collective choice must be globally uniform:
        // MPI_Dist_graph_create_adjacent is collective over MPI_COMM_WORLD, so
        // every rank has to agree on whether it runs. Reduce the local partner
        // count to a global max and branch on that -- take the cheap
        // point-to-point path only when *every* rank is sparse, else all ranks
        // build the graph together. (A per-rank predicate would deadlock when
        // ranks straddle the threshold, e.g. empty ranks vs dense interior ones.)
        const int local_neighbors = static_cast<int>(std::max(dests.size(), srcs.size()));
        int n_neighbors = 0;
        MPI_Allreduce(&local_neighbors, &n_neighbors, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        // 3a. Point-to-point fallback for the very sparse case.
        if (n_neighbors <= SPARSE_NEIGHBOR_LIMIT) {
            std::vector<MPI_Request> reqs;
            reqs.reserve(srcs.size() + dests.size());
            for (int r : srcs) {
                reqs.emplace_back();
                MPI_Irecv(recvs[r].data(), rc[r], MPI_BYTE, r, 0, MPI_COMM_WORLD, &reqs.back());
            }
            for (int r : dests) {
                reqs.emplace_back();
                MPI_Isend(const_cast<T*>(sends[r].data()),
                          sc[r],
                          MPI_BYTE,
                          r,
                          0,
                          MPI_COMM_WORLD,
                          &reqs.back());
            }
            if (!reqs.empty())
                MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
            return;
        }

        // 3b. Neighbourhood collective over a distributed-graph communicator.
        //     create_adjacent preserves the source/destination order we pass,
        //     so the flattened buffers below line up with srcs/dests directly.
        MPI_Comm graph;
        MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD,
                                       static_cast<int>(srcs.size()),
                                       srcs.data(),
                                       MPI_UNWEIGHTED,
                                       static_cast<int>(dests.size()),
                                       dests.data(),
                                       MPI_UNWEIGHTED,
                                       MPI_INFO_NULL,
                                       0,
                                       &graph);

        std::vector<int> sendcounts(dests.size()), sdispls(dests.size());
        std::vector<int> recvcounts(srcs.size()), rdispls(srcs.size());
        int soff = 0;
        for (size_t k = 0; k < dests.size(); ++k) {
            sendcounts[k] = sc[dests[k]];
            sdispls[k] = soff;
            soff += sendcounts[k];
        }
        int roff = 0;
        for (size_t k = 0; k < srcs.size(); ++k) {
            recvcounts[k] = rc[srcs[k]];
            rdispls[k] = roff;
            roff += recvcounts[k];
        }
        std::vector<uint8_t> sflat(soff), rflat(roff);
        for (size_t k = 0; k < dests.size(); ++k)
            memcpy(sflat.data() + sdispls[k], sends[dests[k]].data(), sendcounts[k]);

        MPI_Neighbor_alltoallv(sflat.data(),
                               sendcounts.data(),
                               sdispls.data(),
                               MPI_BYTE,
                               rflat.data(),
                               recvcounts.data(),
                               rdispls.data(),
                               MPI_BYTE,
                               graph);

        for (size_t k = 0; k < srcs.size(); ++k)
            memcpy(recvs[srcs[k]].data(), rflat.data() + rdispls[k], recvcounts[k]);

        MPI_Comm_free(&graph);
    }

    template <typename Oracle>
    bool refine_local_pass(const Oracle& oracle) {
        bool changed = false;
        size_t n = leaf_codes.size();
        wksp_counts.resize(n);
        wksp_offsets.resize(n);
#pragma omp parallel for reduction(| : changed)
        for (size_t i = 0; i < n; ++i) {
            int lvl = static_cast<int>(leaf_levels[i]);
            if (lvl < max_level && oracle({MortonCode{leaf_codes[i]}, lvl}, max_level)) {
                wksp_counts[i] = (1ULL << DIM);
                changed = true;
            } else
                wksp_counts[i] = 1;
        }
        if (!changed)
            return false;
        parallel::exclusive_scan(wksp_counts, wksp_offsets, wksp_scan_buffer);
        size_t total = wksp_offsets.back() + wksp_counts.back();
        std::vector<uint64_t> nc(total);
        std::vector<uint8_t> nl(total);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            size_t pos = wksp_offsets[i];
            uint64_t code = leaf_codes[i];
            int lvl = leaf_levels[i];
            if (wksp_counts[i] > 1) {
                int new_lvl = lvl + 1;
                uint64_t shift = (max_level - new_lvl) * DIM;
                for (int k = 0; k < (1 << DIM); ++k) {
                    nc[pos + k] = code | (static_cast<uint64_t>(k) << shift);
                    nl[pos + k] = static_cast<uint8_t>(new_lvl);
                }
            } else {
                nc[pos] = code;
                nl[pos] = lvl;
            }
        }
        leaf_codes = std::move(nc);
        leaf_levels = std::move(nl);
        return true;
    }

    void apply_refinement_from_flags() {
        size_t n = leaf_codes.size();
        wksp_counts.resize(n);
        wksp_offsets.resize(n);
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i)
            wksp_counts[i] = (wksp_flags[i] ? (1ULL << DIM) : 1);
        parallel::exclusive_scan(wksp_counts, wksp_offsets, wksp_scan_buffer);
        size_t new_size = wksp_offsets.back() + wksp_counts.back();
        std::vector<uint64_t> nc(new_size);
        std::vector<uint8_t> nl(new_size);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            size_t pos = wksp_offsets[i];
            if (wksp_flags[i]) {
                int new_lvl = leaf_levels[i] + 1;
                uint64_t shift = (max_level - new_lvl) * DIM;
                for (int k = 0; k < (1 << DIM); ++k) {
                    nc[pos + k] = leaf_codes[i] | (static_cast<uint64_t>(k) << shift);
                    nl[pos + k] = static_cast<uint8_t>(new_lvl);
                }
            } else {
                nc[pos] = leaf_codes[i];
                nl[pos] = leaf_levels[i];
            }
        }
        leaf_codes = std::move(nc);
        leaf_levels = std::move(nl);
    }
};
}  // namespace amr
