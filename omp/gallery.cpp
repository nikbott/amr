/**
 * @file gallery.cpp
 * @brief Emit small, human-inspectable meshes plus a pass/fail summary.
 *
 * `make results` runs this. The unit tests assert the invariants machine-side;
 * this writes the same invariants out as artifacts a person can look at, on
 * meshes small enough to actually see (a few hundred cells, not millions):
 *
 *   mesh_2d_refined.svg    the circle band, before 2:1 balancing
 *   mesh_2d_balanced.svg   after balancing — no cell touches one >1 level apart
 *   mesh_3d_balanced.vtk   the sphere shell octree (open in ParaView, colour by `level`)
 *   index.html             the two SVGs side by side, plus the checks below
 *   summary.txt            the same checks, greppable
 *
 * Each check restates a test assertion in a form the artifact makes visible:
 * partition of unity (cell areas sum to the domain), active-front/reference
 * balance parity (the headline optimization is byte-identical), and the
 * design-by-contract `verify()`.
 */
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "physics.hpp"
#include "tree.hpp"
#include "viz.hpp"

using namespace amr;

namespace {

struct Check {
    std::string name;
    bool passed;
    std::string detail;
};

std::vector<Check> checks;

void record(const std::string& name, bool passed, const std::string& detail) {
    checks.push_back({name, passed, detail});
    std::cout << (passed ? "  [ok]   " : "  [FAIL] ") << std::left << std::setw(34) << name
              << detail << "\n";
}

/// Sum of leaf volumes; must be exactly the unit domain (partition of unity).
template <typename Tree>
double total_volume(const Tree& tree) {
    double v = 0.0;
    for (const auto& node : tree)
        v += std::pow(1.0 / static_cast<double>(1ULL << node.level), Tree::dim);
    return v;
}

/// The active-front balance() must reproduce the whole-mesh balance_ref() exactly.
template <typename Tree, typename Oracle>
bool balance_parity(int max_level, Oracle oracle) {
    Tree reference(max_level), active(max_level);
    while (reference.refine(oracle)) {
    }
    while (active.refine(oracle)) {
    }
    reference.balance_ref();
    active.balance();
    if (reference.size() != active.size())
        return false;
    for (std::size_t i = 0; i < reference.size(); ++i)
        if (reference[i].code.value != active[i].code.value ||
            reference[i].level != active[i].level)
            return false;
    return reference.last_balance_iters == active.last_balance_iters;
}

bool verifies(const std::function<void()>& fn) {
    try {
        fn();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::string join(std::initializer_list<std::string> parts) {
    std::ostringstream os;
    for (const auto& p : parts)
        os << p;
    return os.str();
}

void write_index(const std::filesystem::path& out) {
    std::ofstream f(out / "index.html");
    f << "<!doctype html><meta charset=utf-8><title>AMR validation gallery</title>\n"
         "<style>body{font:15px/1.5 system-ui,sans-serif;margin:2rem "
         "auto;max-width:64rem;padding:0 1rem}"
         "figure{margin:0}img{width:100%;border:1px solid #ccc}"
         "table{border-collapse:collapse;margin-top:1.5rem;width:100%}"
         "td,th{border:1px solid #ccc;padding:.4rem .6rem;text-align:left}"
         ".ok{color:#0a0;font-weight:600}.fail{color:#c00;font-weight:600}"
         ".row{display:grid;grid-template-columns:1fr 1fr;gap:1rem}</style>\n"
         "<h1>AMR validation gallery</h1>\n"
         "<p>Small meshes, emitted by <code>make results</code>, for manual inspection. "
         "Cells are coloured by refinement level (blue = coarse, red = fine).</p>\n"
         "<div class=row>\n"
         "<figure><img src=mesh_2d_refined.svg alt='2D quadtree before balancing'>"
         "<figcaption><b>Before balance.</b> The circle band is refined; neighbouring cells may "
         "differ by more than one level.</figcaption></figure>\n"
         "<figure><img src=mesh_2d_balanced.svg alt='2D quadtree after 2:1 balancing'>"
         "<figcaption><b>After balance.</b> Every face-neighbour is within one level — the 2:1 "
         "grading should be visible as a smooth colour gradient.</figcaption></figure>\n"
         "</div>\n"
         "<p><b>3D:</b> <code>mesh_3d_balanced.vtk</code> — open in ParaView and colour by "
         "<code>level</code> to inspect the sphere-shell octree.</p>\n"
         "<table><tr><th>Check</th><th>Result</th><th>Detail</th></tr>\n";
    for (const auto& c : checks)
        f << "<tr><td>" << c.name << "</td><td class=" << (c.passed ? "ok>PASS" : "fail>FAIL")
          << "</td><td>" << c.detail << "</td></tr>\n";
    f << "</table>\n";
}

void write_summary(const std::filesystem::path& out) {
    std::ofstream f(out / "summary.txt");
    for (const auto& c : checks)
        f << (c.passed ? "PASS  " : "FAIL  ") << std::left << std::setw(34) << c.name << c.detail
          << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path out = (argc > 1) ? argv[1] : "results";
    std::filesystem::create_directories(out);
    std::cout << "=== AMR validation gallery -> " << out << " ===\n";

    // --- 2D: a refined circle band, before and after 2:1 balancing --------------
    Config cfg2;
    cfg2.max_level = 6;
    cfg2.coarse_level = 2;
    cfg2.fine_level = 6;
    cfg2.radius = 0.30;
    cfg2.bandwidth = 0.05;

    Quadtree tree2(cfg2.max_level);
    CircleOracle oracle2(cfg2);
    while (tree2.refine(oracle2)) {
    }
    const std::size_t n_refined = tree2.size();
    viz::write_svg(tree2, (out / "mesh_2d_refined.svg").string());

    tree2.balance();
    viz::write_svg(tree2, (out / "mesh_2d_balanced.svg").string());

    record("2D verify() after balance",
           verifies([&] { tree2.verify(); }),
           join({std::to_string(n_refined),
                 " -> ",
                 std::to_string(tree2.size()),
                 " cells, ",
                 std::to_string(tree2.last_balance_iters),
                 " balance passes"}));

    const double vol2 = total_volume(tree2);
    record("2D partition of unity",
           std::abs(vol2 - 1.0) < 1e-12,
           join({"sum of cell areas = ", std::to_string(vol2)}));

    record("2D active-front == balance_ref",
           balance_parity<Quadtree>(cfg2.max_level, oracle2),
           "byte-identical leaf array and pass count");

    // --- 3D: the sphere-shell octree -------------------------------------------
    Config cfg3;
    cfg3.max_level = 5;
    cfg3.coarse_level = 2;
    cfg3.fine_level = 5;
    cfg3.radius = 0.30;
    cfg3.bandwidth = 0.05;

    Octree tree3(cfg3.max_level);
    SphereOracle oracle3(cfg3);
    while (tree3.refine(oracle3)) {
    }
    tree3.balance();
    viz::write_vtk(tree3, (out / "mesh_3d_balanced.vtk").string());

    record("3D verify() after balance",
           verifies([&] { tree3.verify(); }),
           join({std::to_string(tree3.size()),
                 " cells, ",
                 std::to_string(tree3.last_balance_iters),
                 " balance passes"}));

    const double vol3 = total_volume(tree3);
    record("3D partition of unity",
           std::abs(vol3 - 1.0) < 1e-12,
           join({"sum of cell volumes = ", std::to_string(vol3)}));

    record("3D active-front == balance_ref",
           balance_parity<Octree>(cfg3.max_level, oracle3),
           "byte-identical leaf array and pass count");

    write_index(out);
    write_summary(out);

    const bool all_passed =
        std::all_of(checks.begin(), checks.end(), [](const Check& c) { return c.passed; });
    std::cout << "\nWrote " << (out / "index.html") << " — open it to inspect the meshes.\n";
    std::cout << (all_passed ? "All checks passed.\n" : "SOME CHECKS FAILED (see summary.txt).\n");
    return all_passed ? 0 : 1;
}
