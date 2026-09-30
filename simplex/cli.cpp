/**
 * @file cli.cpp
 * @brief Command-line front end to the unstructured engine over SMX2 files
 * (io.hpp).
 *
 * @details Lets MATLAB drive the engine through files:
 *
 *   amr_simplex refine IN OUT      IN: mesh + int "elements". OUT: refined mesh
 *                                  + prolongation
 *   amr_simplex balance IN OUT     IN: mesh + int "elements". OUT: IN's mesh +
 *                                  int "balanced"
 *   amr_simplex adapt IN OUT       IN: mesh + double "error", "ratio", "params"
 *                                  = [theta, max_refine_fraction,
 *                                  min_element_length, max_elements,
 *                                  min_growth_fraction]. OUT: the refined mesh
 *                                  (empty if nothing was refined), int "status"
 *                                  (AdaptStatus), the element sets "flagged",
 *                                  "candidates", "marked", "selected",
 *                                  "balanced", double "seconds" (marking,
 *                                  balance, refinement), and the prolongation
 *   amr_simplex structured IN OUT  IN: empty mesh + double "dims", "origin",
 *                                  int "nodes". OUT: the seed mesh
 *
 * The prolongation U_new = S U_old keeps every old node and gives new node
 * n_old + k the weighted sum of its parents: int "prolongation_count" (parents
 * per new node), int "prolongation_parents", double "prolongation_weights".
 */

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "adapt.hpp"
#include "io.hpp"
#include "simplex.hpp"
#include "structured.hpp"

namespace {

using namespace amr::simplex;

template <class T>
const std::vector<T>& field(const std::map<std::string, std::vector<T>>& fields, const char* name) {
    const auto it = fields.find(name);
    if (it == fields.end())
        throw std::runtime_error(std::string("missing input field \"") + name + "\"");
    return it->second;
}

void add_prolongation(const Refinement& r,
                      std::map<std::string, std::vector<Index>>& ints,
                      std::map<std::string, std::vector<double>>& doubles) {
    auto& count = ints["prolongation_count"];
    auto& parents = ints["prolongation_parents"];
    auto& weights = doubles["prolongation_weights"];
    for (const auto& row : r.prolongation) {
        count.push_back(static_cast<Index>(row.parent.size()));
        parents.insert(parents.end(), row.parent.begin(), row.parent.end());
        weights.insert(weights.end(), row.weight.begin(), row.weight.end());
    }
}

void run(const std::string& command, const std::string& in_path, const std::string& out_path) {
    const auto in = io::read(in_path);
    std::map<std::string, std::vector<Index>> ints;
    std::map<std::string, std::vector<double>> doubles;
    if (command == "refine") {
        const auto r = refine(in.mesh, field(in.ints, "elements"));
        add_prolongation(r, ints, doubles);
        io::write(out_path, r.mesh, ints, doubles);
    } else if (command == "balance") {
        ints["balanced"] = balance_closure(in.mesh, field(in.ints, "elements"));
        io::write(out_path, in.mesh, ints, doubles);
    } else if (command == "adapt") {
        const auto& v = field(in.doubles, "params");
        if (v.size() != 5)
            throw std::runtime_error("\"params\" needs 5 values");
        AdaptParams p;
        p.marking.theta = v[0];
        p.marking.max_refine_fraction = v[1];
        p.marking.min_element_length = v[2];
        p.marking.max_elements = v[3];
        p.min_growth_fraction = v[4];
        const auto a = adapt(in.mesh, field(in.doubles, "error"), field(in.doubles, "ratio"), p);
        ints["status"] = {static_cast<Index>(a.status)};
        ints["flagged"] = a.marking.flagged;
        ints["candidates"] = a.marking.candidates;
        ints["marked"] = a.marking.marked;
        ints["selected"] = a.marking.selected;
        ints["balanced"] = a.balanced;
        doubles["seconds"] = {a.seconds.begin(), a.seconds.end()};
        const bool refined = a.status == AdaptStatus::refined || a.status == AdaptStatus::stagnated;
        if (refined)
            add_prolongation(a.refinement, ints, doubles);
        Mesh unchanged;  // the caller keeps its mesh when nothing was refined
        unchanged.dim = in.mesh.dim;
        io::write(out_path, refined ? a.refinement.mesh : unchanged, ints, doubles);
    } else if (command == "structured") {
        io::write(
            out_path,
            structured(
                field(in.doubles, "dims"), field(in.ints, "nodes"), field(in.doubles, "origin")));
    } else {
        throw std::runtime_error("unknown command \"" + command + "\"");
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    const std::vector<std::string> commands{"refine", "balance", "adapt", "structured"};
    if (args.size() != 3 ||
        std::find(commands.begin(), commands.end(), args[0]) == commands.end()) {
        std::cerr << "usage: amr_simplex {refine|balance|adapt|structured} IN.smx OUT.smx\n";
        return 2;
    }
    try {
        run(args[0], args[1], args[2]);
    } catch (const std::exception& e) {
        std::cerr << "amr_simplex: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
