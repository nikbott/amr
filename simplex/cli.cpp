/**
 * @file cli.cpp
 * @brief Command-line front end to the simplex engine over SMX2 files (io.hpp).
 *
 * @details Lets MATLAB drive the engine through files:
 *
 *   amr_simplex refine IN OUT      IN: mesh + int "elements". OUT: refined mesh
 *                                  + prolongation
 *   amr_simplex balance IN OUT     IN: mesh + int "elements". OUT: IN's mesh +
 *                                  int "balanced"
 *
 * The prolongation U_new = S U_old keeps every old node and gives new node
 * n_old + k the weighted sum of its parents: int "prolongation_count" (parents
 * per new node), int "prolongation_parents", double "prolongation_weights".
 */

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "io.hpp"
#include "simplex.hpp"

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
    for (const auto& p : r.parents) {
        count.push_back(2);
        parents.insert(parents.end(), p.begin(), p.end());
        weights.insert(weights.end(), {0.5, 0.5});
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
    } else {
        throw std::runtime_error("unknown command \"" + command + "\"");
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    if (args.size() != 3) {
        std::cerr << "usage: amr_simplex {refine|balance} IN.smx OUT.smx\n";
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
