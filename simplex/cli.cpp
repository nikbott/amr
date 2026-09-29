/**
 * @file cli.cpp
 * @brief Command-line front end to simplex.hpp over SMX1 files (io.hpp).
 *
 * @details Lets MATLAB drive the engine until the MEX interface exists:
 *
 *   amr_simplex balance IN OUT   OUT holds IN's mesh; its indices are the
 *                                balance closure of IN's element list
 *   amr_simplex refine IN OUT    OUT holds the refined mesh; its indices are
 *                                each new node's two parents, in node order
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "io.hpp"
#include "simplex.hpp"

int main(int argc, char** argv) {
    using namespace amr::simplex;
    const std::vector<std::string> args(argv + 1, argv + argc);
    if (args.size() != 3 || (args[0] != "balance" && args[0] != "refine")) {
        std::cerr << "usage: amr_simplex {balance|refine} IN.smx OUT.smx\n";
        return 2;
    }
    try {
        const auto in = io::read(args[1]);
        if (args[0] == "balance") {
            io::write(args[2], in.mesh, balance_closure(in.mesh, in.indices));
        } else {
            const auto r = refine(in.mesh, in.indices);
            std::vector<Index> parents;
            parents.reserve(2 * r.parents.size());
            for (const auto& p : r.parents)
                parents.insert(parents.end(), p.begin(), p.end());
            io::write(args[2], r.mesh, parents);
        }
    } catch (const std::exception& e) {
        std::cerr << "amr_simplex: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
