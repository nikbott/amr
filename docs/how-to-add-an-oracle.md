# How to add a refinement oracle

An **oracle** decides, per leaf, whether to refine. It's the one thing you
change to refine on a new criterion. An oracle is just a callable:

```cpp
bool operator()(const Node& node, int max_lvl) const;   // true = refine this leaf
```

Any type with that signature satisfies the `RefinementOracle` concept — no base
class, no registration. See `omp/physics.hpp` for the built-in `CircleOracle`
(2D) and `SphereOracle` (3D).

## Steps

1. **Write the predicate** in `physics.hpp` (next to the existing oracles):

   ```cpp
   class MyOracle {
       Config cfg;
   public:
       explicit MyOracle(const Config& c) : cfg(c) {}
       bool operator()(const Node& node, int max_lvl) const {
           // decode the leaf's box from node.code / node.level, decide:
           return /* refine? */;
       }
   };
   ```

2. **Use it** where the tree is built (`omp/main.cpp`):

   ```cpp
   tree.refine(MyOracle{cfg});   // one pass; the tree passes max_level to the oracle
   ```

   `refine()` runs a single level-pass and returns `true` while it still split
   something, so drive it in a loop (`while (tree.refine(MyOracle{cfg})) {}`) to
   refine to convergence.

3. **Test it** — add a Catch2 case in `omp/tests.cpp` asserting the expected
   leaf set / invariants on a small fixture.

Keep oracles pure and cheap: they run once per leaf per refinement pass. To
refine from external per-leaf data (e.g. a DIC error field), have the oracle
read a sorted array indexed by Morton code rather than recompute geometry.

The same pattern holds in `mpi/` and `cuda/` (the CUDA version is a device
functor). Keep the three in sync — the cross-backend tests check it.
