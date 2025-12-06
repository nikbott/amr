#pragma once

#ifdef USE_CUDA

#include <cstdint>

// Forward declarations
void refineCUDA_pure(uint64_t** d_codes_ptr, int** d_levels_ptr, int* n_nodes,
                     const void* oracle_data, int max_level);

void balanceCUDA_pure(uint64_t** d_codes_ptr, int** d_levels_ptr, int* n_nodes,
                      int max_level, int max_iter);

#endif // USE_CUDA
